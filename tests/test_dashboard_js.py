"""I5: the dashboard's in-browser feature builder has never been executed by
this test suite -- everything else here tests the Python side only. This
extracts the actual `predict`/`featureRow` functions straight out of
`scripts/dashboard_template.html` (not a hand-copied re-implementation, so it
can't silently drift from what ships), runs them under Node against real
sector/society encodings, and asserts the resulting feature vectors are
bit-for-bit what `homecast.features.build_features` computes for the same
rows. A regression in the JS-side society/sector fallback chain (the kind
that caused the pre-Phase-2 global-median bug) would fail this test even
though it is invisible to every purely-Python test in this suite.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from homecast.features import Encoders, FEATURE_COLUMNS, build_features

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "scripts" / "dashboard_template.html"
MODEL_JSON = ROOT / "models" / "gurgaon" / "model.json"
CLEAN_CSV = ROOT / "data" / "gurgaon" / "processed" / "listings_clean.csv"

NODE = shutil.which("node")
_skip_reason = "node is not installed; cannot verify dashboard JS featureRow() against Python build_features()"


def _extract_feature_row_js(payload: dict) -> str:
    """Pull the `const DATA = ...` through `estimateFor` block verbatim out of
    the template (this is `predict`, `featureRow`, `cmpSector`, `estimateFor`
    and their supporting consts -- no DOM access, so it runs standalone under
    plain Node) and splice in a real exported payload in place of the
    `/*__DATA__*/` build-time placeholder."""
    html = TEMPLATE.read_text(encoding="utf-8")
    start_marker = "const DATA = /*__DATA__*/;"
    end_marker = "/* ── self-test hook"
    start = html.index(start_marker)
    end = html.index(end_marker, start)
    assert start != -1 and end != -1, "dashboard_template.html markers moved -- update this test"
    block = html[start:end]
    return block.replace(start_marker, f"const DATA = {json.dumps(payload)};")


def _query_from_row(row: pd.Series) -> dict:
    def none_if_missing(v):
        return None if pd.isna(v) else v

    return {
        "sector": row["sector"],
        "type": row["property_type"],
        "bedrooms": float(row["bedrooms"]),
        "bathrooms": float(row["bathrooms"]),
        "area": float(row["area"]),
        "furnishing": row["furnishing_type"],
        "luxury": float(row["luxury_score"]),
        "age": none_if_missing(row["age_possession"]),
        "society": none_if_missing(row["society"]),
        "balcony": none_if_missing(row["balcony"]),
    }


def _run_feature_rows_under_node(js_block: str, queries: list[dict], tmp_path: Path) -> list[list[float]]:
    driver = js_block + """
const fs = require('fs');
const queries = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
process.stdout.write(JSON.stringify(queries.map(featureRow)));
"""
    script_path = tmp_path / "feature_row_driver.js"
    queries_path = tmp_path / "queries.json"
    script_path.write_text(driver, encoding="utf-8")
    queries_path.write_text(json.dumps(queries), encoding="utf-8")
    result = subprocess.run([NODE, str(script_path), str(queries_path)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, f"node failed:\n{result.stderr}"
    return json.loads(result.stdout)


@pytest.mark.skipif(NODE is None, reason=_skip_reason)
def test_dashboard_js_feature_row_matches_python_build_features():
    assert MODEL_JSON.exists(), f"{MODEL_JSON} missing -- run `homecast train --city gurgaon` first"
    assert CLEAN_CSV.exists(), f"{CLEAN_CSV} missing -- run `homecast clean --city gurgaon` first"

    payload = json.loads(MODEL_JSON.read_text())
    enc_json = payload["encodings"]
    encoders = Encoders(
        sector_ppsf=enc_json["sector_ppsf"],
        sector_ppsf_mean=enc_json["sector_ppsf_mean"],
        sector_ppsf_std=enc_json["sector_ppsf_std"],
        sector_count=enc_json["sector_count"],
        society_ppsf=enc_json["society_ppsf"],
    )

    df = pd.read_csv(CLEAN_CSV)
    # A broad random sample of real rows, plus every row with a missing
    # society/age (rare in this dataset but exactly the fallback-chain edge
    # case I5 exists to cover) so the comparison isn't limited to the common
    # "everything present" path.
    sample = pd.concat([
        df.sample(n=60, random_state=11),
        df[df["society"].isna()],
        df[df["age_possession"].isna()].head(5),
    ]).drop_duplicates().reset_index(drop=True)

    want = build_features(sample, encoders)[FEATURE_COLUMNS].to_numpy(dtype=float)

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        js_block = _extract_feature_row_js(payload)
        queries = [_query_from_row(row) for _, row in sample.iterrows()]
        got = _run_feature_rows_under_node(js_block, queries, Path(td))

    got = np.array(got, dtype=float)
    assert got.shape == want.shape
    assert np.allclose(got, want, rtol=0, atol=1e-6), (
        "dashboard_template.html's featureRow() disagrees with "
        "homecast.features.build_features() on at least one real row -- "
        "the JS-side fallback chain (society/sector encoding, unknown "
        "labels) has drifted from the Python model it's supposed to mirror")
