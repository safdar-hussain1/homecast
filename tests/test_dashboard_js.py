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


# --- The page must survive a city with a shorter feature set ---------------
#
# The template was written around Gurgaon's thirteen features and read
# several payload fields unguarded. Two of those were genuinely dangerous:
# `DATA.ranges.luxury_score` at module scope aborted the ENTIRE script for a
# city with no amenity score (blank page, every figure left as a dash), and
# `feature_order.map(k => v[k])` yielded `undefined` for any feature the page
# could not build -- which compares false against every tree threshold, so
# the walker returned a confident, plausible, wrong price instead of failing.
# These run the real extracted JS against reduced payloads.

def _reduced_payload(base: dict, keep: list[str], *, ranges: dict,
                     drop_societies: bool = True) -> dict:
    p = json.loads(json.dumps(base))
    p["feature_order"] = keep
    p["feature_importances"] = {k: 1.0 / len(keep) for k in keep}
    p["ranges"] = ranges
    p["narrative"] = False
    if drop_societies:
        p["encodings"]["society_ppsf"] = {
            "__global__": base["encodings"]["society_ppsf"]["__global__"]}
    return p


def _eval_js(js_block: str, expr: str, tmp_path: Path) -> str:
    script = js_block + "\nprocess.stdout.write(String(" + expr + "));\n"
    path = tmp_path / "probe.js"
    path.write_text(script, encoding="utf-8")
    r = subprocess.run([NODE, str(path)], capture_output=True, text=True, timeout=60)
    return r.stdout if r.returncode == 0 else "ERROR:" + r.stderr


@pytest.fixture()
def base_payload():
    assert MODEL_JSON.exists(), f"{MODEL_JSON} missing -- run `homecast train --city gurgaon`"
    return json.loads(MODEL_JSON.read_text())


@pytest.mark.skipif(NODE is None, reason=_skip_reason)
def test_page_script_runs_for_a_city_with_no_amenity_score(base_payload, tmp_path):
    """`ranges.luxury_score` was read unguarded at module scope. Its absence
    threw before a single line of the page had run."""
    keep = ["area", "bedrooms", "sector_ppsf", "sector_ppsf_mean",
            "sector_ppsf_std", "sector_count"]
    payload = _reduced_payload(base_payload, keep,
                               ranges={"area": [200.0, 5000.0], "bedrooms": [1.0, 6.0]})
    out = _eval_js(_extract_feature_row_js(payload),
                   "featureRow({sector: Object.keys(SP)[0], area: 1200, bedrooms: 3}).length",
                   tmp_path)
    assert out == str(len(keep)), out


@pytest.mark.skipif(NODE is None, reason=_skip_reason)
def test_feature_row_builds_the_two_optional_extra_features(base_payload, tmp_path):
    """amenity_count/is_resale exist only in some feeds. The page must be able
    to build them, and must apply the same -1 'not stated' code Python does."""
    keep = ["area", "bedrooms", "sector_ppsf", "sector_ppsf_mean",
            "sector_ppsf_std", "sector_count", "amenity_count", "is_resale"]
    payload = _reduced_payload(base_payload, keep,
                               ranges={"area": [200.0, 5000.0], "bedrooms": [1.0, 6.0]})
    js = _extract_feature_row_js(payload)
    q = "{sector: Object.keys(SP)[0], area: 1200, bedrooms: 3, amenity: 7, resale: '1'}"
    assert _eval_js(js, f"featureRow({q}).slice(-2).join(',')", tmp_path) == "7,1"
    q_blank = "{sector: Object.keys(SP)[0], area: 1200, bedrooms: 3}"
    assert _eval_js(js, f"featureRow({q_blank}).slice(-2).join(',')", tmp_path) == "-1,-1"


@pytest.mark.skipif(NODE is None, reason=_skip_reason)
def test_an_unbuildable_feature_throws_instead_of_predicting_undefined(base_payload, tmp_path):
    """The silent-failure guard. `undefined <= threshold` is false, so a
    feature the page cannot build would send every node down its right branch
    and still print a confident price. It must throw instead."""
    payload = _reduced_payload(base_payload, ["area", "not_a_real_feature"],
                               ranges={"area": [200.0, 5000.0]})
    out = _eval_js(_extract_feature_row_js(payload),
                   "(() => { try { featureRow({sector:'x', area:1200}); return 'NO THROW'; }"
                   " catch (e) { return e.message; } })()", tmp_path)
    assert "not_a_real_feature" in out, out


@pytest.mark.skipif(NODE is None, reason=_skip_reason)
def test_reduced_payload_still_matches_python_build_features(base_payload, tmp_path):
    """Parity, but for a city that trained on a subset. The JS picks its
    columns out of feature_order, so a divergence here means the two sides
    disagree about what the shorter vector contains."""
    keep = ["area", "bedrooms", "sector_ppsf", "sector_ppsf_mean",
            "sector_ppsf_std", "sector_count"]
    payload = _reduced_payload(base_payload, keep,
                               ranges={"area": [200.0, 5000.0], "bedrooms": [1.0, 6.0]})
    enc_json = payload["encodings"]
    encoders = Encoders(
        sector_ppsf=enc_json["sector_ppsf"], sector_ppsf_mean=enc_json["sector_ppsf_mean"],
        sector_ppsf_std=enc_json["sector_ppsf_std"], sector_count=enc_json["sector_count"],
        society_ppsf=enc_json["society_ppsf"])
    df = pd.read_csv(CLEAN_CSV).sample(n=25, random_state=5).reset_index(drop=True)
    want = build_features(df, encoders, keep).to_numpy(dtype=float)

    queries = [{"sector": r["sector"], "area": float(r["area"]),
                "bedrooms": float(r["bedrooms"])} for _, r in df.iterrows()]
    got = np.array(_run_feature_rows_under_node(
        _extract_feature_row_js(payload), queries, tmp_path), dtype=float)
    assert got.shape == want.shape
    assert np.allclose(got, want, rtol=0, atol=1e-6)


# --- The area label is a factual claim, so it follows the payload ----------
#
# Carpet, built-up and super-built-up areas differ by 25-35% for the same
# flat. Before this, the template hardcoded "Built-up area" for every city,
# so a feed that never stated its basis -- which three of the private cities
# genuinely do not -- asserted a specific basis in the one place a user
# reads. These run the template's real label block under Node.

def _extract_area_label_js(payload: dict) -> str:
    """Pull the AREA_BASIS/AREA_LABEL const block verbatim out of the template
    and give it a DATA to read, so the test exercises the shipping logic
    rather than a copy of it."""
    html = TEMPLATE.read_text(encoding="utf-8")
    start = html.index("const AREA_BASIS")
    end = html.index("\n", html.index("const AREA_LABEL_LC"))
    return f"const DATA = {json.dumps(payload)};\n" + html[start:end]


@pytest.mark.skipif(NODE is None, reason=_skip_reason)
@pytest.mark.parametrize("basis,expected", [
    ("superbuiltup", "Super built-up area"),
    ("builtup", "Built-up area"),
    ("carpet", "Carpet area"),
    ("unknown", "Area"),
])
def test_area_label_follows_the_declared_basis(basis, expected, tmp_path):
    js = _extract_area_label_js({"area_basis": basis})
    assert _eval_js(js, "AREA_LABEL", tmp_path) == expected


@pytest.mark.skipif(NODE is None, reason=_skip_reason)
def test_unknown_basis_never_borrows_the_built_up_wording(tmp_path):
    """The whole point: "unknown" must not render as any specific basis."""
    js = _extract_area_label_js({"area_basis": "unknown"})
    label = _eval_js(js, "AREA_LABEL", tmp_path)
    assert "built-up" not in label.lower()
    assert "carpet" not in label.lower()


@pytest.mark.skipif(NODE is None, reason=_skip_reason)
def test_payload_without_a_basis_keeps_the_original_wording(tmp_path):
    """Gurgaon's payload has no area_basis key. The public page's label must
    be exactly what it always was."""
    js = _extract_area_label_js({"city": "Gurgaon"})
    assert _eval_js(js, "AREA_LABEL", tmp_path) == "Built-up area"
    assert _eval_js(js, "String(AREA_BASIS)", tmp_path) == "null"


@pytest.mark.skipif(NODE is None, reason=_skip_reason)
def test_an_unrecognised_basis_degrades_to_the_neutral_noun(tmp_path):
    """A basis this template has no wording for must not fall through to
    "Built-up area" -- the neutral noun is the safe answer."""
    js = _extract_area_label_js({"area_basis": "some_future_basis"})
    assert _eval_js(js, "AREA_LABEL", tmp_path) == "Area"
