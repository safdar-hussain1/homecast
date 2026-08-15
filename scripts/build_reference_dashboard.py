#!/usr/bin/env python3
"""Build the rate-reference page for a market that has no model.

Some markets cannot support a trained model — Amaravathi has single-digit live
listings and no dataset, so the honest product is a reference sheet built from
published government rates, not a prediction. This script bakes such a rate
table into `scripts/reference_template.html` exactly the way
`build_dashboard.py` bakes a `model.json` into the model dashboard.

    python scripts/build_reference_dashboard.py [--rates PATH] [--out PATH]

Output always lands under `private/` and the script refuses to write anywhere
under `docs/`: the rate tables are hand-maintained private data and must never
reach the published site.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PLACEHOLDER = "/*__DATA__*/"
ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "scripts" / "reference_template.html"
DEFAULT_RATES = ROOT / "private" / "amaravathi" / "rates.toml"
DEFAULT_OUT = ROOT / "private" / "dashboards" / "amaravathi_ap.html"
DOCS_DIR = ROOT / "docs"


def _payload(rates_path: Path) -> dict:
    # Imported lazily so the script still runs with a bare checkout.
    sys.path.insert(0, str(ROOT / "src"))
    from homecast import reference as ref

    table = ref.load_rate_table(rates_path)
    return {
        "city_display": ref.CITY_DISPLAY,
        "last_updated": table.last_updated,
        "staleness_days": ref.STALENESS_DAYS,
        "entries": [
            {
                "zone": e.zone,
                "property_type": e.property_type,
                "rate_low_per_sqft": e.rate_low_per_sqft,
                "rate_high_per_sqft": e.rate_high_per_sqft,
                "source": e.source,
                "source_date": e.source_date,
            }
            for e in table.entries
        ],
    }


def build(rates: Path = DEFAULT_RATES,
          output: Path = DEFAULT_OUT,
          template: Path = TEMPLATE) -> Path:
    output = output.resolve()
    # A rate table is private, hand-maintained data. Publishing it would put
    # unverifiable figures on the public site under the project's name.
    if output == DOCS_DIR or DOCS_DIR in output.parents:
        raise ValueError(
            f"refusing to write a rate-reference page into {DOCS_DIR} — rate "
            f"tables are private data and must not be published")
    if not template.exists():
        raise FileNotFoundError(f"reference template missing: {template}")
    if not rates.exists():
        raise FileNotFoundError(
            f"no rate table at {rates} — this market is priced from a "
            f"hand-maintained table, not a trained model. Copy "
            f"config/amaravathi.rates.example.toml to {rates} and replace the "
            f"placeholder figures with real, dated, sourced rates.")

    html = template.read_text(encoding="utf-8")
    if html.count(PLACEHOLDER) != 1:
        raise ValueError(
            f"expected exactly one {PLACEHOLDER} placeholder in {template}, "
            f"found {html.count(PLACEHOLDER)}")

    payload = _payload(rates)
    for key in ("city_display", "last_updated", "staleness_days", "entries"):
        if key not in payload:
            raise ValueError(f"rate payload is missing the '{key}' key")
    if not payload["entries"]:
        raise ValueError(f"{rates} produced no rate entries — nothing to show")

    data = json.dumps(payload, allow_nan=False, separators=(",", ":"))
    if PLACEHOLDER in data:
        raise ValueError("payload contains the placeholder text; refusing to build")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html.replace(PLACEHOLDER, data), encoding="utf-8")
    return output


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Build the rate-reference page for a market with no model")
    p.add_argument("--rates", default=str(DEFAULT_RATES))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    a = p.parse_args(argv)
    try:
        out = build(Path(a.rates), Path(a.out))
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {out} ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
