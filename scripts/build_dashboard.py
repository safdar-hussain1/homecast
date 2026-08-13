#!/usr/bin/env python3
"""Inline a city's exported model payload into the dashboard template.

The template carries a single `/*__DATA__*/` placeholder inside its script
block; this script drops the minified `model.json` in its place and writes the
result to `docs/index.html`. Nothing else is templated, so the page in `docs/`
is always a byte-for-byte function of the template plus the payload.

    python scripts/build_dashboard.py [--city gurgaon]

It reads only files, so it needs no PYTHONPATH and no installed package —
`homecast train` (or `homecast export-dashboard`) must have written
`models/<city>/model.json` first.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PLACEHOLDER = "/*__DATA__*/"
ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "scripts" / "dashboard_template.html"
OUTPUT = ROOT / "docs" / "index.html"


def build(city: str = "gurgaon",
          template: Path = TEMPLATE,
          output: Path = OUTPUT) -> Path:
    model_path = ROOT / "models" / city / "model.json"

    if not template.exists():
        raise FileNotFoundError(f"dashboard template missing: {template}")
    if not model_path.exists():
        raise FileNotFoundError(
            f"model payload missing: {model_path} — run "
            f"`homecast train --city {city}` (or `homecast export-dashboard`) first")

    html = template.read_text(encoding="utf-8")
    if html.count(PLACEHOLDER) != 1:
        raise ValueError(
            f"expected exactly one {PLACEHOLDER} placeholder in {template}, "
            f"found {html.count(PLACEHOLDER)}")

    payload = json.loads(model_path.read_text(encoding="utf-8"))
    for key in ("city", "model", "feature_order", "feature_importances",
                "encodings", "band", "ranges", "metrics", "residual_hist",
                "sectors", "sample"):
        if key not in payload:
            raise ValueError(f"{model_path} is missing the '{key}' key")
    for enc_key in ("furnishing", "age", "balcony", "sector_ppsf",
                    "sector_ppsf_mean", "sector_ppsf_std", "sector_count",
                    "society_ppsf"):
        if enc_key not in payload["encodings"]:
            raise ValueError(f"{model_path} encodings is missing the '{enc_key}' key")
    for metric_key in ("model", "model_no_society", "baseline_sector",
                       "baseline_global", "n", "n_splits"):
        if metric_key not in payload["metrics"]:
            raise ValueError(f"{model_path} metrics is missing the '{metric_key}' key")

    data = json.dumps(payload, allow_nan=False, separators=(",", ":"))
    if PLACEHOLDER in data:
        raise ValueError("payload contains the placeholder text; refusing to build")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html.replace(PLACEHOLDER, data), encoding="utf-8")
    return output


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Build docs/index.html from the dashboard template")
    p.add_argument("--city", default="gurgaon")
    a = p.parse_args(argv)
    try:
        out = build(a.city)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {out.resolve()} ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
