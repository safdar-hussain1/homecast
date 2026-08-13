#!/usr/bin/env python3
"""Inline a city's exported model payload into the dashboard template.

The template carries a single `/*__DATA__*/` placeholder inside its script
block; this script drops the minified `model.json` in its place. Nothing
else is templated, so the built page is always a byte-for-byte function of
the template plus the payload.

    python scripts/build_dashboard.py [--city gurgaon]

Where the result is written is NOT a free choice of the caller's — it is
resolved from the city registry (`homecast.cities.get_city`), the same
authority everything else in this project uses for `public`/`models_dir`,
so a private city cannot land in the public dashboard by a path typo or a
missing file being coincidentally absent:

- a **public** city (`get_city(...).public is True`) builds to `docs/index.html`;
- a **private** city builds to `private/dashboards/<key>.html` instead, and
  this script refuses outright — before touching the filesystem — if asked
  to write a private city anywhere under `docs/`.

This script inserts `src/` onto `sys.path` itself (see below), so it still
needs no `pip install -e .` and no PYTHONPATH env var to run from a plain
checkout — `homecast train` (or `homecast export-dashboard`) must have
written `models/<city>/model.json` (or `private/models/<city>/model.json`
for a private city) first.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PLACEHOLDER = "/*__DATA__*/"
ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "scripts" / "dashboard_template.html"
DOCS_DIR = ROOT / "docs"
PUBLIC_OUTPUT = DOCS_DIR / "index.html"
PRIVATE_DASHBOARDS_DIR = ROOT / "private" / "dashboards"

sys.path.insert(0, str(ROOT / "src"))
from homecast.cities import get_city  # noqa: E402  (after the sys.path insert, by design)


def _default_output(city_key: str) -> Path:
    c = get_city(city_key)
    return PUBLIC_OUTPUT if c.public else PRIVATE_DASHBOARDS_DIR / f"{c.key}.html"


def _guard_private_city_never_writes_to_docs(city_key: str, output: Path) -> None:
    c = get_city(city_key)
    if c.public:
        return
    try:
        output.resolve().relative_to(DOCS_DIR.resolve())
    except ValueError:
        return  # output is not under docs/ -- fine
    raise ValueError(
        f"refusing to build private city '{c.key}' to {output} -- private "
        f"cities (public=False in the city registry) must never be written "
        f"under {DOCS_DIR}, which is the public GitHub Pages build. This is "
        f"a hard refusal, not a default that can be overridden by an "
        f"--output-style path: pass an output outside docs/ instead.")


def build(city: str = "gurgaon",
          template: Path = TEMPLATE,
          output: Path | None = None) -> Path:
    c = get_city(city)
    model_path = c.models_dir / "model.json"
    if output is None:
        output = _default_output(city)
    _guard_private_city_never_writes_to_docs(city, output)

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
    p = argparse.ArgumentParser(
        description="Build the dashboard from the template (docs/index.html "
                    "for a public city, private/dashboards/<key>.html for a private one)")
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
