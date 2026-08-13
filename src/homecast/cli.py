"""HomeCast command line: clean, train, evaluate, predict, export-dashboard."""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from contextlib import contextmanager

import joblib

from homecast.cities import get_city
from homecast.cleaning import clean_city
from homecast.export import export_city, write_export
from homecast.model import evaluate as evaluate_cv
from homecast.model import train_final
from homecast.valuation import Query, estimate


@contextmanager
def _quiet_joblib():
    """joblib 1.5 + numpy 2.5 emit a DeprecationWarning from inside joblib's
    own pickler; it is not actionable by a user of this CLI."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="joblib")
        yield


def _load_processed(city):
    import pandas as pd
    if not city.processed_path.exists():
        _do_clean(city)
    return pd.read_csv(city.processed_path)


def _load_fitted(city):
    """Load the trained model, failing with an actionable message if absent."""
    mpath = city.models_dir / "model.joblib"
    if not mpath.exists():
        raise ValueError(f"No trained model at {mpath} — run `homecast train` first")
    with _quiet_joblib():
        return joblib.load(mpath)


def _do_clean(city) -> None:
    cleaned, log = clean_city(city.key)
    city.processed_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(city.processed_path, index=False)
    for line in log:
        print(f"  - {line}")
    print(f"Saved {len(cleaned)} listings -> {city.processed_path}")


def _fmt_metrics(m: dict) -> str:
    rows = [("HomeCast model", m["model"]), ("Sector-median baseline", m["baseline_sector"]),
            ("Global-median baseline", m["baseline_global"])]
    out = [f"{'':26s} {'MAE (lakh)':>10s} {'MAPE %':>8s} {'R2':>6s}"]
    for name, r in rows:
        out.append(f"{name:26s} {r['mae_lakh']:10.1f} {r['mape_pct']:8.1f} {r['r2']:6.3f}")
    return "\n".join(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="homecast",
                                description="Residential property price intelligence")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("clean", "train", "evaluate", "export-dashboard"):
        sub.add_parser(name).add_argument("--city", default="gurgaon")
    pr = sub.add_parser("predict")
    pr.add_argument("--city", default="gurgaon")
    pr.add_argument("--sector", required=True)
    pr.add_argument("--type", required=True, dest="ptype")
    pr.add_argument("--bhk", type=int, required=True)
    pr.add_argument("--bath", type=int, required=True)
    pr.add_argument("--area", type=float, required=True)
    pr.add_argument("--furnishing", default="semi-furnished")
    pr.add_argument("--luxury", type=int, default=50)
    pr.add_argument("--age", default=None)
    pr.add_argument("--society", default=None,
                    help="optional; unrecognised names fall back to the "
                         "city-wide median instead of erroring")
    pr.add_argument("--balcony", default=None)
    a = p.parse_args(argv)

    try:
        city = get_city(a.city)
        if a.cmd == "clean":
            _do_clean(city)
        elif a.cmd == "train":
            df = _load_processed(city)
            fitted = train_final(df)
            city.models_dir.mkdir(parents=True, exist_ok=True)
            with _quiet_joblib():
                joblib.dump(fitted, city.models_dir / "model.joblib")
            write_export(export_city(fitted, df, city), city.models_dir / "model.json")
            (city.models_dir / "metrics.json").write_text(
                json.dumps({k: fitted.metrics[k] for k in
                            ("model", "baseline_sector", "baseline_global",
                             "n", "n_splits", "params")}, indent=2))
            print(_fmt_metrics(fitted.metrics))
            print(f"Artifacts -> {city.models_dir}")
        elif a.cmd == "evaluate":
            print(_fmt_metrics(evaluate_cv(_load_processed(city))))
        elif a.cmd == "export-dashboard":
            fitted = _load_fitted(city)
            write_export(export_city(fitted, _load_processed(city), city),
                         city.models_dir / "model.json")
            print(f"Wrote {city.models_dir / 'model.json'}")
        elif a.cmd == "predict":
            # _load_fitted unpickles model.joblib. That is safe here because
            # the file is only ever produced by this machine's own
            # `homecast train` run (gitignored, never downloaded).
            fitted = _load_fitted(city)
            q = Query(sector=a.sector, property_type=a.ptype, bedrooms=a.bhk,
                      bathrooms=a.bath, area=a.area, furnishing=a.furnishing,
                      luxury_score=a.luxury, age=a.age, society=a.society,
                      balcony=a.balcony)
            e = estimate(fitted, q)
            print(f"Estimated price: Rs {e['price_cr']:.2f} Cr "
                  f"(range {e['lo_cr']:.2f} - {e['hi_cr']:.2f} Cr)")
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
