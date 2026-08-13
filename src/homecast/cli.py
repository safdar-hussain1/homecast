"""HomeCast command line: clean, train, evaluate, predict, export-dashboard."""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

import joblib

from homecast.cities import get_city
from homecast.cleaning import clean_city
from homecast.export import export_city, write_export
from homecast.ingest import ingest_city
from homecast.model import MODELS
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


def _do_ingest(city, config_path: str) -> None:
    cleaned, log = ingest_city(city.raw_path, Path(config_path),
                               existing_processed_path=city.processed_path)
    city.processed_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(city.processed_path, index=False)
    for line in log:
        print(f"  - {line}")
    print(f"Saved {len(cleaned)} listings -> {city.processed_path}")


def _fmt_metrics(m: dict) -> str:
    rows = [("HomeCast model", m["model"])]
    if "model_no_society" in m:
        # honest second number: what a caller who can't supply a society
        # actually gets (see homecast.model.SOCIETY_MASK_FRACTION)
        rows.append(("  ...without a society given", m["model_no_society"]))
    rows += [("Sector-median baseline", m["baseline_sector"]),
             ("Global-median baseline", m["baseline_global"])]
    out = [f"{'':28s} {'MAE (lakh)':>10s} {'MAPE %':>8s} {'R2':>6s}"]
    for name, r in rows:
        out.append(f"{name:28s} {r['mae_lakh']:10.1f} {r['mape_pct']:8.1f} {r['r2']:6.3f}")
    if m.get("baseline_served"):
        # Printed on every train/evaluate for this city, not only on
        # `predict` -- whoever reads this table must not walk away thinking
        # "HomeCast model" above is what a caller actually gets.
        out.append("NOTE: this model does not beat the sector-median baseline "
                   "-- predict/export present the baseline as the estimate.")
    return "\n".join(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="homecast",
                                description="Residential property price intelligence")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("clean", "export-dashboard"):
        sub.add_parser(name).add_argument("--city", default="gurgaon")
    for name in ("train", "evaluate"):
        sp = sub.add_parser(name)
        sp.add_argument("--city", default="gurgaon")
        # "accurate" (ExtraTrees) is CLI-only: it cannot be exported to the
        # browser dashboard (see homecast.export._require_exportable).
        sp.add_argument("--model", default="default", choices=sorted(MODELS))
    ig = sub.add_parser("ingest")
    ig.add_argument("--city", required=True)
    ig.add_argument("--config", required=True,
                    help="path to the city's ingestion TOML config "
                         "(column mapping, area unit/basis, price unit, "
                         "locality column)")
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
                         "sector's rate instead of erroring")
    pr.add_argument("--balcony", default=None)
    a = p.parse_args(argv)

    try:
        city = get_city(a.city)
        if a.cmd == "clean":
            _do_clean(city)
        elif a.cmd == "ingest":
            _do_ingest(city, a.config)
        elif a.cmd == "train":
            df = _load_processed(city)
            fitted = train_final(df, model=a.model)
            city.models_dir.mkdir(parents=True, exist_ok=True)
            with _quiet_joblib():
                joblib.dump(fitted, city.models_dir / "model.joblib")
            try:
                write_export(export_city(fitted, df, city), city.models_dir / "model.json")
            except ValueError as exc:
                # e.g. --model accurate: never exported to the browser, but
                # the CLI artifacts (joblib/metrics) are still produced.
                print(f"note: dashboard export skipped ({exc})")
            (city.models_dir / "metrics.json").write_text(
                json.dumps({k: fitted.metrics[k] for k in
                            ("model", "model_no_society", "baseline_sector",
                             "baseline_global", "baseline_served", "n", "n_splits", "params",
                             "model_name")}, indent=2))
            print(_fmt_metrics(fitted.metrics))
            print(f"Artifacts -> {city.models_dir}")
        elif a.cmd == "evaluate":
            print(_fmt_metrics(evaluate_cv(_load_processed(city), model=a.model)))
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
            if e["served_by"] == "baseline":
                # This city's model does not beat its own locality-median
                # rule of thumb (see homecast.model.model_beats_baseline) --
                # the rule IS the estimate here, not the model's number with
                # a caveat attached. See e["note"] for why.
                print(f"Estimated price (₹/sq.ft. rule of thumb -- the "
                      f"learned model did not beat it for this market): "
                      f"Rs {e['baseline_price_cr']:.2f} Cr (range "
                      f"{e['baseline_lo_cr']:.2f} - {e['baseline_hi_cr']:.2f} Cr)")
                print(f"note: {e['note']}")
            else:
                print(f"Estimated price: Rs {e['price_cr']:.2f} Cr "
                      f"(range {e['lo_cr']:.2f} - {e['hi_cr']:.2f} Cr)")
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
