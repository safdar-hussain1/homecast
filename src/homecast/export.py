"""Export a fitted model and market stats as plain JSON for the dashboard.

The browser evaluates the ensemble with a small tree-walker; tests assert its
predictions match sklearn's to 1e-9, so the dashboard can never drift from
the Python model.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from homecast.cities import City
from homecast.features import AGE_CODES, BALCONY_CODES, FEATURE_COLUMNS, FURNISHING_CODES
from homecast.model import FittedModel


def tree_to_arrays(tree) -> dict:
    return {"f": tree.feature.tolist(), "t": tree.threshold.tolist(),
            "l": tree.children_left.tolist(), "r": tree.children_right.tolist(),
            "v": tree.value.squeeze(axis=(1, 2)).tolist()}


def export_city(fitted: FittedModel, df: pd.DataFrame, city: City) -> dict:
    res = np.asarray(fitted.metrics["residuals_log"])
    counts, edges = np.histogram(res, bins=20)
    by_sector = (df.groupby("sector")
                   .agg(n=("price", "size"), median_price_cr=("price", "median"),
                        median_ppsf=("price_per_sqft", "median"))
                   .reset_index()
                   .rename(columns={"sector": "name"}))
    by_sector = by_sector[by_sector["n"] >= 30].sort_values(
        "median_price_cr", ascending=False)
    sample = (df[["sector", "property_type", "bedrooms", "area", "price"]]
              .sample(n=min(400, len(df)), random_state=7))
    metrics = {k: fitted.metrics[k] for k in
               ("model", "baseline_sector", "baseline_global", "n", "n_splits")}
    return {
        "city": city.display,
        "generated_n": int(len(df)),
        "model": {"init": float(np.squeeze(fitted.model.init_.constant_)),
                  "learning_rate": float(fitted.model.learning_rate),
                  "trees": [tree_to_arrays(e[0].tree_) for e in fitted.model.estimators_]},
        "feature_order": list(FEATURE_COLUMNS),
        # keyed by feature name so the dashboard never carries its own copy
        "feature_importances": {name: float(v) for name, v in
                                zip(FEATURE_COLUMNS, fitted.model.feature_importances_)},
        "encodings": {
            "furnishing": FURNISHING_CODES, "age": AGE_CODES, "balcony": BALCONY_CODES,
            # each *_ppsf/sector_count map includes the internal "__global__"
            # fallback key alongside real sector names -- don't iterate these
            # as "all sectors"
            "sector_ppsf": {k: float(v) for k, v in fitted.encoders.sector_ppsf.items()},
            "sector_ppsf_mean": {k: float(v) for k, v in fitted.encoders.sector_ppsf_mean.items()},
            "sector_ppsf_std": {k: float(v) for k, v in fitted.encoders.sector_ppsf_std.items()},
            "sector_count": {k: float(v) for k, v in fitted.encoders.sector_count.items()},
            "society_ppsf": {k: float(v) for k, v in fitted.encoders.society_ppsf.items()},
        },
        "band": [float(fitted.band[0]), float(fitted.band[1])],
        "ranges": fitted.ranges,
        "metrics": metrics,
        "residual_hist": {"edges": edges.tolist(), "counts": counts.tolist()},
        "sectors": by_sector.to_dict(orient="records"),
        "sample": sample.to_dict(orient="records"),
    }


def predict_from_export(payload: dict, feature_row) -> float:
    """Reference implementation of the JS tree-walker (kept in lockstep)."""
    x = list(map(float, feature_row))
    total = payload["model"]["init"]
    lr = payload["model"]["learning_rate"]
    for tr in payload["model"]["trees"]:
        i = 0
        while tr["l"][i] != -1:
            i = tr["l"][i] if x[tr["f"][i]] <= tr["t"][i] else tr["r"][i]
        total += lr * tr["v"][i]
    return float(np.exp(total))


def write_export(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, allow_nan=False, separators=(",", ":"))
