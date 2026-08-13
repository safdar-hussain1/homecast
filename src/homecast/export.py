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
from homecast.features import AGE_CODES, BALCONY_CODES, FURNISHING_CODES
from homecast.model import FittedModel

# Columns the browser's comparables table shows, when the city has them.
# property_type is Gurgaon-only; a feed that doesn't say flat-vs-house
# simply gets a narrower table rather than a KeyError.
SAMPLE_COLUMNS = ("sector", "property_type", "bedrooms", "area", "price")


def tree_to_arrays(tree) -> dict:
    return {"f": tree.feature.tolist(), "t": tree.threshold.tolist(),
            "l": tree.children_left.tolist(), "r": tree.children_right.tolist(),
            "v": tree.value.squeeze(axis=(1, 2)).tolist()}


def _require_exportable(model) -> None:
    """The browser walker only understands a GradientBoostingRegressor-style
    staged ensemble (a constant ``init_`` plus additive shrunk trees). An
    estimator like ExtraTrees has no ``init_``/``learning_rate`` and averages
    trees instead of walking them additively, so it cannot be represented by
    ``tree_to_arrays``/``predict_from_export`` at all -- fail loudly instead
    of silently exporting a payload that would mispredict in the browser."""
    if not hasattr(model, "init_") or not hasattr(model, "learning_rate"):
        raise ValueError(
            f"{type(model).__name__} cannot be exported to the browser "
            f"dashboard (missing 'init_'/'learning_rate' -- only a "
            f"GradientBoostingRegressor-style staged ensemble is "
            f"exportable). Train with model=\"default\" to get an "
            f"exportable model; \"accurate\" is CLI-only.")


def _area_basis(df: pd.DataFrame) -> str | None:
    """The one area basis this city's rows are measured on, or None.

    The dashboard writes this into the label above its area input. Getting it
    wrong is not cosmetic: carpet, built-up and super-built-up differ by
    25-35%, so labelling an unknown-basis feed "Built-up area" would assert
    in the page precisely what the ingestion layer refused to assume.

    None (no ``area_basis`` column at all) means the frame predates the
    ingestion config -- the page keeps its existing wording rather than
    inventing a basis. A frame carrying TWO bases is refused: there is no
    honest single label for it, and its price_per_sqft would be mixing
    measurements that are not the same quantity.
    """
    if "area_basis" not in df.columns:
        return None
    found = sorted(set(df["area_basis"].dropna().astype(str)))
    if len(found) > 1:
        raise ValueError(
            f"cannot export a single area_basis: this frame mixes "
            f"{', '.join(found)}. Rows measured on different bases are not "
            f"comparable per sq.ft.; filter to one basis at ingestion "
            f"(see the ingestion config's area_basis/area_basis_column).")
    return found[0] if found else None


def export_city(fitted: FittedModel, df: pd.DataFrame, city: City) -> dict:
    _require_exportable(fitted.model)
    area_basis = _area_basis(df)
    res = np.asarray(fitted.metrics["residuals_log"])
    counts, edges = np.histogram(res, bins=20)
    by_sector = (df.groupby("sector")
                   .agg(n=("price", "size"), median_price_cr=("price", "median"),
                        median_ppsf=("price_per_sqft", "median"))
                   .reset_index()
                   .rename(columns={"sector": "name"}))
    by_sector = by_sector[by_sector["n"] >= 30].sort_values(
        "median_price_cr", ascending=False)
    sample = (df[[c for c in SAMPLE_COLUMNS if c in df.columns]]
              .sample(n=min(400, len(df)), random_state=7))
    metrics = {k: fitted.metrics[k] for k in
               ("model", "model_no_society", "baseline_sector", "baseline_global",
                "n", "n_splits")}
    # This city's own feature list, not the catalogue: a city that can't
    # supply furnishing/age/society has a shorter vector, and the browser
    # builds its rows by iterating feature_order (see the dashboard's
    # featureRow), so the two stay in step automatically.
    columns = list(fitted.columns)
    payload = {
        "city": city.display,
        "generated_n": int(len(df)),
        # False for a city whose page must not carry the hand-written
        # Gurgaon narrative (findings, methodology, hardcoded figures).
        "narrative": bool(city.public),
        "model": {"init": float(np.squeeze(fitted.model.init_.constant_)),
                  "learning_rate": float(fitted.model.learning_rate),
                  "trees": [tree_to_arrays(e[0].tree_) for e in fitted.model.estimators_]},
        "feature_order": columns,
        # keyed by feature name so the dashboard never carries its own copy
        "feature_importances": {name: float(v) for name, v in
                                zip(columns, fitted.model.feature_importances_)},
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
    # Added only when the city actually declared one, so a frame that never
    # recorded a basis (Gurgaon) produces a byte-identical payload to before.
    if area_basis is not None:
        payload["area_basis"] = area_basis
    return payload


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
