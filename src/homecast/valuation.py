"""Turn a buyer's description of a property into a price estimate."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from homecast.features import FURNISHING_CODES, build_features
from homecast.model import FittedModel, predict_price


@dataclass(frozen=True)
class Query:
    sector: str
    property_type: str
    bedrooms: int
    bathrooms: int
    area: float
    furnishing: str
    luxury_score: int
    age: str | None = None


def query_to_row(q: Query) -> pd.DataFrame:
    return pd.DataFrame([{
        "sector": q.sector, "property_type": q.property_type,
        "bedrooms": q.bedrooms, "bathrooms": q.bathrooms, "area": q.area,
        "furnishing_type": q.furnishing, "luxury_score": q.luxury_score,
        "age_possession": q.age, "price_per_sqft": np.nan,
    }])


def estimate(fitted: FittedModel, q: Query) -> dict:
    if q.property_type not in ("flat", "house"):
        raise ValueError(f"property_type must be 'flat' or 'house', got '{q.property_type}'")
    if q.furnishing not in FURNISHING_CODES:
        raise ValueError(f"Unknown furnishing '{q.furnishing}'")
    lo_a, hi_a = fitted.ranges["area"]
    if not lo_a <= q.area <= hi_a:
        raise ValueError(f"area {q.area:.0f} outside supported range "
                         f"{lo_a:.0f}-{hi_a:.0f} sq.ft.")
    X = build_features(query_to_row(q), fitted.sector_map)
    price = float(predict_price(fitted, X)[0])
    # The band holds the 10th/90th percentile of residual = log(pred) - log(actual),
    # so actual = pred * exp(-residual). The SIGN IS NEGATED on purpose: the q90
    # residual (the model's biggest overestimates) maps to the LOW end of the true
    # price, and the q10 residual maps to the HIGH end. Do not "fix" this back.
    q10, q90 = fitted.band
    return {"price_cr": price,
            "lo_cr": price * float(np.exp(-q90)),
            "hi_cr": price * float(np.exp(-q10))}


def comparables(df: pd.DataFrame, q: Query, k: int = 5) -> pd.DataFrame:
    pool = df[df["sector"] == q.sector]
    if pool.empty:
        pool = df
    ranked = pool.iloc[(pool["area"] - q.area).abs().argsort()]
    return ranked.head(k)[["sector", "property_type", "bedrooms", "area", "price"]]
