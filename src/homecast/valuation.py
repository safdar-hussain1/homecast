"""Turn a buyer's description of a property into a price estimate."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from homecast.features import AGE_CODES, BALCONY_CODES, FURNISHING_CODES, build_features
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
    # Both optional: a caller who doesn't know the society or balcony count
    # simply omits it. society falls back to THIS listing's sector rate (a
    # building we can't identify is still in a known sector -- see
    # build_features), only reaching the global median if the sector is also
    # unknown; balcony falls back to the documented missing code -1.
    # society_ppsf is a very strong feature (see fit_encoders), so the model
    # is trained with it randomly masked to the sector rate on part of the
    # training data (homecast.model.SOCIETY_MASK_FRACTION) specifically so
    # an omitted society degrades gracefully instead of badly mispricing.
    society: str | None = None
    balcony: str | None = None


def query_to_row(q: Query) -> pd.DataFrame:
    return pd.DataFrame([{
        "sector": q.sector, "property_type": q.property_type,
        "bedrooms": q.bedrooms, "bathrooms": q.bathrooms, "area": q.area,
        "furnishing_type": q.furnishing, "luxury_score": q.luxury_score,
        "age_possession": q.age, "price_per_sqft": np.nan,
        "society": q.society, "balcony": q.balcony,
    }])


def _check_range(fitted: FittedModel, field: str, value: float, unit: str = "") -> None:
    """Reject a numeric input outside the range seen in training.

    Gradient-boosted trees do not extrapolate: past the training range they
    return a flat, unreliable value, so refusing is more honest than answering.
    """
    lo, hi = fitted.ranges[field]
    if not lo <= value <= hi:
        tail = f" {unit}" if unit else ""
        raise ValueError(f"{field} {value:.0f} outside supported range "
                         f"{lo:.0f}-{hi:.0f}{tail}")


def estimate(fitted: FittedModel, q: Query) -> dict:
    if q.property_type not in ("flat", "house"):
        raise ValueError(f"property_type must be 'flat' or 'house', got '{q.property_type}'")
    if q.furnishing not in FURNISHING_CODES:
        raise ValueError(f"Unknown furnishing '{q.furnishing}'")
    # `build_features` silently falls back to the global median for an unseen
    # sector — correct inside the CV loop, wrong for a human typing a name. A
    # typo like "Sector 65" would otherwise return a confident, badly wrong
    # number, so this entry point rejects it instead.
    known_sectors = [s for s in fitted.encoders.sector_ppsf if s != "__global__"]
    if q.sector not in fitted.encoders.sector_ppsf or q.sector == "__global__":
        raise ValueError(f"Unknown sector '{q.sector}' — not one of the "
                         f"{len(known_sectors)} sectors in the trained set "
                         f"(names are lower-case, e.g. 'sector 65'; check the case)")
    if q.age is not None and q.age not in AGE_CODES:
        raise ValueError(f"Unknown age '{q.age}'. Valid: "
                         f"{', '.join(AGE_CODES)}")
    if q.balcony is not None and q.balcony not in BALCONY_CODES:
        raise ValueError(f"Unknown balcony '{q.balcony}'. Valid: "
                         f"{', '.join(BALCONY_CODES)}")
    # Unlike sector, society is not rejected when unrecognised: it is an
    # optional field, and a name the model never trained on legitimately
    # falls back to this query's own sector rate (note "independent" --
    # ~13% of listings, no named society -- is itself a real, well-populated
    # category with its own learned encoding, not a fallback).
    _check_range(fitted, "area", q.area, "sq.ft.")
    _check_range(fitted, "bedrooms", q.bedrooms)
    _check_range(fitted, "bathrooms", q.bathrooms)
    _check_range(fitted, "luxury_score", q.luxury_score)
    X = build_features(query_to_row(q), fitted.encoders)
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
