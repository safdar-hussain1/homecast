"""Turn a buyer's description of a property into a price estimate."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from homecast.export import SAMPLE_COLUMNS
from homecast.features import AGE_CODES, BALCONY_CODES, FURNISHING_CODES, build_features
from homecast.model import (FittedModel, baseline_served_reason,
                            not_served_reason, predict_price, serving_status)


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
    # Only meaningful for a city whose feed carries them (see
    # homecast.features): a bank of amenity flags, and a resale marker.
    # Left as None for Gurgaon, whose model has neither feature.
    amenity_count: float | None = None
    is_resale: float | None = None


def query_to_row(q: Query) -> pd.DataFrame:
    """One query as a single-row frame carrying EVERY catalogue source column.

    Deliberately complete even where the caller supplied nothing: the row is
    only ever consumed with an explicit column list from the fitted model, so
    the extra placeholders are never selected, and a model that does want one
    of them always finds the column rather than raising.
    """
    return pd.DataFrame([{
        "sector": q.sector, "property_type": q.property_type,
        "bedrooms": q.bedrooms, "bathrooms": q.bathrooms, "area": q.area,
        "furnishing_type": q.furnishing, "luxury_score": q.luxury_score,
        "age_possession": q.age, "price_per_sqft": np.nan,
        "society": q.society, "balcony": q.balcony,
        "amenity_count": q.amenity_count, "is_resale": q.is_resale,
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
    # Each input is validated only if this city's model actually uses it: a
    # feed with no furnishing column has no furnishing_code feature, and
    # rejecting a query over a field the model ignores would be noise.
    used = set(fitted.columns)
    if "is_house" in used and q.property_type not in ("flat", "house"):
        raise ValueError(f"property_type must be 'flat' or 'house', got '{q.property_type}'")
    if "furnishing_code" in used and q.furnishing not in FURNISHING_CODES:
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
    if "age_code" in used and q.age is not None and q.age not in AGE_CODES:
        raise ValueError(f"Unknown age '{q.age}'. Valid: "
                         f"{', '.join(AGE_CODES)}")
    if "balcony_code" in used and q.balcony is not None and q.balcony not in BALCONY_CODES:
        raise ValueError(f"Unknown balcony '{q.balcony}'. Valid: "
                         f"{', '.join(BALCONY_CODES)}")
    # Unlike sector, society is not rejected when unrecognised: it is an
    # optional field, and a name the model never trained on legitimately
    # falls back to this query's own sector rate (note "independent" --
    # ~13% of listings, no named society -- is itself a real, well-populated
    # category with its own learned encoding, not a fallback).
    # fitted.ranges only holds the numeric inputs this city actually has, so
    # a market with no amenity score is not asked to range-check one.
    for field, value, unit in (("area", q.area, "sq.ft."),
                               ("bedrooms", q.bedrooms, ""),
                               ("bathrooms", q.bathrooms, ""),
                               ("luxury_score", q.luxury_score, "")):
        if field in fitted.ranges:
            _check_range(fitted, field, value, unit)
    X = build_features(query_to_row(q), fitted.encoders, list(fitted.columns))
    price = float(predict_price(fitted, X)[0])
    # The band holds the 10th/90th percentile of residual = log(pred) - log(actual),
    # so actual = pred * exp(-residual). The SIGN IS NEGATED on purpose: the q90
    # residual (the model's biggest overestimates) maps to the LOW end of the true
    # price, and the q10 residual maps to the HIGH end. Do not "fix" this back.
    q10, q90 = fitted.band
    result = {"price_cr": price,
             "lo_cr": price * float(np.exp(-q90)),
             "hi_cr": price * float(np.exp(-q10))}

    # The locality-median (₹/sq.ft. rule of thumb) alternative, computed the
    # SAME way regardless of which one is "served" -- this is not only
    # computed when needed, so a caller can always show "what the rule says"
    # alongside the model, not only on the losing cities. q.sector is
    # already validated above to be a real, known sector (not "__global__"),
    # so this indexes fitted.encoders.sector_ppsf directly rather than
    # falling back.
    sector_ppsf = float(fitted.encoders.sector_ppsf[q.sector])
    baseline_price = sector_ppsf * q.area / 1e7
    bq10, bq90 = fitted.baseline_band
    result["baseline_price_cr"] = baseline_price
    result["baseline_lo_cr"] = baseline_price * float(np.exp(-bq90))
    result["baseline_hi_cr"] = baseline_price * float(np.exp(-bq10))

    # served_by is the answer to "which of the two numbers above is THE
    # estimate for this city, if either" -- see homecast.model.serving_status,
    # which layers a quality floor UNDERNEATH the plain model-vs-baseline
    # comparison. A city whose model does not beat its own locality-median
    # rule must not have the model's number presented as if it were the
    # estimate, AND a city where even the better of the two is still too
    # unreliable (both above the floor) must not have EITHER number presented
    # as an estimate at all. Every caller (the CLI, the dashboard export) is
    # expected to read this field and act on it rather than always defaulting
    # to price_cr.
    status = serving_status(fitted.metrics)
    result["served_by"] = status
    if status == "baseline":
        result["note"] = baseline_served_reason(fitted.metrics, fitted.columns)
    elif status == "not_served":
        result["note"] = not_served_reason(fitted.metrics, fitted.columns)
    return result


def comparables(df: pd.DataFrame, q: Query, k: int = 5) -> pd.DataFrame:
    pool = df[df["sector"] == q.sector]
    if pool.empty:
        pool = df
    ranked = pool.iloc[(pool["area"] - q.area).abs().argsort()]
    cols = [c for c in SAMPLE_COLUMNS if c in df.columns]
    return ranked.head(k)[cols]
