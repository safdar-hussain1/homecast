"""Feature engineering for the valuation model.

Every target-derived (fold-local) encoding is learned from a training frame
by ``fit_encoders`` and carried in an ``Encoders`` value; ``build_features``
never looks at prices itself except through the maps it is handed, so it can
be applied identically to a training fold and its held-out rows without
leaking the target.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "area", "bedrooms", "bathrooms", "is_house", "furnishing_code",
    "luxury_score", "age_code", "sector_ppsf", "sector_ppsf_mean",
    "sector_ppsf_std", "sector_count", "society_ppsf", "balcony_code",
]

FURNISHING_CODES = {"unfurnished": 0, "semi-furnished": 1, "furnished": 2}
AGE_CODES = {"Under Construction": 0, "New Property": 1, "Relatively New": 2,
             "Moderately Old": 3, "Old Property": 4}
BALCONY_CODES = {"0": 0, "1": 1, "2": 2, "3": 3, "3+": 4}

# Smoothed target-encoding shrinkage for society -> price_per_sqft:
# (n * society_median + m * global_median) / (n + m). Hardcoded, not tuned
# on the reported CV (the lab tuned m in an inner loop; 10 was in the
# winning range).
SOCIETY_SMOOTHING_M = 10.0


@dataclass(frozen=True)
class Encoders:
    """Every fold-local (target-derived) map, learned from training rows only.

    Each map carries a ``"__global__"`` fallback key for values unseen in the
    training frame it was fit on.
    """
    sector_ppsf: dict[str, float]        # sector -> median price_per_sqft
    sector_ppsf_mean: dict[str, float]   # sector -> mean price_per_sqft
    sector_ppsf_std: dict[str, float]    # sector -> std of price_per_sqft
    sector_count: dict[str, float]       # sector -> training listing count
    society_ppsf: dict[str, float]       # society -> smoothed median price_per_sqft


def fit_encoders(train: pd.DataFrame) -> Encoders:
    """Learn every fold-local encoding from training rows only."""
    global_median = float(train["price_per_sqft"].median())
    global_mean = float(train["price_per_sqft"].mean())
    global_std = float(train["price_per_sqft"].std())

    sector_ppsf = {**train.groupby("sector")["price_per_sqft"].median().to_dict(),
                   "__global__": global_median}

    sgrp = train.groupby("sector")["price_per_sqft"].agg(["mean", "std", "count"])
    sector_ppsf_mean = {**sgrp["mean"].to_dict(), "__global__": global_mean}
    sector_ppsf_std = {**sgrp["std"].fillna(global_std).to_dict(), "__global__": global_std}
    # A sector unseen in this training fold has zero training rows -- fall
    # back to 0, not an average count (matches the lab reference).
    sector_count = {**sgrp["count"].to_dict(), "__global__": 0.0}

    # A missing society (NaN) is dropped by groupby, not bucketed as its own
    # pseudo-society -- a listing with no society name has no group to shrink
    # toward, so it and any genuinely unseen society both fall straight back
    # to the global median (see build_features).
    ggrp = train.groupby("society")["price_per_sqft"].agg(["median", "count"])
    m = SOCIETY_SMOOTHING_M
    smoothed = (ggrp["count"] * ggrp["median"] + m * global_median) / (ggrp["count"] + m)
    society_ppsf = {**smoothed.to_dict(), "__global__": global_median}

    return Encoders(sector_ppsf=sector_ppsf, sector_ppsf_mean=sector_ppsf_mean,
                     sector_ppsf_std=sector_ppsf_std, sector_count=sector_count,
                     society_ppsf=society_ppsf)


def build_features(df: pd.DataFrame, encoders: Encoders) -> pd.DataFrame:
    unknown = set(df["furnishing_type"].dropna().unique()) - set(FURNISHING_CODES)
    if unknown:
        raise ValueError(f"Unknown furnishing labels: {sorted(unknown)}")
    balcony = df["balcony"].astype(str)
    out = pd.DataFrame({
        "area": df["area"].astype(float),
        "bedrooms": df["bedrooms"].astype(float),
        "bathrooms": df["bathrooms"].astype(float),
        "is_house": (df["property_type"] == "house").astype(float),
        "furnishing_code": df["furnishing_type"].map(FURNISHING_CODES).astype(float),
        "luxury_score": df["luxury_score"].astype(float),
        "age_code": df["age_possession"].map(AGE_CODES).fillna(-1).astype(float),
        "sector_ppsf": df["sector"].map(encoders.sector_ppsf)
                         .fillna(encoders.sector_ppsf["__global__"]).astype(float),
        "sector_ppsf_mean": df["sector"].map(encoders.sector_ppsf_mean)
                         .fillna(encoders.sector_ppsf_mean["__global__"]).astype(float),
        "sector_ppsf_std": df["sector"].map(encoders.sector_ppsf_std)
                         .fillna(encoders.sector_ppsf_std["__global__"]).astype(float),
        "sector_count": df["sector"].map(encoders.sector_count).fillna(0.0).astype(float),
        "society_ppsf": df["society"].map(encoders.society_ppsf)
                         .fillna(encoders.society_ppsf["__global__"]).astype(float),
        "balcony_code": balcony.map(BALCONY_CODES).fillna(-1).astype(float),
    })
    return out[FEATURE_COLUMNS]


def target(df: pd.DataFrame) -> np.ndarray:
    """Model target: natural log of price (in crore)."""
    return np.log(df["price"].to_numpy(dtype=float))
