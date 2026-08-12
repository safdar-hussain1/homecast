"""Feature engineering for the valuation model.

The sector is target-encoded as its median listing price per sq.ft. To keep
evaluation honest the encoding must be computed on training data only and
passed in — ``build_features`` never looks at prices itself except through
the supplied map.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = ["area", "bedrooms", "bathrooms", "is_house",
                   "furnishing_code", "luxury_score", "age_code", "sector_ppsf"]

FURNISHING_CODES = {"unfurnished": 0, "semi-furnished": 1, "furnished": 2}
AGE_CODES = {"Under Construction": 0, "New Property": 1, "Relatively New": 2,
             "Moderately Old": 3, "Old Property": 4}


def sector_encoding(train: pd.DataFrame) -> dict[str, float]:
    """Sector -> median price per sq.ft., learned from training rows only."""
    m = train.groupby("sector")["price_per_sqft"].median().to_dict()
    m["__global__"] = float(train["price_per_sqft"].median())
    return m


def build_features(df: pd.DataFrame, sector_map: dict[str, float]) -> pd.DataFrame:
    unknown = set(df["furnishing_type"].dropna().unique()) - set(FURNISHING_CODES)
    if unknown:
        raise ValueError(f"Unknown furnishing labels: {sorted(unknown)}")
    out = pd.DataFrame({
        "area": df["area"].astype(float),
        "bedrooms": df["bedrooms"].astype(float),
        "bathrooms": df["bathrooms"].astype(float),
        "is_house": (df["property_type"] == "house").astype(float),
        "furnishing_code": df["furnishing_type"].map(FURNISHING_CODES).astype(float),
        "luxury_score": df["luxury_score"].astype(float),
        "age_code": df["age_possession"].map(AGE_CODES).fillna(-1).astype(float),
        "sector_ppsf": df["sector"].map(sector_map)
                         .fillna(sector_map["__global__"]).astype(float),
    })
    return out[FEATURE_COLUMNS]


def target(df: pd.DataFrame) -> np.ndarray:
    """Model target: natural log of price (in crore)."""
    return np.log(df["price"].to_numpy(dtype=float))
