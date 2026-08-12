import numpy as np
import pandas as pd
import pytest

from homecast.cleaning import clean_raw_data


@pytest.fixture()
def raw_fixture() -> pd.DataFrame:
    """Small synthetic frame shaped like the raw Gurgaon feed."""
    rng = np.random.default_rng(7)
    n = 45
    sectors = rng.choice(["sector 1", "sector 2", "sector 3"], n)
    area = rng.uniform(500, 3000, n).round()
    ppsf = rng.uniform(4000, 20000, n).round()
    df = pd.DataFrame({
        "property_type": rng.choice(["flat", "house"], n, p=[0.8, 0.2]),
        "society": "test society",
        "sector": sectors,
        "price": (area * ppsf / 1e7).round(2),          # crore
        "price_per_sqft": ppsf,
        "area": area,
        "areaWithType": "Carpet area: x",
        "bedRoom": rng.integers(1, 6, n),
        "bathroom": rng.integers(1, 6, n),
        "balcony": rng.choice(["0", "1", "2", "3", "3+"], n),
        "floorNum": rng.integers(0, 20, n).astype(float),
        "facing": "East",
        "agePossession": rng.choice(
            ["New Property", "Relatively New", "Moderately Old",
             "Old Property", "Under Construction", "Undefined"], n),
        "super_built_up_area": np.nan, "built_up_area": np.nan, "carpet_area": area,
        "study room": 0, "servant room": 0, "store room": 0,
        "pooja room": 0, "others": 0,
        "furnishing_type": rng.integers(0, 3, n),
        "luxury_score": rng.integers(0, 175, n),
    })
    df.loc[43, "price"] = np.nan                        # unpriced row
    df.loc[44, ["price_per_sqft", "area"]] = [600000, 875000]  # unit-error outlier
    return pd.concat([df, df.iloc[[0, 1]]], ignore_index=True)  # 2 exact duplicates


@pytest.fixture()
def clean_fixture(raw_fixture) -> pd.DataFrame:
    cleaned, _ = clean_raw_data(raw_fixture)
    return cleaned
