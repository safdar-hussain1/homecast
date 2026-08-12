"""Data-cleaning pipeline for the Gurgaon property listings dataset.

Every step is a small, testable function; ``clean_raw_data`` orchestrates them
and returns both the cleaned frame and a human-readable log of what changed,
so the notebook (and the README) can report the exact effect of each rule.
"""

from __future__ import annotations

import pandas as pd

# Raw column name -> tidy snake_case name.
COLUMN_RENAMES = {
    "bedRoom": "bedrooms",
    "bathroom": "bathrooms",
    "floorNum": "floor_num",
    "areaWithType": "area_with_type",
    "agePossession": "age_possession",
    "study room": "has_study_room",
    "servant room": "has_servant_room",
    "store room": "has_store_room",
    "pooja room": "has_pooja_room",
    "others": "has_other_rooms",
}

FURNISHING_LABELS = {0: "unfurnished", 1: "semi-furnished", 2: "furnished"}
BALCONY_ORDER = ["0", "1", "2", "3", "3+"]

# Percentile bounds used to trim implausible listings (data-entry errors such as
# a 875,000 sq.ft. flat or Rs. 600,000 per sq.ft.).
TRIM_COLUMNS = ("price_per_sqft", "area")
TRIM_QUANTILES = (0.005, 0.995)


def load_raw(path) -> pd.DataFrame:
    """Load the raw listings CSV."""
    return pd.read_csv(path)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to consistent snake_case."""
    return df.rename(columns=COLUMN_RENAMES)


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate listings."""
    return df.drop_duplicates().reset_index(drop=True)


def encode_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Give categorical columns meaningful labels and orderings."""
    df = df.copy()
    df["furnishing_type"] = df["furnishing_type"].map(FURNISHING_LABELS)
    df["balcony"] = pd.Categorical(df["balcony"], categories=BALCONY_ORDER, ordered=True)
    # 'Undefined' is a placeholder, not a real possession status.
    df["age_possession"] = df["age_possession"].replace("Undefined", pd.NA)
    return df


def drop_unpriced(df: pd.DataFrame) -> pd.DataFrame:
    """Drop listings with no price — they cannot support any price analysis."""
    return df.dropna(subset=["price"]).reset_index(drop=True)


def trim_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Trim listings outside the 0.5th-99.5th percentile of price_per_sqft / area.

    The raw feed contains obvious unit errors (e.g. areas recorded in sq.yards or
    whole-building totals). Percentile trimming removes those without touching
    the genuine luxury tail.
    """
    keep = pd.Series(True, index=df.index)
    for col in TRIM_COLUMNS:
        lo, hi = df[col].quantile(TRIM_QUANTILES)
        keep &= df[col].between(lo, hi)
    return df[keep].reset_index(drop=True)


def clean_raw_data(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Run the full cleaning pipeline and return (cleaned_df, step log)."""
    log: list[str] = []
    steps = [
        ("standardize column names", standardize_columns),
        ("drop exact duplicate listings", drop_duplicates),
        ("label categorical columns", encode_categories),
        ("drop listings without a price", drop_unpriced),
        ("trim percentile outliers (price_per_sqft, area)", trim_outliers),
    ]
    for name, step in steps:
        before = len(df)
        df = step(df)
        removed = before - len(df)
        log.append(f"{name}: {removed} rows removed ({len(df)} remain)" if removed else f"{name}: ok")
    return df, log
