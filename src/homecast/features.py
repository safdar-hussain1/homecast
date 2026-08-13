"""Feature engineering for the valuation model.

Every target-derived (fold-local) encoding is learned from a training frame
by ``fit_encoders`` and carried in an ``Encoders`` value; ``build_features``
never looks at prices itself except through the maps it is handed, so it can
be applied identically to a training fold and its held-out rows without
leaking the target.

Heterogeneous cities
--------------------
Not every market's feed carries every column. Gurgaon's does; a third-party
Bengaluru or Mumbai feed typically has no furnishing, no amenity/luxury
score, no possession age, and often no building name at all -- while some
carry things Gurgaon's does not (a bank of amenity flags, a resale flag).

Rather than demand a fixed schema, the feature set is a CATALOGUE
(``ALL_FEATURE_COLUMNS``) split into two parts:

* ``CORE_FEATURE_COLUMNS`` -- area, bedrooms and the four locality
  encodings. Without these there is no model, so a frame missing any of
  their source columns is rejected outright.
* everything else is OPTIONAL: a feature is included for a city if and only
  if every source column it needs (``FEATURE_SOURCES``) is present in that
  city's frame, and omitted entirely otherwise.

Omitted, not sentinel-filled: a feature nobody can supply carries no
information, and a constant column only invites the model to split on
nothing. The one place sentinels are still used is a value that is missing
on SOME rows of a city that does have the column (an unknown possession age,
an unrecorded balcony count, a listing whose amenities were never captured)
-- those keep the documented ``-1`` "not stated" code, which is a real,
learnable distinction from a genuine zero.

The selected list is resolved ONCE per city by ``feature_columns_for`` and
then carried explicitly: it is stored on the ``FittedModel``, re-used at
predict time, and exported to the dashboard as ``feature_order``. Train and
predict therefore cannot disagree about which columns exist or what order
they are in.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Canonical catalogue order. Every per-city feature list is a SUBSET of this,
# in this order, so two cities that share a feature always agree on where it
# sits relative to the others. Gurgaon's 13 come first and in their original
# order, so the public model's feature vector is byte-identical to what it
# was before the catalogue existed.
ALL_FEATURE_COLUMNS = [
    "area", "bedrooms", "bathrooms", "is_house", "furnishing_code",
    "luxury_score", "age_code", "sector_ppsf", "sector_ppsf_mean",
    "sector_ppsf_std", "sector_count", "society_ppsf", "balcony_code",
    "amenity_count", "is_resale",
]

# The Gurgaon (public) feature set, kept under its original name and value.
FEATURE_COLUMNS = [c for c in ALL_FEATURE_COLUMNS
                   if c not in ("amenity_count", "is_resale")]

# Source column(s) each catalogue feature needs. A feature whose sources are
# not all present in a city's frame is dropped from that city's feature list.
# The locality ("sector") encodings need no raw column beyond `sector`, which
# is required of every city by the ingestion config's `locality_column`.
FEATURE_SOURCES: dict[str, tuple[str, ...]] = {
    "area": ("area",),
    "bedrooms": ("bedrooms",),
    "bathrooms": ("bathrooms",),
    "is_house": ("property_type",),
    "furnishing_code": ("furnishing_type",),
    "luxury_score": ("luxury_score",),
    "age_code": ("age_possession",),
    "sector_ppsf": ("sector",),
    "sector_ppsf_mean": ("sector",),
    "sector_ppsf_std": ("sector",),
    "sector_count": ("sector",),
    "society_ppsf": ("society",),
    "balcony_code": ("balcony",),
    "amenity_count": ("amenity_count",),
    "is_resale": ("is_resale",),
}

# Present for every city or there is no model to fit.
CORE_FEATURE_COLUMNS = ["area", "bedrooms", "sector_ppsf", "sector_ppsf_mean",
                        "sector_ppsf_std", "sector_count"]

# "Not stated", for a column a city HAS but that is blank on some rows. Trees
# isolate it as its own branch, so it stays distinguishable from a real 0.
MISSING_CODE = -1.0

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


def feature_columns_for(df: pd.DataFrame) -> list[str]:
    """The catalogue features this frame can actually supply, in catalogue order.

    A city gets a feature iff every source column it needs is present. This
    is inferred from the frame rather than declared per city on purpose: the
    rule stays the same for every market, so adding a city never means
    editing a list of city names here.
    """
    cols = set(df.columns)
    selected = [f for f in ALL_FEATURE_COLUMNS
                if all(src in cols for src in FEATURE_SOURCES[f])]
    missing_core = [f for f in CORE_FEATURE_COLUMNS if f not in selected]
    if missing_core:
        needed = sorted({src for f in missing_core for src in FEATURE_SOURCES[f]}
                        - cols)
        raise ValueError(
            f"frame cannot supply the core feature(s) {missing_core}: no model "
            f"is possible without them. Missing source column(s): "
            f"{', '.join(needed)}. Check the city's ingestion config -- "
            f"'columns' must map price, area and bedrooms, and "
            f"'locality_column' must name the column that plays the role of "
            f"a sector.")
    return selected


def fit_encoders(train: pd.DataFrame) -> Encoders:
    """Learn every fold-local encoding from training rows only.

    A city with no ``society`` column gets an encoder holding only the
    ``__global__`` fallback; ``society_ppsf`` will not be in its feature list
    (see ``feature_columns_for``), so the empty map is never consulted.
    """
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
    if "society" in train.columns:
        ggrp = train.groupby("society")["price_per_sqft"].agg(["median", "count"])
        m = SOCIETY_SMOOTHING_M
        smoothed = (ggrp["count"] * ggrp["median"] + m * global_median) / (ggrp["count"] + m)
        society_ppsf = {**smoothed.to_dict(), "__global__": global_median}
    else:
        society_ppsf = {"__global__": global_median}

    return Encoders(sector_ppsf=sector_ppsf, sector_ppsf_mean=sector_ppsf_mean,
                     sector_ppsf_std=sector_ppsf_std, sector_count=sector_count,
                     society_ppsf=society_ppsf)


def build_features(df: pd.DataFrame, encoders: Encoders,
                   columns: list[str] | None = None) -> pd.DataFrame:
    """Build the feature matrix for ``columns`` (default: whatever this frame
    can supply, per ``feature_columns_for``).

    Pass ``columns`` explicitly -- from ``FittedModel.columns`` -- at predict
    time. Inference is right for a city's own processed frame, but a
    single-row query frame carries placeholder columns for fields the caller
    left blank, so re-inferring there could select a feature the model was
    never fit on.
    """
    if columns is None:
        columns = feature_columns_for(df)
    unknown_cols = [c for c in columns if c not in FEATURE_SOURCES]
    if unknown_cols:
        raise ValueError(f"Not catalogue feature(s): {unknown_cols}. Valid: "
                         f"{', '.join(ALL_FEATURE_COLUMNS)}")
    missing_src = sorted({src for c in columns for src in FEATURE_SOURCES[c]
                          if src not in df.columns})
    if missing_src:
        raise ValueError(
            f"cannot build features {columns}: frame is missing source "
            f"column(s) {', '.join(missing_src)}. Train and predict must use "
            f"the same feature list (see FittedModel.columns).")

    wanted = set(columns)

    def sector_rate() -> pd.Series:
        return df["sector"].map(encoders.sector_ppsf) \
                 .fillna(encoders.sector_ppsf["__global__"]).astype(float)

    # Only the selected features are computed -- a city with no society
    # column must not have df["society"] touched at all.
    builders: dict[str, callable] = {
        "area": lambda: df["area"].astype(float),
        "bedrooms": lambda: df["bedrooms"].astype(float),
        "bathrooms": lambda: df["bathrooms"].astype(float),
        "is_house": lambda: (df["property_type"] == "house").astype(float),
        "furnishing_code": lambda: df["furnishing_type"].map(FURNISHING_CODES).astype(float),
        "luxury_score": lambda: df["luxury_score"].astype(float),
        "age_code": lambda: df["age_possession"].map(AGE_CODES)
                              .fillna(MISSING_CODE).astype(float),
        "sector_ppsf": sector_rate,
        "sector_ppsf_mean": lambda: df["sector"].map(encoders.sector_ppsf_mean)
                              .fillna(encoders.sector_ppsf_mean["__global__"]).astype(float),
        "sector_ppsf_std": lambda: df["sector"].map(encoders.sector_ppsf_std)
                              .fillna(encoders.sector_ppsf_std["__global__"]).astype(float),
        "sector_count": lambda: df["sector"].map(encoders.sector_count)
                              .fillna(0.0).astype(float),
        # Fallback chain for society_ppsf: known society -> its own smoothed
        # rate; unknown/missing society -> that LISTING's sector rate (a
        # building we can't identify is still in a known sector -- a far
        # better prior than the city-wide median); sector also unknown ->
        # sector_ppsf has already fallen back to the global median, so this
        # chains through automatically (row-aligned fillna, not a scalar).
        "society_ppsf": lambda: df["society"].map(encoders.society_ppsf)
                              .astype(float).fillna(sector_rate()),
        "balcony_code": lambda: df["balcony"].astype(str).map(BALCONY_CODES)
                              .fillna(MISSING_CODE).astype(float),
        # Both may be blank on individual rows of a city that has the column
        # (e.g. a listing whose amenities were simply never captured) -- that
        # is "not stated", not zero amenities / not a new-build.
        "amenity_count": lambda: pd.to_numeric(df["amenity_count"], errors="coerce")
                              .fillna(MISSING_CODE).astype(float),
        "is_resale": lambda: pd.to_numeric(df["is_resale"], errors="coerce")
                              .fillna(MISSING_CODE).astype(float),
    }

    if "furnishing_code" in wanted:
        unknown = set(df["furnishing_type"].dropna().unique()) - set(FURNISHING_CODES)
        if unknown:
            raise ValueError(f"Unknown furnishing labels: {sorted(unknown)}")

    out = pd.DataFrame({name: builders[name]() for name in columns}, index=df.index)
    return out[columns]


def target(df: pd.DataFrame) -> np.ndarray:
    """Model target: natural log of price (in crore)."""
    return np.log(df["price"].to_numpy(dtype=float))
