"""Config-driven ingestion of third-party listing CSVs, with fail-fast unit
validation.

Indian property data mixes sq.ft./sq.m./sq.yards and rupees/lakh/crore, and
different portals/cities quote area on different bases (carpet vs.
super-built-up) that are NOT interconvertible without knowing the building's
loading factor. Silently guessing any of these would corrupt price_per_sqft
with no error thrown -- Mumbai listings in particular typically quote carpet
area where Gurgaon quotes super-built-up, so mixing them across cities is
exactly the kind of mistake this module exists to catch.

A per-city TOML config declares the column mapping, area unit, area basis,
price unit, and locality column explicitly. Ingestion refuses to proceed if
a declaration is missing, or if the numbers it produces are implausible for
an Indian metro -- and every such error names the declaration that is
probably wrong, not just "value out of range".
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from homecast.cleaning import drop_duplicates, drop_unpriced, trim_outliers

AREA_UNITS = {"sqft", "sqm", "sqyd"}
AREA_BASES = {"carpet", "builtup", "superbuiltup"}
PRICE_UNITS = {"rupees", "lakh", "crore"}

# Conversion factors to sq.ft.
_AREA_TO_SQFT = {"sqft": 1.0, "sqm": 10.7639, "sqyd": 9.0}
# Conversion factors to crore (HomeCast's internal price unit).
_PRICE_TO_CRORE = {"rupees": 1e-7, "lakh": 1e-2, "crore": 1.0}

# Plausibility bands used to catch a wrong unit/basis declaration before it
# silently corrupts every downstream price_per_sqft comparison.
PPSF_BAND = (1_000, 100_000)   # Rs per sq.ft, sane range for an Indian metro
AREA_BAND = (100, 20_000)      # sq.ft., sane range for a residential listing

REQUIRED_DECLARATIONS = ("area_unit", "area_basis", "price_unit", "columns",
                          "locality_column")
REQUIRED_COLUMN_TARGETS = ("price", "area")


@dataclass(frozen=True)
class IngestConfig:
    city_key: str
    columns: dict            # target field -> raw column name
    area_unit: str
    area_basis: str
    price_unit: str
    locality_column: str


def load_config(path: Path) -> IngestConfig:
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    missing = [k for k in REQUIRED_DECLARATIONS if k not in data or not data[k]]
    if missing:
        raise ValueError(
            f"{path} is missing required declaration(s): {', '.join(missing)}. "
            f"An ingestion config must declare all of: "
            f"{', '.join(REQUIRED_DECLARATIONS)}")
    area_unit, area_basis, price_unit = data["area_unit"], data["area_basis"], data["price_unit"]
    if area_unit not in AREA_UNITS:
        raise ValueError(f"{path}: area_unit '{area_unit}' is not one of "
                         f"{sorted(AREA_UNITS)}")
    if area_basis not in AREA_BASES:
        raise ValueError(f"{path}: area_basis '{area_basis}' is not one of "
                         f"{sorted(AREA_BASES)} -- carpet and super-built-up "
                         f"are not interconvertible, so this must be stated "
                         f"explicitly, not guessed")
    if price_unit not in PRICE_UNITS:
        raise ValueError(f"{path}: price_unit '{price_unit}' is not one of "
                         f"{sorted(PRICE_UNITS)}")
    return IngestConfig(
        city_key=data.get("city_key", path.stem),
        columns=data["columns"], area_unit=area_unit, area_basis=area_basis,
        price_unit=price_unit, locality_column=data["locality_column"])


def _map_and_convert(raw: pd.DataFrame, cfg: IngestConfig) -> pd.DataFrame:
    missing_targets = [t for t in REQUIRED_COLUMN_TARGETS if t not in cfg.columns]
    if missing_targets:
        raise ValueError(
            f"ingestion config 'columns' mapping is missing required target "
            f"field(s): {', '.join(missing_targets)}")
    rename = {raw_col: target for target, raw_col in cfg.columns.items()}
    rename[cfg.locality_column] = "sector"
    missing_source = [c for c in rename if c not in raw.columns]
    if missing_source:
        raise ValueError(
            f"source CSV is missing declared column(s): {', '.join(missing_source)}")
    df = raw.rename(columns=rename).copy()
    df["area"] = df["area"].astype(float) * _AREA_TO_SQFT[cfg.area_unit]
    df["price"] = df["price"].astype(float) * _PRICE_TO_CRORE[cfg.price_unit]
    df["price_per_sqft"] = df["price"] * 1e7 / df["area"]
    return df


def _suggest_price_unit(median_ppsf: float, declared: str) -> str:
    # rupees (x1e-7) < lakh (x1e-2) < crore (x1.0) as conversion factors to
    # crore. A ppsf that came out too HIGH means the declared unit's factor
    # is too large (inflated the price) -- the fix is a SMALLER unit; too
    # LOW means the opposite.
    if median_ppsf > PPSF_BAND[1]:
        smaller = {"crore": "lakh", "lakh": "rupees"}.get(declared)
        return f"try '{smaller}' instead of '{declared}'" if smaller else \
            "check the raw price column for a units mismatch"
    bigger = {"rupees": "lakh", "lakh": "crore"}.get(declared)
    return f"try '{bigger}' instead of '{declared}'" if bigger else \
        "check the raw price column for a units mismatch"


def _plausibility_check(df: pd.DataFrame, cfg: IngestConfig) -> None:
    non_positive = df["price"].notna() & (df["price"] <= 0)
    if non_positive.any():
        raise ValueError(
            f"{int(non_positive.sum())} row(s) have a non-positive price "
            f"after converting from '{cfg.price_unit}' -- check the source "
            f"data and the price_unit declaration")

    median_area = float(df["area"].median())
    if not (AREA_BAND[0] <= median_area <= AREA_BAND[1]):
        raise ValueError(
            f"median area is {median_area:,.0f} sq.ft. after converting "
            f"from '{cfg.area_unit}', outside the plausible "
            f"{AREA_BAND[0]}-{AREA_BAND[1]} sq.ft. band for a residential "
            f"listing -- is area_unit really '{cfg.area_unit}'? A wrong "
            f"area_unit (e.g. sq.m or sq.yd declared as sq.ft) produces "
            f"exactly this kind of shift.")

    median_ppsf = float(df["price_per_sqft"].median())
    if not (PPSF_BAND[0] <= median_ppsf <= PPSF_BAND[1]):
        raise ValueError(
            f"median price is Rs {median_ppsf:,.0f} per sq.ft., outside the "
            f"plausible Rs {PPSF_BAND[0]:,}-{PPSF_BAND[1]:,} band for an "
            f"Indian metro. This is almost always a wrong price_unit or "
            f"area_unit declaration: price_unit is currently "
            f"'{cfg.price_unit}' ({_suggest_price_unit(median_ppsf, cfg.price_unit)}); "
            f"area_unit is currently '{cfg.area_unit}' (if the source "
            f"actually reports sq.yd or sq.m as sq.ft, ppsf shifts by "
            f"~9x or ~10.8x respectively).")


def ingest_city(raw_path: Path, config_path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load a third-party CSV through its ingestion config and return
    (cleaned DataFrame in HomeCast's schema, step log) -- or raise loudly
    on a missing/implausible unit declaration."""
    cfg = load_config(config_path)
    raw = pd.read_csv(raw_path)
    df = _map_and_convert(raw, cfg)
    _plausibility_check(df, cfg)

    log: list[str] = []
    steps = [
        ("drop exact duplicate listings", drop_duplicates),
        ("drop listings without a price", drop_unpriced),
        ("trim percentile outliers (price_per_sqft, area)", trim_outliers),
    ]
    for name, step in steps:
        before = len(df)
        df = step(df)
        removed = before - len(df)
        log.append(f"{name}: {removed} rows removed ({len(df)} remain)" if removed
                   else f"{name}: ok")

    # area_basis is recorded on every row, not just the config -- carpet vs.
    # super-built-up is not convertible, so anything that later displays or
    # compares price_per_sqft across cities must be able to see the basis
    # without going back to this city's ingestion config.
    df["area_basis"] = cfg.area_basis
    return df, log
