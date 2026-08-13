"""Config-driven ingestion of third-party listing CSVs, with fail-fast unit
validation.

Indian property data mixes sq.ft./sq.m./sq.yards and rupees/lakh/crore, and
different portals/cities quote area on different bases (carpet vs.
super-built-up) that are NOT interconvertible without knowing the building's
loading factor. Silently guessing any of these would corrupt price_per_sqft
with no error thrown -- Mumbai listings in particular typically quote carpet
area where Gurgaon quotes super-built-up, so mixing them is exactly the kind
of mistake this module exists to catch.

A per-city TOML config declares the column mapping, area unit, area basis,
price unit, and locality column explicitly. Ingestion refuses to proceed if
a declaration is missing, or if the numbers it produces are implausible for
an Indian metro -- and every such error names the declaration that is
probably wrong, not just "value out of range".

``area_basis`` is persisted onto every ingested row (see ``ingest_city``) so
anything reading the processed CSV can see the basis without going back to
the config, and re-ingesting a city (``ingest_city(..., existing_processed_path=...)``)
is refused outright if the newly-declared ``area_basis`` disagrees with what
is already recorded for that city -- carpet and super-built-up area cannot
be silently reconciled, so a change in basis has to be a deliberate,
explicit re-ingestion (delete/rename the old processed file first), not an
accidental overwrite. There is no cross-city comparison feature in this
project yet, so that is the actual, narrower scope of what this module
enforces today: consistency of one city's own data over time, not a check
across every city HomeCast knows about.
"""
from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from homecast.cleaning import drop_duplicates, drop_unpriced, trim_outliers
from homecast.reference import guard_maharashtra_contamination

AREA_UNITS = {"sqft", "sqm", "sqyd"}
AREA_BASES = {"carpet", "builtup", "superbuiltup"}
PRICE_UNITS = {"rupees", "lakh", "crore"}

# Conversion factors to sq.ft.
_AREA_TO_SQFT = {"sqft": 1.0, "sqm": 10.7639, "sqyd": 9.0}
# Conversion factors to crore (HomeCast's internal price unit).
_PRICE_TO_CRORE = {"rupees": 1e-7, "lakh": 1e-2, "crore": 1.0}

# Plausibility bands used to catch a wrong unit/basis declaration before it
# silently corrupts every downstream price_per_sqft comparison. These are
# MEDIAN-based checks (see _plausibility_check): a unit error affecting only
# a minority of rows (e.g. 10% of listings mis-entered) will not move the
# median enough to trip either band, and will not be caught here.
#
# PPSF_BAND's Rs 100,000/sq.ft. ceiling is loose enough to admit a dataset
# that is ENTIRELY prime South Mumbai (Malabar Hill, Altamount Road, and
# similar) without a false positive -- which also means it is far too loose
# to catch a real unit error in a low-rate market (see the sq.yd-as-sq.ft
# case below). A per-city `expected_ppsf_range` (declared in the config, see
# IngestConfig) is the escape hatch for exactly that: a tighter band this
# module can check IN ADDITION to this one, once a city's market is known
# well enough to state its own range.
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
    # Optional, tighter than PPSF_BAND: a city-specific plausible Rs/sq.ft
    # range, checked in addition to (not instead of) the global PPSF_BAND.
    # The single global band is calibrated to admit prime South Mumbai on
    # the high end, which makes it ~15x too loose for a low-rate market like
    # Amaravathi (~Rs 4,000/sq.ft) -- exactly where a sq.yd plot declared as
    # sq.ft (a real 9x inflation) would otherwise sail through undetected.
    expected_ppsf_range: tuple[float, float] | None = None


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
    expected_ppsf_range = None
    if data.get("expected_ppsf_range") is not None:
        raw_range = data["expected_ppsf_range"]
        if len(raw_range) != 2:
            raise ValueError(
                f"{path}: expected_ppsf_range must be a [low, high] pair, "
                f"got {raw_range!r}")
        lo, hi = float(raw_range[0]), float(raw_range[1])
        if not (0 < lo < hi):
            raise ValueError(
                f"{path}: expected_ppsf_range [{lo}, {hi}] must have "
                f"0 < low < high")
        expected_ppsf_range = (lo, hi)
    return IngestConfig(
        city_key=data.get("city_key", path.stem),
        columns=data["columns"], area_unit=area_unit, area_basis=area_basis,
        price_unit=price_unit, locality_column=data["locality_column"],
        expected_ppsf_range=expected_ppsf_range)


def _map_and_convert(raw: pd.DataFrame, cfg: IngestConfig) -> pd.DataFrame:
    missing_targets = [t for t in REQUIRED_COLUMN_TARGETS if t not in cfg.columns]
    if missing_targets:
        raise ValueError(
            f"ingestion config 'columns' mapping is missing required target "
            f"field(s): {', '.join(missing_targets)}")

    # cfg.columns is declared target -> raw_col; inverting it below to build
    # the rename map silently collapses two targets that name the SAME raw
    # column to whichever one the dict comprehension visits last. Catch that
    # here, while it's still visible in the target -> raw_col direction.
    raw_to_targets: dict[str, list[str]] = {}
    for target, raw_col in cfg.columns.items():
        raw_to_targets.setdefault(raw_col, []).append(target)
    raw_to_targets.setdefault(cfg.locality_column, []).append("sector (locality_column)")
    dupes = {raw_col: targets for raw_col, targets in raw_to_targets.items() if len(targets) > 1}
    if dupes:
        detail = "; ".join(f"'{raw_col}' -> {targets}" for raw_col, targets in dupes.items())
        raise ValueError(
            f"ingestion config maps more than one target field to the same "
            f"raw column: {detail}. Each raw column may feed exactly one "
            f"target field.")

    rename = {raw_col: target for target, raw_col in cfg.columns.items()}
    rename[cfg.locality_column] = "sector"
    missing_source = [c for c in rename if c not in raw.columns]
    if missing_source:
        raise ValueError(
            f"source CSV is missing declared column(s): {', '.join(missing_source)}")

    # A raw column NOT being renamed can still collide with a rename TARGET
    # -- e.g. the source CSV already has its own "price" column while a
    # different column is declared to become "price". pandas' rename() does
    # not error on the resulting duplicate label; every later df["price"]
    # access would silently return a 2-column frame instead of a Series.
    unrenamed = [c for c in raw.columns if c not in rename]
    collisions = sorted(set(unrenamed) & set(rename.values()))
    if collisions:
        raise ValueError(
            f"source CSV already has column(s) {collisions} that collide "
            f"with a rename target declared in 'columns'/'locality_column' "
            f"-- the result would have duplicate '{collisions[0]}' column(s); "
            f"rename or drop the source column(s) first")

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

    # Optional, tighter, city-specific check -- see IngestConfig.expected_ppsf_range.
    # The global PPSF_BAND above stays enforced regardless; this is a second,
    # narrower band applied ON TOP of it when the city declares one, for
    # exactly the failure mode the global band is too loose to catch (e.g. a
    # sq.yd-declared-as-sq.ft error in a low-rate market -- the resulting
    # ~9x-inflated median still lands inside PPSF_BAND, but not inside a
    # city's own expected range).
    if cfg.expected_ppsf_range is not None:
        lo, hi = cfg.expected_ppsf_range
        if not (lo <= median_ppsf <= hi):
            raise ValueError(
                f"median price is Rs {median_ppsf:,.0f} per sq.ft., outside "
                f"this city's declared expected_ppsf_range of Rs "
                f"{lo:,.0f}-{hi:,.0f} (tighter than the global "
                f"Rs {PPSF_BAND[0]:,}-{PPSF_BAND[1]:,} band, which this "
                f"figure is still inside). This is almost always a wrong "
                f"price_unit or area_unit declaration: price_unit is "
                f"currently '{cfg.price_unit}'; area_unit is currently "
                f"'{cfg.area_unit}' (a sq.yd plot declared as sq.ft inflates "
                f"ppsf by ~9x, exactly the kind of error the global band "
                f"alone is too loose to catch in a low-rate market).")


_AMARAVATHI_KEY_PATTERN = re.compile(r"amaravathi", re.IGNORECASE)


def _looks_like_amaravathi_ap(cfg: IngestConfig) -> bool:
    """Whether this config is (almost certainly) for Amaravathi, AP -- used
    to decide whether the Maharashtra-contamination guard below should run.

    An exact ``cfg.city_key != "amaravathi_ap"`` check is fragile:
    ``load_config`` defaults ``city_key`` to the config file's stem when it
    isn't declared explicitly, so a config saved as ``amaravathi.toml``
    (city_key defaults to ``"amaravathi"``, not the exact ``"amaravathi_ap"``)
    would silently never trigger the guard it obviously should. Match on the
    distinctive "amaravathi" spelling in the key instead -- it's the AP
    city's own name and, per reference.py's MAHARASHTRA_MARKERS comment, not
    a substring of "amravati" (the unrelated Maharashtra city this guard
    exists to distinguish from), so this can't false-trigger on that city's
    own name either."""
    return bool(_AMARAVATHI_KEY_PATTERN.search(cfg.city_key))


def _maharashtra_contamination_log(df: pd.DataFrame, cfg: IngestConfig) -> list[str]:
    """When ingesting for Amaravathi, AP specifically, warn (do not reject --
    this is a data-quality signal, not necessarily a fatal error) if any
    locality string looks like it names Amravati, MAHARASHTRA instead: a
    real, unrelated, much larger city with a near-identical name."""
    if not _looks_like_amaravathi_ap(cfg):
        return []
    flagged = [loc for loc in df["sector"].dropna().unique()
              if guard_maharashtra_contamination(str(loc))]
    if not flagged:
        return []
    examples = ", ".join(map(str, flagged[:3]))
    return [f"WARNING: {len(flagged)} distinct locality value(s) look like "
           f"Amravati, Maharashtra rather than Amaravathi, AP (e.g. "
           f"{examples}) -- verify before trusting these rows"]


def _existing_area_basis(processed_path: Path) -> str | None:
    """The area_basis already recorded for a city's processed CSV, or None if
    there's nothing to compare against (no file yet, or a file predating this
    column -- e.g. one written by ``clean_city`` rather than ``ingest_city``)."""
    if not processed_path.exists():
        return None
    try:
        existing = pd.read_csv(processed_path, usecols=["area_basis"])
    except ValueError:
        return None
    if existing.empty:
        return None
    return str(existing["area_basis"].iloc[0])


def _guard_area_basis_consistency(cfg: IngestConfig, config_path: Path,
                                  existing_processed_path: Path | None) -> None:
    if existing_processed_path is None:
        return
    prior = _existing_area_basis(existing_processed_path)
    if prior is not None and prior != cfg.area_basis:
        raise ValueError(
            f"{config_path} declares area_basis='{cfg.area_basis}', but "
            f"{existing_processed_path} already has area_basis='{prior}' "
            f"recorded for this city. Carpet and super-built-up area are not "
            f"interconvertible, so re-ingesting with a different basis would "
            f"silently mix the two under one label. If the basis genuinely "
            f"changed, delete or rename the existing processed file first.")


def ingest_city(raw_path: Path, config_path: Path, *,
                existing_processed_path: Path | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Load a third-party CSV through its ingestion config and return
    (cleaned DataFrame in HomeCast's schema, step log) -- or raise loudly
    on a missing/implausible unit declaration.

    ``existing_processed_path``, when given, is checked against the newly
    declared ``area_basis`` (see ``_guard_area_basis_consistency``): the same
    city being re-ingested under a different, silently incompatible area
    basis is a fail-fast case, not a silent overwrite."""
    cfg = load_config(config_path)
    _guard_area_basis_consistency(cfg, config_path, existing_processed_path)
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
    log.extend(_maharashtra_contamination_log(df, cfg))
    return df, log
