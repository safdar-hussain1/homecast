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
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from homecast.cleaning import drop_duplicates, drop_unpriced, trim_outliers
from homecast.features import BALCONY_CODES
from homecast.reference import guard_maharashtra_contamination

AREA_UNITS = {"sqft", "sqm", "sqyd"}

# The three bases a listing's area can actually be quoted on. They are not
# interconvertible without the building's loading factor, so a config must
# say which one it is.
KNOWN_AREA_BASES = {"carpet", "builtup", "superbuiltup"}

# ...and the honest fourth answer. Plenty of third-party feeds simply never
# state their basis. Declaring "unknown" is ACCEPTED -- refusing the dataset
# outright would only push people into guessing "superbuiltup" because it is
# the most common, which is the exact silent corruption this module exists to
# prevent. What "unknown" buys is that the uncertainty is recorded rather
# than laundered: it is written onto every ingested row like any other basis,
# it makes ingest_city emit a caveat line into its log, and the
# re-ingestion guard still treats it as incompatible with every known basis,
# so a later config cannot quietly promote "unknown" to "carpet". A city
# ingested this way must never have its price_per_sqft compared with a city
# whose basis IS known -- the difference between carpet and super-built-up
# is routinely 25-35% of the area, i.e. bigger than most of the effects a
# model like this is trying to measure.
UNKNOWN_AREA_BASIS = "unknown"
AREA_BASES = KNOWN_AREA_BASES | {UNKNOWN_AREA_BASIS}

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

# Rows missing any of these are unusable, so they are always dropped. A city
# can extend the list via `required_columns` (e.g. a feed whose bedroom count
# is blank on some rows, where a fitted model needs it non-null).
ALWAYS_REQUIRED_ROW_VALUES = ("price", "area")


# --- Named value parsers ---------------------------------------------------
#
# Third-party feeds put real information in free text: a size of "3 BHK", an
# area of "1133 - 1384", a balcony count as a float. A config names the
# parser each mapped column needs; the parsers themselves are generic (Indian
# listing conventions, not any one city's), so a new market reuses them
# instead of adding city-specific branches here.
#
# Every parser returns NaN for anything it cannot read rather than guessing.
# Unreadable values in a required column drop the row, and the count is
# reported in the ingestion log -- silence about how many rows a parser
# failed on is how a "clean" dataset ends up quietly missing a third of a
# market segment.

_RANGE_RE = re.compile(r"^\s*([\d.]+)\s*-\s*([\d.]+)\s*$")
_LEADING_NUM_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)")


def _parse_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _parse_numeric_or_range(s: pd.Series) -> pd.Series:
    """A number, or a "low - high" range quoted as one value -> its midpoint.

    Listing sites publish a range when a project sells several layouts under
    one size label. The midpoint is the honest single number: it is what the
    range's own author would call typical, and picking either end would bias
    every derived price_per_sqft in one direction.

    A value carrying its OWN unit suffix ("34.46Sq. Meter", "4125Perch") is
    deliberately NOT converted: the config declares one area_unit for the
    column, and a row that disagrees with it is a row whose units this
    config cannot vouch for. They become NaN and are dropped and counted,
    rather than being silently converted -- which for the plot-sized units
    in these feeds (Acres, Guntha, Grounds) produces "residential listings"
    of several lakh sq.ft. that then distort every percentile trim.
    """
    text = s.astype(str).str.strip()
    out = pd.to_numeric(text, errors="coerce")
    todo = out.isna() & s.notna()
    if todo.any():
        m = text[todo].str.extract(_RANGE_RE)
        mid = (pd.to_numeric(m[0], errors="coerce")
               + pd.to_numeric(m[1], errors="coerce")) / 2.0
        out.loc[todo] = mid
    return out


def _parse_leading_number(s: pd.Series) -> pd.Series:
    """The leading count in a size label: "3 BHK" -> 3, "4 Bedroom" -> 4.

    "1 RK" (one room + kitchen) parses to 1 by the same rule, which is the
    right answer: it is a one-room home.
    """
    return pd.to_numeric(s.astype(str).str.extract(_LEADING_NUM_RE)[0],
                         errors="coerce")


def _parse_balcony_label(s: pd.Series) -> pd.Series:
    """Normalise a balcony count onto HomeCast's canonical labels.

    Feeds carry this as a float ("2.0"), an int, or already as a label. The
    model's BALCONY_CODES tops out at "3+", so 3 or more collapses there;
    blank stays blank and becomes the documented "not stated" code
    downstream, never a fabricated zero.
    """
    n = pd.to_numeric(s, errors="coerce")
    labels = n.map(lambda v: pd.NA if pd.isna(v)
                   else ("3+" if v >= 3 else str(int(v))))
    # Anything non-numeric that is ALREADY a valid label passes through.
    passthrough = n.isna() & s.notna()
    if passthrough.any():
        text = s[passthrough].astype(str).str.strip()
        labels.loc[passthrough] = text.where(text.isin(BALCONY_CODES), pd.NA)
    return labels


PARSERS = {
    "numeric": _parse_numeric,
    "numeric_or_range": _parse_numeric_or_range,
    "leading_number": _parse_leading_number,
    "balcony_label": _parse_balcony_label,
}


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
    # target field -> named parser (see PARSERS). Unlisted fields are read as
    # plain numbers, which is what the mapped price/area/bedroom columns
    # usually are.
    parse: dict[str, str] = field(default_factory=dict)
    # Target fields that must be non-null on a row for it to survive. price
    # and area are always required (ALWAYS_REQUIRED_ROW_VALUES); a city adds
    # whatever else its model needs.
    required_columns: tuple[str, ...] = ()
    # A raw column stating each ROW's own area basis, plus the mapping from
    # its labels to HomeCast's canonical bases. When declared, only rows
    # whose own basis equals the declared `area_basis` are kept -- see
    # _filter_to_declared_basis for why this is a filter and not a merge.
    area_basis_column: str | None = None
    area_basis_labels: dict[str, str] = field(default_factory=dict)
    # Raw 0/1 columns to count into a single `amenity_count` feature, and the
    # value this feed uses to mean "not recorded" (see _derive_amenity_count).
    amenity_columns: tuple[str, ...] = ()
    amenity_unknown_value: float | None = None


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
                         f"explicitly, not guessed (declare "
                         f"'{UNKNOWN_AREA_BASIS}' if the source genuinely "
                         f"never says, rather than picking the likeliest one)")
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

    parse = dict(data.get("parse") or {})
    bad_parsers = {f: p for f, p in parse.items() if p not in PARSERS}
    if bad_parsers:
        detail = "; ".join(f"'{f}' -> '{p}'" for f, p in sorted(bad_parsers.items()))
        raise ValueError(f"{path}: unknown parser(s) in [parse]: {detail}. "
                         f"Valid parsers: {', '.join(sorted(PARSERS))}")
    unmapped = sorted(set(parse) - set(data["columns"]))
    if unmapped:
        raise ValueError(
            f"{path}: [parse] names field(s) {unmapped} that 'columns' does "
            f"not map -- a parser can only be applied to a mapped column")

    area_basis_column = data.get("area_basis_column")
    area_basis_labels = dict(data.get("area_basis_labels") or {})
    if area_basis_column and not area_basis_labels:
        raise ValueError(
            f"{path}: area_basis_column '{area_basis_column}' is declared but "
            f"area_basis_labels is empty -- the raw labels in that column have "
            f"to be mapped onto {sorted(AREA_BASES)} explicitly")
    if area_basis_labels and not area_basis_column:
        # Failing open here is the dangerous direction: the filter would
        # simply never run, every row of a mixed-basis feed would be kept,
        # and all of them stamped with the declared basis.
        raise ValueError(
            f"{path}: area_basis_labels is declared but area_basis_column is "
            f"not, so nothing says which column those labels live in. The "
            f"per-row basis filter would silently never run and every row "
            f"would be recorded as '{area_basis}' regardless of its real "
            f"basis -- declare area_basis_column, or drop the labels table.")
    bad_labels = {k: v for k, v in area_basis_labels.items() if v not in AREA_BASES}
    if bad_labels:
        raise ValueError(
            f"{path}: area_basis_labels maps to value(s) that are not a "
            f"recognised basis: {bad_labels}. Valid: {sorted(AREA_BASES)}")
    if area_basis_labels and area_basis not in set(area_basis_labels.values()):
        raise ValueError(
            f"{path}: area_basis is '{area_basis}' but area_basis_labels never "
            f"produces it (it produces {sorted(set(area_basis_labels.values()))}) "
            f"-- filtering to the declared basis would keep zero rows")

    required_columns = tuple(data.get("required_columns") or ())
    # "sector" is produced by locality_column, not by the columns mapping.
    declarable = set(data["columns"]) | {"sector"}
    unknown_required = sorted(set(required_columns) - declarable)
    if unknown_required:
        raise ValueError(
            f"{path}: required_columns names field(s) {unknown_required} that "
            f"are not mapped by 'columns' (and are not 'sector'). A misspelt "
            f"name here fails silently in the unsafe direction -- no rows get "
            f"filtered while the config says they are. Mapped fields: "
            f"{sorted(declarable)}")

    amenity_columns = tuple(data.get("amenity_columns") or ())
    amenity_unknown = data.get("amenity_unknown_value")

    return IngestConfig(
        city_key=data.get("city_key", path.stem),
        columns=data["columns"], area_unit=area_unit, area_basis=area_basis,
        price_unit=price_unit, locality_column=data["locality_column"],
        expected_ppsf_range=expected_ppsf_range, parse=parse,
        required_columns=required_columns,
        area_basis_column=area_basis_column,
        area_basis_labels=area_basis_labels,
        amenity_columns=amenity_columns,
        amenity_unknown_value=(None if amenity_unknown is None
                               else float(amenity_unknown)))


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

    # Free-text columns become numbers/labels BEFORE any unit conversion --
    # "1133 - 1384" has no .astype(float).
    for target_field, parser_name in cfg.parse.items():
        df[target_field] = PARSERS[parser_name](df[target_field])

    df["sector"] = _normalise_locality(df["sector"])

    df["area"] = df["area"].astype(float) * _AREA_TO_SQFT[cfg.area_unit]
    df["price"] = df["price"].astype(float) * _PRICE_TO_CRORE[cfg.price_unit]
    df["price_per_sqft"] = df["price"] * 1e7 / df["area"]
    return df


def _normalise_locality(s: pd.Series) -> pd.Series:
    """Fold locality strings that differ only in spacing or case into one key.

    The locality is the model's strongest location signal and the thing both
    baselines group by, so " Banaswadi", "Banaswadi" and "banaswadi" being
    three separate keys is not cosmetic: it splits one locality's listings
    across three encodings, each fit on a third of the evidence, and leaves
    a user who picks the wrong one with a rate learned from a handful of
    rows. Scraped feeds produce exactly this. Lower-case is HomeCast's
    existing convention for locality keys (see the public Gurgaon data);
    presentation is the dashboard's job, not the data's.
    """
    return (s.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
             .str.lower().where(s.notna()))


def _locality_normalisation_log(before: pd.Series, after: pd.Series) -> list[str]:
    merged = before.astype(str).nunique() - after.nunique()
    if merged <= 0:
        return []
    return [f"normalise locality names (trim, collapse spaces, lower-case): "
            f"{merged} duplicate spelling(s) folded into an existing locality "
            f"({after.nunique()} distinct localities remain)"]


def _parse_loss_log(raw: pd.DataFrame, df: pd.DataFrame,
                    cfg: IngestConfig) -> list[str]:
    """How many values each named parser could not read.

    Reported even though the rows may survive: a column that silently failed
    to parse on 30% of rows is a broken config, and the row counts alone
    would not show it.
    """
    log = []
    for target_field in cfg.parse:
        raw_col = cfg.columns[target_field]
        lost = int(df[target_field].isna().sum() - raw[raw_col].isna().sum())
        if lost > 0:
            log.append(f"parse '{raw_col}' -> {target_field} "
                       f"({cfg.parse[target_field]}): {lost} value(s) "
                       f"unreadable, left blank")
    return log


def _filter_to_declared_basis(df: pd.DataFrame,
                              cfg: IngestConfig) -> tuple[pd.DataFrame, list[str]]:
    """Keep only rows whose OWN stated area basis is the declared one.

    A feed that states the basis per row is the good case, and the temptation
    is to keep every row and record the basis alongside. That would be wrong
    here: price_per_sqft is the model's central location signal and the
    target of both baselines, and a super-built-up rate and a plot rate are
    different quantities. Mixing them under one model teaches it that a
    locality is cheap or dear when all that changed is what the area column
    was measuring. So the rows that disagree are dropped, loudly and with
    counts, rather than merged.
    """
    if not cfg.area_basis_column:
        return df, []
    col = cfg.area_basis_column
    if col not in df.columns:
        raise ValueError(
            f"source CSV is missing the declared area_basis_column '{col}'")
    stated = df[col].notna()
    labels = df[col].astype(str).str.strip().where(stated)
    unmapped = sorted(set(labels.dropna().unique()) - set(cfg.area_basis_labels))
    if unmapped:
        raise ValueError(
            f"area_basis_column '{col}' contains label(s) not covered by "
            f"area_basis_labels: {unmapped}. Every distinct label must be "
            f"mapped -- an unmapped one would be dropped without anyone "
            f"deciding it should be.")
    row_basis = labels.map(cfg.area_basis_labels)
    # A row that does not state its basis cannot be vouched for, so it is
    # dropped like any other mismatch rather than being assumed to match.
    keep = row_basis == cfg.area_basis
    counts = row_basis[~keep].value_counts().to_dict()
    n_blank = int((~stated).sum())
    if n_blank:
        counts["(not stated)"] = n_blank
    log = [f"filter to declared area_basis '{cfg.area_basis}' (per-row "
           f"'{col}'): {int((~keep).sum())} row(s) removed "
           f"({int(keep.sum())} remain)"]
    if counts:
        breakdown = ", ".join(f"{basis}: {n}" for basis, n in sorted(counts.items()))
        log.append(f"  ...dropped by their own basis -- {breakdown}. These are "
                   f"real listings, excluded because their area is not "
                   f"measured the same way as the kept rows.")
    return df[keep].reset_index(drop=True), log


def _derive_amenity_count(df: pd.DataFrame,
                          cfg: IngestConfig) -> tuple[pd.DataFrame, list[str]]:
    """Collapse a bank of 0/1 amenity flags into one `amenity_count` feature.

    The trap this handles: these feeds mark "we never captured amenities for
    this listing" with a sentinel value in EVERY flag column, not with a
    blank. Summing naively turns "unknown" into a confident zero -- the
    single most common way an amenity feature ends up looking predictive of
    cheapness when all it has learnt is which listings were poorly recorded.
    A row whose flags are all the unknown sentinel gets NaN, which
    build_features carries through as the documented "not stated" code.
    """
    if not cfg.amenity_columns:
        return df, []
    missing = [c for c in cfg.amenity_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"source CSV is missing declared amenity_columns: "
            f"{', '.join(missing)}")
    flags = df[list(cfg.amenity_columns)].apply(pd.to_numeric, errors="coerce")
    if cfg.amenity_unknown_value is not None:
        flags = flags.mask(flags == cfg.amenity_unknown_value)
    # "Nothing recorded" covers both conventions these feeds use: an explicit
    # sentinel in every flag column, and simply leaving them all blank.
    unknown_row = flags.isna().all(axis=1)
    count = (flags == 1).sum(axis=1).astype(float)
    count[unknown_row] = np.nan
    df = df.copy()
    df["amenity_count"] = count
    n_unknown = int(unknown_row.sum())
    log = [f"derive amenity_count from {len(cfg.amenity_columns)} flag "
           f"column(s): {len(df) - n_unknown} row(s) counted, {n_unknown} "
           f"with no amenities recorded at all (left blank, NOT counted as "
           f"zero)"]
    if n_unknown:
        log.append(f"  ...that is {n_unknown / len(df):.0%} of rows -- treat "
                   f"amenity_count's usefulness for this city with suspicion.")
    return df, log


def _drop_rows_missing_required(df: pd.DataFrame,
                                cfg: IngestConfig) -> tuple[pd.DataFrame, list[str]]:
    """Drop rows with no value in a column the model cannot do without."""
    required = [c for c in dict.fromkeys(
        (*ALWAYS_REQUIRED_ROW_VALUES, *cfg.required_columns)) if c in df.columns]
    before = len(df)
    out = df.dropna(subset=required).reset_index(drop=True)
    removed = before - len(out)
    if not removed:
        return out, [f"drop rows missing a required value ({', '.join(required)}): ok"]
    per_col = ", ".join(f"{c}: {int(df[c].isna().sum())}"
                        for c in required if df[c].isna().any())
    return out, [f"drop rows missing a required value ({', '.join(required)}): "
                 f"{removed} rows removed ({len(out)} remain); blanks per "
                 f"column -- {per_col}"]


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


def _trim_implausible_ppsf(df: pd.DataFrame,
                           cfg: IngestConfig) -> tuple[pd.DataFrame, list[str]]:
    """Drop individual rows whose price_per_sqft falls outside this city's own
    ``expected_ppsf_range`` -- a genuine PER-ROW plausibility trim, distinct
    from the median-only gate in ``_plausibility_check`` above.

    ``_plausibility_check`` only ever looks at the file's MEDIAN
    price_per_sqft: a single mis-typed area, a placeholder price, or an
    otherwise-corrupted listing can sit far outside the plausible band for
    this market while the dataset's median stays exactly where it should be
    -- the median gate has nothing to say about that one row, because a
    handful of outliers among thousands of listings barely moves a median at
    all. This is the row-level complement, and both checks are kept: they
    catch different failures. The median gate catches a wrong unit/basis
    declaration for the WHOLE file (see its own docstring); this one removes
    individual implausible LISTINGS once the file's units are already known
    to be right.

    A no-op -- nothing dropped, the frame returned unchanged -- for a city
    that declares no ``expected_ppsf_range`` at all (e.g. Gurgaon, which does
    not go through this module at all, or any city whose config has not
    stated one yet): there is no per-city band to check a row against, so
    every row is kept exactly as it is.
    """
    if cfg.expected_ppsf_range is None:
        return df, ["trim rows outside expected_ppsf_range: ok (no "
                    "expected_ppsf_range configured for this city)"]
    lo, hi = cfg.expected_ppsf_range
    keep = df["price_per_sqft"].between(lo, hi)
    removed = int((~keep).sum())
    out = df[keep].reset_index(drop=True)
    name = (f"trim rows outside expected_ppsf_range "
            f"(Rs {lo:,.0f}-{hi:,.0f}/sq.ft.)")
    if not removed:
        return out, [f"{name}: ok"]
    return out, [f"{name}: {removed} row(s) removed ({len(out)} remain)"]


def _area_basis_caveat(cfg: IngestConfig) -> list[str]:
    """Surface an undeclarable area basis as a caveat on every ingestion.

    ``area_basis = "unknown"`` is a legitimate declaration (see
    UNKNOWN_AREA_BASIS) but it must never pass quietly: the whole point is
    that the reader of this log learns the basis is unknown at the moment
    the data enters the project, not later from a footnote.
    """
    if cfg.area_basis != UNKNOWN_AREA_BASIS:
        return []
    return [
        f"CAVEAT: area_basis is '{UNKNOWN_AREA_BASIS}' -- this source never "
        f"states whether its areas are carpet, built-up or super-built-up. "
        f"Every price_per_sqft below is therefore on an unknown basis: it is "
        f"internally consistent for this city (so within-city comparisons "
        f"and the model's own baselines are sound), but it is NOT comparable "
        f"with a city whose basis is known, and the true rate could be "
        f"~25-35% higher if these areas turn out to be super-built-up."]


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

    log: list[str] = [f"read {len(raw)} raw rows from {raw_path.name}"]
    log.extend(_parse_loss_log(raw, df, cfg))
    log.extend(_locality_normalisation_log(raw[cfg.locality_column], df["sector"]))

    # Basis filtering comes before the plausibility check: on a mixed feed the
    # bases have very different Rs/sq.ft. (a plot rate is nothing like a
    # super-built-up rate), so the median worth checking is the one for the
    # rows actually being kept.
    df, basis_log = _filter_to_declared_basis(df, cfg)
    log.extend(basis_log)

    _plausibility_check(df, cfg)

    # Per-row plausibility trim (Correction 1), on top of -- not instead of
    # -- the median gate just above. Runs before dedup/amenity/required-column
    # filtering so its own dropped-row count is not diluted by rows that would
    # have been removed for some other reason anyway.
    df, ppsf_trim_log = _trim_implausible_ppsf(df, cfg)
    log.extend(ppsf_trim_log)

    df, amenity_log = _derive_amenity_count(df, cfg)
    log.extend(amenity_log)

    steps = [
        ("drop exact duplicate listings", drop_duplicates),
        ("drop listings without a price", drop_unpriced),
    ]
    for name, step in steps:
        before = len(df)
        df = step(df)
        removed = before - len(df)
        log.append(f"{name}: {removed} rows removed ({len(df)} remain)" if removed
                   else f"{name}: ok")

    df, required_log = _drop_rows_missing_required(df, cfg)
    log.extend(required_log)

    before = len(df)
    df = trim_outliers(df)
    removed = before - len(df)
    name = "trim percentile outliers (price_per_sqft, area)"
    log.append(f"{name}: {removed} rows removed ({len(df)} remain)" if removed
               else f"{name}: ok")

    # area_basis is recorded on every row, not just the config -- carpet vs.
    # super-built-up is not convertible, so anything that later displays or
    # compares price_per_sqft across cities must be able to see the basis
    # without going back to this city's ingestion config.
    df["area_basis"] = cfg.area_basis
    log.extend(_area_basis_caveat(cfg))
    log.extend(_maharashtra_contamination_log(df, cfg))
    return df, log
