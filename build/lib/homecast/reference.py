"""Rate-based reference calculator for Amaravathi, Andhra Pradesh.

Amaravathi has single-digit live listings and no usable dataset (see the
Phase 2 research note on city data availability) -- a trained model here
would be fabrication dressed up as a prediction. This module is deliberately
NOT a model: it looks up a manually-maintained rate table (APCRDA zone
rates / IGRS AP registration values / Vijayawada-Guntur comparables), shows
the arithmetic and every input rate's own source and date, and refuses
every trapping of a statistical prediction -- no confidence band, an
explicit "not a prediction" label, and a staleness warning once the table
is more than ~6 months old.

City identity: key ``amaravathi_ap``, display "Amaravathi, Andhra Pradesh"
-- deliberately distinct from Amravati, Maharashtra, an unrelated city with
a near-identical name and a much larger, unrelated listings market. See
``guard_maharashtra_contamination``.
"""
from __future__ import annotations

import datetime as _dt
import tomllib
from dataclasses import dataclass
from pathlib import Path

CITY_KEY = "amaravathi_ap"
CITY_DISPLAY = "Amaravathi, Andhra Pradesh"

STALENESS_DAYS = 183  # ~6 months

REFERENCE_LABEL = ("Rate-based reference for Amaravathi, Andhra Pradesh -- "
                   "not a prediction")

# Substrings that flag a locality string as probably naming Amravati,
# MAHARASHTRA (an unrelated city, PIN range 444xxx, Vidarbha region) rather
# than Amaravathi, ANDHRA PRADESH (the AP capital-region city this
# calculator covers). "amravati" (without the extra 'a' that "amaravathi"
# has) is not a substring of "amaravathi", so this does not false-positive
# on the correct city's own name -- see test_guard_does_not_flag_*.
MAHARASHTRA_MARKERS = ("amravati", "maharashtra", "vidarbha")


@dataclass(frozen=True)
class RateEntry:
    zone: str
    property_type: str
    rate_low_per_sqft: float
    rate_high_per_sqft: float
    source: str
    source_date: str


@dataclass(frozen=True)
class RateTable:
    last_updated: str
    entries: tuple[RateEntry, ...]

    def lookup(self, zone: str, property_type: str) -> tuple[RateEntry, ...]:
        matches = tuple(e for e in self.entries
                        if e.zone == zone and e.property_type == property_type)
        if not matches:
            zones = sorted({e.zone for e in self.entries})
            raise ValueError(
                f"No rate on file for zone='{zone}', property_type='{property_type}'. "
                f"Known zones: {', '.join(zones)}")
        return matches


def load_rate_table(path: Path) -> RateTable:
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    if "last_updated" not in data:
        raise ValueError(f"{path} must declare 'last_updated' (ISO date) so "
                         f"the calculator can warn when the table goes stale")
    if not data.get("rate"):
        raise ValueError(f"{path} has no [[rate]] entries -- nothing to look up")
    entries = []
    for r in data["rate"]:
        lo, hi = float(r["rate_low_per_sqft"]), float(r["rate_high_per_sqft"])
        if lo > hi:
            raise ValueError(
                f"{path}: rate for zone='{r['zone']}' property_type="
                f"'{r['property_type']}' has rate_low_per_sqft ({lo}) > "
                f"rate_high_per_sqft ({hi}) -- check the entry")
        entries.append(RateEntry(zone=r["zone"], property_type=r["property_type"],
                                 rate_low_per_sqft=lo, rate_high_per_sqft=hi,
                                 source=r["source"], source_date=r["source_date"]))
    return RateTable(last_updated=data["last_updated"], entries=tuple(entries))


def staleness_warning(table: RateTable, as_of: _dt.date | None = None) -> str | None:
    as_of = as_of or _dt.date.today()
    updated = _dt.date.fromisoformat(table.last_updated)
    age_days = (as_of - updated).days
    if age_days > STALENESS_DAYS:
        return (f"WARNING: rate table last updated {table.last_updated} "
                f"({age_days} days ago) -- stale, past the {STALENESS_DAYS}-day "
                f"(~6 month) threshold. Verify against current APCRDA/IGRS AP "
                f"figures before relying on this estimate.")
    return None


def guard_maharashtra_contamination(locality: str) -> str | None:
    """Return a warning if `locality` looks like it names Amravati,
    MAHARASHTRA rather than Amaravathi, Andhra Pradesh -- None otherwise."""
    low = locality.lower()
    for marker in MAHARASHTRA_MARKERS:
        if marker in low:
            return (f"WARNING: locality '{locality}' looks like it may refer to "
                    f"Amravati, Maharashtra (a different, unrelated city) rather "
                    f"than Amaravathi, Andhra Pradesh -- verify before trusting "
                    f"this row.")
    return None


@dataclass(frozen=True)
class LineItem:
    source: str
    source_date: str
    rate_low_per_sqft: float
    rate_high_per_sqft: float
    low_cr: float
    high_cr: float


@dataclass(frozen=True)
class ReferenceEstimate:
    label: str
    zone: str
    property_type: str
    area_sqft: float
    low_cr: float
    high_cr: float
    line_items: tuple[LineItem, ...]
    table_last_updated: str
    warnings: tuple[str, ...]

    def explain(self) -> str:
        lines = [
            self.label,
            f"  zone: {self.zone} | property_type: {self.property_type} | "
            f"area: {self.area_sqft:,.0f} sqft",
            f"  rate table last updated: {self.table_last_updated}",
            "  inputs:",
        ]
        for li in self.line_items:
            lines.append(
                f"    - {li.source} (as of {li.source_date}): "
                f"Rs {li.rate_low_per_sqft:,.0f}-{li.rate_high_per_sqft:,.0f}/sqft "
                f"x {self.area_sqft:,.0f} sqft = Rs {li.low_cr:.2f}-{li.high_cr:.2f} Cr")
        lines.append(
            f"  range shown: Rs {self.low_cr:.2f}-{self.high_cr:.2f} Cr -- the "
            f"span across the input rates above (arithmetic, not a statistical "
            f"confidence interval). This is a rate-based reference, not a "
            f"prediction.")
        for w in self.warnings:
            lines.append(f"  {w}")
        return "\n".join(lines)


def estimate(table: RateTable, zone: str, property_type: str, area_sqft: float,
            locality: str | None = None, as_of: _dt.date | None = None) -> ReferenceEstimate:
    if area_sqft <= 0:
        raise ValueError(f"area_sqft must be positive, got {area_sqft}")
    entries = table.lookup(zone, property_type)
    line_items = tuple(
        LineItem(source=e.source, source_date=e.source_date,
                 rate_low_per_sqft=e.rate_low_per_sqft,
                 rate_high_per_sqft=e.rate_high_per_sqft,
                 low_cr=e.rate_low_per_sqft * area_sqft / 1e7,
                 high_cr=e.rate_high_per_sqft * area_sqft / 1e7)
        for e in entries)
    low_cr = min(li.low_cr for li in line_items)
    high_cr = max(li.high_cr for li in line_items)

    warnings = []
    stale = staleness_warning(table, as_of)
    if stale:
        warnings.append(stale)
    if locality is not None:
        mh_warning = guard_maharashtra_contamination(locality)
        if mh_warning:
            warnings.append(mh_warning)

    return ReferenceEstimate(
        label=REFERENCE_LABEL, zone=zone, property_type=property_type,
        area_sqft=area_sqft, low_cr=low_cr, high_cr=high_cr,
        line_items=line_items, table_last_updated=table.last_updated,
        warnings=tuple(warnings))
