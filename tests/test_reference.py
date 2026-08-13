"""Tests for the Amaravathi reference calculator.

Amaravathi has single-digit live listings and no usable dataset -- a
trained model here would be fabrication. This module must therefore never
present itself as a prediction: no statistical confidence band, every input
rate shown with its source and date, an explicit "not a prediction" label,
and a staleness warning past 6 months. It must also guard against
contamination from Amravati, Maharashtra -- an unrelated city with a
near-identical name and far more listings.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from homecast.reference import (
    CITY_DISPLAY, CITY_KEY, STALENESS_DAYS, estimate, guard_maharashtra_contamination,
    load_rate_table, staleness_warning,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = REPO_ROOT / "config" / "amaravathi.rates.example.toml"

GOOD_TABLE = '''
last_updated = "2026-01-01"

[[rate]]
zone = "Seed Capital Zone"
property_type = "flat"
rate_low_per_sqft = 4200
rate_high_per_sqft = 4800
source = "APCRDA zone rate schedule (test fixture)"
source_date = "2026-01-01"

[[rate]]
zone = "Seed Capital Zone"
property_type = "flat"
rate_low_per_sqft = 3900
rate_high_per_sqft = 5100
source = "Vijayawada/Guntur comparable listings (test fixture, regional proxy)"
source_date = "2025-08-01"

[[rate]]
zone = "Thullur"
property_type = "house"
rate_low_per_sqft = 3000
rate_high_per_sqft = 3400
source = "IGRS AP registration value (test fixture)"
source_date = "2025-11-01"
'''


def _write(tmp_path, text) -> Path:
    p = tmp_path / "rates.toml"
    p.write_text(text)
    return p


# --- identity: amaravathi_ap, not amravati/maharashtra ---------------------

def test_city_identity_constants():
    assert CITY_KEY == "amaravathi_ap"
    assert CITY_DISPLAY == "Amaravathi, Andhra Pradesh"


# --- arithmetic correctness -------------------------------------------------

def test_estimate_arithmetic_single_source(tmp_path):
    table = load_rate_table(_write(tmp_path, GOOD_TABLE))
    e = estimate(table, zone="Thullur", property_type="house", area_sqft=1000,
                as_of=dt.date(2026, 1, 15))
    # single matching entry: 3000-3400/sqft * 1000 sqft = 0.30-0.34 Cr
    assert e.low_cr == pytest.approx(0.30)
    assert e.high_cr == pytest.approx(0.34)

def test_estimate_arithmetic_spans_multiple_sources(tmp_path):
    table = load_rate_table(_write(tmp_path, GOOD_TABLE))
    e = estimate(table, zone="Seed Capital Zone", property_type="flat", area_sqft=1000,
                as_of=dt.date(2026, 1, 15))
    # two entries: (4200-4800) and (3900-5100) per sqft * 1000 sqft
    # overall span = min(low)-max(high) = 3900-5100 -> 0.39-0.51 Cr
    assert e.low_cr == pytest.approx(0.39)
    assert e.high_cr == pytest.approx(0.51)
    assert len(e.line_items) == 2

def test_every_line_item_carries_its_own_source_and_date(tmp_path):
    table = load_rate_table(_write(tmp_path, GOOD_TABLE))
    e = estimate(table, zone="Seed Capital Zone", property_type="flat", area_sqft=1000,
                as_of=dt.date(2026, 1, 15))
    sources = {li.source for li in e.line_items}
    assert "APCRDA zone rate schedule (test fixture)" in sources
    assert "Vijayawada/Guntur comparable listings (test fixture, regional proxy)" in sources
    for li in e.line_items:
        assert li.source_date  # every rate shows when it's from
    text = e.explain()
    assert "APCRDA zone rate schedule (test fixture)" in text
    assert "2026-01-01" in text
    assert "Vijayawada/Guntur comparable listings" in text
    assert "2025-08-01" in text


# --- explicit "not a prediction" labelling ----------------------------------

def test_output_is_labelled_not_a_prediction(tmp_path):
    table = load_rate_table(_write(tmp_path, GOOD_TABLE))
    e = estimate(table, zone="Thullur", property_type="house", area_sqft=1000,
                as_of=dt.date(2026, 1, 15))
    assert "not a prediction" in e.label.lower()
    assert "not a prediction" in e.explain().lower()

def test_output_refuses_statistical_confidence_framing(tmp_path):
    """No p10/p90-style band, no 'confidence interval' language implying
    statistical inference -- this is arithmetic over a rate table."""
    table = load_rate_table(_write(tmp_path, GOOD_TABLE))
    e = estimate(table, zone="Thullur", property_type="house", area_sqft=1000,
                as_of=dt.date(2026, 1, 15))
    text = e.explain().lower()
    # the disclaimer is allowed to name and explicitly NEGATE "confidence
    # interval"; what must never appear is percentile/statistical-inference
    # language presented as if it were real (that's the model's job, not
    # this calculator's).
    assert "not a statistical" in text
    assert "percentile" not in text and "p10" not in text and "p90" not in text


# --- staleness ---------------------------------------------------------------

def test_staleness_warning_fires_past_six_months(tmp_path):
    table = load_rate_table(_write(tmp_path, GOOD_TABLE))  # last_updated 2026-01-01
    warning = staleness_warning(table, as_of=dt.date(2026, 8, 15))  # ~7 months later
    assert warning is not None
    assert "stale" in warning.lower() or "warning" in warning.lower()

def test_staleness_warning_silent_when_recent(tmp_path):
    table = load_rate_table(_write(tmp_path, GOOD_TABLE))
    warning = staleness_warning(table, as_of=dt.date(2026, 2, 1))  # ~1 month later
    assert warning is None

def test_staleness_days_is_about_six_months():
    assert 175 <= STALENESS_DAYS <= 190

def test_stale_table_warning_surfaces_in_estimate(tmp_path):
    table = load_rate_table(_write(tmp_path, GOOD_TABLE))
    e = estimate(table, zone="Thullur", property_type="house", area_sqft=1000,
                as_of=dt.date(2026, 9, 1))
    assert any("stale" in w.lower() or "warning" in w.lower() for w in e.warnings)
    assert any("stale" in w.lower() or "warning" in w.lower() for w in e.explain().lower().split("\n"))


# --- refusal paths -------------------------------------------------------

def test_unknown_zone_is_rejected(tmp_path):
    table = load_rate_table(_write(tmp_path, GOOD_TABLE))
    with pytest.raises(ValueError, match="Thullur"):
        estimate(table, zone="Atlantis Zone", property_type="flat", area_sqft=1000)

def test_non_positive_area_is_rejected(tmp_path):
    table = load_rate_table(_write(tmp_path, GOOD_TABLE))
    with pytest.raises(ValueError, match="area_sqft"):
        estimate(table, zone="Thullur", property_type="house", area_sqft=0)

def test_table_missing_last_updated_is_rejected(tmp_path):
    bad = GOOD_TABLE.replace('last_updated = "2026-01-01"\n', '')
    with pytest.raises(ValueError, match="last_updated"):
        load_rate_table(_write(tmp_path, bad))

def test_table_with_no_entries_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="rate"):
        load_rate_table(_write(tmp_path, 'last_updated = "2026-01-01"\n'))

def test_inverted_low_high_rate_is_rejected(tmp_path):
    bad = GOOD_TABLE.replace("rate_low_per_sqft = 3000", "rate_low_per_sqft = 9000")
    with pytest.raises(ValueError, match="Thullur"):
        load_rate_table(_write(tmp_path, bad))


# --- Amravati, Maharashtra contamination guard ------------------------------

def test_guard_flags_maharashtra_locality():
    w = guard_maharashtra_contamination("Amravati, Maharashtra")
    assert w is not None
    assert "Maharashtra" in w

def test_guard_does_not_flag_amaravathi_ap_locality():
    assert guard_maharashtra_contamination("Thullur, Amaravathi, Andhra Pradesh") is None
    assert guard_maharashtra_contamination("Seed Capital Zone") is None

def test_guard_flags_vidarbha_region_mention():
    assert guard_maharashtra_contamination("Camp Area, Vidarbha region") is not None

def test_estimate_surfaces_maharashtra_guard_via_locality(tmp_path):
    table = load_rate_table(_write(tmp_path, GOOD_TABLE))
    e = estimate(table, zone="Thullur", property_type="house", area_sqft=1000,
                locality="Amravati, Maharashtra", as_of=dt.date(2026, 1, 15))
    assert any("Maharashtra" in w for w in e.warnings)

def test_estimate_no_warning_for_clean_ap_locality(tmp_path):
    table = load_rate_table(_write(tmp_path, GOOD_TABLE))
    e = estimate(table, zone="Thullur", property_type="house", area_sqft=1000,
                locality="Thullur, Amaravathi", as_of=dt.date(2026, 1, 15))
    assert e.warnings == ()


# --- the committed example config -------------------------------------------

def test_example_config_exists_and_loads():
    assert EXAMPLE_CONFIG.exists(), (
        "config/amaravathi.rates.example.toml must be committed")
    table = load_rate_table(EXAMPLE_CONFIG)
    assert len(table.entries) > 0

def test_example_config_is_clearly_marked_as_fake():
    text = EXAMPLE_CONFIG.read_text().lower()
    assert "fake" in text or "placeholder" in text or "example" in text

def test_example_config_cites_real_sources_in_comments():
    text = EXAMPLE_CONFIG.read_text().lower()
    assert "apcrda" in text
    assert "igrs" in text
    assert "vijayawada" in text or "guntur" in text
    assert "private/" in text  # points to where the real table lives
