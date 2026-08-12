import pandas as pd
import pytest

from homecast.cleaning import clean_raw_data, clean_city


def test_drops_exact_duplicates(raw_fixture):
    cleaned, log = clean_raw_data(raw_fixture)
    assert not cleaned.duplicated().any()
    assert any("duplicate" in line and "2 rows removed" in line for line in log)

def test_drops_unpriced_rows(clean_fixture):
    assert clean_fixture["price"].notna().all()

def test_trims_unit_error_outliers(clean_fixture):
    assert clean_fixture["area"].max() < 875000
    assert clean_fixture["price_per_sqft"].max() < 600000

def test_labels_furnishing(clean_fixture):
    assert set(clean_fixture["furnishing_type"].unique()) <= {
        "unfurnished", "semi-furnished", "furnished"}

def test_undefined_age_becomes_missing(clean_fixture):
    assert "Undefined" not in clean_fixture["age_possession"].dropna().values

def test_row_accounting_matches_log(raw_fixture):
    cleaned, log = clean_raw_data(raw_fixture)
    removed = sum(int(l.split(": ")[1].split(" ")[0]) for l in log if "removed" in l)
    assert len(raw_fixture) - removed == len(cleaned)

def test_clean_city_gurgaon_real_data():
    cleaned, log = clean_city("gurgaon")
    assert len(cleaned) == 3600          # the published number — must reproduce
    assert "sector" in cleaned.columns

def test_clean_city_unknown():
    with pytest.raises(ValueError):
        clean_city("atlantis")
