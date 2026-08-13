"""Tests for config-driven ingestion with fail-fast unit validation.

Indian property data mixes sq.ft./sq.m./sq.yards, rupees/lakh/crore, and
carpet/built-up/super-built-up area bases. Silently guessing any of these
would corrupt price_per_sqft with no error thrown, so every source declares
its units explicitly and ingestion refuses to proceed on a missing or
implausible declaration -- each crafted bad-CSV test below is one such
failure mode.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from homecast.ingest import ingest_city, load_config


def _write_csv(path: Path, df: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _write_config(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


GOOD_CONFIG = '''
city_key = "testcity"
area_unit = "sqft"
area_basis = "superbuiltup"
price_unit = "lakh"
locality_column = "locality"

[columns]
price = "price_lakh"
area = "area_sqft"
property_type = "ptype"
bedrooms = "bhk"
bathrooms = "bath"
'''


def _good_raw_df(n=20, price_lakh=None, area_sqft=None, locality=None):
    price_lakh = price_lakh if price_lakh is not None else [80 + i for i in range(n)]
    area_sqft = area_sqft if area_sqft is not None else [1200 + 10 * i for i in range(n)]
    locality = locality if locality is not None else ["thullur"] * n
    return pd.DataFrame({
        "price_lakh": price_lakh,
        "area_sqft": area_sqft,
        "ptype": ["flat"] * n,
        "bhk": [3] * n,
        "bath": [2] * n,
        "locality": locality,
    })


# --- happy path --------------------------------------------------------

def test_ingest_produces_cleaned_schema_with_area_basis(tmp_path):
    raw = _write_csv(tmp_path / "raw.csv", _good_raw_df())
    cfg = _write_config(tmp_path / "config.toml", GOOD_CONFIG)
    df, log = ingest_city(raw, cfg)
    assert (df["area_basis"] == "superbuiltup").all()
    assert "price_per_sqft" in df.columns
    assert "sector" in df.columns  # locality_column renamed to sector
    assert any("duplicate" in line for line in log)

def test_ingest_price_per_sqft_arithmetic_is_correct(tmp_path):
    # 1 row: price = 80 lakh = 0.8 crore, area = 1600 sqft.
    # price_per_sqft = 0.8 * 1e7 / 1600 = 5000
    raw_df = _good_raw_df(n=1, price_lakh=[80], area_sqft=[1600])
    raw = _write_csv(tmp_path / "raw.csv", raw_df)
    cfg = _write_config(tmp_path / "config.toml", GOOD_CONFIG)
    df, _ = ingest_city(raw, cfg)
    assert df["price_per_sqft"].iloc[0] == pytest.approx(5000.0)

def test_ingest_reuses_duplicate_and_outlier_handling(tmp_path):
    raw_df = _good_raw_df(n=200)
    raw_df = pd.concat([raw_df, raw_df.iloc[[0]]], ignore_index=True)  # exact dup
    raw = _write_csv(tmp_path / "raw.csv", raw_df)
    cfg = _write_config(tmp_path / "config.toml", GOOD_CONFIG)
    df, log = ingest_city(raw, cfg)
    assert not df.duplicated().any()
    assert any("duplicate" in line and "1 rows removed" in line for line in log)


# --- failure mode 1: missing declaration --------------------------------

def test_missing_area_basis_declaration_is_rejected(tmp_path):
    bad_config = GOOD_CONFIG.replace('area_basis = "superbuiltup"\n', '')
    cfg = _write_config(tmp_path / "config.toml", bad_config)
    with pytest.raises(ValueError, match="area_basis"):
        load_config(cfg)

def test_missing_locality_column_declaration_is_rejected(tmp_path):
    bad_config = GOOD_CONFIG.replace('locality_column = "locality"\n', '')
    cfg = _write_config(tmp_path / "config.toml", bad_config)
    with pytest.raises(ValueError, match="locality_column"):
        load_config(cfg)

def test_missing_columns_price_or_area_target_is_rejected(tmp_path):
    bad_config = GOOD_CONFIG.replace('price = "price_lakh"\n', '')
    raw = _write_csv(tmp_path / "raw.csv", _good_raw_df())
    cfg = _write_config(tmp_path / "config.toml", bad_config)
    with pytest.raises(ValueError, match="price"):
        ingest_city(raw, cfg)


# --- failure mode 2: invalid unit token ---------------------------------

def test_invalid_area_unit_token_is_rejected(tmp_path):
    bad_config = GOOD_CONFIG.replace('area_unit = "sqft"', 'area_unit = "hectares"')
    cfg = _write_config(tmp_path / "config.toml", bad_config)
    with pytest.raises(ValueError, match="area_unit"):
        load_config(cfg)

def test_invalid_price_unit_token_is_rejected(tmp_path):
    bad_config = GOOD_CONFIG.replace('price_unit = "lakh"', 'price_unit = "dollars"')
    cfg = _write_config(tmp_path / "config.toml", bad_config)
    with pytest.raises(ValueError, match="price_unit"):
        load_config(cfg)

def test_invalid_area_basis_token_is_rejected(tmp_path):
    bad_config = GOOD_CONFIG.replace('area_basis = "superbuiltup"', 'area_basis = "plot"')
    cfg = _write_config(tmp_path / "config.toml", bad_config)
    with pytest.raises(ValueError, match="area_basis"):
        load_config(cfg)


# --- failure mode 3: wrong price_unit (plausibility) ---------------------

def test_price_declared_rupees_but_actually_lakh_is_rejected(tmp_path):
    """Prices that are genuinely lakh-scale (e.g. 80) but declared 'rupees'
    would imply Rs 80/sqft-ish flats -- price_per_sqft collapses far below
    the sane band. The error must name price_unit as the likely culprit."""
    wrong_config = GOOD_CONFIG.replace('price_unit = "lakh"', 'price_unit = "rupees"')
    raw = _write_csv(tmp_path / "raw.csv", _good_raw_df())
    cfg = _write_config(tmp_path / "config.toml", wrong_config)
    with pytest.raises(ValueError, match="price_unit"):
        ingest_city(raw, cfg)

def test_price_unit_suggestion_points_at_a_smaller_unit_when_too_high(tmp_path):
    """declared 'crore' but the raw numbers are actually lakh-scale inflates
    price by 100x -- too HIGH a ppsf means the declared factor is too big,
    so the fix must be a SMALLER unit ('lakh'), not a bigger one."""
    wrong_config = GOOD_CONFIG.replace('price_unit = "lakh"', 'price_unit = "crore"')
    raw = _write_csv(tmp_path / "raw.csv", _good_raw_df())
    cfg = _write_config(tmp_path / "config.toml", wrong_config)
    with pytest.raises(ValueError, match=r"try 'lakh' instead of 'crore'"):
        ingest_city(raw, cfg)

def test_price_declared_lakh_but_actually_rupees_is_rejected(tmp_path):
    """Full-rupee prices (e.g. 8,000,000) mislabelled as 'lakh' inflate
    price_per_sqft by 1e5x -- the error must flag price_unit."""
    wrong_config = GOOD_CONFIG.replace('price_unit = "lakh"', 'price_unit = "crore"')
    raw_df = _good_raw_df(n=5, price_lakh=[80] * 5)  # actually crore-scale numbers
    raw = _write_csv(tmp_path / "raw.csv", raw_df)
    cfg = _write_config(tmp_path / "config.toml", wrong_config)
    with pytest.raises(ValueError, match="price_unit") as exc_info:
        ingest_city(raw, cfg)
    # the message must name the actual computed number, not just say "out of range"
    assert "Rs" in str(exc_info.value)


# --- failure mode 4: sq.yd/sq.m declared as sq.ft (plausibility) ---------

def test_area_declared_sqft_but_actually_sqm_is_rejected(tmp_path):
    """Typical small-flat areas in sq.m (40-90) treated as sq.ft directly
    (no 10.76x conversion) collapse the median area below the plausible
    100 sqft floor for a residence -- the error must flag area_unit."""
    raw_df = _good_raw_df(n=5, area_sqft=[40, 55, 65, 75, 90])  # actually sq.m values
    raw = _write_csv(tmp_path / "raw.csv", raw_df)
    cfg = _write_config(tmp_path / "config.toml", GOOD_CONFIG)  # declares sqft
    with pytest.raises(ValueError, match="area_unit"):
        ingest_city(raw, cfg)

def test_area_declared_sqft_but_actually_sqyd_inflates_ppsf(tmp_path):
    """A plot genuinely 110 sq.yd, treated as 110 sq.ft directly (missing
    the 9x conversion), pushes price_per_sqft far past the sane band even
    though the area itself still looks plausible -- the error must mention
    both area_unit and price_unit as possible causes of an out-of-band ppsf."""
    raw_df = _good_raw_df(n=5, price_lakh=[200] * 5, area_sqft=[110] * 5)
    raw = _write_csv(tmp_path / "raw.csv", raw_df)
    cfg = _write_config(tmp_path / "config.toml", GOOD_CONFIG)  # declares sqft
    with pytest.raises(ValueError) as exc_info:
        ingest_city(raw, cfg)
    msg = str(exc_info.value)
    assert "price_unit" in msg and "area_unit" in msg


# --- failure mode 5: non-positive price -----------------------------------

def test_non_positive_price_is_rejected(tmp_path):
    raw_df = _good_raw_df(n=5, price_lakh=[80, 0, -5, 90, 85])
    raw = _write_csv(tmp_path / "raw.csv", raw_df)
    cfg = _write_config(tmp_path / "config.toml", GOOD_CONFIG)
    with pytest.raises(ValueError, match="non-positive"):
        ingest_city(raw, cfg)


# --- Amravati, Maharashtra contamination guard (Task 7) -----------------

def test_ingest_warns_on_maharashtra_localities_for_amaravathi_ap(tmp_path):
    ap_config = GOOD_CONFIG.replace('city_key = "testcity"', 'city_key = "amaravathi_ap"')
    raw_df = _good_raw_df(n=6, locality=["Thullur"] * 3 + ["Amravati, Maharashtra"] * 3)
    raw = _write_csv(tmp_path / "raw.csv", raw_df)
    cfg = _write_config(tmp_path / "config.toml", ap_config)
    df, log = ingest_city(raw, cfg)
    assert any("Maharashtra" in line for line in log)

def test_ingest_silent_for_other_cities_even_with_odd_localities(tmp_path):
    """The guard is specific to amaravathi_ap -- a Gurgaon-style config with
    a locality that happens to contain 'maharashtra' must not warn."""
    raw_df = _good_raw_df(n=3, locality=["some place in maharashtra"] * 3)
    raw = _write_csv(tmp_path / "raw.csv", raw_df)
    cfg = _write_config(tmp_path / "config.toml", GOOD_CONFIG)  # city_key = testcity
    df, log = ingest_city(raw, cfg)
    assert not any("Maharashtra" in line for line in log)

def test_ingest_warns_even_when_city_key_defaults_from_a_amaravathi_filename_stem(tmp_path):
    """I9: load_config defaults city_key to the config file's stem when it
    isn't declared explicitly. A config saved as amaravathi.toml (stem
    'amaravathi', not the exact 'amaravathi_ap') must still trigger the
    guard -- an exact-match check would silently skip it."""
    no_explicit_key = GOOD_CONFIG.replace('city_key = "testcity"\n', '')
    raw_df = _good_raw_df(n=6, locality=["Thullur"] * 3 + ["Amravati, Maharashtra"] * 3)
    raw = _write_csv(tmp_path / "raw.csv", raw_df)
    cfg = _write_config(tmp_path / "amaravathi.toml", no_explicit_key)
    df, log = ingest_city(raw, cfg)
    assert any("Maharashtra" in line for line in log)


# --- area_basis consistency across re-ingestion (Task I7) -----------------

def test_reingest_with_a_different_area_basis_is_rejected(tmp_path):
    raw = _write_csv(tmp_path / "raw.csv", _good_raw_df())
    cfg = _write_config(tmp_path / "config.toml", GOOD_CONFIG)  # declares superbuiltup
    processed = tmp_path / "processed.csv"
    df, _ = ingest_city(raw, cfg)
    df.to_csv(processed, index=False)

    changed_basis = GOOD_CONFIG.replace('area_basis = "superbuiltup"', 'area_basis = "carpet"')
    changed_cfg = _write_config(tmp_path / "config2.toml", changed_basis)
    with pytest.raises(ValueError, match="area_basis"):
        ingest_city(raw, changed_cfg, existing_processed_path=processed)

def test_reingest_with_the_same_area_basis_is_allowed(tmp_path):
    raw = _write_csv(tmp_path / "raw.csv", _good_raw_df())
    cfg = _write_config(tmp_path / "config.toml", GOOD_CONFIG)
    processed = tmp_path / "processed.csv"
    df, _ = ingest_city(raw, cfg)
    df.to_csv(processed, index=False)

    df2, _ = ingest_city(raw, cfg, existing_processed_path=processed)
    assert (df2["area_basis"] == "superbuiltup").all()

def test_first_ingestion_with_no_existing_processed_file_is_unaffected(tmp_path):
    """existing_processed_path pointing at a file that doesn't exist yet (the
    very first ingestion for a city) must not raise."""
    raw = _write_csv(tmp_path / "raw.csv", _good_raw_df())
    cfg = _write_config(tmp_path / "config.toml", GOOD_CONFIG)
    df, _ = ingest_city(raw, cfg, existing_processed_path=tmp_path / "does_not_exist.csv")
    assert (df["area_basis"] == "superbuiltup").all()


# --- per-city expected_ppsf_range (Task I8) --------------------------------

def _sqyd_declared_as_sqft_raw_df():
    """200-500 sq.yd plots, entered directly as if they were already sq.ft
    (no 9x conversion applied) -- median declared area lands at 350, still
    comfortably inside AREA_BAND, and a constant Rs 67,148/sq.ft. median
    lands comfortably inside the global PPSF_BAND (1,000-100,000) too. Both
    global checks pass; only a tighter, city-specific expected_ppsf_range
    catches this."""
    area_sqft = [200, 250, 300, 350, 400, 450, 500]
    ppsf = 67_148.0
    price_lakh = [ppsf * a / 1e5 for a in area_sqft]
    return _good_raw_df(n=len(area_sqft), price_lakh=price_lakh, area_sqft=area_sqft)

def test_sqyd_as_sqft_error_slips_through_without_a_per_city_range(tmp_path):
    """Documents the known gap I8 exists to close: the global bands alone
    let this through silently."""
    raw = _write_csv(tmp_path / "raw.csv", _sqyd_declared_as_sqft_raw_df())
    cfg = _write_config(tmp_path / "config.toml", GOOD_CONFIG)
    df, _ = ingest_city(raw, cfg)  # does not raise
    assert df["area"].median() == pytest.approx(350.0)

def test_sqyd_as_sqft_error_is_caught_with_a_per_city_range(tmp_path):
    # Must be inserted BEFORE the `[columns]` table header -- appended after
    # it, a bare `key = value` line would parse as columns.expected_ppsf_range
    # (TOML: unheadered keys belong to the most recently opened table).
    tight_range_config = GOOD_CONFIG.replace(
        "\n[columns]", "\nexpected_ppsf_range = [3000, 8000]\n\n[columns]")
    raw = _write_csv(tmp_path / "raw.csv", _sqyd_declared_as_sqft_raw_df())
    cfg = _write_config(tmp_path / "config.toml", tight_range_config)
    with pytest.raises(ValueError, match="expected_ppsf_range"):
        ingest_city(raw, cfg)

def test_expected_ppsf_range_is_optional(tmp_path):
    """A config with no expected_ppsf_range at all must ingest exactly as
    before -- this is an additive, opt-in check."""
    raw = _write_csv(tmp_path / "raw.csv", _good_raw_df())
    cfg = _write_config(tmp_path / "config.toml", GOOD_CONFIG)
    cfg_obj = load_config(cfg)
    assert cfg_obj.expected_ppsf_range is None

def test_invalid_expected_ppsf_range_is_rejected(tmp_path):
    bad_range_config = GOOD_CONFIG.replace(       # low > high
        "\n[columns]", "\nexpected_ppsf_range = [8000, 3000]\n\n[columns]")
    cfg = _write_config(tmp_path / "config.toml", bad_range_config)
    with pytest.raises(ValueError, match="expected_ppsf_range"):
        load_config(cfg)


# --- duplicate rename targets (Task M16) -----------------------------------

def test_two_targets_mapped_to_the_same_raw_column_is_rejected(tmp_path):
    dup_config = GOOD_CONFIG.replace('area = "area_sqft"', 'area = "price_lakh"')  # both -> price_lakh/area now collide
    raw = _write_csv(tmp_path / "raw.csv", _good_raw_df())
    cfg = _write_config(tmp_path / "config.toml", dup_config)
    with pytest.raises(ValueError, match="more than one target field"):
        ingest_city(raw, cfg)

def test_source_column_colliding_with_a_rename_target_is_rejected(tmp_path):
    """The raw CSV already has a column literally called 'price' (not one of
    the declared source columns) while 'price_lakh' is separately mapped to
    become 'price' -- pandas' rename() would silently produce two 'price'
    columns."""
    raw_df = _good_raw_df()
    raw_df["price"] = 999.0  # pre-existing column that collides with the rename target
    raw = _write_csv(tmp_path / "raw.csv", raw_df)
    cfg = _write_config(tmp_path / "config.toml", GOOD_CONFIG)
    with pytest.raises(ValueError, match="collide"):
        ingest_city(raw, cfg)


# --- CLI wiring -------------------------------------------------------

def test_cli_ingest_writes_processed_csv(tmp_path, monkeypatch, capsys):
    import homecast.cities as cities_module
    from homecast import cli

    raw_path = tmp_path / "raw" / "testcity.csv"
    _write_csv(raw_path, _good_raw_df())
    city = cities_module.City("testcity", "Test City", raw_path,
                              tmp_path / "processed" / "listings_clean.csv",
                              tmp_path / "models")
    monkeypatch.setitem(cities_module.CITIES, "testcity", city)
    cfg = _write_config(tmp_path / "config.toml", GOOD_CONFIG)

    rc = cli.main(["ingest", "--city", "testcity", "--config", str(cfg)])
    assert rc == 0
    assert city.processed_path.exists()
    out_df = pd.read_csv(city.processed_path)
    assert "area_basis" in out_df.columns
    assert "Saved" in capsys.readouterr().out

def test_cli_ingest_bad_config_is_friendly(tmp_path, monkeypatch, capsys):
    import homecast.cities as cities_module
    from homecast import cli

    raw_path = tmp_path / "raw" / "testcity.csv"
    _write_csv(raw_path, _good_raw_df())
    city = cities_module.City("testcity", "Test City", raw_path,
                              tmp_path / "processed" / "listings_clean.csv",
                              tmp_path / "models")
    monkeypatch.setitem(cities_module.CITIES, "testcity", city)
    bad_config = GOOD_CONFIG.replace('area_basis = "superbuiltup"\n', '')
    cfg = _write_config(tmp_path / "config.toml", bad_config)

    rc = cli.main(["ingest", "--city", "testcity", "--config", str(cfg)])
    assert rc == 2
    assert "area_basis" in capsys.readouterr().err
