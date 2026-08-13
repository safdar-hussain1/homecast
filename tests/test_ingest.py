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
from homecast.ingest import (AREA_BASES, KNOWN_AREA_BASES, PARSERS,
                             UNKNOWN_AREA_BASIS)


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


# --- shared helpers for the newer declarations ----------------------------

def _config_with_top_level(extra: str) -> str:
    """GOOD_CONFIG with extra top-level key(s) spliced in before `[columns]`.

    TOML puts an unheadered `key = value` line into the most recently opened
    table, so appending one after `[columns]` would silently create
    `columns.<key>` instead -- the declaration would be read as a column
    mapping and the feature under test would simply never switch on.
    """
    return GOOD_CONFIG.replace("\n[columns]", f"\n{extra}\n\n[columns]")


def _flat_raw_df(n, locality=None, **extra_columns):
    """`n` rows that survive every plausibility band and every trim.

    price and area are held CONSTANT (80 lakh / 1600 sqft -> Rs 5,000 per
    sq.ft), so the 0.5th/99.5th percentile trim in `trim_outliers` cannot
    quietly remove the very rows a test is asserting about -- with varying
    values it removes the min and max row of both columns. Localities are
    made distinct so `drop_duplicates` does not collapse otherwise-identical
    rows either.
    """
    locality = locality if locality is not None else [f"sector{i}" for i in range(n)]
    df = _good_raw_df(n=n, price_lakh=[80] * n, area_sqft=[1600] * n,
                      locality=locality)
    for name, values in extra_columns.items():
        df[name] = values
    return df


# --- area_basis = "unknown": recorded, not laundered ----------------------

UNKNOWN_CONFIG = GOOD_CONFIG.replace('area_basis = "superbuiltup"',
                                     'area_basis = "unknown"')


def test_area_basis_unknown_is_an_accepted_declaration(tmp_path):
    """Plenty of third-party feeds never state their basis. If "unknown" were
    rejected outright, the only way to ingest such a city would be to guess
    "superbuiltup" because it is the commonest -- i.e. the refusal would
    manufacture exactly the silent corruption this module exists to prevent."""
    cfg = _write_config(tmp_path / "config.toml", UNKNOWN_CONFIG)
    cfg_obj = load_config(cfg)                      # must not raise
    assert cfg_obj.area_basis == UNKNOWN_AREA_BASIS
    assert UNKNOWN_AREA_BASIS not in KNOWN_AREA_BASES
    assert AREA_BASES == KNOWN_AREA_BASES | {UNKNOWN_AREA_BASIS}


def test_area_basis_unknown_is_recorded_on_every_ingested_row(tmp_path):
    """The uncertainty has to travel with the data. If the rows carried no
    basis (or, worse, a plausible-looking default) anything later reading the
    processed CSV would compare this city's price_per_sqft against a
    known-basis city and be wrong by the 25-35% loading factor."""
    raw = _write_csv(tmp_path / "raw.csv", _good_raw_df())
    cfg = _write_config(tmp_path / "config.toml", UNKNOWN_CONFIG)
    df, _ = ingest_city(raw, cfg)
    assert len(df) > 0
    assert (df["area_basis"] == "unknown").all()


def test_area_basis_unknown_emits_a_caveat_in_the_ingestion_log(tmp_path):
    """"unknown" passing silently is the whole failure mode: the reader of
    the log has to learn the basis is unstated at the moment the data enters
    the project, not months later from a footnote nobody reads."""
    raw = _write_csv(tmp_path / "raw.csv", _good_raw_df())
    cfg = _write_config(tmp_path / "config.toml", UNKNOWN_CONFIG)
    _, log = ingest_city(raw, cfg)
    caveats = [line for line in log if "CAVEAT" in line]
    assert caveats, f"no CAVEAT line in the ingestion log: {log}"
    assert any("unknown" in line for line in caveats)


def test_a_known_area_basis_emits_no_caveat(tmp_path):
    """A caveat on every ingestion would train people to ignore it. It must
    appear only when the basis genuinely is not known."""
    raw = _write_csv(tmp_path / "raw.csv", _good_raw_df())
    cfg = _write_config(tmp_path / "config.toml", GOOD_CONFIG)  # superbuiltup
    _, log = ingest_city(raw, cfg)
    assert not any("CAVEAT" in line for line in log)


def test_reingest_promoting_unknown_to_a_known_basis_is_rejected(tmp_path):
    """A city ingested as "unknown" must not be quietly re-declared "carpet"
    later. Nothing about the data changed -- only someone's guess did -- and
    the processed file would then claim a precision it does not have."""
    raw = _write_csv(tmp_path / "raw.csv", _good_raw_df())
    unknown_cfg = _write_config(tmp_path / "config.toml", UNKNOWN_CONFIG)
    processed = tmp_path / "processed.csv"
    df, _ = ingest_city(raw, unknown_cfg)
    df.to_csv(processed, index=False)

    promoted = _write_config(tmp_path / "config2.toml", GOOD_CONFIG)  # superbuiltup
    with pytest.raises(ValueError, match="area_basis"):
        ingest_city(raw, promoted, existing_processed_path=processed)


def test_reingest_demoting_a_known_basis_to_unknown_is_rejected(tmp_path):
    """The guard has to run in both directions: overwriting known-basis rows
    with unknown-basis ones mixes two incompatible quantities under one label
    just as thoroughly as the other way round."""
    raw = _write_csv(tmp_path / "raw.csv", _good_raw_df())
    known_cfg = _write_config(tmp_path / "config.toml", GOOD_CONFIG)
    processed = tmp_path / "processed.csv"
    df, _ = ingest_city(raw, known_cfg)
    df.to_csv(processed, index=False)

    demoted = _write_config(tmp_path / "config2.toml", UNKNOWN_CONFIG)
    with pytest.raises(ValueError, match="area_basis"):
        ingest_city(raw, demoted, existing_processed_path=processed)


# --- named parsers: values -------------------------------------------------

def test_numeric_or_range_reads_a_plain_number_and_a_range_midpoint():
    """Listing sites publish "1133 - 1384" when a project sells several
    layouts under one size label. Dropping those rows would throw away whole
    projects; taking either end would bias every derived price_per_sqft in
    one direction, so the midpoint is the honest single number."""
    out = PARSERS["numeric_or_range"](pd.Series(["1200", 950.0, "1133 - 1384"]))
    assert out.tolist() == [1200.0, 950.0, 1258.5]


def test_numeric_or_range_refuses_a_value_carrying_its_own_unit():
    """A row that states its own unit is a row whose units the config's single
    `area_unit` declaration cannot vouch for. Stripping the suffix and keeping
    the number is how a 4,125-perch plot becomes a "4,125 sq.ft. flat" and
    then drags the percentile trim off every genuine listing."""
    out = PARSERS["numeric_or_range"](
        pd.Series(["34.46Sq. Meter", "4125Perch", "1200"]))
    assert out.isna().tolist() == [True, True, False]
    assert out.iloc[2] == 1200.0


def test_leading_number_reads_indian_size_labels():
    """Bedroom count lives in free text ("3 BHK", "4 Bedroom"). "1 RK" is a
    one-room home and must read as 1, not be discarded; genuine junk must
    become NaN rather than a fabricated count the model would then fit on."""
    out = PARSERS["leading_number"](
        pd.Series(["3 BHK", "4 Bedroom", "1 RK", "studio", None]))
    assert out.iloc[:3].tolist() == [3.0, 4.0, 1.0]
    assert out.isna().tolist() == [False, False, False, True, True]


def test_balcony_label_normalises_onto_the_models_own_codes():
    """build_features maps balcony through BALCONY_CODES, which tops out at
    "3+". A raw 5 that does not normalise becomes an unmapped NaN downstream,
    and a blank that turns into "0" invents a fact about the listing -- both
    show up as quiet feature corruption rather than an error."""
    out = PARSERS["balcony_label"](pd.Series([2.0, 5, None, "3+"]))
    assert out.iloc[0] == "2"
    assert out.iloc[1] == "3+"          # capped, not left as "5"
    assert pd.isna(out.iloc[2])         # blank stays blank, never a zero
    assert out.iloc[3] == "3+"          # already a valid label, passes through


# --- named parsers: config validation and wiring ---------------------------

def test_unknown_parser_name_in_the_parse_table_is_rejected(tmp_path):
    """A typo'd parser name that was ignored would leave the column read as
    raw text, and the failure would only surface much later as an
    unexplained astype error or a column of NaN."""
    bad = GOOD_CONFIG + '\n[parse]\narea = "midpoint_of_range"\n'
    cfg = _write_config(tmp_path / "config.toml", bad)
    with pytest.raises(ValueError, match="midpoint_of_range"):
        load_config(cfg)


def test_parse_naming_a_field_columns_does_not_map_is_rejected(tmp_path):
    """A parser declared for a field that was never mapped does nothing at
    all. Accepting it lets a config look like it handles a column it has in
    fact forgotten to declare."""
    bad = GOOD_CONFIG + '\n[parse]\nbalcony = "balcony_label"\n'
    cfg = _write_config(tmp_path / "config.toml", bad)
    with pytest.raises(ValueError, match="balcony"):
        load_config(cfg)


def test_declared_parsers_are_applied_during_ingestion(tmp_path):
    """The parse table has to run BEFORE unit conversion -- "1133 - 1384" has
    no .astype(float) -- and its output is what every downstream number is
    computed from, so a table that silently did not run would leave the whole
    ingestion falling over on free text."""
    n = 5
    raw_df = _flat_raw_df(n, area_sqft=["1133 - 1384"] * n, bhk=["3 BHK"] * n)
    raw = _write_csv(tmp_path / "raw.csv", raw_df)
    parse_config = (GOOD_CONFIG
                    + '\n[parse]\narea = "numeric_or_range"\n'
                      'bedrooms = "leading_number"\n')
    cfg = _write_config(tmp_path / "config.toml", parse_config)
    df, _ = ingest_city(raw, cfg)
    assert len(df) == n
    assert df["area"].tolist() == pytest.approx([1258.5] * n)
    assert (df["bedrooms"] == 3).all()


def test_ingestion_log_reports_how_many_values_a_parser_could_not_read(tmp_path):
    """Row counts alone hide a broken parser: a column that failed on a third
    of its rows still leaves a plausible-looking dataset behind. The count of
    unreadable values is the only signal that the config, not the market,
    is why a segment vanished."""
    areas = ["1500", "1600", "1700", "34.46Sq. Meter", "1800", "4125Perch"]
    raw_df = _flat_raw_df(len(areas), area_sqft=areas)
    raw = _write_csv(tmp_path / "raw.csv", raw_df)
    parse_config = GOOD_CONFIG + '\n[parse]\narea = "numeric_or_range"\n'
    cfg = _write_config(tmp_path / "config.toml", parse_config)
    _, log = ingest_city(raw, cfg)
    loss_lines = [line for line in log if "unreadable" in line]
    assert loss_lines, f"no parse-loss line in the ingestion log: {log}"
    assert any("2 value(s) unreadable" in line for line in loss_lines)
    assert any("numeric_or_range" in line for line in loss_lines)


# --- per-row area basis ----------------------------------------------------

PER_ROW_BASIS_LABELS = '''
[area_basis_labels]
"Super built-up  Area" = "superbuiltup"
"Carpet  Area" = "carpet"
"Built Up  Area" = "builtup"
'''

PER_ROW_BASIS_CONFIG = (_config_with_top_level('area_basis_column = "area_type"')
                        + PER_ROW_BASIS_LABELS)


def test_rows_stating_a_different_basis_are_dropped_and_counted(tmp_path):
    """price_per_sqft is the model's central location signal. Keeping carpet
    and super-built-up rows side by side teaches it that a locality got 30%
    cheaper when all that changed is what the area column was measuring --
    and the log has to say how many real listings that cost, or the exclusion
    is indistinguishable from the market simply being small."""
    types = (["Super built-up  Area"] * 3 + ["Carpet  Area"] * 2
             + ["Built Up  Area"])
    raw_df = _flat_raw_df(len(types), area_type=types)
    raw = _write_csv(tmp_path / "raw.csv", raw_df)
    cfg = _write_config(tmp_path / "config.toml", PER_ROW_BASIS_CONFIG)
    df, log = ingest_city(raw, cfg)
    assert len(df) == 3
    assert (df["area_basis"] == "superbuiltup").all()
    filter_lines = [line for line in log if "filter to declared area_basis" in line]
    assert filter_lines, f"no basis-filter line in the ingestion log: {log}"
    assert any("3 row(s) removed" in line and "3 remain" in line
               for line in filter_lines)
    assert any("builtup: 1" in line and "carpet: 2" in line for line in log)


def test_an_area_basis_label_absent_from_the_mapping_is_rejected(tmp_path):
    """An unmapped label would be dropped by the equality filter without
    anyone having decided it should be -- a whole basis (say plot area) could
    disappear from a city silently. Every distinct label has to be a
    deliberate mapping."""
    types = ["Super built-up  Area"] * 4 + ["Plot  Area"] * 2
    raw_df = _flat_raw_df(len(types), area_type=types)
    raw = _write_csv(tmp_path / "raw.csv", raw_df)
    cfg = _write_config(tmp_path / "config.toml", PER_ROW_BASIS_CONFIG)
    with pytest.raises(ValueError, match="Plot  Area"):
        ingest_city(raw, cfg)


def test_area_basis_column_with_no_labels_is_rejected(tmp_path):
    """Declaring the column but not the mapping would make every raw label
    unrecognised, so the filter would keep zero rows -- an empty processed
    file rather than an error naming the missing declaration."""
    cfg = _write_config(tmp_path / "config.toml",
                        _config_with_top_level('area_basis_column = "area_type"'))
    with pytest.raises(ValueError, match="area_basis_labels"):
        load_config(cfg)


def test_labels_that_never_produce_the_declared_basis_are_rejected(tmp_path):
    """A config declaring superbuiltup while its labels only ever produce
    carpet/builtup keeps nothing. Caught at config load, this names the
    contradiction; uncaught, it looks like a city whose feed happens to be
    empty this month."""
    labels_without_declared = '''
[area_basis_labels]
"Carpet  Area" = "carpet"
"Built Up  Area" = "builtup"
'''
    bad = (_config_with_top_level('area_basis_column = "area_type"')
           + labels_without_declared)
    cfg = _write_config(tmp_path / "config.toml", bad)
    with pytest.raises(ValueError, match="never produces it"):
        load_config(cfg)


# --- amenity_count ---------------------------------------------------------

AMENITY_CONFIG = _config_with_top_level(
    'amenity_columns = ["lift", "parking", "gym"]\n'
    'amenity_unknown_value = -1')


def _amenity_raw_df(flag_rows):
    return _flat_raw_df(len(flag_rows),
                        lift=[r[0] for r in flag_rows],
                        parking=[r[1] for r in flag_rows],
                        gym=[r[2] for r in flag_rows])


def test_amenity_flags_are_collapsed_into_a_count(tmp_path):
    raw_df = _amenity_raw_df([(1, 1, 1), (1, 1, 0), (1, 0, 0), (0, 0, 0)])
    raw = _write_csv(tmp_path / "raw.csv", raw_df)
    cfg = _write_config(tmp_path / "config.toml", AMENITY_CONFIG)
    df, _ = ingest_city(raw, cfg)
    assert df["amenity_count"].tolist() == [3.0, 2.0, 1.0, 0.0]


def test_a_row_with_every_amenity_unknown_is_blank_not_zero(tmp_path):
    """THE trap this feature exists for. These feeds mark "we never captured
    amenities for this listing" with a sentinel in EVERY flag column, not
    with a blank. A plain sum turns that into a confident zero, and the model
    then learns that badly-recorded listings are cheap -- an amenity feature
    that looks predictive while measuring nothing but data-collection quality.

    If someone "simplifies" _derive_amenity_count back to (flags == 1).sum(),
    row 1 below comes out as 0.0 and this test fails on both assertions."""
    raw_df = _amenity_raw_df([(1, 1, 0), (-1, -1, -1), (0, 0, 0)])
    raw = _write_csv(tmp_path / "raw.csv", raw_df)
    cfg = _write_config(tmp_path / "config.toml", AMENITY_CONFIG)
    df, _ = ingest_city(raw, cfg)
    assert len(df) == 3
    all_unknown = df["amenity_count"].iloc[1]
    assert pd.isna(all_unknown), (
        f"a row whose every amenity flag is the unknown sentinel must be "
        f"blank, got {all_unknown!r}")
    assert all_unknown != 0, "unknown amenities must not become a confident zero"
    # ...and a genuine zero is still a zero, so the two are distinguishable.
    assert df["amenity_count"].iloc[2] == 0.0
    assert df["amenity_count"].iloc[0] == 2.0


def test_a_row_mixing_the_sentinel_with_real_flags_counts_only_the_real_ones(tmp_path):
    """Partially-recorded rows are the common case, and voiding them entirely
    would throw away the amenities that WERE recorded. Only the sentinel
    cells are masked; the row still counts its genuine 1s."""
    raw_df = _amenity_raw_df([(1, -1, 1), (1, -1, 0), (-1, -1, 1)])
    raw = _write_csv(tmp_path / "raw.csv", raw_df)
    cfg = _write_config(tmp_path / "config.toml", AMENITY_CONFIG)
    df, _ = ingest_city(raw, cfg)
    assert df["amenity_count"].tolist() == [2.0, 1.0, 1.0]
    assert not df["amenity_count"].isna().any()


def test_amenity_log_reports_how_many_rows_recorded_nothing(tmp_path):
    """A city where most rows have no amenities recorded has an
    amenity_count worth distrusting. That is only visible if ingestion says
    how many rows it left blank."""
    raw_df = _amenity_raw_df([(1, 1, 0), (-1, -1, -1), (-1, -1, -1), (0, 1, 0)])
    raw = _write_csv(tmp_path / "raw.csv", raw_df)
    cfg = _write_config(tmp_path / "config.toml", AMENITY_CONFIG)
    _, log = ingest_city(raw, cfg)
    amenity_lines = [line for line in log if "amenity_count" in line]
    assert amenity_lines, f"no amenity line in the ingestion log: {log}"
    assert any("2 with no amenities recorded" in line for line in amenity_lines)
    assert any("NOT counted as zero" in line for line in amenity_lines)


def test_a_declared_amenity_column_missing_from_the_csv_is_rejected(tmp_path):
    """Silently skipping a missing flag column would make amenity_count mean
    something different from what the config says it means, and every row's
    count would be quietly one lower."""
    raw_df = _amenity_raw_df([(1, 1, 0), (0, 1, 1), (1, 0, 1)])
    raw_df = raw_df.drop(columns=["gym"])
    raw = _write_csv(tmp_path / "raw.csv", raw_df)
    cfg = _write_config(tmp_path / "config.toml", AMENITY_CONFIG)
    with pytest.raises(ValueError, match="gym"):
        ingest_city(raw, cfg)


# --- required_columns ------------------------------------------------------

def test_price_and_area_are_required_even_when_not_declared(tmp_path):
    """A row with no area has no price_per_sqft, so it contributes nothing but
    NaN to the model's central signal. This must hold for a config that says
    nothing about required_columns at all -- opting in should not be what
    makes the dataset usable."""
    n = 6
    raw_df = _flat_raw_df(n)
    raw_df.loc[2, "area_sqft"] = None
    raw = _write_csv(tmp_path / "raw.csv", raw_df)
    cfg = _write_config(tmp_path / "config.toml", GOOD_CONFIG)
    assert load_config(cfg).required_columns == ()
    df, log = ingest_city(raw, cfg)
    assert len(df) == n - 1
    assert not df["area"].isna().any()
    required_lines = [line for line in log if "required value" in line]
    assert required_lines, f"no required-value line in the ingestion log: {log}"
    assert any("price, area" in line and "area: 1" in line
               for line in required_lines)


def test_a_city_declared_required_column_drops_rows_blank_in_it(tmp_path):
    """Some feeds leave bedroom count blank. A model fitted with bedrooms as a
    feature cannot use those rows, and dropping them at ingestion (with the
    per-column blank counts said out loud) is what keeps "we have 6,000
    listings" from meaning something different to every consumer."""
    n = 6
    raw_df = _flat_raw_df(n, bhk=[3, 3, None, 3, None, 3])
    raw = _write_csv(tmp_path / "raw.csv", raw_df)

    lenient = _write_config(tmp_path / "lenient.toml", GOOD_CONFIG)
    kept, _ = ingest_city(raw, lenient)
    assert len(kept) == n            # blanks survive when nobody asked for them

    strict = _write_config(tmp_path / "strict.toml",
                           _config_with_top_level('required_columns = ["bedrooms"]'))
    df, log = ingest_city(raw, strict)
    assert len(df) == n - 2
    assert not df["bedrooms"].isna().any()
    required_lines = [line for line in log if "required value" in line]
    assert any("bedrooms" in line and "bedrooms: 2" in line
               for line in required_lines)
    assert any("2 rows removed" in line for line in required_lines)
