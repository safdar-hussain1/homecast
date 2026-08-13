import json

import numpy as np
import pandas as pd
import pytest

from homecast import cli


@pytest.fixture()
def city_env(tmp_path, monkeypatch, raw_fixture):
    """Point the gurgaon registry entry at a tmp copy so the CLI is testable."""
    import homecast.cities as cities
    raw = tmp_path / "raw" / "gurgaon_properties.csv"
    raw.parent.mkdir()
    raw_fixture.to_csv(raw, index=False)
    c = cities.City("gurgaon", "Gurgaon", raw,
                    tmp_path / "processed" / "listings_clean.csv",
                    tmp_path / "models")
    monkeypatch.setitem(cities.CITIES, "gurgaon", c)
    return c


@pytest.fixture()
def servable_raw_fixture() -> pd.DataFrame:
    """Same raw schema as conftest.py's raw_fixture, but with genuine
    per-sector price signal (not independent noise) and a larger sample --
    realistic enough that the trained model or its baseline clears the
    quality floor (QUALITY_FLOOR_MAPE_PCT). raw_fixture itself is
    deliberately small and noisy for OTHER tests that need a hopeless model
    (see test_model.py/test_valuation.py's not_served coverage); this is the
    servable counterpart, used here only to prove the CLI's happy predict
    path still prints a headline price end to end."""
    rng = np.random.default_rng(11)
    n = 300
    sectors = rng.choice(["sector 1", "sector 2", "sector 3", "sector 4"], n)
    base_rate = {"sector 1": 6000.0, "sector 2": 9000.0,
                 "sector 3": 12000.0, "sector 4": 15000.0}
    area = rng.uniform(500, 3000, n).round()
    ppsf = np.array([base_rate[s] for s in sectors]) * np.exp(rng.normal(0.0, 0.05, n))
    return pd.DataFrame({
        "property_type": rng.choice(["flat", "house"], n, p=[0.8, 0.2]),
        "society": "test society",
        "sector": sectors,
        "price": (area * ppsf / 1e7).round(4),
        "price_per_sqft": ppsf,
        "area": area,
        "areaWithType": "Carpet area: x",
        "bedRoom": rng.integers(1, 6, n),
        "bathroom": rng.integers(1, 6, n),
        "balcony": rng.choice(["0", "1", "2", "3", "3+"], n),
        "floorNum": rng.integers(0, 20, n).astype(float),
        "facing": "East",
        "agePossession": rng.choice(
            ["New Property", "Relatively New", "Moderately Old",
             "Old Property", "Under Construction", "Undefined"], n),
        "super_built_up_area": np.nan, "built_up_area": np.nan, "carpet_area": area,
        "study room": 0, "servant room": 0, "store room": 0,
        "pooja room": 0, "others": 0,
        "furnishing_type": rng.integers(0, 3, n),
        "luxury_score": rng.integers(0, 175, n),
    })


@pytest.fixture()
def servable_city_env(tmp_path, monkeypatch, servable_raw_fixture):
    import homecast.cities as cities
    raw = tmp_path / "raw" / "gurgaon_properties.csv"
    raw.parent.mkdir()
    servable_raw_fixture.to_csv(raw, index=False)
    c = cities.City("gurgaon", "Gurgaon", raw,
                    tmp_path / "processed" / "listings_clean.csv",
                    tmp_path / "models")
    monkeypatch.setitem(cities.CITIES, "gurgaon", c)
    return c


def test_clean_writes_processed(city_env, capsys):
    assert cli.main(["clean", "--city", "gurgaon"]) == 0
    assert city_env.processed_path.exists()
    assert "rows removed" in capsys.readouterr().out

def test_train_writes_artifacts(city_env):
    assert cli.main(["train"]) == 0
    for name in ("model.joblib", "model.json", "metrics.json"):
        assert (city_env.models_dir / name).exists()
    m = json.loads((city_env.models_dir / "metrics.json").read_text())
    assert "baseline_sector" in m
    assert "model_no_society" in m
    assert "serving_status" in m

def test_predict_not_served_prints_no_headline_price(city_env, capsys):
    """city_env's data (raw_fixture, see conftest.py) is deliberately small
    and noisy -- both the model and the sector baseline land above the
    quality floor, so predict must print NO headline price at all, only the
    plain-language reason. See test_predict_prints_estimate_for_a_servable_city
    below for the counterpart where a headline price IS printed."""
    cli.main(["train"])
    rc = cli.main(["predict", "--sector", "sector 1", "--type", "flat",
                   "--bhk", "3", "--bath", "2", "--area", "1500",
                   "--furnishing", "semi-furnished", "--luxury", "50"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "No price estimate" in out
    assert "not good enough to price a property responsibly" in out
    assert "Cr" not in out

def test_predict_prints_estimate_for_a_servable_city(servable_city_env, capsys):
    cli.main(["train"])
    rc = cli.main(["predict", "--sector", "sector 1", "--type", "flat",
                   "--bhk", "3", "--bath", "2", "--area", "1500",
                   "--furnishing", "semi-furnished", "--luxury", "50"])
    assert rc == 0
    assert "Cr" in capsys.readouterr().out

def test_predict_out_of_range_is_friendly(city_env, capsys):
    cli.main(["train"])
    rc = cli.main(["predict", "--sector", "sector 1", "--type", "flat",
                   "--bhk", "3", "--bath", "2", "--area", "999999",
                   "--furnishing", "semi-furnished", "--luxury", "50"])
    assert rc == 2
    assert "range" in capsys.readouterr().err.lower()

def test_predict_unknown_sector_is_friendly(city_env, capsys):
    """A case typo in the sector must exit 2 with a message, not a wrong price."""
    cli.main(["train"])
    rc = cli.main(["predict", "--sector", "Sector 1", "--type", "flat",
                   "--bhk", "3", "--bath", "2", "--area", "1500",
                   "--furnishing", "semi-furnished", "--luxury", "50"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "error:" in err and "Unknown sector 'Sector 1'" in err

def test_predict_unknown_age_is_friendly(city_env, capsys):
    cli.main(["train"])
    rc = cli.main(["predict", "--sector", "sector 1", "--type", "flat",
                   "--bhk", "3", "--bath", "2", "--area", "1500",
                   "--furnishing", "semi-furnished", "--luxury", "50",
                   "--age", "Brand New"])
    assert rc == 2
    assert "Unknown age" in capsys.readouterr().err

def test_clean_city_registered_without_pipeline_is_friendly(monkeypatch, tmp_path, capsys):
    """`--city <registered-but-unpipelined>` must not raise a raw KeyError."""
    import homecast.cities as cities
    monkeypatch.setitem(cities.CITIES, "atlantis", cities.City(
        "atlantis", "Atlantis", tmp_path / "raw.csv",
        tmp_path / "clean.csv", tmp_path / "models"))
    assert cli.main(["clean", "--city", "atlantis"]) == 2
    assert "PIPELINES" in capsys.readouterr().err

def test_unknown_city_is_friendly(capsys):
    assert cli.main(["clean", "--city", "atlantis"]) == 2
    assert "gurgaon" in capsys.readouterr().err

def test_city_flag_only_after_subcommand(city_env, capsys):
    """The top-level parser must not advertise a --city it would discard."""
    with pytest.raises(SystemExit):
        cli.main(["--city", "gurgaon", "clean"])

def test_evaluate_prints_metrics_table(city_env, capsys):
    rc = cli.main(["evaluate", "--city", "gurgaon"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "HomeCast model" in out
    assert "Sector-median baseline" in out
    assert "Global-median baseline" in out

def test_export_dashboard_rewrites_model_json(city_env, capsys):
    cli.main(["train"])
    model_json = city_env.models_dir / "model.json"
    before = model_json.read_text()
    model_json.unlink()
    rc = cli.main(["export-dashboard", "--city", "gurgaon"])
    assert rc == 0
    assert model_json.exists()
    assert model_json.read_text() == before

def test_export_dashboard_without_trained_model_is_friendly(city_env, capsys):
    rc = cli.main(["export-dashboard", "--city", "gurgaon"])
    assert rc == 2
    assert "homecast train" in capsys.readouterr().err

def test_train_accurate_model_skips_dashboard_export(city_env, capsys):
    """--model accurate must still produce a usable trained model, but never
    a browser payload (ExtraTrees has no init_/learning_rate to walk)."""
    rc = cli.main(["train", "--model", "accurate"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "note: dashboard export skipped" in out
    assert (city_env.models_dir / "model.joblib").exists()
    assert (city_env.models_dir / "metrics.json").exists()
    assert not (city_env.models_dir / "model.json").exists()

def test_predict_works_after_training_the_accurate_model(servable_city_env, capsys):
    cli.main(["train", "--model", "accurate"])
    rc = cli.main(["predict", "--sector", "sector 1", "--type", "flat",
                   "--bhk", "3", "--bath", "2", "--area", "1500",
                   "--furnishing", "semi-furnished", "--luxury", "50"])
    assert rc == 0
    assert "Cr" in capsys.readouterr().out

def test_train_default_model_still_writes_dashboard_export(city_env, capsys):
    rc = cli.main(["train", "--model", "default"])
    assert rc == 0
    assert "note: dashboard export skipped" not in capsys.readouterr().out
    assert (city_env.models_dir / "model.json").exists()

def test_evaluate_accepts_model_flag(city_env, capsys):
    rc = cli.main(["evaluate", "--city", "gurgaon", "--model", "accurate"])
    assert rc == 0
    assert "HomeCast model" in capsys.readouterr().out

def test_bad_model_name_is_friendly(city_env, capsys):
    with pytest.raises(SystemExit):
        cli.main(["train", "--model", "fastest"])

def test_predict_accepts_optional_society_and_balcony(servable_city_env, capsys):
    cli.main(["train"])
    rc = cli.main(["predict", "--sector", "sector 1", "--type", "flat",
                   "--bhk", "3", "--bath", "2", "--area", "1500",
                   "--furnishing", "semi-furnished", "--luxury", "50",
                   "--society", "test society", "--balcony", "2"])
    assert rc == 0
    assert "Cr" in capsys.readouterr().out

def test_predict_unrecognised_society_is_not_an_error(city_env, capsys):
    """Society is optional and not a typo trap, unlike sector."""
    cli.main(["train"])
    rc = cli.main(["predict", "--sector", "sector 1", "--type", "flat",
                   "--bhk", "3", "--bath", "2", "--area", "1500",
                   "--furnishing", "semi-furnished", "--luxury", "50",
                   "--society", "not a real society"])
    assert rc == 0

def test_predict_bad_balcony_is_friendly(city_env, capsys):
    cli.main(["train"])
    rc = cli.main(["predict", "--sector", "sector 1", "--type", "flat",
                   "--bhk", "3", "--bath", "2", "--area", "1500",
                   "--furnishing", "semi-furnished", "--luxury", "50",
                   "--balcony", "9"])
    assert rc == 2
    assert "Unknown balcony" in capsys.readouterr().err


@pytest.fixture()
def society_signal_city_env(tmp_path, monkeypatch, raw_fixture):
    """A city whose price depends on the SOCIETY as well as the sector, so the
    learned model genuinely beats the sector-median rule of thumb and the
    model-served branch is reached. servable_raw_fixture prices at exactly
    sector-rate x area, which makes the baseline optimal by construction and
    therefore can't exercise this path."""
    import homecast.cities as cities
    rng = np.random.default_rng(23)
    n = 600
    sectors = rng.choice(["sector 1", "sector 2"], n)
    socs = rng.choice(["alpha towers", "beta greens", "gamma residency"], n)
    sector_rate = {"sector 1": 7000.0, "sector 2": 12000.0}
    soc_mult = {"alpha towers": 0.72, "beta greens": 1.0, "gamma residency": 1.35}
    area = rng.uniform(600, 2600, n).round()
    ppsf = np.array([sector_rate[s] * soc_mult[c] for s, c in zip(sectors, socs)])
    ppsf = ppsf * np.exp(rng.normal(0.0, 0.03, n))
    price = (area * ppsf / 1e7).round(3)

    base = raw_fixture.iloc[0].to_dict()
    df = pd.DataFrame([base] * n).reset_index(drop=True)
    df["sector"], df["society"] = sectors, socs
    df["area"], df["price_per_sqft"], df["price"] = area, ppsf.round(), price
    df["bedRoom"] = rng.integers(2, 5, n)
    df["bathroom"] = rng.integers(1, 4, n)
    df["property_type"] = "flat"
    df["balcony"] = rng.choice(["0", "1", "2", "3", "3+"], n)
    df["agePossession"] = "New Property"
    df["furnishing_type"] = rng.integers(0, 3, n)
    df["luxury_score"] = rng.integers(0, 175, n)

    raw = tmp_path / "raw" / "gurgaon_properties.csv"
    raw.parent.mkdir()
    df.to_csv(raw, index=False)
    c = cities.City("gurgaon", "Gurgaon", raw,
                    tmp_path / "processed" / "listings_clean.csv",
                    tmp_path / "models")
    monkeypatch.setitem(cities.CITIES, "gurgaon", c)
    return c


def test_predict_reports_which_accuracy_regime_it_used(society_signal_city_env, capsys):
    """A mistyped society silently falls back to the sector rate; the CLI must
    say which regime it used rather than printing an equally confident number
    either way (the dashboard already does this)."""
    cli.main(["train"])
    capsys.readouterr()
    base = ["predict", "--sector", "sector 1", "--type", "flat", "--bhk", "3",
            "--bath", "2", "--area", "1500", "--furnishing", "semi-furnished",
            "--luxury", "50"]

    assert cli.main(base) == 0
    out = capsys.readouterr().out
    assert "no society given" in out and "MAPE" in out

    assert cli.main(base + ["--society", "alpha towers"]) == 0
    out = capsys.readouterr().out
    assert "matched" in out and "MAPE" in out

    assert cli.main(base + ["--society", "definitely not a real society"]) == 0
    out = capsys.readouterr().out
    assert "not in the training data" in out and "MAPE" in out
