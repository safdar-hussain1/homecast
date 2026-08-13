import json

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

def test_predict_prints_estimate(city_env, capsys):
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
