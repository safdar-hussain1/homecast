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

def test_unknown_city_is_friendly(capsys):
    assert cli.main(["clean", "--city", "atlantis"]) == 2
    assert "gurgaon" in capsys.readouterr().err
