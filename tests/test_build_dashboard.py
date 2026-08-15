"""Tests for scripts/build_dashboard.py's public/private routing.

`build_dashboard.py` used to hardcode `docs/index.html` and `models/<city>/`
-- a private city only failed safe today by accident (its model happens to
live under `private/models/`, so the hardcoded public path never resolved).
These tests pin the actual design: the city registry (`public`/`models_dir`)
is authoritative, a private city's default build lands under
`private/dashboards/<key>.html`, and writing a private city anywhere under
`docs/` is refused outright, not merely unlikely.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from homecast.cities import City
from homecast.export import export_city
from homecast.model import train_final

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "build_dashboard", ROOT / "scripts" / "build_dashboard.py")
build_dashboard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_dashboard)


@pytest.fixture()
def synthetic_private_city(tmp_path, monkeypatch, clean_fixture):
    """A temporary, synthetic private city with a real (small) model.json
    payload, registered into the live city registry for the duration of the
    test only (monkeypatch undoes it automatically) -- entirely under
    tmp_path except for the actual dashboard file the test writes, which is
    the thing under test and is cleaned up explicitly."""
    import homecast.cities as cities_module

    key = "shadowcity_guard_probe"
    models_dir = tmp_path / "private_models" / key
    models_dir.mkdir(parents=True)
    city = City(key=key, display="Shadow City",
               raw_path=tmp_path / "raw.csv",
               processed_path=tmp_path / "processed.csv",
               models_dir=models_dir, public=False)

    fitted = train_final(clean_fixture, model="default")
    payload = export_city(fitted, clean_fixture, city)
    (models_dir / "model.json").write_text(json.dumps(payload))

    merged = {**cities_module.CITIES, key: city}
    monkeypatch.setattr(cities_module, "CITIES", merged)
    return city


def test_public_city_default_output_is_docs_index_html():
    assert build_dashboard._default_output("gurgaon") == build_dashboard.PUBLIC_OUTPUT
    assert build_dashboard.PUBLIC_OUTPUT == ROOT / "docs" / "index.html"


def test_building_gurgaon_is_unchanged_by_the_refactor():
    """The public build is unaffected: still resolves to, and writes,
    docs/index.html -- rebuilding it here is exactly what running this
    script normally does, so it's not a test-only side effect."""
    out = build_dashboard.build("gurgaon")
    assert out == build_dashboard.PUBLIC_OUTPUT
    assert out.exists()
    html = out.read_text(encoding="utf-8")
    assert build_dashboard.PLACEHOLDER not in html
    assert '"city":"Gurgaon"' in html


def test_private_city_default_output_is_under_private_dashboards(synthetic_private_city):
    key = synthetic_private_city.key
    expected = build_dashboard.PRIVATE_DASHBOARDS_DIR / f"{key}.html"
    assert build_dashboard._default_output(key) == expected


def test_private_city_build_lands_in_private_dashboards_not_docs(synthetic_private_city):
    key = synthetic_private_city.key
    out_path = build_dashboard.PRIVATE_DASHBOARDS_DIR / f"{key}.html"
    try:
        out = build_dashboard.build(key)
        assert out == out_path
        assert out.exists()
        # Never under docs/, by construction of the path itself.
        with pytest.raises(ValueError):
            out.resolve().relative_to((ROOT / "docs").resolve())
    finally:
        if out_path.exists():
            out_path.unlink()
        if build_dashboard.PRIVATE_DASHBOARDS_DIR.exists() and not any(
                build_dashboard.PRIVATE_DASHBOARDS_DIR.iterdir()):
            build_dashboard.PRIVATE_DASHBOARDS_DIR.rmdir()


def test_private_city_refuses_to_write_into_docs_even_if_explicitly_asked(synthetic_private_city, tmp_path):
    key = synthetic_private_city.key
    forced_path = ROOT / "docs" / f"{key}-sneaked-in.html"
    with pytest.raises(ValueError, match="refusing to build private city"):
        build_dashboard.build(key, output=forced_path)
    assert not forced_path.exists()


def test_private_city_never_writes_into_docs_dir_end_to_end(synthetic_private_city):
    """No matter how it's invoked, a private city's build must never produce
    a file under docs/."""
    before = set(p.resolve() for p in (ROOT / "docs").rglob("*"))
    key = synthetic_private_city.key
    out_path = build_dashboard.PRIVATE_DASHBOARDS_DIR / f"{key}.html"
    try:
        build_dashboard.build(key)
    finally:
        if out_path.exists():
            out_path.unlink()
        if build_dashboard.PRIVATE_DASHBOARDS_DIR.exists() and not any(
                build_dashboard.PRIVATE_DASHBOARDS_DIR.iterdir()):
            build_dashboard.PRIVATE_DASHBOARDS_DIR.rmdir()
    after = set(p.resolve() for p in (ROOT / "docs").rglob("*"))
    assert before == after


# ── reference-market page + local all-cities hub ───────────────────────────

def _rates_toml(p):
    p.write_text(
        'last_updated = "2026-08-13"\n\n'
        '[[rate]]\nzone = "Test Zone"\nproperty_type = "plot"\n'
        'rate_low_per_sqft = 1944\nrate_high_per_sqft = 1978\n'
        'source = "an official notification"\nsource_date = "2022-01-10"\n')
    return p


def test_reference_page_bakes_the_rate_table(tmp_path):
    import scripts.build_reference_dashboard as brd
    out = brd.build(_rates_toml(tmp_path / "rates.toml"), tmp_path / "ref.html")
    html = out.read_text()
    assert "/*__DATA__*/" not in html          # placeholder consumed
    assert "an official notification" in html  # source travelled through
    assert "2022-01-10" in html                # and its date
    assert "not a prediction" in html          # framing survives


def test_reference_page_refuses_to_write_into_docs(tmp_path):
    import scripts.build_reference_dashboard as brd
    with pytest.raises(ValueError, match="refusing to write"):
        brd.build(_rates_toml(tmp_path / "rates.toml"),
                  brd.DOCS_DIR / "index.html")


def test_reference_page_errors_helpfully_without_a_rate_table(tmp_path):
    import scripts.build_reference_dashboard as brd
    with pytest.raises(FileNotFoundError, match="example.toml"):
        brd.build(tmp_path / "absent.toml", tmp_path / "ref.html")


def test_hub_refuses_to_write_into_docs():
    import scripts.build_hub as bh
    with pytest.raises(ValueError, match="refusing to write"):
        bh.build(bh.DOCS_DIR / "hub.html")
