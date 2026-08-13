"""Tests for the private multi-city tier.

The single most important property under test: private data must be
structurally incapable of reaching the public repo. The guard tests below
do not just assert a line exists in .gitignore -- they create a real file
under the repo's own private/ directory and prove `git` itself refuses to
treat it as addable/trackable. If someone weakens the ignore rule (or
force-adds a private path), these tests fail loudly.
"""
from __future__ import annotations

import subprocess

import pytest

from homecast import cities as cities_module
from homecast import private as private_module
from homecast.cities import City, PROJECT_ROOT, get_city
from homecast.private import load_private_cities


def _git(*args):
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT,
                          capture_output=True, text=True)


# --- Guard: private/ must be structurally incapable of reaching git -------

def test_private_dir_pattern_is_gitignored():
    result = _git("check-ignore", "-q", "private/")
    assert result.returncode == 0, (
        "private/ is not covered by .gitignore -- private data could be "
        "committed by accident")


def test_a_real_file_under_private_is_ignored_not_addable():
    """Adversarial: actually create a file under the repo's real private/
    directory (not some unrelated tmp_path) and prove git treats it as
    ignored, not as something `git add -A`/`git status` would ever surface."""
    probe_dir = PROJECT_ROOT / "private" / "_guard_probe_city" / "raw"
    probe_dir.mkdir(parents=True, exist_ok=True)
    probe_file = probe_dir / "listings.csv"
    probe_file.write_text("sector,price\nsector 1,1.0\n")
    try:
        ignored = _git("check-ignore", "-q", str(probe_file))
        assert ignored.returncode == 0, (
            f"{probe_file} would NOT be ignored by git -- a genuine private "
            f"data file could be staged")

        # The check that would actually catch a careless `git add -A`: list
        # untracked files git considers addable (i.e. NOT already excluded).
        addable = _git("ls-files", "--others", "--exclude-standard", "private")
        assert addable.stdout.strip() == "", (
            f"file(s) under private/ show up as addable/untracked-and-not-"
            f"ignored: {addable.stdout!r}")
    finally:
        probe_file.unlink()
        probe_dir.rmdir()
        (PROJECT_ROOT / "private" / "_guard_probe_city").rmdir()


def test_no_private_path_is_currently_tracked():
    tracked = _git("ls-files", "private")
    assert tracked.stdout.strip() == "", (
        f"private data is tracked in git: {tracked.stdout!r}")


# --- City gains a public flag ----------------------------------------------

def test_public_city_defaults_public_true():
    assert get_city("gurgaon").public is True


def test_public_cities_excludes_non_public(monkeypatch):
    private_city = City("shadowcity", "Shadow City", PROJECT_ROOT / "x",
                        PROJECT_ROOT / "y", PROJECT_ROOT / "z", public=False)
    merged = {**cities_module.CITIES, "shadowcity": private_city}
    monkeypatch.setattr(cities_module, "CITIES", merged)
    pub = cities_module.public_cities()
    assert "gurgaon" in pub
    assert "shadowcity" not in pub


# --- private/cities.toml: present vs. absent behave identically ------------

def test_load_private_cities_empty_when_config_absent(tmp_path):
    assert load_private_cities(tmp_path / "does_not_exist" / "cities.toml") == {}


def test_load_private_cities_empty_when_default_path_absent(monkeypatch, tmp_path):
    """Simulates a fresh public clone: no private/ directory at all."""
    monkeypatch.setattr(private_module, "PRIVATE_CONFIG",
                        tmp_path / "private" / "cities.toml")
    assert load_private_cities() == {}


def test_public_registry_unaffected_when_private_config_absent():
    """Prove the public city is exactly what it would be without the
    private-tier machinery at all -- an invariant, not a snapshot of the
    fresh-clone case. Asserting `set(CITIES) == {"gurgaon"}` would break the
    moment a real `private/cities.toml` is configured (a legitimate,
    supported state, not a bug), so instead assert the two things that must
    always hold: gurgaon is present and public, and anything else in the
    merged registry -- which can only have come from the private tier -- is
    marked non-public."""
    assert "gurgaon" in cities_module.CITIES
    c = get_city("gurgaon")
    assert c.display == "Gurgaon"
    assert c.public is True
    for key, city in cities_module.CITIES.items():
        if key != "gurgaon":
            assert city.public is False, (
                f"'{key}' is in CITIES but not marked private -- only gurgaon "
                f"may be public")


def test_load_private_cities_parses_toml(tmp_path):
    private_root = tmp_path / "private"
    private_root.mkdir()
    config = private_root / "cities.toml"
    config.write_text(
        '[amaravathi_ap]\n'
        'display = "Amaravathi, Andhra Pradesh"\n'
        'raw_name = "amaravathi_ap.csv"\n'
    )
    loaded = load_private_cities(config)
    assert set(loaded) == {"amaravathi_ap"}
    c = loaded["amaravathi_ap"]
    assert c.key == "amaravathi_ap"
    assert c.display == "Amaravathi, Andhra Pradesh"
    assert c.public is False
    assert c.raw_path == private_root / "amaravathi_ap" / "raw" / "amaravathi_ap.csv"
    assert c.processed_path == (private_root / "amaravathi_ap" / "processed"
                                / "listings_clean.csv")
    # models_dir is private/models/<key>, NOT private/<key>/models -- keeps
    # every private city's models under one root, mirroring the public
    # models/<key>/ layout one level down inside private/.
    assert c.models_dir == private_root / "models" / "amaravathi_ap"


def test_load_private_cities_missing_field_is_friendly(tmp_path):
    config = tmp_path / "cities.toml"
    config.write_text('[somecity]\ndisplay = "Some City"\n')  # no raw_name
    with pytest.raises(ValueError, match="somecity"):
        load_private_cities(config)
