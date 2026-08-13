"""Optional private city registry.

Private cities are additional markets whose raw data cannot be published in
this public repo (licensing, scope, or simply not ready). They are declared
in a gitignored ``private/cities.toml`` and live under ``private/<city>/``.

When that file -- or the whole ``private/`` directory -- is absent, as it is
on any fresh public clone, this module contributes nothing: the package
behaves identically to the public-only build. See tests/test_private.py for
the guard that keeps private/ out of git, and the test proving the absent
case is unaffected.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

from homecast.cities import City, PROJECT_ROOT

PRIVATE_CONFIG = PROJECT_ROOT / "private" / "cities.toml"

REQUIRED_FIELDS = ("display", "raw_name")


def _private_city(key: str, entry: dict, private_root: Path) -> City:
    missing = [f for f in REQUIRED_FIELDS if f not in entry]
    if missing:
        raise ValueError(
            f"private city '{key}' in cities.toml is missing required "
            f"field(s): {', '.join(missing)} (needs: {', '.join(REQUIRED_FIELDS)})")
    base = private_root / key
    return City(
        key=key,
        display=entry["display"],
        raw_path=base / "raw" / entry["raw_name"],
        processed_path=base / "processed" / "listings_clean.csv",
        # private/models/<key>, not private/<key>/models -- every private
        # city's trained artifacts live under one root, one level inside
        # private/, mirroring the public models/<key>/ layout.
        models_dir=private_root / "models" / key,
        public=False,
    )


def load_private_cities(config_path: Path | None = None) -> dict[str, City]:
    """Load the private city registry from ``private/cities.toml``.

    Returns an empty dict -- never an error -- when the config (or the
    whole ``private/`` directory) does not exist, which is the state of any
    fresh public clone.
    """
    path = config_path if config_path is not None else PRIVATE_CONFIG
    if not path.exists():
        return {}
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    private_root = path.parent
    return {key: _private_city(key, entry, private_root) for key, entry in data.items()}
