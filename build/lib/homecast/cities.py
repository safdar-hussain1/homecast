"""City registry: every dataset HomeCast knows about, and where it lives."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class City:
    key: str
    display: str
    raw_path: Path
    processed_path: Path
    models_dir: Path
    # True for every city shippable in the public repo/dashboard. Private
    # cities (see homecast.private) are registered with public=False so
    # anything that must only ever see the public surface can filter on it.
    public: bool = True


def _city(key: str, display: str, raw_name: str) -> City:
    base = PROJECT_ROOT / "data" / key
    return City(
        key=key,
        display=display,
        raw_path=base / "raw" / raw_name,
        processed_path=base / "processed" / "listings_clean.csv",
        models_dir=PROJECT_ROOT / "models" / key,
    )


def _load_all_cities() -> dict[str, City]:
    # Deferred import: homecast.private imports City/PROJECT_ROOT from this
    # module, so importing it at module scope here would be circular.
    from homecast.private import load_private_cities
    cities: dict[str, City] = {
        "gurgaon": _city("gurgaon", "Gurgaon", "gurgaon_properties.csv"),
    }
    cities.update(load_private_cities())
    return cities


CITIES: dict[str, City] = _load_all_cities()


def public_cities() -> dict[str, City]:
    """Cities safe for public commands / the public dashboard build to see."""
    return {k: c for k, c in CITIES.items() if c.public}


def get_city(key: str) -> City:
    try:
        return CITIES[key]
    except KeyError:
        valid = ", ".join(sorted(CITIES))
        raise ValueError(f"Unknown city '{key}'. Valid cities: {valid}") from None
