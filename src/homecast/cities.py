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


def _city(key: str, display: str, raw_name: str) -> City:
    base = PROJECT_ROOT / "data" / key
    return City(
        key=key,
        display=display,
        raw_path=base / "raw" / raw_name,
        processed_path=base / "processed" / "listings_clean.csv",
        models_dir=PROJECT_ROOT / "models" / key,
    )


CITIES: dict[str, City] = {
    "gurgaon": _city("gurgaon", "Gurgaon", "gurgaon_properties.csv"),
}


def get_city(key: str) -> City:
    try:
        return CITIES[key]
    except KeyError:
        valid = ", ".join(sorted(CITIES))
        raise ValueError(f"Unknown city '{key}'. Valid cities: {valid}") from None
