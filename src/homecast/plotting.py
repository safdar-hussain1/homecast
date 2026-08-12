"""Project-wide paths and plotting configuration."""

from pathlib import Path

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

FIGURES_DIR = Path(__file__).resolve().parents[2] / "reports" / "figures"

# Categorical palette (fixed order, CVD-safe adjacent separation).
PALETTE = {
    "blue": "#2a78d6",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "green": "#008300",
    "violet": "#4a3aa7",
    "red": "#e34948",
}
SERIES = list(PALETTE.values())

# Single-hue sequential ramp (light -> dark) for magnitude encodings such as heatmaps.
BLUES = LinearSegmentedColormap.from_list(
    "seq_blues",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"


def apply_plot_style() -> None:
    """Set a recessive, publication-style matplotlib theme for the whole project."""
    mpl.rcParams.update(
        {
            "figure.figsize": (8, 4.5),
            "figure.dpi": 110,
            "axes.prop_cycle": mpl.cycler(color=SERIES),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#c9c8c2",
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT_PRIMARY,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.grid": True,
            "grid.color": "#e8e7e2",
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "font.size": 10,
            "legend.frameon": False,
        }
    )


def save_figure(fig, name: str) -> Path:
    """Save a figure to reports/figures/<name>.png and return the path."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    return path
