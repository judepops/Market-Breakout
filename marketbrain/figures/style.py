"""Vela house-style matplotlib contract for market-brain figures.

Mirrors the ``/figure-vela`` styling contract (GraphFounder ``figures/_render.py``):
IEEE-compact sans-serif, the dark-red / slate / grey palette, no grid, top & right
spines off, inward ticks, framed legends, 600-DPI PNG + a vector PDF on save. The
constants are *vendored* (not imported from GraphFounder) so market-brain has no
dependency on that repo. Keep in sync with the house figure style.
"""

from __future__ import annotations

from pathlib import Path

# --- canonical palette (use the names, never inline raw hexes) ---
C_GNN = "#8b0000"              # hero / accent (dark red)
C_BASELINE_STRONG = "#2c3e50"  # dark slate
C_BASELINE = "#6b7a85"
C_CONTROL = "#cfcfcf"          # light grey controls
C_BASE_RATE = "#555555"        # reference / annotation dotted lines
C_FAIL = "#bdbdbd"

# --- IEEE-compact rcParams (mirrors _RC) ---
_RC = {
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.titleweight": "bold",
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.9,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.frameon": True,
    "legend.edgecolor": "black",
    "legend.framealpha": 1.0,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
}

FULL_WIDTH = 7.1  # inches, two-column

# Per-dimension styling for the combined multi-signal view (colour + linestyle + marker so the
# four lines stay distinct within the red/slate/grey house palette).
DIM_STYLE = {
    "research": (C_GNN, "-", "o"),
    "adoption": (C_BASELINE_STRONG, "-", "s"),
    "attention": ("#6b7a85", "--", "^"),
    "funding": ("#c65353", "--", "D"),
}


def heatmap_cmap():
    """Diverging grey→white→red colormap (house style); use with a Normalize symmetric about 0."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "mb_diverging", [(0.0, "#4a4a4a"), (0.5, "#f5f2f2"), (1.0, "#8b0000")]
    )


def apply_style() -> None:
    """Apply the house rcParams. Call before plotting."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(_RC)


def save(fig, out_dir: "str | Path", stem: str) -> tuple[Path, Path]:
    """Save ``fig`` as a 600-DPI PNG and a vector PDF, then close it."""
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    png, pdf = out / f"{stem}.png", out / f"{stem}.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf
