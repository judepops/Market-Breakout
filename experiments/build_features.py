"""Build the combined feature view from the corpus (normalize + correlation).

    python -m experiments.build_features --root data/raw --out artifacts/figures

Produces the standardized combined view (4 dimensions; standardized levels + YoY growth) and a
correlation matrix across the 6 core signals (printed + heatmap). This is NOT prediction — it
normalizes the hand-built signals into one comparable view (npm + PyPI combined into adoption).
"""

from __future__ import annotations

import argparse

import pandas as pd

from marketbrain.corpus import CorpusStore
from marketbrain.features import DIMENSIONS, build_feature_panel, correlation_matrix
from marketbrain.figures.style import (
    C_BASE_RATE,
    DIM_STYLE,
    FULL_WIDTH,
    apply_style,
    heatmap_cmap,
    save,
)


def _combined_figure(panel: pd.DataFrame, out: str):
    apply_style()
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    months = pd.to_datetime(pd.Index(panel.index) + "-01")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FULL_WIDTH, 5.2), sharex=True)
    for dim in DIMENSIONS:
        c, ls, mk = DIM_STYLE[dim]
        if f"dim_{dim}__level_z" in panel:
            ax1.plot(months, panel[f"dim_{dim}__level_z"], color=c, ls=ls, marker=mk, ms=2.5,
                     lw=1.5, label=dim, zorder=5 if dim == "research" else 4)
        if f"dim_{dim}__yoy_growth" in panel:
            ax2.plot(months, panel[f"dim_{dim}__yoy_growth"], color=c, ls=ls, marker=mk, ms=2.5,
                     lw=1.5, label=dim)
    for ax in (ax1, ax2):
        ax.axhline(0, color=C_BASE_RATE, lw=0.6, ls=":")
    ax1.set_title("(a) LLM dev-tools — standardized signal levels (npm+PyPI = one adoption dim)")
    ax1.set_ylabel("level (z-score)")
    ax1.legend(loc="upper left", ncol=4, fontsize=6)
    ax2.set_title("(b) year-over-year growth — research/adoption compound, attention/funding spike")
    ax2.set_ylabel("YoY log growth")
    ax2.set_xlabel("month (knowledge-dated, leakage-safe)")
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    return save(fig, out, "combined_signals")


def _corr_figure(corr: pd.DataFrame, out: str):
    apply_style()
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    labels = [c.replace("_submissions", "").replace("_value_usd", "").replace("_", " ")[:13]
              for c in corr.columns]
    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.8, FULL_WIDTH * 0.66))
    im = ax.imshow(corr.values, cmap=heatmap_cmap(), norm=Normalize(-1, 1))
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(v) > 0.6 else "black")
    ax.set_title("(c) signal correlation (levels)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return save(fig, out, "signal_correlation")


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description="Build the combined feature view + correlation.")
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--market", default="llm-dev-tools")
    ap.add_argument("--out", default="artifacts/figures")
    args = ap.parse_args(argv)

    obs = CorpusStore(args.root).load_observations()
    panel = build_feature_panel(obs, args.market)
    corr = correlation_matrix(obs, args.market)
    print("=== signal correlation (levels) ===")
    print(corr.round(2).to_string())
    print("saved", _combined_figure(panel, args.out)[0])
    print("saved", _corr_figure(corr, args.out)[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
