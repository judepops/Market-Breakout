"""Regenerate the attention-vs-funding eyeball figure with honest, consistent SHARE labels.

    python -m experiments.fig_eyeball

Both axes are a market's SHARE of the total that month (as a percent), so the secular growth of the whole VC
market and the whole attention stream is removed and only the market-specific dynamics remain. The point is
that the two lines do not track.
"""
from __future__ import annotations
import pathlib
import warnings

warnings.filterwarnings("ignore")
TS = pathlib.Path("data/raw/timeseries")
FIG = pathlib.Path("artifacts/figures")
RED, SLATE, GREY = "#8b0000", "#2c3e50", "#b9bcc4"
MARKETS = ["machine-learning", "robotics", "fintech", "cyber-security"]


def main():
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

    floor = pd.Timestamp("2012-01-01")   # before this, total VC is tiny so a single market can be ~100% of it
    att = pd.read_parquet(TS / "monthly_attention.parquet")
    att["m"] = pd.to_datetime(dict(year=att["year"], month=att["month"], day=1))
    att = att[att["m"] >= floor]
    fund = pd.read_parquet(TS / "monthly_funding.parquet")
    fund["m"] = pd.to_datetime(fund["month"])
    fund = fund[fund["m"] >= floor]

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), dpi=150)
    for ax, mk in zip(axes.ravel(), MARKETS):
        a = att[att["market"] == mk].set_index("m")["share"].sort_index() * 100
        f = fund[fund["market"] == mk].set_index("m")["usd_share"].sort_index() * 100
        a6 = a.rolling(6, min_periods=2).mean()
        f6 = f.rolling(6, min_periods=2).mean()
        ax.plot(a6.index, a6.values, color=SLATE, lw=1.8, zorder=3)
        ax.set_ylabel("attention share of all HN (%)", color=SLATE, fontsize=8.5)
        ax2 = ax.twinx()
        ax2.plot(f6.index, f6.values, color=RED, lw=1.8, zorder=3)
        ax2.set_ylabel("funding share of all VC (%)", color=RED, fontsize=8.5)
        ax.set_title(mk.replace("-", " "), color=SLATE, fontweight="bold", fontsize=10, loc="left")
        for a_ in (ax, ax2):
            for s in ("top",):
                a_.spines[s].set_visible(False)
            a_.tick_params(length=0, labelsize=8)
        ax.tick_params(axis="y", colors=SLATE)
        ax2.tick_params(axis="y", colors=RED)
    fig.suptitle("A market's SHARE of total attention (slate) vs its SHARE of total VC funding (red), "
                 "6mo smoothed: do they track?",
                 color=SLATE, fontweight="bold", fontsize=11, x=0.02, ha="left")
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIG / "attention_vs_funding_eyeball.png", bbox_inches="tight", facecolor="white")
    print("wrote attention_vs_funding_eyeball.png (both axes now a clear share-of-total percent)")


if __name__ == "__main__":
    main()
