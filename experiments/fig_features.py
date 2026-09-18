"""Companion to fig_timing: two panels (durable | fad) showing how the KEY STRUCTURAL FEATURES rise or
fail to rise relative to raw attention. The point the detector rests on: a fad's *attention* can spike
just as high as a durable idea's, but its *structure* (how many distinct authors adopt it, whether it
reaches strong venues, whether it persists) never follows — and that gap is legible at month 9, when the
raw attention curves still look alike. Illustrative/schematic, not fitted to data.

    ~/miniconda3/envs/marketbrain/bin/python -m experiments.fig_features
"""
from __future__ import annotations
import pathlib, warnings
warnings.filterwarnings("ignore")
FIG = pathlib.Path("artifacts/figures")


def main():
    import numpy as np, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from marketbrain.figures.style import apply_style, save, C_GNN, C_BASELINE_STRONG, C_BASE_RATE
    apply_style()

    t = np.linspace(0, 42, 400)
    sig = lambda c, s: 1.0 / (1.0 + np.exp(-(t - c) / s))
    bump = lambda c, s: np.exp(-((t - c) ** 2) / (2 * s ** 2))

    # DURABLE: attention rises and holds; breadth and venue rise and hold with it
    att_d = sig(8, 2.2); att_d /= att_d.max()
    breadth_d = sig(10, 2.6) * 0.96
    venue_d = sig(11.5, 3.0) * 0.90

    # FAD: attention spikes just as high, then decays; structure never follows (stays low)
    att_f = bump(12, 5.5); att_f /= att_f.max()
    breadth_f = bump(12, 4.2) * 0.34          # a few authors, never broad
    venue_f = bump(13, 4.0) * 0.22            # weak venue penetration

    GREY = "#9a9aa0"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.8), sharey=True)

    for ax, (att, breadth, venue, title) in [
        (ax1, (att_d, breadth_d, venue_d, "Durable idea")),
        (ax2, (att_f, breadth_f, venue_f, "Fad idea")),
    ]:
        ax.fill_between(t, 0, att, color=GREY, alpha=0.16, zorder=0)
        ax.plot(t, att, color=GREY, lw=2.6, label="attention (raw activity)", zorder=2)
        ax.plot(t, breadth, color=C_GNN, lw=2.2, label="author breadth", zorder=3)
        ax.plot(t, venue, color=C_BASELINE_STRONG, lw=2.0, ls=(0, (4, 2)), label="venue penetration", zorder=3)
        ax.axvline(9, color=C_BASE_RATE, lw=1.1)
        ax.text(8.4, 0.52, "decision · month 9", rotation=90, va="center", ha="right",
                fontsize=6.6, color=C_BASE_RATE)
        ax.set_title(title, fontsize=9.5, fontweight="bold", pad=8)
        ax.set_xlim(0, 42); ax.set_ylim(0, 1.12)
        ax.set_yticks([]); ax.set_xlabel("months since idea emerged")
    ax1.set_ylabel("relative level")

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Same attention spike — different structure", fontsize=10.5, fontweight="bold", y=1.02)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, FIG, "deck_features")
    print("[fig] wrote deck_features", flush=True)


if __name__ == "__main__":
    main()
