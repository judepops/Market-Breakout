"""Illustrative 'how the timing works' figure for the radar's live-prediction section: two ideas that look
alike at the 9-month decision point but diverge later — the detector scores the shape at month 9; the true
outcome only resolves ~2 years on.

    ~/miniconda3/envs/marketbrain/bin/python -m experiments.fig_timing
"""
from __future__ import annotations
import pathlib, warnings
warnings.filterwarnings("ignore")
FIG = pathlib.Path("artifacts/figures")


def main():
    import numpy as np, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from marketbrain.figures.style import apply_style, save, C_GNN, C_BASELINE_STRONG, C_CONTROL, C_BASE_RATE
    apply_style()

    t = np.linspace(0, 42, 400)
    # both rise similarly early; durable persists, fad collapses
    durable = 1 / (1 + np.exp(-(t - 8) / 2.2))                      # sigmoid to a high plateau
    fad = np.exp(-((t - 12) ** 2) / (2 * 5.5 ** 2)) * 1.02          # bump that decays
    # make the first ~9 months nearly identical
    durable = durable / durable.max(); fad = fad / fad.max()

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.plot(t, durable, color=C_GNN, lw=2.4, label="durable idea", zorder=3)
    ax.plot(t, fad, color=C_BASELINE_STRONG, lw=2.2, ls="--", label="fad (spikes, fades)", zorder=3)

    ax.axvline(9, color=C_BASE_RATE, lw=1.2)
    ax.text(9, 1.06, "decision point\nT₀ + 9 months", ha="center", va="bottom", fontsize=7.5, color=C_BASE_RATE)
    ax.axvspan(28, 40, color=C_CONTROL, alpha=0.45, zorder=0)
    ax.text(34, 0.14, "outcome resolves\n~2 years later", ha="center", va="center", fontsize=7.5, color="#555")

    # shade the "they look alike here" early region
    ax.axvspan(0, 9, color=C_GNN, alpha=0.05, zorder=0)
    ax.annotate("alike here", xy=(5, 0.35), fontsize=7, color=C_BASE_RATE, ha="center")

    ax.set_xlabel("months since the idea emerged (T₀)")
    ax.set_ylabel("research activity")
    ax.set_ylim(0, 1.22); ax.set_xlim(0, 42)
    ax.set_yticks([])
    ax.set_title("Called at 9 months — confirmed ~2 years later")
    ax.legend(loc="center", fontsize=7, frameon=True)
    save(fig, FIG, "deck_timing")
    print("[fig] wrote deck_timing", flush=True)


if __name__ == "__main__":
    main()
