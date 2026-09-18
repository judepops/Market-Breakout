"""Paper A / meeting figures — Vela house style (marketbrain.figures.style). Built from cached result
artifacts; no recompute. Four panels:

    fig_detector      signature vs momentum vs random AUPRC (+0.23)
    fig_orthogonality signature vs +momentum (subsumed)
    fig_baserate      converter base-rate: durable vs fad (~20% vs ~16%)
    fig_ablation      the 400-combo robustness band (chosen +0.23 / floor +0.11)

    ~/miniconda3/envs/marketbrain/bin/python -m experiments.fig_paperA
"""
from __future__ import annotations
import json
import pathlib
import warnings

warnings.filterwarnings("ignore")
FIG = pathlib.Path("artifacts/figures")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from marketbrain.figures.style import (apply_style, save, C_GNN, C_BASELINE_STRONG,
                                           C_BASELINE, C_CONTROL, C_BASE_RATE)
    apply_style()

    # ---- 1. detector: signature vs momentum vs random (AI/ML domain) ----
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    labels = ["signature", "momentum", "random"]
    vals = [0.659, 0.432, 0.30]
    colors = [C_GNN, C_BASELINE_STRONG, C_CONTROL]
    bars = ax.bar(labels, vals, color=colors, width=0.62, edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.annotate("", xy=(0.5, 0.659), xytext=(0.5, 0.432),
                arrowprops=dict(arrowstyle="<->", color=C_GNN, lw=1.0))
    ax.text(0.6, (0.659 + 0.432) / 2, "+0.23", color=C_GNN, fontsize=8, va="center", fontweight="bold")
    ax.set_ylabel("AUPRC (twin-safe OOF)")
    ax.set_ylim(0, 0.75)
    ax.set_title("Durable-breakout detection (AI/ML)")
    save(fig, FIG, "paperA_detector")

    # ---- 2. orthogonality: does momentum add to the signature? ----
    o = json.load(open("artifacts/phase1_orthogonality/summary.json"))
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    keys = ["A signature (11)", "D logistic stack", "C sig+momentum GBM (12)", "B momentum alone"]
    disp = ["signature", "+ momentum\n(stack)", "+ momentum\n(feature)", "momentum\nalone"]
    vals = [o[k] for k in keys]
    colors = [C_GNN, C_BASELINE, C_BASELINE, C_BASELINE_STRONG]
    bars = ax.bar(range(len(vals)), vals, color=colors, width=0.66, edgecolor="black", linewidth=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.axhline(o["A signature (11)"], color=C_GNN, ls=":", lw=0.9)
    ax.set_xticks(range(len(disp))); ax.set_xticklabels(disp, fontsize=6.5)
    ax.set_ylabel("AUPRC")
    ax.set_ylim(0, 0.75)
    ax.set_title("Momentum is subsumed (adds nothing)")
    save(fig, FIG, "paperA_orthogonality")

    # ---- 3. converter base-rate: durable vs fad ----
    b = pd.read_parquet("artifacts/stageB/converter_baserate.parquet")
    sub = b[(b.window == "founded>=T0") & (b.tau == 0.60) & (b.x == 500)].set_index("group")
    fig, ax = plt.subplots(figsize=(3.2, 2.5))
    groups = ["positive", "hyped_died"]; disp = ["durable\nthemes", "hyped-died\nfads"]
    rate = [sub.loc[g, "rate"] * 100 for g in groups]
    lo = [(sub.loc[g, "rate"] - sub.loc[g, "ci_lo"]) * 100 for g in groups]
    hi = [(sub.loc[g, "ci_hi"] - sub.loc[g, "rate"]) * 100 for g in groups]
    bars = ax.bar(disp, rate, yerr=[lo, hi], capsize=3, color=[C_GNN, C_BASELINE], width=0.58,
                  edgecolor="black", linewidth=0.6, error_kw=dict(lw=0.9))
    for i, v in enumerate(rate):
        ax.text(i, v + max(hi) + 1.2, f"{v:.0f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax.set_ylabel("themes → ≥\\$500M company (%)")
    ax.set_ylim(0, 40)
    ax.set_title("Converter base-rate (≈ indistinguishable)")
    save(fig, FIG, "paperA_baserate")

    # ---- 4. ablation robustness band ----
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.axhspan(0.10, 0.12, color=C_CONTROL, alpha=0.7, label="random-corpus floor ~+0.11")
    pts = {"random AI\nmixes (median)": (0.11, C_BASELINE), "chosen field\n(AI/ML core)": (0.23, C_GNN),
           "strongest\ncoherent AI": (0.33, C_BASELINE_STRONG)}
    for i, (lab, (v, c)) in enumerate(pts.items()):
        ax.scatter(i, v, s=42, color=c, edgecolor="black", linewidth=0.6, zorder=5)
        ax.text(i, v + 0.012, f"+{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.axhline(0, color=C_BASE_RATE, ls="--", lw=0.8)
    ax.set_xticks(range(len(pts))); ax.set_xticklabels(list(pts), fontsize=6.5)
    ax.set_ylabel("Δ AUPRC over momentum")
    ax.set_ylim(-0.02, 0.4)
    ax.set_title("400-combo ablation: a robust regime")
    save(fig, FIG, "paperA_ablation")

    # ---- 5. cross-domain transfer: AI/ML vs biology (Δ over momentum, twin-safe) ----
    fig, ax = plt.subplots(figsize=(3.2, 2.5))
    doms = ["AI/ML\n(arXiv)", "biology\n(bioRxiv)"]
    delta = [0.227, 0.162]
    err_lo = [0.227 - 0.122, 0.162 - 0.054]; err_hi = [0.330 - 0.227, 0.254 - 0.162]
    bars = ax.bar(doms, delta, yerr=[err_lo, err_hi], capsize=3, color=[C_GNN, C_BASELINE_STRONG],
                  width=0.56, edgecolor="black", linewidth=0.6, error_kw=dict(lw=0.9))
    for i, v in enumerate(delta):
        ax.text(i, v + err_hi[i] + 0.012, f"+{v:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.axhline(0, color=C_BASE_RATE, ls="--", lw=0.8)
    ax.set_ylabel("Δ AUPRC over momentum")
    ax.set_ylim(0, 0.42)
    ax.set_title("The signature transfers across science")
    save(fig, FIG, "paperA_transfer")

    print("[fig] wrote paperA_detector / _orthogonality / _baserate / _ablation / _transfer (.png + .pdf) to", FIG)


if __name__ == "__main__":
    main()
