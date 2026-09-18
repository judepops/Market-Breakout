"""Two figures for the 8-July deck: interpretability (logistic coefficients + simple-vs-boosted) and the
full 6-field translational transfer law.

    ~/miniconda3/envs/marketbrain/bin/python -m experiments.fig_deck
"""
from __future__ import annotations
import pathlib, warnings
warnings.filterwarnings("ignore")
FIG = pathlib.Path("artifacts/figures")


def main():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from marketbrain.figures.style import apply_style, save, C_GNN, C_BASELINE_STRONG, C_BASELINE, C_CONTROL, C_BASE_RATE
    apply_style()

    # ---- interpretability: logistic coefficients (what drives durability) ----
    coefs = [("venue penetration", 0.664), ("persistence", 0.566), ("emergence slope", 0.481),
             ("author breadth", 0.445), ("acceleration", 0.377), ("author inflow", 0.353),
             ("volume share", 0.205), ("cross-listing", -0.019), ("cross-domain spread", -0.177),
             ("emergence level", -0.228), ("growth ratio", -0.251)]
    coefs = sorted(coefs, key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    labels = [c[0] for c in coefs]; vals = [c[1] for c in coefs]
    colors = [C_GNN if v >= 0 else C_BASELINE_STRONG for v in vals]
    ax.barh(range(len(vals)), vals, color=colors, edgecolor="black", linewidth=0.5, height=0.72)
    ax.axvline(0, color=C_BASE_RATE, lw=0.9)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("logistic coefficient  (+ = more durable)")
    ax.set_title("A simple model, and what drives it")
    ax.set_xlim(-0.4, 0.8)
    save(fig, FIG, "deck_interp_coefs")

    # ---- simple vs boosted AUPRC ----
    fig, ax = plt.subplots(figsize=(2.7, 3.0))
    names = ["boosted\ndetector", "logistic\n(11 feats)", "tree\n(depth 3)", "momentum"]
    ap = [0.651, 0.625, 0.603, 0.432]
    cols = [C_GNN, C_BASELINE_STRONG, C_BASELINE, C_CONTROL]
    b = ax.bar(range(4), ap, color=cols, edgecolor="black", linewidth=0.6, width=0.66)
    for i, v in enumerate(ap):
        ax.text(i, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(4)); ax.set_xticklabels(names, fontsize=6.5)
    ax.set_ylabel("AUPRC"); ax.set_ylim(0, 0.75)
    ax.set_title("Simple ≈ boosted")
    save(fig, FIG, "deck_interp_bars")

    # ---- 6-field transfer law ----
    fields = [("AI/ML", 0.227, 0.122, 0.330, True), ("biology", 0.162, 0.054, 0.254, True),
              ("signal proc.", 0.127, 0.035, 0.217, True), ("quantum", -0.016, -0.117, 0.078, False),
              ("theor. physics", -0.015, -0.108, 0.072, False), ("gravitation", 0.025, -0.058, 0.108, False)]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ys = range(len(fields))[::-1]
    for y, (nm, d, lo, hi, fire) in zip(ys, fields):
        c = C_GNN if fire else C_CONTROL
        ax.plot([lo, hi], [y, y], color=c, lw=2.2, solid_capstyle="round", zorder=2)
        ax.scatter([d], [y], s=34, color=c, edgecolor="black", linewidth=0.6, zorder=3)
        ax.text(hi + 0.01, y, ("FIRE" if fire else "null"), va="center", fontsize=7,
                color=(C_GNN if fire else C_BASE_RATE), fontweight=("bold" if fire else "normal"))
    ax.axvline(0, color=C_BASE_RATE, ls="--", lw=0.9)
    ax.set_yticks(list(ys)); ax.set_yticklabels([f[0] for f in fields], fontsize=8)
    ax.set_xlabel("Δ AUPRC over momentum (twin-safe, 95% CI)")
    ax.set_xlim(-0.20, 0.42)
    ax.set_title("Fires in translational fields, null in foundational — 6/6")
    save(fig, FIG, "deck_transfer6")
    print("[fig] wrote deck_interp_coefs / deck_interp_bars / deck_transfer6", flush=True)


if __name__ == "__main__":
    main()
