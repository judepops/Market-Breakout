"""Schematic: the pipeline generalises to a new market via a config + a registry (the engineering proof).

    python -m experiments.plot_pipeline_generalises --out artifacts/figures

A diagram (not a data plot): one per-market config + a registry that dispatches the roster by name means the
SAME signal collectors + label code run on any market with zero new signal code. The numbers are real
(llm-dev-tools vs data-infra outputs from the identical pipeline).
"""

from __future__ import annotations

import argparse


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/figures")
    args = ap.parse_args(argv)

    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    from marketbrain.figures.style import (
        C_BASELINE, C_BASELINE_STRONG, C_GNN, apply_style, save,
    )
    apply_style()
    fig, ax = plt.subplots(figsize=(7.1, 4.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(cx, cy, w, h, text, *, edge, face, tw="normal", fs=8.5):
        ax.add_patch(mpatches.FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.4, edgecolor=edge, facecolor=face, zorder=2))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, fontweight=tw, zorder=3, color="#1c1c1e")

    def arrow(x, y0, y1):
        ax.annotate("", xy=(x, y1), xytext=(x, y0), zorder=1,
                    arrowprops=dict(arrowstyle="-|>", color=C_BASELINE, lw=1.6))

    box(0.5, 0.90, 0.66, 0.11,
        "Market config (per market)\nroster  ·  GitHub repos  ·  patent assignees  ·  arXiv terms",
        edge=C_BASELINE_STRONG, face="#f3f6f9")
    arrow(0.5, 0.845, 0.775)
    box(0.5, 0.71, 0.52, 0.10, "Market registry — get_market(name)\ndispatches the roster by name",
        edge=C_BASELINE_STRONG, face="#eef1f4")
    arrow(0.5, 0.66, 0.575)
    box(0.5, 0.49, 0.78, 0.12,
        "THE SAME signal collectors  +  label code\n(zero new signal code)",
        edge=C_GNN, face="#fbf4f4", tw="bold", fs=9.5)
    arrow(0.30, 0.43, 0.275)
    arrow(0.70, 0.43, 0.275)
    box(0.27, 0.17, 0.42, 0.16,
        "llm-dev-tools\n9 features (leakage-clean)\n12/21 >= $500M\n172 patents",
        edge=C_BASELINE, face="#ffffff")
    box(0.73, 0.17, 0.42, 0.16,
        "data-infra  (new, via config)\nsame 9 features (leakage-clean)\n21/21 >= $500M  ·  6.8-yr gap\n"
        "1,977 patents",
        edge=C_GNN, face="#ffffff", tw="bold")
    ax.text(0.5, 0.015,
            "one config + a registry dispatch  ->  the whole pipeline runs unchanged   ·   "
            "78 tests green   ·   llm-dev-tools untouched",
            ha="center", va="bottom", fontsize=7.5, color=C_BASELINE_STRONG, fontstyle="italic")
    ax.set_title("The machinery generalises: a new market is a config file, not a rebuild")

    png, _ = save(fig, args.out, "pipeline_generalises")
    print(f"saved {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
