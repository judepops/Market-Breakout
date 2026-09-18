"""Lead-lag analysis: does attention lead adoption / research lead adoption / adoption lead funding?

    python -m experiments.lagged_correlation --root data/raw --out artifacts/figures

EXPLORATORY (n~30 YoY-growth points, one market) — hypothesis-generating, not inference. Runs on the
**stationary** (YoY-growth) series, reports the peak lead-lag lag + correlation + a circular-shift
permutation p-value per pair, and saves a (pair x lag) correlation heatmap. See `features/leadlag.py`.
"""

from __future__ import annotations

import argparse

import pandas as pd

from marketbrain.corpus import CorpusStore
from marketbrain.features import leadlag
from marketbrain.figures.style import apply_style, heatmap_cmap, save

# short name -> corpus feature
FEATURES = {
    "arxiv": "arxiv_submissions", "wikipedia": "wikipedia_pageviews", "hackernews": "hn_story_count",
    "npm": "npm_downloads", "pypi": "pypi_downloads", "github": "github_stars",
    "crunchbase": "crunchbase_funding_usd", "emergence": "arxiv_emergence",
}
# hypothesised driver -> target pairs (does the driver lead the target?)
PAIRS = [
    ("hackernews", "github"), ("hackernews", "pypi"), ("wikipedia", "github"), ("wikipedia", "pypi"),
    ("arxiv", "github"), ("arxiv", "pypi"), ("emergence", "github"),
    ("github", "crunchbase"), ("pypi", "crunchbase"), ("arxiv", "crunchbase"),
]
MAX_LAG = 6


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description="Lead-lag (time-lag) analysis on the stationary signals.")
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--market", default="llm-dev-tools")
    ap.add_argument("--out", default="artifacts/figures")
    ap.add_argument("--max-lag", type=int, default=MAX_LAG)
    args = ap.parse_args(argv)

    df = CorpusStore(args.root).load_observations()
    df = df[df["market"] == args.market].copy()
    df["m"] = pd.to_datetime(df["reference_date"]).dt.strftime("%Y-%m")
    wide = df.pivot_table(index="m", columns="feature", values="value", aggfunc="first").sort_index()

    yoy = {s: leadlag.yoy_growth(wide[f]) for s, f in FEATURES.items() if f in wide.columns}
    pairs = [(d, t) for d, t in PAIRS if d in yoy and t in yoy]

    print(f"Lead-lag (YoY-growth, n={wide.shape[0]} months, lags 0..{args.max_lag}, {args.market}) "
          "— EXPLORATORY, one market:\n")
    print(f"  {'driver -> target':28s} {'best lag':>8s} {'corr':>7s} {'perm p':>7s}")
    rows, labels = [], []
    for d, t in pairs:
        dv, tv = yoy[d].to_numpy(), yoy[t].to_numpy()
        k, c = leadlag.best_lag(dv, tv, args.max_lag)
        p = leadlag.permutation_pvalue(dv, tv, args.max_lag)
        flag = "*" if (p is not None and p < 0.1) else " "
        print(f"  {d + ' -> ' + t:28s} {str(k):>8s} {c:>7.2f} {p:>7.3f} {flag}")
        rows.append([leadlag.lagged_corr(dv, tv, j) for j in range(args.max_lag + 1)])
        labels.append(f"{d}→{t}")

    _plot(rows, labels, args.max_lag, args.out, args.market)
    print("\n* = circular-shift permutation p < 0.1. Lead-lag = predictive structure; treat as hypotheses.")
    return 0


def _plot(rows, labels, max_lag, out, market):
    import matplotlib.pyplot as plt
    import numpy as np

    apply_style()
    mat = np.array(rows)
    fig, ax = plt.subplots(figsize=(6.6, 0.42 * len(labels) + 1.4))
    im = ax.imshow(mat, cmap=heatmap_cmap(), vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(max_lag + 1))
    ax.set_xticklabels([f"+{k}" for k in range(max_lag + 1)])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("driver leads target by k months")
    ax.set_title(f"(k) {market} — lead-lag correlation (YoY growth)")
    for i in range(mat.shape[0]):
        finite = [j for j in range(mat.shape[1]) if np.isfinite(mat[i, j])]
        best = max(finite, key=lambda j: abs(mat[i, j])) if finite else None
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                w = "bold" if j == best else "normal"
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6, fontweight=w,
                        color="white" if abs(mat[i, j]) > 0.6 else "#222")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="corr")
    png, pdf = save(fig, out, f"leadlag_{market}")
    print(f"saved {png}")


if __name__ == "__main__":
    raise SystemExit(main())
