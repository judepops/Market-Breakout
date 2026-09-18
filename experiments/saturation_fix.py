"""The saturation fix, end to end: curated winner-rosters -> full fundable universes with a real rate.

    python -m experiments.saturation_fix --out artifacts/figures

Builds the percentile-per-segment label over each market's FULL fundable (>= $5M) universe and shows the
headline: the curated rosters saturated (llm-dev-tools 57%, data-infra 100% — no negatives), while the full
universe yields a real, learnable outlier RATE with a long tail of negatives. Saves per-company percentiles
to data/raw/labels/<market>_universe.parquet and two figures:
  * saturation_fix_rates: curated roster rate vs full-universe rate, per market (the before/after).
  * outcome_distribution: the universe outcome distribution (log $) with the $500M line + p50/p90/p99 — the
    continuous target the binary flattened.
"""

from __future__ import annotations

import argparse
from pathlib import Path

CURATED = {  # the saturated curated-roster rates, for the before/after contrast
    "llm-dev-tools": (12, 21),
    "data-infra": (21, 21),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/figures")
    ap.add_argument("--labels-dir", default="data/raw/labels")
    ap.add_argument("--project", default=None)
    args = ap.parse_args(argv)

    import pandas as pd

    from marketbrain.labels.universe_label import run

    results = {}
    Path(args.labels_dir).mkdir(parents=True, exist_ok=True)
    for market in CURATED:
        label, per = run(market, project=args.project)
        results[market] = (label, per)
        df = pd.DataFrame([{
            "permalink": c.permalink, "name": c.name, "outcome_usd": c.outcome_usd,
            "percentile": c.percentile, "is_outlier": c.is_outlier,
        } for c in per])
        out = Path(args.labels_dir) / f"{market}_universe.parquet"
        df.to_parquet(out, index=False)
        ow, on = CURATED[market]
        print(f"\n{market}:")
        print(f"  curated roster : {ow}/{on} = {ow/on:.0%} outliers  (saturated — no negatives)")
        print(f"  full universe  : {label.n_outliers:,}/{label.n_universe:,} = {label.rate:.1%} outliers  "
              f"({label.n_universe - label.n_outliers:,} negatives)")
        print(f"  total funding  : ${label.total_funding_usd/1e9:.0f}B   "
              f"p50/p90/p99 outcome = ${label.p50_outcome_usd/1e6:.0f}M / "
              f"${label.p90_outcome_usd/1e6:.0f}M / ${label.p99_outcome_usd/1e9:.1f}B")
        print(f"  wrote {out} ({len(df):,} companies)")

    _plot_rates(results, args.out)
    _plot_distribution(results, args.out)
    return 0


def _plot_rates(results, out):
    import matplotlib.pyplot as plt
    import numpy as np

    from marketbrain.figures.style import C_BASELINE_STRONG, C_GNN, apply_style, save
    apply_style()
    markets = list(results)
    curated = [CURATED[m][0] / CURATED[m][1] for m in markets]
    universe = [results[m][0].rate for m in markets]
    y = np.arange(len(markets))
    fig, ax = plt.subplots(figsize=(7.0, 2.2))
    h = 0.34
    ax.barh(y + h / 2, [c * 100 for c in curated], height=h, color=C_BASELINE_STRONG,
            label="curated roster (saturated)")
    ax.barh(y - h / 2, [u * 100 for u in universe], height=h, color=C_GNN,
            label="full fundable universe (the target)")
    for i, m in enumerate(markets):
        ax.text(curated[i] * 100, i + h / 2, f" {curated[i]:.0%}", va="center", fontsize=7,
                color=C_BASELINE_STRONG, fontweight="bold")
        lab = results[m][0]
        ax.text(universe[i] * 100, i - h / 2,
                f" {universe[i]:.1%}  ({lab.n_outliers:,}/{lab.n_universe:,})", va="center", fontsize=7,
                color=C_GNN, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(markets)
    ax.set_xlim(0, 112)
    ax.set_xlabel(r"\$500M outlier rate (%)")
    ax.set_title("Saturation fixed: curated winner-rosters vs full fundable universe")
    ax.legend(loc="lower right", fontsize=6.5)
    png, _ = save(fig, out, "saturation_fix_rates")
    print(f"\nsaved {png}")


def _plot_distribution(results, out):
    import matplotlib.pyplot as plt
    import numpy as np

    from marketbrain.figures.style import C_BASELINE_STRONG, C_GNN, apply_style, save
    apply_style()
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    colors = {"llm-dev-tools": C_GNN, "data-infra": C_BASELINE_STRONG}
    for market, (label, per) in results.items():
        vals = np.array([max(c.outcome_usd, 1) for c in per], dtype=float) / 1e6  # $M, floor at $1 for log
        vals = np.sort(vals)
        ccdf = 1.0 - np.arange(len(vals)) / len(vals)  # P(outcome >= x)
        ax.step(vals, ccdf, where="post", color=colors[market], lw=1.6,
                label=f"{market}  (rate {label.rate:.1%}, n={label.n_universe:,})")
    ax.axvline(500, color="#444", lw=1.0, ls="--")
    ax.text(500, 0.92, r" \$500M outlier line", fontsize=6.5, color="#444")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"company outcome = max(raised, valuation)  (\$M, log)")
    ax.set_ylabel("fraction of universe with outcome >= x")
    ax.set_title("Outcome distribution per market: the continuous target the binary flattened")
    ax.legend(loc="lower left", fontsize=6.5)
    png, _ = save(fig, out, "outcome_distribution")
    print(f"saved {png}")


if __name__ == "__main__":
    raise SystemExit(main())
