"""Broad build — the N≈10 rate-target spine for the prediction dataset (step 2).

    python -m experiments.broad_build --out artifacts/figures

For every market in the catalog, resolve its full fundable (>= $5M) universe and compute the
percentile-per-segment label -> the market's outlier RATE (the cross-market target), total funding, and
universe size. Assembles the master target table (one row per market) + per-company percentiles, and a
figure of the rate target across markets (positives vs deliberate negatives), total funding annotated.

Cost: one cost-capped scan per market (~270-500 MB each). NO Gemini (pure Crunchbase at universe scale).
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/figures")
    ap.add_argument("--labels-dir", default="data/raw/labels")
    ap.add_argument("--project", default=None)
    args = ap.parse_args(argv)

    import pandas as pd

    from marketbrain.labels.universe_label import build_from_universe
    from marketbrain.markets.catalog import CATALOG

    Path(args.labels_dir).mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"{'market':22s} {'kind':9s} {'univ':>7s} {'>=500M':>7s} {'rate':>7s} "
          f"{'total$':>8s} {'p90 outcome':>12s}")
    for spec in CATALOG:
        comps = spec.universe(project=args.project)
        label, per = build_from_universe(comps, spec.name)
        # per-company percentiles -> parquet (the continuous target store)
        pd.DataFrame([{
            "permalink": c.permalink, "name": c.name, "outcome_usd": c.outcome_usd,
            "percentile": c.percentile, "is_outlier": c.is_outlier,
        } for c in per]).to_parquet(Path(args.labels_dir) / f"{spec.name}_universe.parquet", index=False)
        rows.append({
            "market": spec.name, "kind": spec.kind, "n_universe": label.n_universe,
            "n_outliers": label.n_outliers, "rate": label.rate,
            "total_funding_usd": label.total_funding_usd,
            "p50_outcome_usd": label.p50_outcome_usd, "p90_outcome_usd": label.p90_outcome_usd,
            "p99_outcome_usd": label.p99_outcome_usd,
        })
        print(f"{spec.name:22s} {spec.kind:9s} {label.n_universe:>7,} {label.n_outliers:>7,} "
              f"{label.rate:>6.1%} {label.total_funding_usd/1e9:>6.0f}B "
              f"{label.p90_outcome_usd/1e6:>10.0f}M")

    table = pd.DataFrame(rows).sort_values("rate", ascending=False)
    target_path = Path(args.labels_dir) / "broad_build_targets.parquet"
    table.to_parquet(target_path, index=False)
    lo, hi = table["rate"].min(), table["rate"].max()
    print(f"\n{len(table)} markets; outlier rate {lo:.1%}..{hi:.1%} = {hi/lo:.1f}x spread. "
          f"wrote {target_path}")
    _plot(table, args.out)
    return 0


def _plot(table, out):
    import matplotlib.pyplot as plt

    from marketbrain.figures.style import C_BASELINE_STRONG, C_GNN, apply_style, save
    apply_style()
    t = table.sort_values("rate")
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(t) + 1.2))
    for i, (_, r) in enumerate(t.iterrows()):
        color = C_BASELINE_STRONG if r["kind"] == "negative" else C_GNN
        ax.plot([0, r["rate"] * 100], [i, i], color="#e2e2e6", lw=0.9, zorder=1)
        ax.scatter([r["rate"] * 100], [i], color=color, s=36, zorder=3)
        ax.text(r["rate"] * 100, i, f"  {r['rate']:.1%}  (n={r['n_universe']:,}, \\${r['total_funding_usd']/1e9:.0f}B)",
                va="center", fontsize=6.5, color=color)
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels(t["market"])
    ax.set_xlim(0, max(t["rate"]) * 100 * 1.5)
    ax.set_xlabel(r"\$500M outlier rate over the \$5M-fundable universe (%)")
    ax.set_title(f"Broad build: outlier-rate target across {len(t)} sub-markets (the prediction spine)")
    ax.scatter([], [], color=C_GNN, s=36, label="dev-facing target market")
    ax.scatter([], [], color=C_BASELINE_STRONG, s=36, label="deliberate low-rate negative")
    ax.legend(loc="lower right", fontsize=6.5)
    png, _ = save(fig, out, "broad_build_targets")
    print(f"saved {png}")


if __name__ == "__main__":
    raise SystemExit(main())
