"""Phase 1 — the first (crude, exploratory) features->rate predictor over the mature markets.

    python -m experiments.predict_rate --features data/raw/labels/market_features.parquet

N is tiny (the mature markets only), so this is NOT a validated model — it is the end-to-end harness + an
honest read. Two outputs:
  1. Spearman rank-correlation of each structural feature with the outlier rate (which features TRACK outlier
     propensity), the main honest signal at this N.
  2. Leave-one-market-out (LOMO) with the single best feature, vs a predict-the-mean baseline + a shuffle
     null, to see whether the out-of-sample RANKING beats chance at all.
Reports a figure (LOMO predicted vs actual, + the feature correlations). Frame strictly as exploratory.
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np

warnings.filterwarnings("ignore")

# candidate features (derived ones added in _matrix); kept interpretable, not exhaustive (N is tiny)
BASE = ["cohort_size", "early_round_count", "early_usd", "median_first_round_usd",
        "entrant_growth", "seed_hhi", "mega_round_count"]


def _rank(x):
    """Ranks (average ties) for a Spearman correlation = Pearson on ranks."""
    order = np.argsort(np.argsort(x))
    return order.astype(float)


def _spearman(x, y):
    rx, ry = _rank(x), _rank(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    d = np.sqrt((rx @ rx) * (ry @ ry))
    return float(rx @ ry / d) if d else 0.0


def _matrix(df):
    f = {k: df[k].to_numpy(float) for k in BASE}
    f["early_usd_per_co"] = df["early_usd"].to_numpy(float) / np.maximum(df["cohort_size"].to_numpy(float), 1)
    f["log_cohort"] = np.log10(np.maximum(df["cohort_size"].to_numpy(float), 1))
    return f


def _lomo_one_feature(x, y):
    """Leave-one-out linear fit on a single standardized feature; return predictions."""
    n = len(y)
    pred = np.zeros(n)
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        xs, ys = x[tr], y[tr]
        xm, xsd = xs.mean(), xs.std() or 1.0
        b1, b0 = np.polyfit((xs - xm) / xsd, ys, 1)
        pred[i] = b0 + b1 * ((x[i] - xm) / xsd)
    return pred


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/raw/labels/market_features.parquet")
    ap.add_argument("--out", default="artifacts/figures")
    args = ap.parse_args(argv)

    import pandas as pd

    df = pd.read_parquet(args.features)
    df = df[df["mature"] & df["footprint_rich"]].reset_index(drop=True)
    y = df["target_rate"].to_numpy(float)
    feats = _matrix(df)
    n = len(df)
    print(f"N = {n} mature footprint-rich markets (exploratory — not a validated model)\n")

    # 1. feature -> rate correlations
    corrs = sorted(((name, _spearman(x, y)) for name, x in feats.items()),
                   key=lambda t: -abs(t[1]))
    print(f"{'feature':22s} {'spearman r vs rate':>20s}")
    for name, r in corrs:
        print(f"{name:22s} {r:>20.2f}")

    # 2. LOMO with the single best feature, vs baselines
    best_name, best_r = corrs[0]
    x = feats[best_name]
    pred = _lomo_one_feature(x, y)
    mae = np.mean(np.abs(pred - y))
    base_mae = np.mean(np.abs(y - np.array([y[[j for j in range(n) if j != i]].mean() for i in range(n)])))
    rank_skill = _spearman(pred, y)
    # shuffle null for the rank skill
    rng = np.random.RandomState(0)
    null = [_spearman(pred, rng.permutation(y)) for _ in range(2000)]
    p = float((np.sum(np.abs(null) >= abs(rank_skill)) + 1) / (len(null) + 1))
    print(f"\nLOMO single best feature: {best_name} (r={best_r:.2f})")
    print(f"  MAE {mae*100:.2f}pp vs predict-the-mean baseline {base_mae*100:.2f}pp")
    print(f"  out-of-sample rank skill (spearman pred vs actual) = {rank_skill:.2f}  "
          f"(shuffle p = {p:.3f}, N={n})")
    # 3. domain split — does the signal HOLD within software vs deep-tech, or is it a pooling artifact?
    def _domain(cov):
        c = set(str(cov).split(","))
        if "gh" in c and "pat" in c:
            return "hybrid"
        return "deep-tech" if "pat" in c else "software"
    df["domain"] = df["coverage"].map(_domain)
    print(f"\nDoes the top feature ({best_name}) hold WITHIN each domain?")
    for dom, g in df.groupby("domain"):
        idx = g.index.to_numpy()
        if len(idx) >= 4:
            r = _spearman(feats[best_name][idx], y[idx])
            print(f"  {dom:10s} N={len(idx):>2d}  spearman r = {r:>5.2f}")
    print("\nHonest read: exploratory. The pooled signal can be a cross-domain artifact (deep-tech raises more "
          "capital AND has higher rates), so the within-domain correlations are the real test. Phase 2 signal "
          "features must beat this Crunchbase-only baseline.")
    _plot(df, feats, corrs, best_name, pred, y, args.out)
    return 0


def _plot(df, feats, corrs, best_name, pred, y, out):
    import matplotlib.pyplot as plt

    from marketbrain.figures.style import C_BASELINE_STRONG, C_GNN, apply_style, save
    apply_style()
    fig, (axc, axp) = plt.subplots(1, 2, figsize=(7.6, 3.4))
    names = [c[0] for c in corrs][::-1]
    vals = [c[1] for c in corrs][::-1]
    axc.barh(range(len(names)), vals, color=[C_GNN if v >= 0 else C_BASELINE_STRONG for v in vals])
    axc.set_yticks(range(len(names)))
    axc.set_yticklabels(names, fontsize=6.5)
    axc.axvline(0, color="#888", lw=0.8)
    axc.set_xlabel("Spearman r vs outlier rate")
    axc.set_title("Which structural features track the rate", fontsize=8)

    axp.scatter(pred * 100, y * 100, color=C_GNN, s=34, zorder=3)
    for _, r in df.iterrows():
        pass
    lim = [min(pred.min(), y.min()) * 100 - 0.5, max(pred.max(), y.max()) * 100 + 0.5]
    axp.plot(lim, lim, color="#888", lw=0.8, ls="--")
    for i, name in enumerate(df["market"]):
        axp.annotate(name, (pred[i] * 100, y[i] * 100), fontsize=5.2, color="#444",
                     xytext=(3, 2), textcoords="offset points")
    axp.set_xlabel(f"LOMO predicted rate (%), from {best_name}")
    axp.set_ylabel("actual outlier rate (%)")
    axp.set_title("Leave-one-market-out (exploratory, N small)", fontsize=8)
    png, _ = save(fig, out, "predict_rate_phase1")
    print(f"saved {png}")


if __name__ == "__main__":
    raise SystemExit(main())
