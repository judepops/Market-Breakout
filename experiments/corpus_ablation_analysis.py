"""Analyse the corpus-combination ablation (artifacts/corpus_ablation/results.parquet).

    python -m experiments.corpus_ablation_analysis

Answers the two questions that decide the honest headline:
  Q1  Is the sweet spot PRINCIPLED?  Among random medium-breadth subsets, what fraction clear a
      HELD-OUT (half-B) delta threshold?  A regime = many combos clear it; a fluke = ~only cand-9.
  Q2  Which categories SYSTEMATICALLY carry the lift?  Marginal inclusion effect + a linear
      importance regression of held-out delta on the 28 category-inclusion indicators.
Plus: winner's-curse shrinkage (delta_A selection vs delta_B held-out) for the greedy winners, and the
best defensible (held-out) combos. Runs against partial results too (safe to call mid-search).
"""
from __future__ import annotations
import pathlib

import numpy as np
import pandas as pd

OUT = pathlib.Path("artifacts/corpus_ablation")


def _held(df):
    """Held-out delta: prefer delta_B, fall back to delta where B is missing/degenerate."""
    return df["delta_B"].where(df["delta_B"].notna(), df["delta"])


def main():
    f = OUT / "results.parquet"
    if not f.exists():
        print("no results yet"); return
    df = pd.read_parquet(f)
    df["held"] = _held(df)
    print(f"[analysis] {len(df)} evals: " + ", ".join(f"{k}={v}" for k, v in df["note"].value_counts().items()))

    rnd = df[df["note"] == "random"].copy()
    print(f"\n[analysis] === Q1: is the sweet spot PRINCIPLED? (random subsets, held-out half-B delta) ===")
    for lo, hi in [(2, 4), (5, 7), (8, 10), (11, 14)]:
        sub = rnd[(rnd["k"] >= lo) & (rnd["k"] <= hi)]
        if not len(sub):
            continue
        for thr in (0.05, 0.10, 0.15):
            frac = (sub["held"] > thr).mean()
            print(f"   k in [{lo:2d},{hi:2d}] (n_combos={len(sub):3d})  held>+{thr:.2f}: {frac:5.1%}"
                  + ("   <-- " + ("REGIME" if (thr == 0.10 and frac >= 0.4) else "") if thr == 0.10 else ""))
        print(f"       held delta: median={sub['held'].median():+.3f}  mean={sub['held'].mean():+.3f}  "
              f"p90={sub['held'].quantile(0.9):+.3f}  base_rate median={sub['base_rate'].median():.2f}")

    # relationship of lift to structure
    if len(rnd) >= 20:
        print(f"\n[analysis] correlations over random subsets:")
        for col in ["k", "n", "base_rate", "n_papers"]:
            r = np.corrcoef(rnd[col].to_numpy(float), rnd["held"].to_numpy(float))[0, 1]
            print(f"   corr(held_delta, {col:9s}) = {r:+.3f}")

    # ---- Q2: category importance -------------------------------------------------
    print(f"\n[analysis] === Q2: category importance (random subsets) ===")
    universe = sorted({c for cs in rnd["cats"] for c in cs})
    X = pd.DataFrame({c: rnd["cats"].apply(lambda cs, c=c: int(c in cs)) for c in universe}, index=rnd.index)
    y = rnd["held"].to_numpy(float)

    # marginal inclusion effect: mean held-delta WITH cat minus WITHOUT
    marg = []
    for c in universe:
        inc = X[c] == 1
        if inc.sum() >= 5 and (~inc).sum() >= 5:
            marg.append((c, y[inc].mean() - y[~inc].mean(), int(inc.sum())))
    marg.sort(key=lambda t: t[1], reverse=True)

    # linear regression of held-delta on inclusion indicators, bootstrap CI per coef
    Xm = np.column_stack([np.ones(len(rnd)), X.to_numpy(float)])
    coef, *_ = np.linalg.lstsq(Xm, y, rcond=None)
    rng = np.random.RandomState(0)
    boot = []
    for _ in range(1000):
        idx = rng.randint(0, len(rnd), len(rnd))
        try:
            b, *_ = np.linalg.lstsq(Xm[idx], y[idx], rcond=None)
            boot.append(b)
        except Exception:
            pass
    boot = np.array(boot)
    reg = {}
    for i, c in enumerate(universe):
        col = boot[:, i + 1]
        reg[c] = (coef[i + 1], np.percentile(col, 2.5), np.percentile(col, 97.5))

    print(f"   {'cat':10s}{'marg_incl':>11s}{'n_in':>6s}   {'reg_coef':>9s}  {'95% CI':>18s}")
    order = sorted(universe, key=lambda c: reg[c][0], reverse=True)
    mdict = {c: (m, n) for c, m, n in marg}
    for c in order:
        b, lo, hi = reg[c]
        m, nin = mdict.get(c, (float("nan"), 0))
        star = " *" if (lo > 0 or hi < 0) else ""
        print(f"   {c:10s}{m:>+11.3f}{nin:>6d}   {b:>+9.3f}  [{lo:>+7.3f},{hi:>+7.3f}]{star}")

    # ---- best held-out combos + greedy shrinkage --------------------------------
    print(f"\n[analysis] === best HELD-OUT combos (delta_B, n>=100 for reliability) ===")
    rel = df[(df["n"] >= 100) & df["delta_B"].notna()].sort_values("delta_B", ascending=False)
    for _, r in rel.head(12).iterrows():
        cs = ",".join(r["cats"]) if r["k"] <= 8 else f"{r['k']}cats:" + ",".join(r["cats"][:6]) + "..."
        print(f"   held={r['delta_B']:+.3f}  full={r['delta']:+.3f}  n={r['n']:3d} br={r['base_rate']:.2f} "
              f"[{r['note']}]  {cs}")

    greedy = df[df["note"].isin(("greedy_fwd", "greedy_back"))]
    if len(greedy):
        print(f"\n[analysis] === greedy winners: selection (delta_A) vs held-out (delta_B) shrinkage ===")
        for note in ("greedy_fwd", "greedy_back"):
            g = greedy[greedy["note"] == note]
            if not len(g):
                continue
            best = g.loc[g["delta_A"].fillna(g["delta"]).idxmax()]
            shr = (best["delta_A"] - best["delta_B"]) if pd.notna(best.get("delta_A")) and pd.notna(best.get("delta_B")) else float("nan")
            cs = ",".join(best["cats"])
            print(f"   {note:11s} k={best['k']:2d}  delta_A(sel)={best.get('delta_A')}  "
                  f"delta_B(held)={best.get('delta_B')}  shrink={shr:+.3f}\n      {cs}")

    print(f"\n[analysis] HONEST HEADLINE CANDIDATE: median held-delta of robust medium-breadth regime = "
          f"{rnd[(rnd['k']>=5)&(rnd['k']<=12)]['held'].median():+.3f}")


if __name__ == "__main__":
    main()
