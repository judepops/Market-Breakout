"""ADVERSARIAL EFFECT-SIZE VERIFICATION of the rich pre-round signal.

Statistically real (p=0.007, Δ AUC=+0.045) does NOT imply practically useful. This script translates the
signal into deployment terms on the fy>=2020 test set, outcome = total raised >= $50M:

  (1) Fit controls+rich, rank test companies by predicted prob. Report base rate and PRECISION (hit-rate of
      >=$50M) in the top decile (10%) and top ventile (5%), as a LIFT multiple over base rate.
  (2) Same for a controls-ONLY model. Does adding rich features improve top-decile precision in practice?
  (3) Business translation.

Feature/panel code copied VERBATIM from experiments/altdata_richfeatures.py so the model is byte-identical.

    python -m experiments.verify_effectsize
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
DATA_END = 2026 * 12 + 5


def _m(x):
    s = str(x); return int(s[:4]) * 12 + int(s[5:7]) - 1


def build_frame():
    import numpy as np, pandas as pd

    lvl = pd.read_parquet("data/raw/labels/company_level.parquet")
    meta = pd.read_parquet("data/raw/labels/company_meta.parquet")[["permalink", "founded_on"]]
    df = lvl.merge(meta, on="permalink", how="left")
    fd = pd.to_datetime(df["founded_on"], errors="coerce"); frd = pd.to_datetime(df["first_round_date"], errors="coerce")
    df["fy"] = fd.dt.year
    df = df.dropna(subset=["fy"]).copy(); df["fy"] = df["fy"].astype(int)
    df["found_m"] = (fd.dt.year * 12 + fd.dt.month - 1).astype("Int64")
    df["fr_m"] = (frd.dt.year * 12 + frd.dt.month - 1).astype("Int64")
    df = df.dropna(subset=["found_m", "fr_m"]).copy()
    df["found_m"] = df["found_m"].astype(int); df["fr_m"] = df["fr_m"].astype(int)
    df["win_len"] = (df["fr_m"] - df["found_m"]).clip(lower=1)
    df["mkt_code"] = pd.Categorical(df["market"]).codes.astype(float)
    df["obs_age"] = (DATA_END - df["found_m"]).clip(lower=1)
    base = df.set_index("permalink")[["found_m", "fr_m", "win_len"]]

    feat_cols = []
    def rich(panel, val, tag):
        p = pd.read_parquet(f"data/raw/labels/{panel}.parquet")
        p["mm"] = p["month"].astype(str).str[:7].map(_m)
        p = p.join(base, on="permalink", how="inner")
        p = p[(p["mm"] >= p["found_m"]) & (p["mm"] < p["fr_m"])]
        p["rel"] = p["mm"] - p["found_m"]; p["late"] = (p["rel"] >= p["win_len"] / 2).astype(int)
        g = p.groupby("permalink")
        out = pd.DataFrame({
            f"{tag}_logmax": np.log1p(g[val].max()),
            f"{tag}_logmean": np.log1p(g[val].mean()),
            f"{tag}_active": g[val].apply(lambda s: (s > 0).mean()),
            f"{tag}_present": 1.0,
        })
        late = p[p["late"] == 1].groupby("permalink")[val].sum(); early = p[p["late"] == 0].groupby("permalink")[val].sum()
        out[f"{tag}_momentum"] = (np.log1p(late) - np.log1p(early)).reindex(out.index).fillna(0.0)
        for c in out.columns:
            df[c] = df["permalink"].map(out[c]).fillna(0.0); feat_cols.append(c)
    for panel, val, tag in [("company_adoption", "stars", "gh"), ("company_attention", "hn", "hn"), ("company_wiki", "views", "wiki")]:
        rich(panel, val, tag)
    df["breadth"] = df[[c for c in feat_cols if c.endswith("_present")]].sum(1)
    df["log_winlen"] = np.log1p(df["win_len"])
    RICH = feat_cols + ["breadth", "log_winlen"]
    has = set(pd.read_parquet("data/raw/labels/company_adoption.parquet")["permalink"])
    df = df[df["permalink"].isin(has)].reset_index(drop=True)
    raised = pd.read_parquet("data/raw/labels/company_rounds.parquet").groupby("permalink")["raised_usd"].sum()
    df["total_raised"] = df["permalink"].map(raised).fillna(0.0)
    return df, ["fy", "mkt_code", "obs_age"], RICH


def precision_at(y_true, scores, frac):
    """Precision (hit-rate) among the top `frac` fraction ranked by score."""
    import numpy as np
    n = len(scores)
    k = max(1, int(round(frac * n)))
    order = np.argsort(-scores, kind="mergesort")  # deterministic; highest score first
    top = order[:k]
    return y_true[top].mean(), k, int(y_true[top].sum())


def main():
    import numpy as np, pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier

    df, CTRL, RICH = build_frame()
    tr = (df["fy"] < 2020).to_numpy(); te = (df["fy"] >= 2020).to_numpy()
    yv = (df["total_raised"] >= 50e6).astype(int).to_numpy()
    y_te = yv[te]
    n_te = int(te.sum())
    base_rate = y_te.mean()
    print(f"[effectsize] test set (fy>=2020): {n_te} companies, {int(y_te.sum())} reached >=$50M, base rate = {base_rate:.4f} ({base_rate*100:.2f}%)")

    def fit_scores(cols):
        clf = HistGradientBoostingClassifier(max_depth=3, max_iter=400, learning_rate=0.04, min_samples_leaf=15, random_state=0)
        clf.fit(df.loc[tr, cols].to_numpy(), yv[tr])
        p = clf.predict_proba(df.loc[te, cols].to_numpy())[:, 1]
        from sklearn.metrics import roc_auc_score
        return p, roc_auc_score(y_te, p)

    scores_full, auc_full = fit_scores(CTRL + RICH)
    scores_ctrl, auc_ctrl = fit_scores(CTRL)
    print(f"[effectsize] AUC  controls-only={auc_ctrl:.3f}   controls+rich={auc_full:.3f}   Δ={auc_full-auc_ctrl:+.3f}")

    print("\n[effectsize] PRECISION & LIFT over base rate  (higher predicted-prob = screened in):")
    print(f"   {'model':16s} {'tier':10s} {'k':>5s} {'hits':>5s} {'precision':>10s} {'lift':>7s}")
    rows = {}
    for name, scores in [("controls+rich", scores_full), ("controls-only", scores_ctrl)]:
        for label, frac in [("top 10% (decile)", 0.10), ("top 5% (ventile)", 0.05)]:
            prec, k, hits = precision_at(y_te, scores, frac)
            lift = prec / base_rate if base_rate > 0 else float("nan")
            rows[(name, label)] = (prec, lift, k, hits)
            print(f"   {name:16s} {label:16s} {k:5d} {hits:5d} {prec:10.4f} {lift:6.2f}x")

    print("\n[effectsize] DOES RICH BEAT CONTROLS AT THE TOP?")
    for label in ["top 10% (decile)", "top 5% (ventile)"]:
        pf, lf, kf, hf = rows[("controls+rich", label)]
        pc, lc, kc, hc = rows[("controls-only", label)]
        print(f"   {label:16s}: rich precision={pf:.4f} ({lf:.2f}x)  vs  controls precision={pc:.4f} ({lc:.2f}x)   "
              f"Δprecision={pf-pc:+.4f}  Δhits={hf-hc:+d}")

    # Business translation
    pf10, lf10, kf10, hf10 = rows[("controls+rich", "top 10% (decile)")]
    pf5, lf5, kf5, hf5 = rows[("controls+rich", "top 5% (ventile)")]
    print("\n[effectsize] BUSINESS TRANSLATION:")
    print(f"   Screening the top 10% by pre-funding footprint finds >=$50M raisers at {lf10:.2f}x the base rate "
          f"({pf10*100:.1f}% hit-rate vs {base_rate*100:.1f}% overall).")
    print(f"   Screening the top 5%  finds them at {lf5:.2f}x ({pf5*100:.1f}% hit-rate).")

    # KILL CRITERION assessment
    pc10 = rows[("controls-only", "top 10% (decile)")][0]
    print("\n[effectsize] KILL-CRITERION CHECK:")
    print(f"   top-decile lift (rich) = {lf10:.2f}x  (kill if ~1x)")
    print(f"   rich adds at top decile: Δprecision={pf10-pc10:+.4f}  (kill if ~0)")


if __name__ == "__main__":
    main()
