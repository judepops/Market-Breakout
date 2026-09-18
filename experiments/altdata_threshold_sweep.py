"""Alt-data traction premium — OUTCOME-THRESHOLD sweep (pre-registered, all reported). Does the organic
(pre-round) traction premium appear at ANY success threshold, or is it null throughout (bulletproof)?

    python -m experiments.altdata_threshold_sweep

Motivation: at >=$500M there were only ~65 test positives -> the pre-round null could be underpowered. Lower
thresholds (>=$50M/$100M/$250M) give far more positives -> real power. Discipline: report EVERY threshold; the
reverse-causality guard still applies (a premium only counts if it shows PRE-ROUND, not just in the funding-
contaminated window); flag multiple comparisons. Outcome = total raised >= X (raise-based, consistent across X).
No money features as predictors.
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
DATA_END = 2026 * 12 + 5
THRESHOLDS = [50e6, 100e6, 250e6, 500e6]


def _m(x):
    s = str(x); return int(s[:4]) * 12 + int(s[5:7]) - 1


def main():
    import numpy as np, pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

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
    df["mkt_code"] = pd.Categorical(df["market"]).codes.astype(float)
    df["obs_age"] = (DATA_END - df["found_m"]).clip(lower=1)

    def wsig(panel, val, agg, pre):
        p = pd.read_parquet(f"data/raw/labels/{panel}.parquet")
        p["mm"] = p["month"].astype(str).str[:7].map(_m)
        p = p.merge(df[["permalink", "found_m", "fr_m"]], on="permalink", how="inner")
        w = p[(p["mm"] >= p["found_m"]) & ((p["mm"] < p["fr_m"]) if pre else (p["mm"] <= p["found_m"] + 24))]
        return np.log1p(w.groupby("permalink")[val].agg(agg))
    for tag, pre in [("pre", True), ("e24", False)]:
        df[f"{tag}_stars"] = df["permalink"].map(wsig("company_adoption", "stars", "max", pre)).fillna(0.0)
        df[f"{tag}_hn"] = df["permalink"].map(wsig("company_attention", "hn", "sum", pre)).fillna(0.0)
        df[f"{tag}_wiki"] = df["permalink"].map(wsig("company_wiki", "views", "max", pre)).fillna(0.0)

    has = set(pd.read_parquet("data/raw/labels/company_adoption.parquet")["permalink"])
    df = df[df["permalink"].isin(has)].reset_index(drop=True)
    raised = pd.read_parquet("data/raw/labels/company_rounds.parquet").groupby("permalink")["raised_usd"].sum()
    df["total_raised"] = df["permalink"].map(raised).fillna(0.0)

    CTRL = ["fy", "mkt_code", "obs_age"]
    def premium(cols_own, yv):
        tr = (df["fy"] < 2020).to_numpy(); te = (df["fy"] >= 2020).to_numpy()
        if yv[te].sum() < 10 or yv[tr].sum() < 10:
            return np.nan, np.nan, 0
        def fit(cols):
            clf = HistGradientBoostingClassifier(max_depth=3, max_iter=300, learning_rate=0.05, min_samples_leaf=20, random_state=0)
            clf.fit(df.loc[tr, cols].to_numpy(), yv[tr]); return roc_auc_score(yv[te], clf.predict_proba(df.loc[te, cols].to_numpy())[:, 1])
        base = fit(CTRL); full = fit(CTRL + cols_own)
        # permutation null on the traction cols (single refit per permutation; base is fixed)
        rng = np.random.RandomState(0); nd = []; own_tr = df.loc[tr, cols_own].to_numpy(); own_te = df.loc[te, cols_own].to_numpy()
        ctr_tr = df.loc[tr, CTRL].to_numpy(); ctr_te = df.loc[te, CTRL].to_numpy(); ntr = tr.sum(); nte = te.sum()
        for _ in range(150):
            pt = rng.permutation(ntr); pe = rng.permutation(nte)
            clf = HistGradientBoostingClassifier(max_depth=3, max_iter=300, learning_rate=0.05, min_samples_leaf=20, random_state=0)
            clf.fit(np.column_stack([ctr_tr, own_tr[pt]]), yv[tr])
            nd.append(roc_auc_score(yv[te], clf.predict_proba(np.column_stack([ctr_te, own_te[pe]]))[:, 1]) - base)
        p = (np.sum(np.array(nd) >= (full - base)) + 1) / (len(nd) + 1)
        return full - base, p, int(yv[te].sum())

    print(f"{'threshold':>10s}{'test_pos':>9s}   {'PRE-ROUND (organic) Δ / perm-p':>34s}   {'early-24 (contaminated) Δ':>26s}")
    for X in THRESHOLDS:
        yv = (df["total_raised"] >= X).astype(int).to_numpy()
        d_pre, p_pre, npos = premium(["pre_stars", "pre_hn", "pre_wiki"], yv)
        d_e24, _, _ = premium(["e24_stars", "e24_hn", "e24_wiki"], yv)
        flag = "  <-- pre-round SIGNAL" if (not np.isnan(p_pre) and p_pre < 0.05 and d_pre > 0) else ""
        print(f"{'>=$'+str(int(X/1e6))+'M':>10s}{npos:>9d}   Δ={d_pre:+.3f}  perm-p={p_pre:.3f}{'':>10s}   Δ={d_e24:+.3f}{flag}", flush=True)
    print("\n[sweep] discipline: 4 thresholds tested -> multiple-comparison aware (need p<~0.0125 to claim). A pre-round")
    print("[sweep] Δ only counts if it beats the null; the early-24 column shows the funding-contamination baseline.")


if __name__ == "__main__":
    main()
