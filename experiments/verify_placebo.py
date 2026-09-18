"""Adversarial verification of the RICH pre-round alt-data signal (Δ=+0.045, perm-p=0.007).

ATTACK: stricter STRATIFIED permutation null + LABEL placebo.

 (1) STRATIFIED null: shuffle which rich-feature vector is assigned to which company WITHIN
     (market, founding-year) strata. This destroys the company-specific footprint <-> outcome link
     while PRESERVING sector/year composition. If the observed Δ still beats this stricter null,
     the lift is company-specific, not a sector x year composition artifact.

 (2) LABEL placebo: shuffle the outcome (raised>=50M) labels; the rich lift must collapse to ~0.
     Guards against a bug that manufactures lift regardless of labels.

    python -m experiments.verify_placebo

Reuses the EXACT feature-building recipe from altdata_richfeatures.py.
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
DATA_END = 2026 * 12 + 5
N_PERM = 150
SEED = 0


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
    return df, RICH


def main():
    import numpy as np, pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    df, RICH = build_frame()
    CTRL = ["fy", "mkt_code", "obs_age"]
    tr = (df["fy"] < 2020).to_numpy(); te = (df["fy"] >= 2020).to_numpy()
    print(f"[verify] {len(df)} companies, {len(RICH)} rich feats; train(fy<2020)={tr.sum()} test(fy>=2020)={te.sum()}", flush=True)

    def new_clf():
        return HistGradientBoostingClassifier(max_depth=3, max_iter=400, learning_rate=0.04,
                                              min_samples_leaf=15, random_state=0)

    # precompute control/own blocks
    ctr_tr = df.loc[tr, CTRL].to_numpy(); ctr_te = df.loc[te, CTRL].to_numpy()
    own_tr = df.loc[tr, RICH].to_numpy(); own_te = df.loc[te, RICH].to_numpy()

    def fit_auc(Xtr, Xte, yv):
        clf = new_clf(); clf.fit(Xtr, yv[tr]); return roc_auc_score(yv[te], clf.predict_proba(Xte)[:, 1])

    # strata: (market code, founding year) — use global row index so we can map train/test subsets
    idx_tr = np.where(tr)[0]; idx_te = np.where(te)[0]
    # strata computed on the WHOLE frame; permutation done within stratum & within split
    strata_key = list(zip(df["mkt_code"].to_numpy(), df["fy"].to_numpy()))
    strata_arr = np.array([hash(k) for k in strata_key])

    def stratified_perm(rng, which_idx):
        """Return a permutation (in local coordinates of which_idx) that only swaps rows within the same stratum."""
        n = len(which_idx)
        perm = np.arange(n)
        keys = strata_arr[which_idx]
        for k in np.unique(keys):
            pos = np.where(keys == k)[0]
            if len(pos) > 1:
                perm[pos] = pos[rng.permutation(len(pos))]
        return perm

    for X in [50e6, 500e6]:
        yv = (df["total_raised"] >= X).astype(int).to_numpy()
        if yv[te].sum() < 10:
            continue
        b = fit_auc(ctr_tr, ctr_te, yv)
        f = fit_auc(np.column_stack([ctr_tr, own_tr]), np.column_stack([ctr_te, own_te]), yv)
        obs = f - b
        print(f"\n[verify] >=${int(X/1e6)}M ({int(yv[te].sum())} test pos): controls={b:.3f}  +rich={f:.3f}  Δ={obs:+.3f}", flush=True)

        # (1) GLOBAL null (baseline reproduction)
        rng = np.random.RandomState(SEED); g_nd = []
        for _ in range(N_PERM):
            pt = rng.permutation(tr.sum()); pe = rng.permutation(te.sum())
            clf = new_clf(); clf.fit(np.column_stack([ctr_tr, own_tr[pt]]), yv[tr])
            g_nd.append(roc_auc_score(yv[te], clf.predict_proba(np.column_stack([ctr_te, own_te[pe]]))[:, 1]) - b)
        g_nd = np.array(g_nd); g_p = (np.sum(g_nd >= obs) + 1) / (N_PERM + 1)
        print(f"   [global   null] mean Δ_null={g_nd.mean():+.4f}  95%={np.quantile(g_nd,0.95):+.4f}  perm-p={g_p:.3f}", flush=True)

        # (2) STRATIFIED null — shuffle feature vectors WITHIN (market, fy) strata
        rng = np.random.RandomState(SEED); s_nd = []
        for _ in range(N_PERM):
            pt = stratified_perm(rng, idx_tr); pe = stratified_perm(rng, idx_te)
            clf = new_clf(); clf.fit(np.column_stack([ctr_tr, own_tr[pt]]), yv[tr])
            s_nd.append(roc_auc_score(yv[te], clf.predict_proba(np.column_stack([ctr_te, own_te[pe]]))[:, 1]) - b)
        s_nd = np.array(s_nd); s_p = (np.sum(s_nd >= obs) + 1) / (N_PERM + 1)
        verdict = "SURVIVES stricter null" if (s_p < 0.05 and obs > 0) else "COMPOSITION ARTIFACT"
        print(f"   [stratified null] mean Δ_null={s_nd.mean():+.4f}  95%={np.quantile(s_nd,0.95):+.4f}  strat-perm-p={s_p:.3f}  -> {verdict}", flush=True)

        # (3) LABEL placebo — shuffle outcome labels, real feature block; lift must collapse to ~0
        rng = np.random.RandomState(SEED); lp = []
        for _ in range(N_PERM):
            yp = yv.copy()
            yp[tr] = yp[tr][rng.permutation(tr.sum())]
            yp[te] = yp[te][rng.permutation(te.sum())]
            clf_b = new_clf(); clf_b.fit(ctr_tr, yp[tr]); bb = roc_auc_score(yp[te], clf_b.predict_proba(ctr_te)[:, 1])
            clf_f = new_clf(); clf_f.fit(np.column_stack([ctr_tr, own_tr]), yp[tr])
            ff = roc_auc_score(yp[te], clf_f.predict_proba(np.column_stack([ctr_te, own_te]))[:, 1])
            lp.append(ff - bb)
        lp = np.array(lp)
        placebo_ok = "OK (collapses to ~0)" if abs(lp.mean()) < 0.02 and lp.mean() < obs else "SUSPICIOUS"
        print(f"   [label placebo ] mean Δ_placebo={lp.mean():+.4f}  sd={lp.std():.4f}  (real Δ={obs:+.3f})  -> {placebo_ok}", flush=True)

    print("\n[verify] KILL if stratified perm-p > 0.05 at $50M: lift is a sector/year composition artifact.")


if __name__ == "__main__":
    main()
