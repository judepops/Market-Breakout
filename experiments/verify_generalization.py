"""ADVERSARIAL VERIFICATION — out-of-sector & alternative-split robustness of the +rich pre-round signal.

    python -m experiments.verify_generalization

Reproduces the baseline (controls vs +rich, $50M outcome, fy<2020/>=2020 split), then attacks:
  (1) SECTOR-HOLDOUT (leave-markets-out): train on one half of markets, test on the other half.
  (2) ALTERNATIVE TEMPORAL SPLITS: fy<2019/>=2019 and fy<2021/>=2021.
  (3) PER-TEST-YEAR: within fy>=2020, break the +rich vs controls gap by founding-year cohort.

All features strictly BEFORE first funding. No money features. Same model/hyperparams as the reference.
"""
from __future__ import annotations
import warnings, hashlib
warnings.filterwarnings("ignore")
DATA_END = 2026 * 12 + 5


def _m(x):
    s = str(x); return int(s[:4]) * 12 + int(s[5:7]) - 1


def _clf():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_depth=3, max_iter=400, learning_rate=0.04,
                                          min_samples_leaf=15, random_state=0)


def build_df():
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
    df["y50"] = (df["total_raised"] >= 50e6).astype(int)
    return df, RICH


CTRL = ["fy", "mkt_code", "obs_age"]


def auc_split(df, cols, tr, te, y):
    from sklearn.metrics import roc_auc_score
    import numpy as np
    clf = _clf()
    clf.fit(df.loc[tr, cols].to_numpy(), y[tr])
    return roc_auc_score(y[te], clf.predict_proba(df.loc[te, cols].to_numpy())[:, 1])


def perm_p(df, RICH, tr, te, y, b, f, nperm=150, seed=0):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    rng = np.random.RandomState(seed); nd = []
    ot, oe = df.loc[tr, RICH].to_numpy(), df.loc[te, RICH].to_numpy()
    ct, ce = df.loc[tr, CTRL].to_numpy(), df.loc[te, CTRL].to_numpy()
    for _ in range(nperm):
        clf = _clf()
        clf.fit(np.column_stack([ct, ot[rng.permutation(tr.sum())]]), y[tr])
        nd.append(roc_auc_score(y[te], clf.predict_proba(np.column_stack([ce, oe[rng.permutation(te.sum())]]))[:, 1]) - b)
    return (np.sum(np.array(nd) >= (f - b)) + 1) / (len(nd) + 1)


def main():
    import numpy as np, pandas as pd
    df, RICH = build_df()
    y = df["y50"].to_numpy()
    print(f"[verify] {len(df)} companies, {len(RICH)} rich feats, {int(y.sum())} total $50M+ ({y.mean():.1%})", flush=True)

    # ---- BASELINE reproduction (fy<2020 / >=2020) ----
    print("\n=== BASELINE (fy<2020 train / fy>=2020 test), $50M ===", flush=True)
    tr = (df["fy"] < 2020).to_numpy(); te = (df["fy"] >= 2020).to_numpy()
    b = auc_split(df, CTRL, tr, te, y); f = auc_split(df, CTRL + RICH, tr, te, y)
    p = perm_p(df, RICH, tr, te, y, b, f)
    print(f"  n_train={tr.sum()} n_test={te.sum()} test_pos={y[te].sum()}", flush=True)
    print(f"  controls={b:.3f}  +rich={f:.3f}  Δ={f-b:+.3f}  perm-p={p:.3f}", flush=True)

    # ---- (2) ALTERNATIVE TEMPORAL SPLITS ----
    print("\n=== (2) ALTERNATIVE TEMPORAL SPLITS ($50M) ===", flush=True)
    for cut in [2019, 2020, 2021]:
        tr = (df["fy"] < cut).to_numpy(); te = (df["fy"] >= cut).to_numpy()
        if y[te].sum() < 10 or y[tr].sum() < 10:
            print(f"  cut fy<{cut}/>={cut}: too few positives (train_pos={y[tr].sum()} test_pos={y[te].sum()})", flush=True)
            continue
        b = auc_split(df, CTRL, tr, te, y); f = auc_split(df, CTRL + RICH, tr, te, y)
        p = perm_p(df, RICH, tr, te, y, b, f)
        tag = "  [baseline]" if cut == 2020 else ""
        print(f"  cut fy<{cut}/>={cut}: n_tr={tr.sum()} n_te={te.sum()} te_pos={y[te].sum()}  "
              f"controls={b:.3f}  +rich={f:.3f}  Δ={f-b:+.3f}  perm-p={p:.3f}{tag}", flush=True)

    # ---- (1) SECTOR-HOLDOUT (leave-markets-out) ----
    print("\n=== (1) SECTOR-HOLDOUT (leave-markets-out, ignore year) ($50M) ===", flush=True)
    mkts = df["market"].astype(str)
    def mhash(s):
        return int(hashlib.md5(s.encode()).hexdigest(), 16) % 2
    df["_mhalf"] = mkts.map(mhash)
    # remove market code from controls in this test (it would leak the held-out sector identity is fine, but
    # for a fair leave-markets-out we keep controls; market codes in test are simply unseen categories)
    for testhalf in [0, 1]:
        tr = (df["_mhalf"] != testhalf).to_numpy(); te = (df["_mhalf"] == testhalf).to_numpy()
        b = auc_split(df, CTRL, tr, te, y); f = auc_split(df, CTRL + RICH, tr, te, y)
        p = perm_p(df, RICH, tr, te, y, b, f)
        print(f"  hash-half test={testhalf}: n_tr={tr.sum()} n_te={te.sum()} te_pos={y[te].sum()} "
              f"({df.loc[te,'market'].nunique()} test markets)  "
              f"controls={b:.3f}  +rich={f:.3f}  Δ={f-b:+.3f}  perm-p={p:.3f}", flush=True)
    # also drop mkt_code from controls entirely (pure out-of-sector: model can't lean on sector code at all)
    print("  -- same but controls WITHOUT mkt_code (pure out-of-sector) --", flush=True)
    CTRL2 = ["fy", "obs_age"]
    for testhalf in [0, 1]:
        tr = (df["_mhalf"] != testhalf).to_numpy(); te = (df["_mhalf"] == testhalf).to_numpy()
        b = auc_split(df, CTRL2, tr, te, y); f = auc_split(df, CTRL2 + RICH, tr, te, y)
        print(f"  hash-half test={testhalf} (no mkt): controls={b:.3f}  +rich={f:.3f}  Δ={f-b:+.3f}", flush=True)
    # multi-seed leave-markets-out to check stability across random market partitions
    print("  -- leave-markets-out Δ across 8 random market partitions (50/50) --", flush=True)
    uniq = df["market"].astype(str).unique()
    deltas = []
    for seed in range(8):
        rng = np.random.RandomState(seed)
        assign = {m: rng.randint(2) for m in uniq}
        half = df["market"].astype(str).map(assign)
        d0 = []
        for testhalf in [0, 1]:
            tr = (half != testhalf).to_numpy(); te = (half == testhalf).to_numpy()
            if y[te].sum() < 10 or y[tr].sum() < 10:
                continue
            b = auc_split(df, CTRL, tr, te, y); f = auc_split(df, CTRL + RICH, tr, te, y)
            d0.append(f - b)
        if d0:
            m = float(np.mean(d0)); deltas.append(m)
            print(f"    seed {seed}: mean Δ over 2 folds = {m:+.3f}", flush=True)
    if deltas:
        print(f"    -> across-seed mean Δ = {np.mean(deltas):+.3f}  min={np.min(deltas):+.3f}  max={np.max(deltas):+.3f}  "
              f"frac>0 = {np.mean([d>0 for d in deltas]):.2f}", flush=True)

    # ---- (3) PER-TEST-YEAR within fy>=2020 ----
    print("\n=== (3) PER-TEST-YEAR (train fy<2020; test cohort = each fy>=2020) ($50M) ===", flush=True)
    tr = (df["fy"] < 2020).to_numpy()
    from sklearn.metrics import roc_auc_score
    clf_b = _clf(); clf_b.fit(df.loc[tr, CTRL].to_numpy(), y[tr])
    clf_f = _clf(); clf_f.fit(df.loc[tr, CTRL + RICH].to_numpy(), y[tr])
    pb_all = clf_b.predict_proba(df[CTRL].to_numpy())[:, 1]
    pf_all = clf_f.predict_proba(df[CTRL + RICH].to_numpy())[:, 1]
    for yr in sorted(df.loc[df["fy"] >= 2020, "fy"].unique()):
        m = (df["fy"] == yr).to_numpy()
        yy = y[m]
        if yy.sum() < 3 or (yy == 0).sum() < 3:
            print(f"  fy={yr}: n={m.sum()} pos={yy.sum()} (too few for AUC)", flush=True)
            continue
        ab = roc_auc_score(yy, pb_all[m]); af = roc_auc_score(yy, pf_all[m])
        print(f"  fy={yr}: n={m.sum()} pos={int(yy.sum())}  controls-AUC={ab:.3f}  +rich-AUC={af:.3f}  Δ={af-ab:+.3f}", flush=True)
    # pooled test AUC using the SAME single models (sanity vs baseline refit)
    te = (df["fy"] >= 2020).to_numpy()
    print(f"  [pooled test, same models] controls={roc_auc_score(y[te], pb_all[te]):.3f}  "
          f"+rich={roc_auc_score(y[te], pf_all[te]):.3f}  Δ={roc_auc_score(y[te], pf_all[te])-roc_auc_score(y[te], pb_all[te]):+.3f}", flush=True)

    print("\n[verify] KILL if: lift only in-sector, only in one temporal split, or only in one founding year.", flush=True)


if __name__ == "__main__":
    main()
