"""Alt-data — RICH FEATURES on the ORGANIC (pre-round) window. Decisive test of the "crude features" objection:
if ~17 engineered shape features of pre-funding traction still don't predict outcome, the null is causal
(reverse causality), not representational. Mirrors Stage A's shape-feature approach — applied to the clean window.

    python -m experiments.altdata_richfeatures

Features per signal (GitHub stars / HN / Wikipedia), over [founding, first_round): log-max, log-mean, active
fraction, early-vs-late momentum (shape/acceleration proxy), peak level. Plus cross-signal breadth + window
length. Controls = year + sector + observation age (NO money). Outcome = reached >=$X (raise-based) at two
thresholds for power. Chronological split + permutation null. All computed strictly BEFORE first funding.
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
DATA_END = 2026 * 12 + 5


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
    df["win_len"] = (df["fr_m"] - df["found_m"]).clip(lower=1)
    df["mkt_code"] = pd.Categorical(df["market"]).codes.astype(float)
    df["obs_age"] = (DATA_END - df["found_m"]).clip(lower=1)
    base = df.set_index("permalink")[["found_m", "fr_m", "win_len"]]

    # rich shape features per signal over the PRE-ROUND window
    feat_cols = []
    def rich(panel, val, tag):
        p = pd.read_parquet(f"data/raw/labels/{panel}.parquet")
        p["mm"] = p["month"].astype(str).str[:7].map(_m)
        p = p.join(base, on="permalink", how="inner")
        p = p[(p["mm"] >= p["found_m"]) & (p["mm"] < p["fr_m"])]         # STRICTLY pre-round
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
    print(f"[rich] {len(df)} companies, {len(RICH)} engineered pre-round features: {RICH}", flush=True)

    CTRL = ["fy", "mkt_code", "obs_age"]
    tr = (df["fy"] < 2020).to_numpy(); te = (df["fy"] >= 2020).to_numpy()
    def fit_auc(cols, yv):
        clf = HistGradientBoostingClassifier(max_depth=3, max_iter=400, learning_rate=0.04, min_samples_leaf=15, random_state=0)
        clf.fit(df.loc[tr, cols].to_numpy(), yv[tr]); return roc_auc_score(yv[te], clf.predict_proba(df.loc[te, cols].to_numpy())[:, 1])

    print("\n[rich] RICH pre-round features vs controls (chronological split):")
    for X in [50e6, 500e6]:
        yv = (df["total_raised"] >= X).astype(int).to_numpy()
        if yv[te].sum() < 10:
            continue
        b = fit_auc(CTRL, yv); f = fit_auc(CTRL + RICH, yv)
        rng = np.random.RandomState(0); nd = []
        own_tr = df.loc[tr, RICH].to_numpy(); own_te = df.loc[te, RICH].to_numpy()
        ctr_tr = df.loc[tr, CTRL].to_numpy(); ctr_te = df.loc[te, CTRL].to_numpy()
        for _ in range(150):
            pt = rng.permutation(tr.sum()); pe = rng.permutation(te.sum())
            clf = HistGradientBoostingClassifier(max_depth=3, max_iter=400, learning_rate=0.04, min_samples_leaf=15, random_state=0)
            clf.fit(np.column_stack([ctr_tr, own_tr[pt]]), yv[tr])
            nd.append(roc_auc_score(yv[te], clf.predict_proba(np.column_stack([ctr_te, own_te[pe]]))[:, 1]) - b)
        p = (np.sum(np.array(nd) >= (f - b)) + 1) / (len(nd) + 1)
        star = "  <-- SIGNAL" if (p < 0.05 and f > b) else "  (null)"
        print(f"   >=${int(X/1e6)}M ({int(yv[te].sum())} test pos): controls={b:.3f}  +{len(RICH)} rich feats={f:.3f}  Δ={f-b:+.3f}  perm-p={p:.3f}{star}", flush=True)
    print("\n[rich] if null: the crude-feature objection is closed — the pre-round null is CAUSAL, not representational.")


if __name__ == "__main__":
    main()
