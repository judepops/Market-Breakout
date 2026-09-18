"""Alt-data traction premium — ADVERSARIAL HARDENING. Make "organic traction predicts >=$500M success" bulletproof.

    GOOGLE_APPLICATION_CREDENTIALS unused (no bridge here — detector was null). python -m experiments.altdata_hardened

Threats addressed, in order:
  1. REVERSE CAUSALITY (make-or-break): funding -> attention. Measure traction STRICTLY BEFORE first funding
     (window [founding, first_round)). If the premium survives pre-round, it's organic; if it vanishes, it was
     funding-driven. Reported side-by-side with the funding-contaminated [founding,+24mo] window.
  2. CENSORING: recent companies had less time to reach $500M. Control for observation age (months founding->data end).
  3. NULL: permutation test — shuffle the traction features, does + delta survive?
  4. ROBUSTNESS: multiple chronological split cutoffs + logistic AND gradient-boosted.
  5. SELECTION: does merely HAVING GitHub coverage predict the outcome? report it.
  6. ATTRIBUTION: drop-one-out over stars/hn/wiki — which signal carries it (or is it all three)?
No money features anywhere (money->money is circular). Outcome = is_outlier (>=$500M).
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
DATA_END = 2026 * 12 + 5   # ~2026-06


def _m(x):
    s = str(x); return int(s[:4]) * 12 + int(s[5:7]) - 1


def main():
    import numpy as np, pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    lvl = pd.read_parquet("data/raw/labels/company_level.parquet")
    meta = pd.read_parquet("data/raw/labels/company_meta.parquet")[["permalink", "founded_on"]]
    df = lvl.merge(meta, on="permalink", how="left")
    fd = pd.to_datetime(df["founded_on"], errors="coerce")
    df["fy"] = fd.dt.year
    df = df.dropna(subset=["fy"]).copy(); df["fy"] = df["fy"].astype(int)
    df["found_m"] = (fd.dt.year * 12 + fd.dt.month - 1).astype("Int64")
    frd = pd.to_datetime(df["first_round_date"], errors="coerce")
    df["fr_m"] = (frd.dt.year * 12 + frd.dt.month - 1).astype("Int64")
    df = df.dropna(subset=["found_m", "fr_m"]).copy()
    df["found_m"] = df["found_m"].astype(int); df["fr_m"] = df["fr_m"].astype(int)
    df["mkt_code"] = pd.Categorical(df["market"]).codes.astype(float)
    df["obs_age"] = (DATA_END - df["found_m"]).clip(lower=1)      # censoring control

    # signals over two windows: PRE-ROUND (organic) and EARLY-24 (funding-contaminated, for contrast)
    def wsig(panel, val, agg, lo, hi):
        p = pd.read_parquet(f"data/raw/labels/{panel}.parquet")
        p["mm"] = p["month"].astype(str).str[:7].map(_m)
        p = p.merge(df[["permalink", "found_m", "fr_m"]], on="permalink", how="inner")
        w = p[(p["mm"] >= p[lo]) & (p["mm"] < p[hi] if hi == "fr_m" else p["mm"] <= p[lo] + hi)]
        return np.log1p(w.groupby("permalink")[val].agg(agg))
    for tag, lo, hi in [("pre", "found_m", "fr_m"), ("e24", "found_m", 24)]:
        df[f"{tag}_stars"] = df["permalink"].map(wsig("company_adoption", "stars", "max", lo, hi)).fillna(0.0)
        df[f"{tag}_hn"] = df["permalink"].map(wsig("company_attention", "hn", "sum", lo, hi)).fillna(0.0)
        df[f"{tag}_wiki"] = df["permalink"].map(wsig("company_wiki", "views", "max", lo, hi)).fillna(0.0)

    has_stars = set(pd.read_parquet("data/raw/labels/company_adoption.parquet")["permalink"])
    # ---- 5. SELECTION: does merely having GitHub coverage predict the outcome? ----
    df["has_gh"] = df["permalink"].isin(has_stars).astype(int)
    print(f"[hard] SELECTION: is_outlier rate — with GitHub {df[df.has_gh==1]['is_outlier'].mean():.3f} "
          f"vs without {df[df.has_gh==0]['is_outlier'].mean():.3f}  (coverage {df.has_gh.mean():.1%}) "
          f"-> claim is for the developer-facing population", flush=True)
    D = df[df["has_gh"] == 1].reset_index(drop=True)
    y = D["is_outlier"].astype(int).to_numpy()
    CTRL = ["fy", "mkt_code", "obs_age"]

    def auc_split(cols, cutoff, model="gbm", Dloc=None, yloc=None):
        Dl = D if Dloc is None else Dloc; yl = y if yloc is None else yloc
        tr = (Dl["fy"] < cutoff).to_numpy(); te = (Dl["fy"] >= cutoff).to_numpy()
        if yl[te].sum() < 10 or yl[tr].sum() < 10:
            return np.nan, tr, te
        Xtr, Xte = Dl.loc[tr, cols].to_numpy(), Dl.loc[te, cols].to_numpy()
        if model == "gbm":
            clf = HistGradientBoostingClassifier(max_depth=3, max_iter=300, learning_rate=0.05, min_samples_leaf=20, random_state=0)
            clf.fit(Xtr, yl[tr]); p = clf.predict_proba(Xte)[:, 1]
        else:
            sc = StandardScaler().fit(Xtr)
            clf = LogisticRegression(max_iter=1000, C=1.0).fit(sc.transform(Xtr), yl[tr])
            p = clf.predict_proba(sc.transform(Xte))[:, 1]
        return roc_auc_score(yl[te], p), tr, te

    # ---- 1. REVERSE CAUSALITY: pre-round vs early-24 premium ----
    print("\n[hard] 1. REVERSE-CAUSALITY (window matters) — premium over controls (year+sector+age):")
    for tag, label in [("pre", "PRE-ROUND (organic — the honest test)"), ("e24", "early-24mo (funding-contaminated)")]:
        own = [f"{tag}_stars", f"{tag}_hn", f"{tag}_wiki"]
        base, _, _ = auc_split(CTRL, 2020); full, _, _ = auc_split(CTRL + own, 2020)
        print(f"   {label:42s} controls={base:.3f}  +traction={full:.3f}  Δ={full-base:+.3f}", flush=True)

    # headline = pre-round
    OWN = ["pre_stars", "pre_hn", "pre_wiki"]
    # ---- 3. PERMUTATION NULL on the pre-round traction ----
    base20, tr, te = auc_split(CTRL, 2020); full20, _, _ = auc_split(CTRL + OWN, 2020)
    obs_delta = full20 - base20
    rng = np.random.RandomState(0); nd = []
    for _ in range(300):
        Dp = D.copy(); perm = rng.permutation(len(Dp))
        for c in OWN: Dp[c] = Dp[c].to_numpy()[perm]
        f, _, _ = auc_split(CTRL + OWN, 2020, Dloc=Dp); nd.append(f - base20)
    nd = np.array(nd); pval = (np.sum(nd >= obs_delta) + 1) / (len(nd) + 1)
    print(f"\n[hard] 3. PERMUTATION NULL (pre-round): observed Δ={obs_delta:+.3f}  null mean={nd.mean():+.3f}  p={pval:.3f}"
          f"{'  <-- real' if pval < 0.05 else ''}", flush=True)

    # ---- 4. ROBUSTNESS: multiple cutoffs + logistic vs gbm ----
    print("\n[hard] 4. ROBUSTNESS (pre-round premium across splits & models):")
    for cut in [2018, 2019, 2020, 2021]:
        for mdl in ["gbm", "logit"]:
            b, _, tem = auc_split(CTRL, cut, mdl); f, _, _ = auc_split(CTRL + OWN, cut, mdl)
            npos = int(y[tem].sum()) if not np.isnan(b) else 0
            print(f"   cutoff {cut} · {mdl:5s}  controls={b:.3f}  +traction={f:.3f}  Δ={f-b:+.3f}  (test pos={npos})", flush=True)

    # ---- 6. ATTRIBUTION: drop-one-out over the three signals ----
    print("\n[hard] 6. ATTRIBUTION (pre-round, cutoff 2020) — which signal carries it:")
    for drop in [None, "pre_stars", "pre_hn", "pre_wiki"]:
        cols = CTRL + [c for c in OWN if c != drop]
        f, _, _ = auc_split(cols, 2020)
        print(f"   {'full (all 3)' if drop is None else 'drop '+drop:16s} +traction={f:.3f}", flush=True)


if __name__ == "__main__":
    main()
