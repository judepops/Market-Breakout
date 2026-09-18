"""ADVERSARIAL: COVERAGE-SELECTION CONFOUND attack on the +0.045 rich-feature lift.

Hypothesis: the sample is restricted to GitHub-covered companies, and the level/present/breadth
features may just encode "has a rich public footprint" (correlates with prominence/size). So the
+0.045 might be a footprint-coverage selection effect, not organic traction dynamics.

Two checks:
  (1) Add EXPLICIT coverage indicators as CONTROLS (not features): has_hn, has_wiki. Re-run
      controls vs controls+rich with the permutation null. Does Δ survive with coverage controlled?
  (2) Restrict to FULLY-COVERED subsample (present in GitHub AND HN AND Wikipedia). Within this
      subsample "having a footprint" is constant. Re-run the rich-feature lift + perm null. If the
      signal is merely "has footprint," it should die here.

KILL: if Δ ~0 or perm-p>0.05 in BOTH checks, coverage selection explains it.

    python -m experiments.verify_coverage
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
DATA_END = 2026 * 12 + 5
N_PERM = 150


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

    # coverage universes
    has_gh = set(pd.read_parquet("data/raw/labels/company_adoption.parquet")["permalink"])
    has_hn = set(pd.read_parquet("data/raw/labels/company_attention.parquet")["permalink"])
    has_wiki = set(pd.read_parquet("data/raw/labels/company_wiki.parquet")["permalink"])

    # EXPLICIT coverage indicators as controls
    df["has_hn"] = df["permalink"].isin(has_hn).astype(float)
    df["has_wiki"] = df["permalink"].isin(has_wiki).astype(float)

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

    # restrict to GitHub-covered universe (the baseline 'has')
    df = df[df["permalink"].isin(has_gh)].reset_index(drop=True)
    raised = pd.read_parquet("data/raw/labels/company_rounds.parquet").groupby("permalink")["raised_usd"].sum()
    df["total_raised"] = df["permalink"].map(raised).fillna(0.0)

    def run_block(dsub, CTRL, RICHcols, label, thresholds=(50e6,)):
        tr = (dsub["fy"] < 2020).to_numpy(); te = (dsub["fy"] >= 2020).to_numpy()
        print(f"\n=== {label} | n={len(dsub)} (train={tr.sum()}, test={te.sum()}) | CTRL={CTRL} ===")
        def fit_auc(cols, yv):
            clf = HistGradientBoostingClassifier(max_depth=3, max_iter=400, learning_rate=0.04, min_samples_leaf=15, random_state=0)
            clf.fit(dsub.loc[tr, cols].to_numpy(), yv[tr])
            return roc_auc_score(yv[te], clf.predict_proba(dsub.loc[te, cols].to_numpy())[:, 1])
        results = {}
        for X in thresholds:
            yv = (dsub["total_raised"] >= X).astype(int).to_numpy()
            if yv[te].sum() < 10:
                print(f"   >=${int(X/1e6)}M: too few test positives ({int(yv[te].sum())}), skipped")
                continue
            b = fit_auc(CTRL, yv); f = fit_auc(CTRL + RICHcols, yv)
            rng = np.random.RandomState(0); nd = []
            own_tr = dsub.loc[tr, RICHcols].to_numpy(); own_te = dsub.loc[te, RICHcols].to_numpy()
            ctr_tr = dsub.loc[tr, CTRL].to_numpy(); ctr_te = dsub.loc[te, CTRL].to_numpy()
            for _ in range(N_PERM):
                pt = rng.permutation(tr.sum()); pe = rng.permutation(te.sum())
                clf = HistGradientBoostingClassifier(max_depth=3, max_iter=400, learning_rate=0.04, min_samples_leaf=15, random_state=0)
                clf.fit(np.column_stack([ctr_tr, own_tr[pt]]), yv[tr])
                nd.append(roc_auc_score(yv[te], clf.predict_proba(np.column_stack([ctr_te, own_te[pe]]))[:, 1]) - b)
            p = (np.sum(np.array(nd) >= (f - b)) + 1) / (len(nd) + 1)
            star = "  <-- SIGNAL SURVIVES" if (p < 0.05 and f > b) else "  (null / dead)"
            print(f"   >=${int(X/1e6)}M ({int(yv[te].sum())} test pos): controls={b:.3f}  +rich={f:.3f}  Δ={f-b:+.3f}  perm-p={p:.3f}{star}", flush=True)
            results[X] = (b, f, f - b, p)
        return results

    print("#" * 90)
    print("# CHECK 0 (baseline replication in this script, GitHub universe, CTRL=[fy,mkt,obs_age])")
    print("#" * 90)
    run_block(df, ["fy", "mkt_code", "obs_age"], RICH, "BASELINE (no coverage controls)", thresholds=(50e6, 500e6))

    print("\n" + "#" * 90)
    print("# CHECK 1 — add EXPLICIT coverage indicators (has_hn, has_wiki) as CONTROLS")
    print("#" * 90)
    run_block(df, ["fy", "mkt_code", "obs_age", "has_hn", "has_wiki"], RICH,
              "CHECK1 (+has_hn,+has_wiki as controls)", thresholds=(50e6, 500e6))

    print("\n" + "#" * 90)
    print("# CHECK 2 — restrict to FULLY-COVERED subsample (GitHub AND HN AND Wikipedia)")
    print("#" * 90)
    full = has_gh & has_hn & has_wiki
    print(f"[cover] |GitHub|={len(has_gh)} |HN|={len(has_hn)} |Wiki|={len(has_wiki)} |all-three|={len(full)}", flush=True)
    dfull = df[df["permalink"].isin(full)].reset_index(drop=True)
    # In this subsample every company has all 3 footprints -> _present cols are constant (1),
    # breadth is constant (3). Drop the degenerate coverage-encoding features so the test is
    # purely about SHAPE/level dynamics, not "has footprint".
    RICH_full = [c for c in RICH if not c.endswith("_present") and c != "breadth"]
    print(f"[cover] fully-covered rich feats (present/breadth dropped as constant): {RICH_full}", flush=True)
    # (2a) full rich block (present/breadth are constant here so harmless but keep for parity)
    run_block(dfull, ["fy", "mkt_code", "obs_age"], RICH, "CHECK2a full-covered, ALL rich feats", thresholds=(50e6, 500e6))
    # (2b) shape/level-only (coverage-encoding features removed)
    run_block(dfull, ["fy", "mkt_code", "obs_age"], RICH_full, "CHECK2b full-covered, shape/level only", thresholds=(50e6, 500e6))

    print("\n[verdict] KILL if Δ~0 or perm-p>0.05 in BOTH check1 and check2. Otherwise coverage does NOT explain the lift.")


if __name__ == "__main__":
    main()
