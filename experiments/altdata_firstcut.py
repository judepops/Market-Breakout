"""Alt-data track — first-cut ablation (leakage-safe): does a company's OWN early traction predict its outcome,
and does the Stage-A detector's market-quality INTERACTION add on top? Pre-reg: Plans/2026-07-02_alt-data-valuation-track.md.

    GOOGLE_APPLICATION_CREDENTIALS=... python -m experiments.altdata_firstcut

Panel: one row per company. Signal = own alt-data momentum in [founding, founding+24mo] (backward-only).
Outcome = is_outlier (reached >=$500M, generally years later -> leakage-safe by construction). Comps = first-round
amount, founding year, market. Detector = bridged theme's durable-breakout score (market-quality) + own x quality
interaction. Chronological split (train founded <2020, test >=2020). Nested ablation with company-bootstrap CIs on
the increments. NOTE: reverse-causality guard = signal strictly in the early window; refine with pre-round windows later.
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")


def _m(x):
    s = str(x)
    return int(s[:4]) * 12 + int(s[5:7]) - 1


def main():
    import numpy as np, pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from experiments.stageB_bridge import _embed

    lvl = pd.read_parquet("data/raw/labels/company_level.parquet")
    meta = pd.read_parquet("data/raw/labels/company_meta.parquet")[["permalink", "founded_on"]]
    df = lvl.merge(meta, on="permalink", how="left")
    fd = pd.to_datetime(df["founded_on"], errors="coerce")
    df["fy"] = fd.dt.year
    df = df.dropna(subset=["fy"]).copy()
    df["fy"] = df["fy"].astype(int)
    df["found_m"] = (fd.dt.year * 12 + fd.dt.month - 1).astype("Int64")
    df = df.dropna(subset=["found_m"]).copy(); df["found_m"] = df["found_m"].astype(int)
    df["log_first"] = pd.to_numeric(df.get("log_first"), errors="coerce").fillna(0.0)

    # early-window own signals [founding, founding+24mo] -- strictly backward-looking within the company's early life
    def early(panel, val, agg):
        p = pd.read_parquet(f"data/raw/labels/{panel}.parquet")
        p["mm"] = p["month"].astype(str).str[:7].map(_m)
        p = p.merge(df[["permalink", "found_m"]], on="permalink", how="inner")
        w = p[(p["mm"] >= p["found_m"]) & (p["mm"] <= p["found_m"] + 24)]
        return np.log1p(w.groupby("permalink")[val].agg(agg))
    df["e_stars"] = df["permalink"].map(early("company_adoption", "stars", "max")).fillna(0.0)
    df["e_hn"] = df["permalink"].map(early("company_attention", "hn", "sum")).fillna(0.0)
    df["e_wiki"] = df["permalink"].map(early("company_wiki", "views", "max")).fillna(0.0)
    has_stars = set(pd.read_parquet("data/raw/labels/company_adoption.parquet")["permalink"])
    df = df[df["permalink"].isin(has_stars)].reset_index(drop=True)

    # detector market-quality via bridge (company -> best theme -> durable-breakout score)
    comp = pd.read_parquet("data/raw/labels/company_descriptions.parquet")
    comp = comp[comp["bridge_text"].str.len() > 0].reset_index(drop=True)
    cvec = np.load("artifacts/stageB/company_emb.npy"); ids = list(np.load("artifacts/stageB/company_ids.npy", allow_pickle=True))
    assert ids == list(comp["permalink"])
    th = pd.read_parquet("artifacts/stageB/themes.parquet")
    tvec = _embed(list(th["theme_text"]), label="themes")
    best = (cvec @ tvec.T).argmax(1)
    c2theme = dict(zip(ids, th["theme_id"].to_numpy()[best]))
    c2sig = dict(zip(ids, th["sig_score"].to_numpy()[best]))
    df["theme"] = df["permalink"].map(c2theme).fillna("none")
    df["sig"] = df["permalink"].map(c2sig).fillna(float(th["sig_score"].median()))
    df["mkt_code"] = pd.Categorical(df["market"]).codes.astype(float)
    for s in ["e_stars", "e_hn", "e_wiki"]:
        df[s + "_x_sig"] = df[s] * df["sig"]

    y = df["is_outlier"].astype(int).to_numpy()
    tr = (df["fy"] < 2020).to_numpy(); te = (df["fy"] >= 2020).to_numpy()
    print(f"[altdata] n={len(df)}  train={tr.sum()} ({y[tr].sum()} outliers)  test={te.sum()} ({y[te].sum()} outliers)", flush=True)

    # NON-MONEY controls only: founding year + market (sector). NO funding amount (money->money is circular).
    FS = {
        "controls (year+sector, NO money)": ["fy", "mkt_code"],
        "+ own traction signals": ["fy", "mkt_code", "e_stars", "e_hn", "e_wiki"],
        "+ detector (mkt-quality)": ["fy", "mkt_code", "e_stars", "e_hn", "e_wiki", "sig"],
        "+ interaction (own x quality)": ["fy", "mkt_code", "e_stars", "e_hn", "e_wiki", "sig",
                                          "e_stars_x_sig", "e_hn_x_sig", "e_wiki_x_sig"],
    }
    preds = {}
    print("\n[altdata] nested ablation (chronological split, test AUC on is_outlier):")
    for name, cols in FS.items():
        clf = HistGradientBoostingClassifier(max_depth=3, max_iter=300, learning_rate=0.05,
                                             min_samples_leaf=20, random_state=0).fit(df.loc[tr, cols].to_numpy(), y[tr])
        p = clf.predict_proba(df.loc[te, cols].to_numpy())[:, 1]
        preds[name] = p
        print(f"   {name:32s} AUC = {roc_auc_score(y[te], p):.3f}", flush=True)

    # bootstrap CIs on the two key increments (resample test companies)
    yte = y[te]; rng = np.random.RandomState(0)
    def boot_delta(a, b, B=2000):
        d = []
        n = len(yte)
        for _ in range(B):
            i = rng.randint(0, n, n)
            if yte[i].sum() < 5:
                continue
            d.append(roc_auc_score(yte[i], a[i]) - roc_auc_score(yte[i], b[i]))
        return np.mean(d), np.percentile(d, 2.5), np.percentile(d, 97.5)
    print("\n[altdata] key increments (company-bootstrap 95% CI):")
    for lbl, a, b in [("traction premium (+own vs controls)", "+ own traction signals", "controls (year+sector, NO money)"),
                      ("detector market-quality (+sig vs +own)", "+ detector (mkt-quality)", "+ own traction signals"),
                      ("INTERACTION (own x quality vs +sig)", "+ interaction (own x quality)", "+ detector (mkt-quality)")]:
        m, lo, hi = boot_delta(preds[a], preds[b])
        star = " *" if (lo > 0 or hi < 0) else ""
        print(f"   {lbl:42s} ΔAUC = {m:+.3f}  95% CI [{lo:+.3f},{hi:+.3f}]{star}", flush=True)


if __name__ == "__main__":
    main()
