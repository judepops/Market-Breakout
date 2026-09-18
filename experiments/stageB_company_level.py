"""Company-level commercialization test (NEW, adequately-powered question -- NOT a market-link rescue).

    python -m experiments.stageB_company_level

Question: does a company founded into an EMERGING research theme show better commercialization (GitHub-star
adoption / attention / funding) than one founded into a flat/mature theme? Unit = company (thousands).

Honest power: the signal must vary at the COMPANY level or "thousands of companies" is pseudo-replication that
collapses to ~#themes. So the signal is the theme's emergence state AS OF EACH COMPANY'S FOUNDING month
(escore over the 24mo ending at founding) -- varies by theme x timing (early vs late entry into a research
wave). Leakage-safe by construction: signal as-of founding, outcomes accrue after. Confounds handled: control
for founding-year (age), and CLUSTER-BOOTSTRAP by theme (so the theme-clustering can't inflate significance).
A null here is informative because it is powered.
"""
from __future__ import annotations
import pathlib

import numpy as np
import pandas as pd

SB = pathlib.Path("artifacts/stageB")
TAU = 0.60          # looser floor: we want most companies assigned to their nearest research theme
THEME_FILE = "themes.parquet"       # 364 detector themes = fine bridging resolution + trajectories


def _months(m):
    return int(str(m)[:4]) * 12 + (int(str(m)[5:7]) - 1)


def _embed(texts):
    from marketbrain.labels.gemini_grounding import make_client
    client = make_client()
    out = []
    for i in range(0, len(texts), 100):
        r = client.models.embed_content(model="gemini-embedding-001", contents=[str(t)[:2000] for t in texts[i:i + 100]])
        out.extend(np.asarray(e.values, dtype=np.float32) for e in r.embeddings)
    X = np.vstack(out)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def _rankresid(a, ctrl):
    """rank(a) residualised on rank(ctrl) (linear) -- for partial-Spearman controlling for age."""
    ra = pd.Series(a).rank().to_numpy()
    rc = pd.Series(ctrl).rank().to_numpy()
    rc1 = np.column_stack([np.ones_like(rc), rc])
    beta = np.linalg.lstsq(rc1, ra, rcond=None)[0]
    return ra - rc1 @ beta


def _partial_spearman(x, y, ctrl):
    rx, ry = _rankresid(x, ctrl), _rankresid(y, ctrl)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def _cluster_boot(df, sigcol, outcol, ctrlcol, theme_col, B=2000, seed=0):
    rng = np.random.RandomState(seed)
    themes = df[theme_col].unique()
    obs = _partial_spearman(df[sigcol].to_numpy(float), df[outcol].to_numpy(float), df[ctrlcol].to_numpy(float))
    rs = []
    by = {t: df[df[theme_col] == t] for t in themes}
    for _ in range(B):
        pick = rng.choice(themes, len(themes), replace=True)
        s = pd.concat([by[t] for t in pick], ignore_index=True)
        r = _partial_spearman(s[sigcol].to_numpy(float), s[outcol].to_numpy(float), s[ctrlcol].to_numpy(float))
        if not np.isnan(r):
            rs.append(r)
    lo, hi = (np.percentile(rs, 2.5), np.percentile(rs, 97.5)) if rs else (np.nan, np.nan)
    p = 2 * min((np.array(rs) <= 0).mean(), (np.array(rs) >= 0).mean()) if rs else np.nan
    return obs, lo, hi, p


def main():
    from marketbrain.corpus.load_full import load_corpus
    from marketbrain.eval.corpus_inputs import corpus_term_events
    from marketbrain.features.emergence import EmergenceConfig, escore, month_range_ending

    th = pd.read_parquet(SB / THEME_FILE)
    comp = pd.read_parquet("data/raw/labels/company_descriptions.parquet")
    comp = comp[comp["bridge_text"].str.len() > 0].reset_index(drop=True)
    print(f"[company] {len(th)} themes x {len(comp)} companies; company-timed emergence signal", flush=True)

    # theme monthly trajectories (corpus reload; te for the theme phrases)
    corpus = load_corpus("cs.LG,cs.CL,cs.AI,cs.RO,cs.SE,cs.DC,stat.ML".split(","))
    grid = sorted(corpus["month"].unique())
    te = corpus_term_events(corpus, min_doc_freq=40, ngram_max=3)
    te = te[te["term"].isin(set(th["theme_id"]))]
    traj = te.groupby(["term", "month"])["record_id"].nunique().unstack(fill_value=0).reindex(columns=grid, fill_value=0)
    ecfg = EmergenceConfig()

    # bridge companies -> nearest theme (cached company embeddings)
    tvec = _embed(list(th["theme_text"]))
    cvec = np.load(SB / "company_emb.npy"); ids = list(np.load(SB / "company_ids.npy", allow_pickle=True))
    assert ids == list(comp["permalink"]), "company embedding cache out of sync"
    sims = cvec @ tvec.T
    best = sims.argmax(1)
    comp = comp.assign(theme_id=th["theme_id"].to_numpy()[best], sim=sims[np.arange(len(comp)), best])
    comp = comp[comp["sim"] >= TAU]

    # founding month + company-timed theme-emergence signal (escore over 24mo ending at founding)
    meta = pd.read_parquet("data/raw/labels/company_meta.parquet")[["permalink", "founded_on"]]
    lvl = pd.read_parquet("data/raw/labels/company_level.parquet")[["permalink", "first_round_date", "is_outlier"]]
    df = comp.merge(meta, on="permalink", how="left").merge(lvl, on="permalink", how="left")
    df["found"] = df["founded_on"].fillna(df["first_round_date"])
    df = df.dropna(subset=["found"])
    df["found_m"] = df["found"].astype(str).str[:7].map(_months)
    gmin, gmax = _months(grid[0]), _months(grid[-1])
    df = df[(df["found_m"] >= gmin + 12) & (df["found_m"] <= gmax - 12)]      # need window before + outcome after

    traj_arr = {t: traj.loc[t] if t in traj.index else pd.Series(0, index=grid) for t in df["theme_id"].unique()}
    def sig_at(theme, fm):
        fmonth = grid[min(max(fm - gmin, 0), len(grid) - 1)]
        w = month_range_ending(fmonth, ecfg.window_months)
        return escore(traj_arr[theme].reindex(w, fill_value=0).to_numpy().tolist())
    df["sig_founding"] = [sig_at(t, fm) for t, fm in zip(df["theme_id"], df["found_m"])]
    df["sig_theme"] = df["theme_id"].map(th.set_index("theme_id")["sig_score"])   # static theme durability (comparison)

    # outcomes (accrue AFTER founding -> leakage-safe by construction)
    adop = pd.read_parquet("data/raw/labels/company_adoption.parquet")
    peak_stars = adop.groupby("permalink")["stars"].max()
    attn = pd.read_parquet("data/raw/labels/company_attention.parquet").groupby("permalink")["hn"].sum()
    raised = pd.read_parquet("data/raw/labels/company_rounds.parquet").groupby("permalink")["raised_usd"].sum()
    df["log_stars"] = np.log1p(df["permalink"].map(peak_stars).fillna(0))
    df["log_hn"] = np.log1p(df["permalink"].map(attn).fillna(0))
    df["log_raised"] = np.log1p(df["permalink"].map(raised).fillna(0))
    df["is_outlier"] = df["is_outlier"].fillna(False).astype(float)
    df["age"] = gmax - df["found_m"]

    n_th = df["theme_id"].nunique()
    print(f"[company] n_companies={len(df)}  n_themes={n_th}  (median companies/theme={int(df.groupby('theme_id').size().median())})", flush=True)
    print(f"[company] outcomes: with stars={int((df['log_stars']>0).sum())}, is_outlier={int(df['is_outlier'].sum())}\n", flush=True)

    print("[company] partial-Spearman(signal, outcome | founding-age), CLUSTER-bootstrap CI by theme:")
    for sig in ["sig_founding", "sig_theme"]:
        for outc in ["log_stars", "is_outlier", "log_raised", "log_hn"]:
            sub = df[df[outc] > 0] if outc in ("log_stars", "log_hn") else df   # stars/hn only where present
            if len(sub) < 50:
                continue
            obs, lo, hi, p = _cluster_boot(sub, sig, outc, "age", "theme_id", B=1500)
            star = "  *" if (lo > 0 or hi < 0) else ""
            print(f"   {sig:12s} -> {outc:10s}  rho_partial={obs:+.3f}  cluster95[{lo:+.3f},{hi:+.3f}]  p={p:.3f}  (n={len(sub)}){star}", flush=True)
    df.to_parquet(SB / "company_level_test.parquet")
    print(f"\n[company] wrote company_level_test.parquet", flush=True)


if __name__ == "__main__":
    main()
