"""Does RICH research-emergence add to the company-footprint signal? The crude test used ONE escore per company
(powered null). This mirrors the footprint win: engineer ~9 emergence SHAPE features from each theme's arXiv
trajectory, as-of the company's first-round decision (leakage-safe), and ask whether they add on top of footprint.

    ~/miniconda3/envs/marketbrain/bin/python -m experiments.altdata_plus_emergence

Universe = companies with BOTH a footprint (GitHub/HN/Wiki) AND a bridged research theme (cached). Footprint block
= the 17 rich pre-round features (the established signal). Emergence block = escore, log-level, log-peak, active-frac,
persistence, slope, momentum, recent-level, acceleration — all over the 30mo arXiv window ending at first_round,
STRICTLY before outcome. Controls=[fy,mkt_code,obs_age], NO money. Outcome = total_raised>=$50M (label only).
Chronological split fy<2020/>=2020. Permutation null on the ADDED block. THE key test: does emergence beat the
null when added ON TOP of footprint (not just on top of controls)?
"""
from __future__ import annotations
import pathlib, warnings
warnings.filterwarnings("ignore")
SB = pathlib.Path("artifacts/stageB")
DATA_END = 2026 * 12 + 5


def _m(x):
    s = str(x); return int(s[:4]) * 12 + int(s[5:7]) - 1


def main():
    import numpy as np, pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from marketbrain.features.emergence import escore, _growth

    # ---- universe: bridged companies with footprint + dates ----
    c = pd.read_parquet(SB / "company_level_test.parquet")[["permalink", "theme_id", "sim", "founded_on", "first_round_date"]]
    mkt = pd.read_parquet("data/raw/labels/company_level.parquet")[["permalink", "market"]]
    c = c.merge(mkt, on="permalink", how="left")
    has = set(pd.read_parquet("data/raw/labels/company_adoption.parquet")["permalink"])
    fd = pd.to_datetime(c["founded_on"], errors="coerce"); frd = pd.to_datetime(c["first_round_date"], errors="coerce")
    c["fy"] = fd.dt.year
    c["found_m"] = (fd.dt.year * 12 + fd.dt.month - 1)
    c["fr_m"] = (frd.dt.year * 12 + frd.dt.month - 1)
    c = c[c["permalink"].isin(has)].dropna(subset=["fy", "found_m", "fr_m"]).copy()
    c["fy"] = c["fy"].astype(int); c["found_m"] = c["found_m"].astype(int); c["fr_m"] = c["fr_m"].astype(int)
    c["mkt_code"] = pd.Categorical(c["market"]).codes.astype(float)
    c["obs_age"] = (DATA_END - c["found_m"]).clip(lower=1)
    raised = pd.read_parquet("data/raw/labels/company_rounds.parquet").groupby("permalink")["raised_usd"].sum()
    c["y"] = (c["permalink"].map(raised).fillna(0.0) >= 50e6).astype(int)
    print(f"[emerge] universe n={len(c)}  themes={c['theme_id'].nunique()}  pos={int(c['y'].sum())}", flush=True)

    # ---- FOOTPRINT block (17 rich pre-round features over [founding, first_round)) ----
    panels = {tag: pd.read_parquet(f"data/raw/labels/{p}.parquet").assign(
                  mm=lambda d: d["month"].astype(str).str[:7].map(_m)).rename(columns={v: "val"})[["permalink", "mm", "val"]]
              for tag, p, v in [("gh", "company_adoption", "stars"), ("hn", "company_attention", "hn"), ("wiki", "company_wiki", "views")]}
    base = c.set_index("permalink")[["found_m", "fr_m"]]
    FP = []
    for tag, p in panels.items():
        q = p.join(base, on="permalink", how="inner")
        q = q[(q["mm"] >= q["found_m"]) & (q["mm"] < q["fr_m"])]
        q["win"] = (q["fr_m"] - q["found_m"]).clip(lower=1); q["late"] = (q["mm"] - q["found_m"] >= q["win"] / 2).astype(int)
        g = q.groupby("permalink")
        out = pd.DataFrame({f"{tag}_logmax": np.log1p(g["val"].max()), f"{tag}_logmean": np.log1p(g["val"].mean()),
                            f"{tag}_active": g["val"].apply(lambda s: (s > 0).mean()), f"{tag}_present": 1.0})
        late = q[q["late"] == 1].groupby("permalink")["val"].sum(); early = q[q["late"] == 0].groupby("permalink")["val"].sum()
        out[f"{tag}_momentum"] = (np.log1p(late) - np.log1p(early)).reindex(out.index).fillna(0.0)
        for col in out.columns:
            c[col] = c["permalink"].map(out[col]).fillna(0.0); FP.append(col)
    c["breadth"] = c[[x for x in FP if x.endswith("_present")]].sum(1)
    c["log_winlen"] = np.log1p((c["fr_m"] - c["found_m"]).clip(lower=1))
    FP += ["breadth", "log_winlen"]

    # ---- EMERGENCE block: rich shape features from theme trajectory, as-of first_round (leakage-safe) ----
    from marketbrain.corpus.load_full import load_corpus
    from marketbrain.eval.corpus_inputs import corpus_term_events
    th = pd.read_parquet(SB / "themes.parquet")
    corpus = load_corpus("cs.LG,cs.CL,cs.AI,cs.RO,cs.SE,cs.DC,stat.ML".split(","))
    grid = sorted(corpus["month"].unique()); gnum = np.array([_m(x) for x in grid])
    te = corpus_term_events(corpus, min_doc_freq=40, ngram_max=3)
    te = te[te["term"].isin(set(th["theme_id"]))]
    traj = te.groupby(["term", "month"])["record_id"].nunique().unstack(fill_value=0).reindex(columns=grid, fill_value=0)
    zero = np.zeros(len(grid))
    tarr = {t: (traj.loc[t].to_numpy() if t in traj.index else zero) for t in c["theme_id"].unique()}

    def emfeats(theme, fm):
        idx = int(np.searchsorted(gnum, fm, side="right")) - 1          # last grid month <= first_round
        idx = min(max(idx, 0), len(grid) - 1)
        a = tarr[theme][max(0, idx - 29): idx + 1].astype(float)         # 30mo window ending at as-of
        if a.size < 4:
            a = np.concatenate([np.zeros(4 - a.size), a])
        n = a.size; half = a[n // 2:]
        mean = a.mean() + 1e-9
        slope = float(np.polyfit(range(len(half)), half, 1)[0]) / mean if len(half) >= 2 else 0.0
        mom = float(np.log1p(a[n // 2:].sum()) - np.log1p(a[:n // 2].sum()))
        return [escore(a.tolist()), float(np.log1p(a.sum())), float(np.log1p(a.max())), float((a > 0).mean()),
                float((a > 0).sum()), slope, mom, float(np.log1p(a[-3:].mean())), float(_growth(a[-2:].sum(), a[-4:-2].sum()))]
    EM = ["em_escore", "em_loglevel", "em_logpeak", "em_active", "em_persist", "em_slope", "em_momentum", "em_recent", "em_accel"]
    M = np.array([emfeats(t, fm) for t, fm in zip(c["theme_id"], c["fr_m"])], dtype=float)
    for j, name in enumerate(EM):
        c[name] = M[:, j]

    # ---- models ----
    CTRL = ["fy", "mkt_code", "obs_age"]
    c = c.reset_index(drop=True)
    tr = (c["fy"] < 2020).to_numpy(); te_m = (c["fy"] >= 2020).to_numpy(); yv = c["y"].to_numpy()
    print(f"[emerge] train={int(tr.sum())} test={int(te_m.sum())} test_pos={int(yv[te_m].sum())}\n", flush=True)

    def auc(cols):
        clf = HistGradientBoostingClassifier(max_depth=3, max_iter=400, learning_rate=0.04, min_samples_leaf=15, random_state=0)
        clf.fit(c.loc[tr, cols].to_numpy(), yv[tr]); return roc_auc_score(yv[te_m], clf.predict_proba(c.loc[te_m, cols].to_numpy())[:, 1])

    def perm_add(fixed, added):
        """p-value for `added` block on top of `fixed` (shuffle only the added block)."""
        b = auc(fixed); f = auc(fixed + added)
        rng = np.random.RandomState(0); nd = []
        ft, fe = c.loc[tr, fixed].to_numpy(), c.loc[te_m, fixed].to_numpy()
        at, ae = c.loc[tr, added].to_numpy(), c.loc[te_m, added].to_numpy()
        for _ in range(150):
            clf = HistGradientBoostingClassifier(max_depth=3, max_iter=400, learning_rate=0.04, min_samples_leaf=15, random_state=0)
            clf.fit(np.column_stack([ft, at[rng.permutation(tr.sum())]]), yv[tr])
            nd.append(roc_auc_score(yv[te_m], clf.predict_proba(np.column_stack([fe, ae[rng.permutation(te_m.sum())]]))[:, 1]) - b)
        p = (np.sum(np.array(nd) >= (f - b)) + 1) / (len(nd) + 1)
        return b, f, p

    print("[emerge] does each block beat its null? (perm on the ADDED block)")
    b, f, p = perm_add(CTRL, FP);        print(f"   controls -> +footprint(17)          {b:.3f} -> {f:.3f}  Δ={f-b:+.3f}  p={p:.3f}  {'SIGNAL' if p<0.05 and f>b else 'null'}", flush=True)
    b, f, p = perm_add(CTRL, EM);        print(f"   controls -> +emergence(9)           {b:.3f} -> {f:.3f}  Δ={f-b:+.3f}  p={p:.3f}  {'SIGNAL' if p<0.05 and f>b else 'null'}", flush=True)
    b, f, p = perm_add(CTRL + FP, EM);   print(f"   footprint -> +emergence  <== KEY     {b:.3f} -> {f:.3f}  Δ={f-b:+.3f}  p={p:.3f}  {'ADDS' if p<0.05 and f>b else 'adds NOTHING'}", flush=True)
    b, f, p = perm_add(CTRL + EM, FP);   print(f"   emergence -> +footprint (symmetry)  {b:.3f} -> {f:.3f}  Δ={f-b:+.3f}  p={p:.3f}  {'SIGNAL' if p<0.05 and f>b else 'null'}", flush=True)
    print("\n[emerge] read: if 'footprint -> +emergence' adds NOTHING, research-emergence carries no company-money signal")
    print("[emerge]       even with rich feature engineering — the powered, rich confirmation of the detector-integration null.")


if __name__ == "__main__":
    main()
