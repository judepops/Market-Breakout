"""Scrutinize the surprising RICH pre-round signal: is it organic, or unrecorded-pre-seed contamination?

    python -m experiments.altdata_scrutiny

Guard 1 (decisive): buffer the window to end K months BEFORE the recorded first round (K=0,6,12). Unrecorded
pre-seed lands months before a priced round, not a year+ — so if a window ending 12mo early still carries
signal, it's very unlikely to be funding's shadow. Guard 2: attribution — drop each signal group + the
maturity proxies (breadth, window-length) to see whether genuine SHAPE (momentum) or a size/age proxy carries it.
No money features. Outcome = total raised >= $50M (power). Chronological split + permutation null.
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
    df["mkt_code"] = pd.Categorical(df["market"]).codes.astype(float)
    df["obs_age"] = (DATA_END - df["found_m"]).clip(lower=1)
    has = set(pd.read_parquet("data/raw/labels/company_adoption.parquet")["permalink"])
    raised = pd.read_parquet("data/raw/labels/company_rounds.parquet").groupby("permalink")["raised_usd"].sum()

    panels = {tag: pd.read_parquet(f"data/raw/labels/{p}.parquet").assign(
                  mm=lambda d: d["month"].astype(str).str[:7].map(_m)).rename(columns={v: "val"})[["permalink", "mm", "val"]]
              for tag, p, v in [("gh", "company_adoption", "stars"), ("hn", "company_attention", "hn"), ("wiki", "company_wiki", "views")]}

    def build(buffer_mo):
        d = df.copy(); d["end_m"] = d["fr_m"] - buffer_mo
        base = d.set_index("permalink")[["found_m", "end_m"]]
        cols = []
        for tag, p in panels.items():
            q = p.join(base, on="permalink", how="inner")
            q = q[(q["mm"] >= q["found_m"]) & (q["mm"] < q["end_m"])]
            q["win"] = (q["end_m"] - q["found_m"]).clip(lower=1); q["late"] = (q["mm"] - q["found_m"] >= q["win"] / 2).astype(int)
            g = q.groupby("permalink")
            out = pd.DataFrame({f"{tag}_logmax": np.log1p(g["val"].max()), f"{tag}_logmean": np.log1p(g["val"].mean()),
                                f"{tag}_active": g["val"].apply(lambda s: (s > 0).mean()), f"{tag}_present": 1.0})
            late = q[q["late"] == 1].groupby("permalink")["val"].sum(); early = q[q["late"] == 0].groupby("permalink")["val"].sum()
            out[f"{tag}_momentum"] = (np.log1p(late) - np.log1p(early)).reindex(out.index).fillna(0.0)
            for c in out.columns:
                d[c] = d["permalink"].map(out[c]).fillna(0.0); cols.append(c)
        d["breadth"] = d[[c for c in cols if c.endswith("_present")]].sum(1)
        d["log_winlen"] = np.log1p((d["end_m"] - d["found_m"]).clip(lower=1))
        d = d[d["permalink"].isin(has)].reset_index(drop=True)
        d["y"] = (d["permalink"].map(raised).fillna(0.0) >= 50e6).astype(int)
        return d, cols + ["breadth", "log_winlen"]

    CTRL = ["fy", "mkt_code", "obs_age"]
    def test(d, feats, perm=True):
        tr = (d["fy"] < 2020).to_numpy(); te = (d["fy"] >= 2020).to_numpy(); yv = d["y"].to_numpy()
        def fit(cols):
            clf = HistGradientBoostingClassifier(max_depth=3, max_iter=400, learning_rate=0.04, min_samples_leaf=15, random_state=0)
            clf.fit(d.loc[tr, cols].to_numpy(), yv[tr]); return roc_auc_score(yv[te], clf.predict_proba(d.loc[te, cols].to_numpy())[:, 1])
        b = fit(CTRL); f = fit(CTRL + feats)
        if not perm:
            return b, f, None
        rng = np.random.RandomState(0); nd = []
        ot, oe = d.loc[tr, feats].to_numpy(), d.loc[te, feats].to_numpy(); ct, ce = d.loc[tr, CTRL].to_numpy(), d.loc[te, CTRL].to_numpy()
        for _ in range(150):
            clf = HistGradientBoostingClassifier(max_depth=3, max_iter=400, learning_rate=0.04, min_samples_leaf=15, random_state=0)
            clf.fit(np.column_stack([ct, ot[rng.permutation(tr.sum())]]), yv[tr])
            nd.append(roc_auc_score(yv[te], clf.predict_proba(np.column_stack([ce, oe[rng.permutation(te.sum())]]))[:, 1]) - b)
        return b, f, (np.sum(np.array(nd) >= (f - b)) + 1) / (len(nd) + 1)

    print("[scrutiny] GUARD 1 — buffer the window to end K months BEFORE the recorded first round:")
    for K in [0, 6, 12]:
        d, feats = build(K)
        b, f, p = test(d, feats)
        verdict = "SURVIVES" if (p < 0.05 and f > b) else "gone"
        print(f"   window ends {K:2d}mo pre-round: controls={b:.3f}  +rich={f:.3f}  Δ={f-b:+.3f}  perm-p={p:.3f}  -> {verdict}", flush=True)

    print("\n[scrutiny] GUARD 2 — attribution (window K=6): which features carry it (drop-group)?")
    d, feats = build(6)
    groups = {"none (full)": [], "drop GitHub": [c for c in feats if c.startswith("gh_")],
              "drop HN": [c for c in feats if c.startswith("hn_")], "drop Wikipedia": [c for c in feats if c.startswith("wiki_")],
              "drop maturity (breadth,winlen)": ["breadth", "log_winlen"],
              "ONLY momentum (shape)": [c for c in feats if "momentum" not in c]}
    for name, drop in groups.items():
        keep = [c for c in feats if c not in drop]
        b, f, _ = test(d, keep, perm=False)
        print(f"   {name:32s} +feats={f:.3f}  Δ={f-b:+.3f}", flush=True)
    print("\n[scrutiny] read: if Δ collapses to ~0 at K=12 -> it was pre-seed contamination (still reverse-causal).")
    print("[scrutiny]       if it SURVIVES K=12 and momentum-only holds -> a genuine organic early-traction signal.")


if __name__ == "__main__":
    main()
