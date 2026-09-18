"""Does the organic footprint signal survive with ZERO funding-derived information? Answers "did we use money
to predict money?" head-on by progressively stripping every funding touchpoint.

    ~/miniconda3/envs/marketbrain/bin/python -m experiments.altdata_fundingfree

Variants (all: footprint features from GitHub/HN/Wikipedia only; controls=[fy,mkt_code,obs_age], NO money;
outcome=total_raised>=$50M as the LABEL only; chronological split; permutation null):
  V1 full (17)          organic window [founding, first_round); incl. log_winlen + breadth  -> baseline
  V2 no log_winlen (16) organic window; drops the one funding-TIMING feature
  V3 footprint-only (15) organic window; drops log_winlen AND breadth -> pure pre-round footprint
  V4 funding-DATE-free  restrict to slow-raisers (first_round >= founding+36mo), FIXED window [founding, founding+24mo)
                        (never references the exact round date to build features; guaranteed pre-round) + footprint-only
If V3/V4 survive -> the signal is NOT money-predicting-money; the footprint itself carries it.
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
    CTRL = ["fy", "mkt_code", "obs_age"]

    def build(d, start_col, end_col, add_winlen, add_breadth):
        d = d.copy(); base = d.set_index("permalink")[[start_col, end_col]].rename(columns={start_col: "s", end_col: "e"})
        cols = []
        for tag, p in panels.items():
            q = p.join(base, on="permalink", how="inner")
            q = q[(q["mm"] >= q["s"]) & (q["mm"] < q["e"])]
            q["win"] = (q["e"] - q["s"]).clip(lower=1); q["late"] = (q["mm"] - q["s"] >= q["win"] / 2).astype(int)
            g = q.groupby("permalink")
            out = pd.DataFrame({f"{tag}_logmax": np.log1p(g["val"].max()), f"{tag}_logmean": np.log1p(g["val"].mean()),
                                f"{tag}_active": g["val"].apply(lambda s: (s > 0).mean()), f"{tag}_present": 1.0})
            late = q[q["late"] == 1].groupby("permalink")["val"].sum(); early = q[q["late"] == 0].groupby("permalink")["val"].sum()
            out[f"{tag}_momentum"] = (np.log1p(late) - np.log1p(early)).reindex(out.index).fillna(0.0)
            for c in out.columns:
                d[c] = d["permalink"].map(out[c]).fillna(0.0); cols.append(c)
        if add_breadth:
            d["breadth"] = d[[c for c in cols if c.endswith("_present")]].sum(1); cols.append("breadth")
        if add_winlen:
            d["log_winlen"] = np.log1p((d[end_col] - d[start_col]).clip(lower=1)); cols.append("log_winlen")
        d = d[d["permalink"].isin(has)].reset_index(drop=True)
        d["y"] = (d["permalink"].map(raised).fillna(0.0) >= 50e6).astype(int)
        return d, cols

    def test(d, feats):
        tr = (d["fy"] < 2020).to_numpy(); te = (d["fy"] >= 2020).to_numpy(); yv = d["y"].to_numpy()
        if yv[te].sum() < 10 or yv[tr].sum() < 10:
            return None
        def fit(cols):
            clf = HistGradientBoostingClassifier(max_depth=3, max_iter=400, learning_rate=0.04, min_samples_leaf=15, random_state=0)
            clf.fit(d.loc[tr, cols].to_numpy(), yv[tr]); return roc_auc_score(yv[te], clf.predict_proba(d.loc[te, cols].to_numpy())[:, 1])
        b = fit(CTRL); f = fit(CTRL + feats)
        rng = np.random.RandomState(0); nd = []
        ot, oe = d.loc[tr, feats].to_numpy(), d.loc[te, feats].to_numpy(); ct, ce = d.loc[tr, CTRL].to_numpy(), d.loc[te, CTRL].to_numpy()
        for _ in range(150):
            clf = HistGradientBoostingClassifier(max_depth=3, max_iter=400, learning_rate=0.04, min_samples_leaf=15, random_state=0)
            clf.fit(np.column_stack([ct, ot[rng.permutation(tr.sum())]]), yv[tr])
            nd.append(roc_auc_score(yv[te], clf.predict_proba(np.column_stack([ce, oe[rng.permutation(te.sum())]]))[:, 1]) - b)
        p = (np.sum(np.array(nd) >= (f - b)) + 1) / (len(nd) + 1)
        return b, f, p, int(yv[te].sum()), len(d)

    print("[fundingfree] progressively stripping every funding touchpoint from the FEATURES:")
    df["end_org"] = df["fr_m"]
    configs = [
        ("V1 full 17 (organic win, +winlen +breadth)", df, "found_m", "end_org", True, True),
        ("V2 no log_winlen 16 (drop funding-timing feat)", df, "found_m", "end_org", False, True),
        ("V3 footprint-only 15 (no winlen, no breadth)", df, "found_m", "end_org", False, False),
    ]
    for name, d0, s, e, wl, br in configs:
        r = test(*build(d0, s, e, wl, br))
        if r:
            b, f, p, npos, n = r
            v = "SURVIVES" if (p < 0.05 and f > b) else "gone"
            print(f"   {name:48s} n={n:5d} pos={npos:4d}  controls={b:.3f}  +feats={f:.3f}  Δ={f-b:+.3f}  perm-p={p:.3f}  -> {v}", flush=True)

    # V4: funding-DATE-free. Slow-raisers only (fr >= found+36), fixed window [found, found+24).
    slow = df[df["fr_m"] >= df["found_m"] + 36].copy()
    slow["end_fix"] = slow["found_m"] + 24
    r = test(*build(slow, "found_m", "end_fix", False, False))
    print("\n[fundingfree] V4 — FUNDING-DATE-FREE features (fixed [founding, founding+24mo), slow-raisers fr>=found+36):")
    if r:
        b, f, p, npos, n = r
        v = "SURVIVES" if (p < 0.05 and f > b) else "gone"
        print(f"   footprint-only 15, no funding date in features  n={n:5d} pos={npos:4d}  controls={b:.3f}  +feats={f:.3f}  Δ={f-b:+.3f}  perm-p={p:.3f}  -> {v}", flush=True)
    else:
        print("   (too few positives in the slow-raiser subsample to test)", flush=True)
    print("\n[fundingfree] read: V3 survives -> footprint (not funding) carries it. V4 survives -> signal holds even when")
    print("[fundingfree]       the funding DATE never touches the features at all. money is ONLY ever the label.")


if __name__ == "__main__":
    main()
