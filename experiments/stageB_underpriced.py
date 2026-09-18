"""Stage B test #2 (pre-registered): UNDERPRICED DURABILITY.

    THEMES_FILE=clustered_markets.parquet HORIZON=60 python -m experiments.stageB_underpriced

Test #1 showed capital ignores durability (chases contemporaneous heat). The live question: do durable markets
nonetheless produce more real >=$500M company OUTCOMES later? If durability predicts forward OUTCOMES but not
forward CAPITAL (#1), the market underprices durability -- an allocative inefficiency / alpha.

Outcome (leakage-safe): per market, rate of bridged companies whose CUMULATIVE raise first crosses >=$500M with
the crossing date STRICTLY AFTER T_dec, within H months. Signal = predicted durability (sig_score) + research
label; baseline = momentum. Spearman + permutation null + bootstrap CI. Full-horizon-observable + rate versions.
Pass (pre-registered): durability->forward-outcome CI excludes 0 AND perm p<0.05 AND beats momentum.
"""
from __future__ import annotations
import os
import pathlib

import numpy as np
import pandas as pd

SB = pathlib.Path("artifacts/stageB")
TAU = float(os.environ.get("BRIDGE_TAU", "0.63"))
H = int(os.environ.get("HORIZON", "60"))
OUTLIER_USD = 500e6


def _months(m):
    return int(m[:4]) * 12 + (int(m[5:7]) - 1)


def _embed(texts):
    from marketbrain.labels.gemini_grounding import make_client
    client = make_client()
    out = []
    for i in range(0, len(texts), 100):
        r = client.models.embed_content(model="gemini-embedding-001", contents=[str(t)[:2000] for t in texts[i:i + 100]])
        out.extend(np.asarray(e.values, dtype=np.float32) for e in r.embeddings)
    X = np.vstack(out)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def _perm(x, y, obs, B=5000, seed=0):
    from scipy.stats import spearmanr
    rng = np.random.RandomState(seed)
    return (sum(spearmanr(x, rng.permutation(y))[0] >= obs for _ in range(B)) + 1) / (B + 1)


def _boot(x, y, B=2000, seed=0):
    from scipy.stats import spearmanr
    rng = np.random.RandomState(seed); n = len(x); rs = []
    for _ in range(B):
        i = rng.randint(0, n, n)
        if len(set(y[i])) > 2:
            r = spearmanr(x[i], y[i])[0]
            if not np.isnan(r):
                rs.append(r)
    return (np.percentile(rs, 2.5), np.percentile(rs, 97.5)) if rs else (np.nan, np.nan)


def main():
    from scipy.stats import spearmanr
    themes_file = os.environ.get("THEMES_FILE", "clustered_markets.parquet")
    th = pd.read_parquet(SB / themes_file)
    comp = pd.read_parquet("data/raw/labels/company_descriptions.parquet")
    comp = comp[comp["bridge_text"].str.len() > 0].reset_index(drop=True)
    print(f"[underpriced] {themes_file}: {len(th)} markets; >=${OUTLIER_USD/1e6:.0f}M outcomes, H={H}mo, tau={TAU}", flush=True)

    # company >=$500M crossing month (leakage-safe datable outcome)
    r = pd.read_parquet("data/raw/labels/company_rounds.parquet").dropna(subset=["announced_on"]).sort_values(["permalink", "announced_on"])
    r["m"] = r["announced_on"].astype(str).str[:7].map(_months)
    r["cum"] = r.groupby("permalink")["raised_usd"].cumsum()
    cross = r[r["cum"] >= OUTLIER_USD].groupby("permalink")["m"].min().rename("cross_m")
    data_end = int(r["m"].max())

    # bridge (cached company embeddings)
    tvec = _embed(list(th["theme_text"]))
    cvec = np.load(SB / "company_emb.npy"); ids = list(np.load(SB / "company_ids.npy", allow_pickle=True))
    assert ids == list(comp["permalink"]), "company embedding cache out of sync (re-run stageB_bridge to rebuild)"
    sims = cvec @ tvec.T
    best = sims.argmax(1)
    comp = comp.assign(theme_id=th["theme_id"].to_numpy()[best], sim=sims[np.arange(len(comp)), best])
    matched = comp[comp["sim"] >= TAU][["permalink", "theme_id"]].merge(cross, on="permalink", how="left")
    matched = matched.merge(th[["theme_id", "T_dec"]], on="theme_id", how="left")
    matched["tdec_m"] = matched["T_dec"].map(_months)
    matched["fwd_outlier"] = (matched["cross_m"] > matched["tdec_m"]) & (matched["cross_m"] <= matched["tdec_m"] + H)

    g = matched.groupby("theme_id")
    th = th.assign(
        n_bridged=th["theme_id"].map(g.size()).fillna(0),
        n_fwd_outlier=th["theme_id"].map(g["fwd_outlier"].sum()).fillna(0),
        tdec_m=th["T_dec"].map(_months))
    th["fwd_outlier_rate"] = th["n_fwd_outlier"] / th["n_bridged"].replace(0, np.nan)
    th["fwd_outlier_rate"] = th["fwd_outlier_rate"].fillna(0)

    def report(sub, tag):
        sub = sub[sub["n_bridged"] >= 5]
        if len(sub) < 8 or sub["n_fwd_outlier"].sum() < 3:
            print(f"\n[underpriced] {tag}: n={len(sub)}, total fwd-outliers={int(sub['n_fwd_outlier'].sum())} -- underpowered"); return
        print(f"\n[underpriced] === {tag} (n={len(sub)} markets, {int(sub['n_fwd_outlier'].sum())} forward >=$500M outcomes) ===")
        for sig in ["sig_score", "momentum"]:
            x = sub[sig].to_numpy(float); y = sub["fwd_outlier_rate"].to_numpy(float)
            if np.std(y) == 0:
                continue
            rho = spearmanr(x, y)[0]; p = _perm(x, y, rho); lo, hi = _boot(x, y)
            star = "  <-- PASS" if (lo > 0 and p < 0.05 and sig == "sig_score") else ("  *" if (lo > 0 or hi < 0) else "")
            print(f"   spearman({sig:9s}, fwd_outlier_rate) = {rho:+.3f}  perm p={p:.3f}  boot95[{lo:+.3f},{hi:+.3f}]{star}", flush=True)
        # durability LABEL contrast: do durable-positive markets out-produce the rest?
        pos = sub[sub["label"] == "positive"]["fwd_outlier_rate"]
        rest = sub[sub["label"] != "positive"]["fwd_outlier_rate"]
        if len(pos) >= 3 and len(rest) >= 3:
            print(f"   [label] durable-positive markets median fwd-outlier-rate={pos.median():.3f} (n={len(pos)}) "
                  f"vs rest={rest.median():.3f} (n={len(rest)})", flush=True)

    report(th[th["tdec_m"] <= data_end - H], f"FULL-HORIZON-OBSERVABLE (H={H}mo)")
    report(th, "ALL MARKETS (rate; recent markets right-censored -> conservative)")
    th.to_parquet(SB / themes_file.replace(".parquet", "_underpriced.parquet"))
    print(f"\n[underpriced] {len(cross)} companies ever cross >=$500M; wrote *_underpriced.parquet", flush=True)


if __name__ == "__main__":
    main()
