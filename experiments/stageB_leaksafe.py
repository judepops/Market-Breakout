"""Stage B-B leakage-safe rung-2: does the as-of signal PREDICT capital raised strictly AFTER detection?

    THEMES_FILE=themes_market_clean.parquet HORIZON=36 python -m experiments.stageB_leaksafe

The contemporaneous correlation (momentum vs all-time capital, +0.36) can't tell "signal predicted the money"
from "hot markets already had money". This converts it to a PREDICTION: bridge markets->companies (cached
embeddings), then count only funding rounds announced STRICTLY AFTER each market's T_dec, within a fixed
forward window H. Reports Spearman(signal, forward_capital) for sig_score AND momentum, with:
  - a full-horizon-observable subset (T_dec + H <= data end; no right-censoring), AND
  - a censoring-free RATE (forward_capital / months_observed) over all markets;
each with a permutation (shuffle) null + a bootstrap CI. A forward null is a committed, honest outcome.
"""
from __future__ import annotations
import os
import pathlib

import numpy as np
import pandas as pd

SB = pathlib.Path("artifacts/stageB")
TAU = float(os.environ.get("BRIDGE_TAU", "0.63"))
H = int(os.environ.get("HORIZON", "36"))          # forward window in months


def _months(m):
    y, mo = int(m[:4]), int(m[5:7])
    return y * 12 + (mo - 1)


def _embed(texts):
    from marketbrain.labels.gemini_grounding import make_client
    client = make_client()
    out = []
    for i in range(0, len(texts), 100):
        r = client.models.embed_content(model="gemini-embedding-001", contents=[str(t)[:2000] for t in texts[i:i + 100]])
        out.extend(np.asarray(e.values, dtype=np.float32) for e in r.embeddings)
    X = np.vstack(out)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def _perm_null(x, y, obs, B=5000, seed=0):
    from scipy.stats import spearmanr
    rng = np.random.RandomState(seed)
    cnt = 0
    for _ in range(B):
        r, _ = spearmanr(x, rng.permutation(y))
        if r >= obs:
            cnt += 1
    return (cnt + 1) / (B + 1)


def _boot_ci(x, y, B=2000, seed=0):
    from scipy.stats import spearmanr
    rng = np.random.RandomState(seed)
    n = len(x); rs = []
    for _ in range(B):
        idx = rng.randint(0, n, n)
        if len(set(y[idx])) < 3:
            continue
        r, _ = spearmanr(x[idx], y[idx])
        if not np.isnan(r):
            rs.append(r)
    return (np.percentile(rs, 2.5), np.percentile(rs, 97.5)) if rs else (np.nan, np.nan)


def main():
    from scipy.stats import spearmanr
    themes_file = os.environ.get("THEMES_FILE", "themes_market_clean.parquet")
    th = pd.read_parquet(SB / themes_file)
    comp = pd.read_parquet("data/raw/labels/company_descriptions.parquet")
    comp = comp[comp["bridge_text"].str.len() > 0].reset_index(drop=True)
    print(f"[leaksafe] {themes_file}: {len(th)} markets; horizon H={H}mo, tau={TAU}", flush=True)

    tvec = _embed(list(th["theme_text"]))
    cvec = np.load(SB / "company_emb.npy")
    ids = list(np.load(SB / "company_ids.npy", allow_pickle=True))
    assert ids == list(comp["permalink"]), "company embedding cache out of sync"
    sims = cvec @ tvec.T
    best = sims.argmax(1); best_sim = sims[np.arange(len(comp)), best]
    comp = comp.assign(theme_id=th["theme_id"].to_numpy()[best], sim=best_sim)
    matched = comp[comp["sim"] >= TAU][["permalink", "theme_id", "sim"]].copy()

    rounds = pd.read_parquet("data/raw/labels/company_rounds.parquet").dropna(subset=["announced_on"])
    rounds["m"] = rounds["announced_on"].astype(str).str[:7].map(_months)
    data_end = int(rounds["m"].max())
    mr = matched.merge(rounds, on="permalink", how="inner")
    mr = mr.merge(th[["theme_id", "T_dec"]], on="theme_id", how="left")
    mr["tdec_m"] = mr["T_dec"].map(_months)

    # forward = rounds strictly after T_dec, within H months; pre = at/before T_dec (the contemporaneous confound)
    fwd = mr[(mr["m"] > mr["tdec_m"]) & (mr["m"] <= mr["tdec_m"] + H)]
    pre = mr[mr["m"] <= mr["tdec_m"]]
    fwd_cap = fwd.groupby("theme_id")["raised_usd"].sum()
    pre_cap = pre.groupby("theme_id")["raised_usd"].sum()
    n_fwd_co = fwd.groupby("theme_id")["permalink"].nunique()

    th = th.assign(
        fwd_cap=th["theme_id"].map(fwd_cap).fillna(0.0),
        pre_cap=th["theme_id"].map(pre_cap).fillna(0.0),
        n_fwd_co=th["theme_id"].map(n_fwd_co).fillna(0),
        tdec_m=th["T_dec"].map(_months))
    th["months_obs"] = (data_end - th["tdec_m"]).clip(lower=1, upper=H)
    th["fwd_rate"] = th["fwd_cap"] / th["months_obs"]
    n_comp = matched.groupby("theme_id").size()
    th["n_bridged"] = th["theme_id"].map(n_comp).fillna(0)

    def report(sub, tag):
        sub = sub[sub["n_bridged"] >= 3]
        if len(sub) < 8:
            print(f"\n[leaksafe] {tag}: only {len(sub)} markets with >=3 companies -- underpowered"); return
        print(f"\n[leaksafe] === {tag} (n={len(sub)} markets, >=3 bridged companies) ===")
        for sig in ["momentum", "sig_score"]:
            for outc in ["fwd_cap", "fwd_rate"]:
                x = sub[sig].to_numpy(float); y = sub[outc].to_numpy(float)
                if np.std(y) == 0:
                    continue
                rho, _ = spearmanr(x, y)
                p = _perm_null(x, y, rho)
                lo, hi = _boot_ci(x, y)
                print(f"   spearman({sig:9s}, {outc:8s}) = {rho:+.3f}  perm-null p={p:.3f}  boot95[{lo:+.3f},{hi:+.3f}]"
                      f"{'  *' if (lo>0 or hi<0) else ''}", flush=True)
        # contemporaneous vs forward, for momentum (the confound check)
        x = sub["momentum"].to_numpy(float)
        rho_pre, _ = spearmanr(x, sub["pre_cap"].to_numpy(float))
        rho_fwd, _ = spearmanr(x, sub["fwd_cap"].to_numpy(float))
        print(f"   [confound] momentum vs PRE-T_dec capital={rho_pre:+.3f}  vs FORWARD capital={rho_fwd:+.3f} "
              f"(forward is the honest prediction)", flush=True)

    obs = th[th["tdec_m"] <= data_end - H]     # full horizon observable (no right-censoring)
    report(obs, f"FULL-HORIZON-OBSERVABLE (T_dec <= {data_end - H} in months; H={H}mo)")
    report(th, "RATE-NORMALISED (all markets; fwd_rate is censoring-free)")
    th.to_parquet(SB / themes_file.replace(".parquet", "_leaksafe.parquet"))
    print(f"\n[leaksafe] data_end month index={data_end} (~{data_end // 12}-{data_end % 12 + 1:02d}); wrote *_leaksafe.parquet")


if __name__ == "__main__":
    main()
