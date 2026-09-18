"""Stage B-B, steps 3-4: semantic theme->company bridge + signal->money correlation.

    python -m experiments.stageB_bridge

Embeds the candidate detector's themes (artifacts/stageB/themes.parquet) and the 34k company descriptions
(data/raw/labels/company_descriptions.parquet) with gemini-embedding-001, assigns each company to its
best-matching theme (cosine >= tau), aggregates the matched companies' funding outcomes to the theme, and
asks the money question two ways:
  (1) CORRELATION (rung 1, descriptive): does the as-of emergence signal (sig_score) rank themes by their
      bridged funding outcome (outlier rate / total capital)?  Spearman + shuffle null.
  (2) NAS GAP: do durable-research POSITIVES convert to capital more than HYPED-DIED themes?  (NAS is the
      named non-converter: research-durable, market-fizzled.)
Company embeddings are cached to artifacts/stageB/company_emb.npy so re-runs are cheap. Reports bridge
eyeball samples so precision can be judged. Leakage note: sig_score is strictly as-of T_dec; the funding
outcome is the target (may use the future) -- rung-2 (outcome strictly after T_dec) is a follow-up.
"""
from __future__ import annotations
import pathlib

import numpy as np
import pandas as pd

SB = pathlib.Path("artifacts/stageB")
import os
TAU = float(os.environ.get("BRIDGE_TAU", "0.63"))   # cross-domain cosines top out ~0.74; 0.63 ~ real matches
TOPK_EYEBALL = 6


def _embed(texts, model="gemini-embedding-001", batch=100, label="", ckpt=None):
    import time
    from marketbrain.labels.gemini_grounding import make_client
    client = make_client()
    out = []
    start = 0
    if ckpt is not None and ckpt.exists():                       # resume a partial company embed
        prev = np.load(ckpt)
        out = [row for row in prev]
        start = len(out)
        print(f"[bridge]   resume {label} from {start}/{len(texts)}", flush=True)
    for i in range(start, len(texts), batch):
        chunk = [str(t)[:2000] for t in texts[i:i + batch]]
        for attempt in range(6):
            try:
                r = client.models.embed_content(model=model, contents=chunk)
                out.extend(np.asarray(e.values, dtype=np.float32) for e in r.embeddings)
                break
            except Exception as e:
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)
        if (i // batch) % 10 == 0:
            print(f"[bridge]   embed {label} {i + len(chunk)}/{len(texts)}", flush=True)
            if ckpt is not None:
                np.save(ckpt, np.vstack(out))
    X = np.vstack(out)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def main():
    from scipy.stats import spearmanr

    themes_file = os.environ.get("THEMES_FILE", "themes.parquet")
    themes = pd.read_parquet(SB / themes_file)
    comp = pd.read_parquet("data/raw/labels/company_descriptions.parquet")
    comp = comp[comp["bridge_text"].str.len() > 0].reset_index(drop=True)
    print(f"[bridge] {len(themes)} themes x {len(comp)} companies", flush=True)

    # ---- embed (cache company vectors; themes are cheap) ----
    tvec = _embed(list(themes["theme_text"]), label="themes")
    cache = SB / "company_emb.npy"
    idcache = SB / "company_ids.npy"
    if cache.exists() and idcache.exists() and list(np.load(idcache, allow_pickle=True)) == list(comp["permalink"]):
        cvec = np.load(cache)
        print("[bridge] loaded cached company embeddings", flush=True)
    else:
        cvec = _embed(list(comp["bridge_text"]), label="companies", ckpt=SB / "company_emb_ckpt.npy")
        np.save(cache, cvec)
        np.save(idcache, comp["permalink"].to_numpy())

    # ---- assign each company to its best-matching theme (cosine), keep >= TAU ----
    sims = cvec @ tvec.T                              # (n_comp, n_theme)
    best = sims.argmax(1)
    best_sim = sims[np.arange(len(comp)), best]
    comp = comp.assign(theme_id=themes["theme_id"].to_numpy()[best], sim=best_sim)
    matched = comp[comp["sim"] >= TAU].copy()
    print(f"[bridge] {len(matched)}/{len(comp)} companies matched at cosine>={TAU} "
          f"({matched['theme_id'].nunique()} themes have >=1 company)", flush=True)

    # ---- outcomes per company (total raised + outlier flag) ----
    rounds = pd.read_parquet("data/raw/labels/company_rounds.parquet")
    raised = rounds.groupby("permalink")["raised_usd"].sum()
    clevel = pd.read_parquet("data/raw/labels/company_level.parquet")[["permalink", "is_outlier"]].drop_duplicates("permalink")
    matched = matched.merge(raised.rename("total_raised"), on="permalink", how="left")
    matched = matched.merge(clevel, on="permalink", how="left")
    matched["total_raised"] = matched["total_raised"].fillna(0.0)
    matched["is_outlier"] = matched["is_outlier"].fillna(False)
    matched["is_500m"] = matched["total_raised"] >= 500e6

    # ---- eyeball a few bridges (precision by inspection) ----
    print("\n[bridge] EYEBALL (top matched companies per high-signal positive theme):", flush=True)
    top_pos = themes[themes["label"] == "positive"].sort_values("sig_score", ascending=False).head(6)
    for _, tr in top_pos.iterrows():
        sub = matched[matched["theme_id"] == tr["theme_id"]].sort_values("sim", ascending=False).head(TOPK_EYEBALL)
        names = [f"{r['name']}({r.sim:.2f}${r.total_raised/1e6:.0f}M)" for _, r in sub.iterrows()]
        print(f"   * {tr['theme_id'][:34]:34s} -> " + ", ".join(names) if names else
              f"   * {tr['theme_id'][:34]:34s} -> (no company >= tau)", flush=True)

    # ---- aggregate to theme + correlate signal vs money ----
    agg = matched.groupby("theme_id").agg(
        n_comp=("permalink", "size"), n_outlier=("is_outlier", "sum"),
        n_500m=("is_500m", "sum"), total_capital=("total_raised", "sum"),
        p90_raised=("total_raised", lambda s: float(np.percentile(s, 90)))).reset_index()
    agg["outlier_rate"] = agg["n_outlier"] / agg["n_comp"]
    th = themes.merge(agg, on="theme_id", how="left").fillna(
        {"n_comp": 0, "n_outlier": 0, "n_500m": 0, "total_capital": 0, "p90_raised": 0, "outlier_rate": 0})
    th.to_parquet(SB / themes_file.replace("themes", "theme_outcomes"))

    # themes with a real bridge (>=3 companies) for the correlation
    ev = th[th["n_comp"] >= 3].copy()
    print(f"\n[bridge] === SIGNAL -> MONEY (rung 1, correlation; {len(ev)} themes with >=3 bridged companies) ===")
    for outcome in ["outlier_rate", "total_capital", "n_500m", "p90_raised"]:
        s = pd.to_numeric(ev[outcome], errors="coerce")
        if len(ev) < 5 or float(s.std() or 0) == 0:
            continue
        rho, p = spearmanr(ev["sig_score"].to_numpy(float), s.to_numpy(float))
        rho_m, p_m = spearmanr(ev["momentum"].to_numpy(float), s.to_numpy(float))
        print(f"   spearman(sig_score, {outcome:13s}) = {rho:+.3f} (p={p:.3f})   | momentum: {rho_m:+.3f} (p={p_m:.3f})")

    # ---- NAS gap: positives vs hyped-died conversion ----
    print(f"\n[bridge] === NAS GAP (do durable POSITIVES convert to capital more than HYPED-DIED?) ===")
    for grp in ("positive", "hyped_died"):
        g = th[th["label"] == grp]
        gg = g[g["n_comp"] >= 1]
        print(f"   {grp:11s}: n={len(g):3d} ({len(gg)} bridged)  median outlier_rate={gg['outlier_rate'].median():.3f}  "
              f"median total_capital=${gg['total_capital'].median()/1e6:.0f}M  "
              f"mean n_500m={gg['n_500m'].mean():.2f}")
    # NAS specifically, if present
    for probe in ["neural architecture search", "nas"]:
        r = th[th["theme_id"] == probe]
        if len(r):
            rr = r.iloc[0]
            print(f"   [NAS probe] '{probe}': label={rr['label']} sig={rr['sig_score']:.2f} "
                  f"n_comp={rr['n_comp']:.0f} outlier_rate={rr['outlier_rate']:.2f} capital=${rr['total_capital']/1e6:.0f}M")
    print(f"\n[bridge] wrote {SB}/theme_outcomes.parquet", flush=True)


if __name__ == "__main__":
    main()
