"""Recompute per-theme durability scores + 11 features on the HEADLINE AI/ML-core corpus (the +0.23 run) and
save them for the radar, so the app's themes/scores match the headline number on screen (fixes the demo
consistency risk of running on the weaker cs.AI-only pool).

    ~/miniconda3/envs/marketbrain/bin/python -m experiments.radar_recompute

Identical pipeline to phase1_detector, cats = the AI/ML core. Saves artifacts/phase1_aiml/{scores,features}.parquet.
"""
from __future__ import annotations
import pathlib, warnings
warnings.filterwarnings("ignore")
OUT = pathlib.Path("artifacts/phase1_aiml")
CATS = ["cs.AI", "cs.LG", "cs.CV", "cs.CL", "cs.NE", "stat.ML"]


def main():
    import numpy as np, pandas as pd
    from marketbrain.corpus.load_full import load_corpus
    from marketbrain.eval.corpus_inputs import cs_denominator, corpus_term_events
    from marketbrain.eval.label import EmergenceLabelConfig, durable_breakout_label
    from marketbrain.eval import splits
    from marketbrain.eval.breakout_signature import build_feature_matrix, fit_predict_oof, SIGNATURE_FEATURES
    from marketbrain.markets.granularity_gates import _pmi_ok, _noun_headed, _is_fragment, GENERIC, DF_CEIL
    from marketbrain.features.emergence import EmergenceConfig, escore, month_range_ending

    cfg = EmergenceLabelConfig()
    corpus = load_corpus(CATS)
    denom = cs_denominator(corpus); grid = list(denom.index); corpus_end = grid[-1]; nP = len(corpus)
    print(f"[recompute] AI/ML core: {nP:,} papers, grid {grid[0]}..{corpus_end}", flush=True)
    te = corpus_term_events(corpus, min_doc_freq=40, ngram_max=3)
    dfc = te.groupby("term")["record_id"].nunique().to_dict(); ceil = DF_CEIL * nP
    gated = [t for t, c in dfc.items() if c <= ceil and not _is_fragment(t)
             and not (len(t.split()) == 1 and t in GENERIC)
             and not (len(t.split()) > 1 and all(w in GENERIC for w in t.split())) and _pmi_ok(t, dfc, nP)]
    gated = set(_noun_headed(gated)); te = te[te["term"].isin(gated)]
    pivot = te.groupby(["term", "month"])["record_id"].nunique().unstack(fill_value=0).reindex(columns=grid, fill_value=0)
    da = denom.reindex(grid).to_numpy()
    rows = []
    for term, nth in pivot.iterrows():
        share = pd.Series(np.divide(nth.to_numpy(), da, out=np.zeros(len(grid)), where=da > 0), index=grid)
        res = durable_breakout_label(share, nth.astype(int), cfg, corpus_end=corpus_end)
        t0 = res.get("T0"); tdec = grid[grid.index(t0) + 9] if t0 in grid and grid.index(t0) + 9 < len(grid) else None
        rows.append({"theme_id": term, "label": res["label"], "T0": t0, "T_dec": tdec, "parent_id": term})
    labels = pd.DataFrame(rows)
    disc = labels[labels["label"].isin(("positive", "hyped_died"))].dropna(subset=["T_dec"]).reset_index(drop=True)
    print(f"[recompute] discriminative pool: {int((disc.label=='positive').sum())} pos / {int((disc.label=='hyped_died').sum())} fad", flush=True)

    themes = {t: {t} for t in disc["theme_id"]}
    term_to_records = te.groupby("term")["record_id"].apply(set).to_dict()
    recs = corpus.set_index("record_id")[["month", "authors", "categories", "comment", "journal_ref"]]
    feats = build_feature_matrix(themes, term_to_records, recs, disc, denom).merge(disc[["theme_id", "T0"]], on="theme_id")

    ecfg = EmergenceConfig()
    b1 = pd.Series({r.theme_id: escore(pivot.loc[r.theme_id].reindex(month_range_ending(r.T_dec, ecfg.window_months), fill_value=0).to_numpy().tolist())
                    for r in disc.itertuples(index=False)})
    folds = splits.grouped_folds(disc, k=5)
    sig = fit_predict_oof(feats, folds)

    OUT.mkdir(parents=True, exist_ok=True)
    scores = disc[["theme_id", "label", "T0"]].copy()
    scores["signature"] = scores["theme_id"].map(sig).fillna(0.0)
    scores["momentum"] = scores["theme_id"].map(b1).fillna(0.0)
    scores.to_parquet(OUT / "scores.parquet")
    feats.to_parquet(OUT / "features.parquet")
    print(f"[recompute] wrote {OUT}/scores.parquet ({len(scores)}) + features.parquet", flush=True)


if __name__ == "__main__":
    main()
