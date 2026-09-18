"""Build a from-scratch-COMPARABLE detector for live scoring of brand-new themes.

    python -m experiments.fix_live_scorer

Problem: corpus 'durable' themes carry pre-emergence history (persistence 12-19mo), but a brand-new theme
scored over its first ~9mo caps at persistence ~10 -> the raw model under-rates from-scratch themes (GRPO).
Fix: replace the raw persistence COUNT with a window-normalized FRACTION (months-active / window-length),
which is age-invariant, retrain, and CONFIRM the detector's edge is preserved. Saves the retrained model +
training distribution for score_new_theme.py to use.
"""
from __future__ import annotations
import pathlib, warnings
warnings.filterwarnings("ignore")
OUT = pathlib.Path("artifacts/live_scorer")
CATS = ["cs.AI", "cs.LG", "cs.CV", "cs.CL", "cs.NE", "stat.ML"]


def main():
    import numpy as np, pandas as pd, joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score
    from marketbrain.corpus.load_full import load_corpus
    from marketbrain.eval.corpus_inputs import cs_denominator, corpus_term_events
    from marketbrain.eval.label import EmergenceLabelConfig, durable_breakout_label
    from marketbrain.eval import splits
    from marketbrain.eval.breakout_signature import build_feature_matrix, SIGNATURE_FEATURES
    from marketbrain.markets.granularity_gates import _pmi_ok, _noun_headed, _is_fragment, GENERIC, DF_CEIL

    cfg = EmergenceLabelConfig()
    corpus = load_corpus(CATS)
    denom = cs_denominator(corpus); grid = list(denom.index); corpus_end = grid[-1]; nP = len(corpus)
    print(f"[fix] AI/ML core: {nP:,} papers, grid {grid[0]}..{corpus_end}", flush=True)
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

    themes = {t: {t} for t in disc["theme_id"]}
    term_to_records = te.groupby("term")["record_id"].apply(set).to_dict()
    recs = corpus.set_index("record_id")[["month", "authors", "categories", "comment", "journal_ref"]]
    feats = build_feature_matrix(themes, term_to_records, recs, disc, denom)   # now includes window_len
    feats["persistence_frac"] = feats["persistence_months"] / feats["window_len"].clip(lower=1)
    print(f"[fix] {int((disc.label=='positive').sum())} pos / {int((disc.label=='hyped_died').sum())} fad; "
          f"persistence_months median={feats['persistence_months'].median():.0f}, "
          f"persistence_frac median={feats['persistence_frac'].median():.2f}", flush=True)

    FRAC = [f if f != "persistence_months" else "persistence_frac" for f in SIGNATURE_FEATURES]
    y = (feats["label"] == "positive").astype(int).to_numpy()

    def oof(cols):
        folds = splits.grouped_folds(disc, k=5); pred = np.full(len(feats), np.nan)
        idx = {t: i for i, t in enumerate(feats["theme_id"])}
        for tr, te_ in folds:
            trm = feats["theme_id"].isin(tr["theme_id"]); tem = feats["theme_id"].isin(te_["theme_id"])
            if trm.sum() == 0 or tem.sum() == 0 or feats[trm]["label"].nunique() < 2:
                continue
            clf = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05,
                                                 min_samples_leaf=10, random_state=0)
            clf.fit(feats[trm][cols].fillna(0).to_numpy(), y[trm.to_numpy()])
            pred[tem.to_numpy()] = clf.predict_proba(feats[tem][cols].fillna(0).to_numpy())[:, 1]
        m = ~np.isnan(pred)
        return average_precision_score(y[m], pred[m])

    ap_raw, ap_frac = oof(SIGNATURE_FEATURES), oof(FRAC)
    base = y.mean()
    print(f"[fix] OOF AUPRC — raw persistence: {ap_raw:.3f} | normalized frac: {ap_frac:.3f} | base rate: {base:.3f}", flush=True)
    print(f"[fix] {'OK: edge preserved' if ap_frac >= ap_raw - 0.02 else 'WARNING: normalization hurt AUPRC'}", flush=True)

    # final model on ALL disc, with the normalized feature
    clf = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05,
                                         min_samples_leaf=10, random_state=0).fit(feats[FRAC].fillna(0).to_numpy(), y)
    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "features": FRAC}, OUT / "model.joblib")
    feats[["theme_id", "label"] + FRAC].to_parquet(OUT / "train_feats_frac.parquet")
    print(f"[fix] saved {OUT}/model.joblib (+train_feats_frac.parquet); features order: {FRAC}", flush=True)


if __name__ == "__main__":
    main()
