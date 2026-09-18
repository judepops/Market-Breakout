"""Shared builder for the Stage-A evals: corpus -> gated discriminative themes + as-of features + momentum.

Returns the {positive, hyped_died} discriminative pool with the durable-breakout labels, the 11 as-of signature
features, per-theme momentum (escore), and the lookups needed to attach titles/records. Single-phrase gated
themes (the detector's unit). Used by phase1_b2 (B2 baseline) and the idea-family-holdout eval so the heavy
corpus pass + gating + labelling live in ONE place.
"""
from __future__ import annotations


def build_discriminative_dataset(cats=None, *, min_docs=40):
    import numpy as np
    import pandas as pd
    from marketbrain.corpus.load_full import load_corpus
    from marketbrain.eval.corpus_inputs import cs_denominator, corpus_term_events
    from marketbrain.eval.label import EmergenceLabelConfig, durable_breakout_label
    from marketbrain.eval.breakout_signature import build_feature_matrix
    from marketbrain.markets.granularity_gates import _pmi_ok, _noun_headed, _is_fragment, GENERIC, DF_CEIL
    from marketbrain.features.emergence import EmergenceConfig, escore, month_range_ending

    cfg = EmergenceLabelConfig()
    corpus = load_corpus(cats)
    denom = cs_denominator(corpus)
    grid = list(denom.index)
    corpus_end, n = grid[-1], len(corpus)
    te = corpus_term_events(corpus, min_doc_freq=min_docs, ngram_max=3)
    df_count = te.groupby("term")["record_id"].nunique().to_dict()
    ceil = DF_CEIL * n
    gated = [t for t, c in df_count.items()
             if c <= ceil and not _is_fragment(t)
             and not (len(t.split()) == 1 and t in GENERIC)
             and not (len(t.split()) > 1 and all(w in GENERIC for w in t.split()))
             and _pmi_ok(t, df_count, n)]
    gated = set(_noun_headed(gated))
    te = te[te["term"].isin(gated)]
    pivot = te.groupby(["term", "month"])["record_id"].nunique().unstack(fill_value=0).reindex(columns=grid, fill_value=0)
    denom_arr = denom.reindex(grid).to_numpy()
    rows = []
    for term, n_theme in pivot.iterrows():
        share = pd.Series(np.divide(n_theme.to_numpy(), denom_arr, out=np.zeros(len(grid)), where=denom_arr > 0), index=grid)
        res = durable_breakout_label(share, n_theme.astype(int), cfg, corpus_end=corpus_end)
        t0 = res.get("T0")
        t_dec = grid[grid.index(t0) + 9] if t0 in grid and grid.index(t0) + 9 < len(grid) else None
        rows.append({"theme_id": term, "label": res["label"], "T0": t0, "T_dec": t_dec, "parent_id": term})
    labels = pd.DataFrame(rows)
    disc = labels[labels["label"].isin(("positive", "hyped_died"))].dropna(subset=["T_dec"]).reset_index(drop=True)

    themes = {t: {t} for t in disc["theme_id"]}
    term_to_records = te.groupby("term")["record_id"].apply(set).to_dict()
    records = corpus.set_index("record_id")[["month", "authors", "categories", "comment", "journal_ref", "title",
                                             "primary_category"]]
    feats = build_feature_matrix(themes, term_to_records, records, disc, denom)
    ecfg = EmergenceConfig()
    tdec = disc.set_index("theme_id")["T_dec"]
    momentum = pd.Series({t: escore(pivot.loc[t].reindex(
        month_range_ending(tdec[t], ecfg.window_months), fill_value=0).to_numpy().tolist()) for t in feats["theme_id"]})
    return {"disc": disc, "labels": labels, "feats": feats, "momentum": momentum, "pivot": pivot,
            "denom": denom, "term_to_records": term_to_records, "records": records, "corpus_end": corpus_end}
