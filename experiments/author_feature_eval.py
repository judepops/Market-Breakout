"""Do in-corpus author-quality features improve the concept detector?

    python -m experiments.author_feature_eval

On the concepts-v1 labeled cohort, compares logistic OOF AUPRC of:
  SIG            the 11 pre-registered signature features
  SIG+AUTH       plus the 4 author features (marketbrain/eval/author_features.py)
  SIG+<one>      add-one ablation per author feature
  AUTH           author features alone (sanity floor)
at horizons t0+1 and t0+9, with a paired bootstrap delta for SIG+AUTH vs SIG.

Writes artifacts/author_feature_eval.json.
"""
from __future__ import annotations

import dataclasses
import json
import pathlib

OUT = pathlib.Path("artifacts/author_feature_eval.json")


def main(argv=None):
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from experiments.build_radar_data import TERM_ARTIFACTS, dedupe_cohort
    from marketbrain.corpus.load_full import load_corpus
    from marketbrain.eval import metrics, splits
    from marketbrain.eval.author_features import (
        AUTHOR_FEATURES, CITATION_FEATURES, build_author_index, build_author_paper_index,
        early_authors, settled_hit_index, theme_author_features, theme_citation_features)
    from marketbrain.eval.breakout_signature import SIGNATURE_FEATURES, build_feature_matrix
    from marketbrain.eval.corpus_inputs import cs_denominator
    from marketbrain.eval.label import EmergenceLabelConfig, _shift_month, durable_breakout_label

    corpus = load_corpus(["cs.AI"], cache=False)
    denom = cs_denominator(corpus)
    grid = list(denom.index)
    corpus_end = grid[-1]
    denom_arr = denom.reindex(grid).to_numpy()
    te = pd.read_parquet("data/derived/concepts/concept_events.parquet")
    df_count = te.groupby("term")["record_id"].nunique()
    te = te[te["term"].isin(df_count[df_count >= 10].index)]
    pivot = (te.groupby(["term", "month"])["record_id"].nunique()
             .unstack(fill_value=0).reindex(columns=grid, fill_value=0))
    term_to_records = te.groupby("term")["record_id"].apply(set).to_dict()
    records_indexed = corpus.set_index("record_id")[["month", "authors", "categories", "comment", "journal_ref"]]

    cfg = dataclasses.replace(EmergenceLabelConfig(), birth_min_records=3)
    rows = []
    for term, n_theme in pivot.iterrows():
        share = pd.Series(np.divide(n_theme.to_numpy(), denom_arr,
                                    out=np.zeros(len(grid)), where=denom_arr > 0), index=grid)
        res = durable_breakout_label(share, n_theme.astype(int), cfg, corpus_end=corpus_end)
        rows.append({"theme_id": term, "label": res["label"], "T0": res.get("T0"),
                     "T_peak": res.get("T_peak"), "parent_id": term})
    labels = pd.DataFrame(rows)
    disc = labels[labels["label"].isin(("positive", "hyped_died"))].dropna(subset=["T0"]).copy()
    disc = disc[~disc["theme_id"].isin(TERM_ARTIFACTS)]
    disc = disc[disc["theme_id"].isin(dedupe_cohort(list(disc["theme_id"]), term_to_records))].reset_index(drop=True)
    print(f"[eval] cohort {len(disc)} ({int((disc['label']=='positive').sum())} durable)", flush=True)

    print("[eval] building author index...", flush=True)
    author_index = build_author_index(corpus)
    index_set = set(records_indexed.index)

    # OpenAlex citation histories (Tier 2), if fetched
    import glob as _glob
    cit_lookup: dict[str, dict] = {}
    for f in _glob.glob("data/derived/openalex/chunk-*.parquet"):
        for r in pd.read_parquet(f).itertuples(index=False):
            cit_lookup[r.record_id] = json.loads(r.counts_by_year)
    paper_index = build_author_paper_index(corpus) if cit_lookup else {}
    print(f"[eval] citation histories for {len(cit_lookup):,} papers", flush=True)

    def theme_recs(tid):
        ids = term_to_records.get(tid, set()) & index_set
        return records_indexed.loc[list(ids)]

    # settled durable themes -> their early authors + settle month (T_peak + persist_hi)
    positives = []
    for r in disc[disc["label"] == "positive"].itertuples(index=False):
        settle = _shift_month(r.T_peak or r.T0, cfg.persist_hi)
        authors = early_authors(theme_recs(r.theme_id), r.T0, as_of=corpus_end)
        positives.append((r.theme_id, settle, authors))
    hits = settled_hit_index(positives)
    print(f"[eval] {len(hits):,} authors with settled durable track records", flush=True)

    def oof_auprc(feats, cols, y):
        X = StandardScaler().fit_transform(feats[cols].fillna(0).to_numpy(float))
        oof = np.zeros(len(feats))
        for tr_df, te_df in splits.grouped_folds(feats, k=5):
            tri = feats["theme_id"].isin(tr_df["theme_id"]).to_numpy()
            tei = feats["theme_id"].isin(te_df["theme_id"]).to_numpy()
            if len(np.unique(y[tri])) < 2:
                continue
            m = LogisticRegression(max_iter=1000).fit(X[tri], y[tri])
            oof[tei] = m.predict_proba(X[tei])[:, 1]
        return oof

    results = {}
    for h in (1, 9):
        lf = disc.copy()
        lf["T_dec"] = lf["T0"].map(lambda t0: _shift_month(t0, h))
        feats = build_feature_matrix({t: {t} for t in lf["theme_id"]},
                                     term_to_records, records_indexed, lf, denom)
        feats = feats.merge(lf[["theme_id", "T0"]], on="theme_id")
        arows = []
        for r in feats.itertuples(index=False):
            authors = early_authors(theme_recs(r.theme_id), r.T0, as_of=r.T_dec)
            row = {"theme_id": r.theme_id,
                   **theme_author_features(authors, r.T0, author_index, hits)}
            if cit_lookup:
                row.update(theme_citation_features(authors, r.T0, paper_index, cit_lookup))
            arows.append(row)
        feats = feats.merge(pd.DataFrame(arows), on="theme_id")
        y = (feats["label"] == "positive").astype(int).to_numpy()
        yl = feats["label"].to_numpy()

        base_cols = SIGNATURE_FEATURES + AUTHOR_FEATURES
        sig = oof_auprc(feats, SIGNATURE_FEATURES, y)
        both = oof_auprc(feats, base_cols, y)
        d, lo, hi, p1 = metrics.paired_delta_ci(both, sig, yl, B=2000, seed=0)
        res = {
            "SIG": round(float(metrics.auprc(sig, yl)), 3),
            "SIG+AUTH": round(float(metrics.auprc(both, yl)), 3),
            "AUTH_only": round(float(metrics.auprc(oof_auprc(feats, AUTHOR_FEATURES, y), yl)), 3),
            "delta": round(float(d), 3), "ci": [round(float(lo), 3), round(float(hi), 3)],
            "p": round(float(p1), 3),
        }
        for f in AUTHOR_FEATURES:
            res[f"SIG+{f}"] = round(float(metrics.auprc(
                oof_auprc(feats, SIGNATURE_FEATURES + [f], y), yl)), 3)
        if cit_lookup:
            cit = oof_auprc(feats, base_cols + CITATION_FEATURES, y)
            dc, lc, hc, pc = metrics.paired_delta_ci(cit, both, yl, B=2000, seed=0)
            res["SIG+AUTH+CIT"] = round(float(metrics.auprc(cit, yl)), 3)
            res["cit_delta"] = round(float(dc), 3)
            res["cit_ci"] = [round(float(lc), 3), round(float(hc), 3)]
            res["cit_p"] = round(float(pc), 3)
            for f in CITATION_FEATURES:
                res[f"BASE+{f}"] = round(float(metrics.auprc(
                    oof_auprc(feats, base_cols + [f], y), yl)), 3)
        results[f"h{h}"] = res
        print(f"[eval] h={h}: {json.dumps(res)}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"[eval] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
