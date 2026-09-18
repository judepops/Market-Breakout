"""Find the emergence gate + decision horizon for the leak-free (venue-dropped) model.

    python -m experiments.emergence_gate_sweep

The business model: no venue feature (lookahead bias), a recency-based emergence gate
("K papers in the last W months" instead of cumulative), and a 3-4 month decision
horizon. This sweeps (K, W) x horizon on the concept themes and reports, per config:
labeled cohort size, prevalence, clean OOF AUPRC, paired delta vs momentum, and the
live+newborn counts a 3-4mo product would surface.

Clean feature set = 10 signature features (venue_appearance dropped) + 4 author features.

Writes artifacts/emergence_gate_sweep.json.
"""
from __future__ import annotations

import dataclasses
import json
import pathlib

OUT = pathlib.Path("artifacts/emergence_gate_sweep.json")

# (birth_min_records K, birth_window W) — "K papers in the last W months"
GATES = [
    (3, 3),    # current concepts-v1
    (8, 3),    # 8 in last 3 months
    (10, 3),   # 10 in last 3 months
    (10, 2),   # 10 in last 2 months (stronger momentum)
    (10, 1),   # 10 in a single month (hottest)
]
HORIZONS = [3, 4]


def main(argv=None):
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from experiments.build_radar_data import TERM_ARTIFACTS, dedupe_cohort
    from marketbrain.corpus.load_full import load_corpus
    from marketbrain.eval import metrics, splits
    from marketbrain.eval.author_features import (
        AUTHOR_FEATURES, build_author_index, early_authors, settled_hit_index,
        theme_author_features)
    from marketbrain.eval.breakout_signature import SIGNATURE_FEATURES, build_feature_matrix
    from marketbrain.eval.corpus_inputs import cs_denominator
    from marketbrain.eval.label import EmergenceLabelConfig, _shift_month, durable_breakout_label
    from marketbrain.features.emergence import EmergenceConfig, escore, month_range_ending

    # leak-free feature set: everything except the venue-acceptance feature
    CLEAN = [f for f in SIGNATURE_FEATURES if f != "venue_appearance"] + AUTHOR_FEATURES

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
    index_set = set(records_indexed.index)
    shares = {t: pd.Series(np.divide(r.to_numpy(), denom_arr, out=np.zeros(len(grid)), where=denom_arr > 0),
                           index=grid) for t, r in pivot.iterrows()}

    def theme_recs(tid):
        return records_indexed.loc[list(term_to_records.get(tid, set()) & index_set)]

    author_index = build_author_index(corpus)
    ecfg = EmergenceConfig()
    results = []

    for K, W in GATES:
        cfg = dataclasses.replace(EmergenceLabelConfig(), birth_min_records=K, birth_window=W)
        rows = []
        for term, n_theme in pivot.iterrows():
            res = durable_breakout_label(shares[term], n_theme.astype(int), cfg, corpus_end=corpus_end)
            rows.append({"theme_id": term, "label": res["label"], "T0": res.get("T0"),
                         "T_peak": res.get("T_peak"), "parent_id": term})
        labels = pd.DataFrame(rows)
        disc = labels[labels["label"].isin(("positive", "hyped_died"))].dropna(subset=["T0"]).copy()
        disc = disc[~disc["theme_id"].isin(TERM_ARTIFACTS)]
        disc = disc[disc["theme_id"].isin(dedupe_cohort(list(disc["theme_id"]), term_to_records))].reset_index(drop=True)
        n_pos = int((disc["label"] == "positive").sum())
        live = int(((labels["label"] == "censored") & (labels["T0"] >= _shift_month(corpus_end, -48)).fillna(False)).sum())
        newborn = int(((labels["label"] == "never") & labels["T0"].notna()
                       & (labels["T0"] >= _shift_month(corpus_end, -12))).sum())

        hits = settled_hit_index([
            (r.theme_id, _shift_month(r.T_peak or r.T0, cfg.persist_hi),
             early_authors(theme_recs(r.theme_id), r.T0, as_of=corpus_end))
            for r in disc[disc["label"] == "positive"].itertuples(index=False)])

        entry = {"K": K, "W": W, "labeled": len(disc), "durable": n_pos,
                 "live": live, "newborn": newborn}
        if len(disc) >= 50 and 0 < n_pos < len(disc):
            for h in HORIZONS:
                lf = disc.copy()
                lf["T_dec"] = lf["T0"].map(lambda t0: _shift_month(t0, h))
                feats = build_feature_matrix({t: {t} for t in lf["theme_id"]},
                                             term_to_records, records_indexed, lf, denom)
                feats = feats.merge(lf[["theme_id", "T0"]], on="theme_id")
                arows = [{"theme_id": r.theme_id,
                          **theme_author_features(early_authors(theme_recs(r.theme_id), r.T0, as_of=r.T_dec),
                                                  r.T0, author_index, hits)}
                         for r in feats.itertuples(index=False)]
                feats = feats.merge(pd.DataFrame(arows), on="theme_id")
                y = (feats["label"] == "positive").astype(int).to_numpy()
                yl = feats["label"].to_numpy()
                X = StandardScaler().fit_transform(feats[CLEAN].fillna(0).to_numpy(float))
                oof = np.zeros(len(feats))
                for tr_df, te_df in splits.grouped_folds(feats, k=5):
                    tri = feats["theme_id"].isin(tr_df["theme_id"]).to_numpy()
                    tei = feats["theme_id"].isin(te_df["theme_id"]).to_numpy()
                    if len(np.unique(y[tri])) < 2:
                        continue
                    m = LogisticRegression(max_iter=1000).fit(X[tri], y[tri])
                    oof[tei] = m.predict_proba(X[tei])[:, 1]
                b1 = np.array([escore(pivot.loc[r.theme_id]
                                      .reindex(month_range_ending(r.T_dec, ecfg.window_months), fill_value=0)
                                      .to_numpy().tolist()) for r in feats.itertuples(index=False)])
                d, lo, hi, p1 = metrics.paired_delta_ci(oof, b1, yl, B=2000, seed=0)
                entry[f"h{h}"] = {"auprc": round(float(metrics.auprc(oof, yl)), 3),
                                  "momentum": round(float(metrics.auprc(b1, yl)), 3),
                                  "delta": round(float(d), 3), "p": round(float(p1), 3),
                                  "prevalence": round(float(y.mean()), 3)}
        results.append(entry)
        print(f"[gate] {json.dumps(entry)}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"[gate] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
