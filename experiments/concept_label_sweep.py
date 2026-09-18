"""Recalibrate durable-breakout label gates for LLM-concept themes.

    python -m experiments.concept_label_sweep

The pre-registered label config (birth 5-in-3mo, growth 4x, scale 0.3%) was frozen
for n-gram MENTION counts. LLM concepts count papers ABOUT an idea (~5x sparser),
so the same gates collapse the cohort (73 decidable of 7,297 themes). This sweep
varies the gates on the concept universe and reports, per config: decidable cohort
size, prevalence, 9-month logistic OOF AUPRC, and the paired delta vs momentum.

The chosen config becomes the documented "concepts-v1" label calibration — a
deliberate, reported departure from the n-gram pre-registration.

Writes artifacts/concept_label_sweep.json.
"""
from __future__ import annotations

import dataclasses
import json
import pathlib

OUT = pathlib.Path("artifacts/concept_label_sweep.json")

CONFIGS = [
    # (birth_min_records, G_REL, G_ABS)
    (5, 4.0, 0.003),    # pre-registered baseline
    (3, 4.0, 0.003),
    (3, 3.0, 0.001),
    (3, 2.5, 0.0005),
    (2, 3.0, 0.0005),
    (2, 2.5, 0.00025),
]


def main(argv=None):
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from experiments.build_radar_data import TERM_ARTIFACTS, dedupe_cohort
    from marketbrain.corpus.load_full import load_corpus
    from marketbrain.eval import metrics, splits
    from marketbrain.eval.breakout_signature import SIGNATURE_FEATURES, build_feature_matrix
    from marketbrain.eval.corpus_inputs import cs_denominator
    from marketbrain.eval.label import EmergenceLabelConfig, _shift_month, durable_breakout_label
    from marketbrain.features.emergence import EmergenceConfig, escore, month_range_ending

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
    shares = {}
    for term, n_theme in pivot.iterrows():
        shares[term] = pd.Series(
            np.divide(n_theme.to_numpy(), denom_arr, out=np.zeros(len(grid)), where=denom_arr > 0),
            index=grid)
    ecfg = EmergenceConfig()
    results = []

    for birth, g_rel, g_abs in CONFIGS:
        cfg = dataclasses.replace(EmergenceLabelConfig(), birth_min_records=birth,
                                  G_REL=g_rel, G_ABS=g_abs)
        rows = []
        for term, n_theme in pivot.iterrows():
            res = durable_breakout_label(shares[term], n_theme.astype(int), cfg, corpus_end=corpus_end)
            t0 = res.get("T0")
            t_dec = grid[grid.index(t0) + 9] if t0 in grid and grid.index(t0) + 9 < len(grid) else None
            rows.append({"theme_id": term, "label": res["label"], "T0": t0, "T_dec": t_dec, "parent_id": term})
        labels = pd.DataFrame(rows)
        disc = labels[labels["label"].isin(("positive", "hyped_died"))].dropna(subset=["T_dec"])
        disc = disc[~disc["theme_id"].isin(TERM_ARTIFACTS)]
        disc = disc[disc["theme_id"].isin(dedupe_cohort(list(disc["theme_id"]), term_to_records))].reset_index(drop=True)
        n_pos = int((disc["label"] == "positive").sum())
        live = int(((labels["label"] == "censored") & (labels["T0"] >= _shift_month(corpus_end, -48))).sum())

        row = {"birth": birth, "g_rel": g_rel, "g_abs": g_abs,
               "labeled": len(disc), "durable": n_pos, "live": live}
        if len(disc) >= 60 and 0 < n_pos < len(disc):
            feats = build_feature_matrix({t: {t} for t in disc["theme_id"]},
                                         term_to_records, records_indexed, disc, denom)
            y = (feats["label"] == "positive").astype(int).to_numpy()
            X = StandardScaler().fit_transform(feats[SIGNATURE_FEATURES].fillna(0).to_numpy(float))
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
                                  .to_numpy().tolist())
                           for r in feats.itertuples(index=False)])
            yl = feats["label"].to_numpy()
            d, lo, hi, p1 = metrics.paired_delta_ci(oof, b1, yl, B=2000, seed=0)
            row.update({"prevalence": round(float(y.mean()), 3),
                        "auprc": round(float(metrics.auprc(oof, yl)), 3),
                        "auprc_momentum": round(float(metrics.auprc(b1, yl)), 3),
                        "delta": round(float(d), 3),
                        "ci": [round(float(lo), 3), round(float(hi), 3)],
                        "p": round(float(p1), 3)})
        results.append(row)
        print(f"[sweep] {row}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"[sweep] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
