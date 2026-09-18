"""Find the ideal vocabulary floor: sweep min_doc_freq and measure what each level buys.

    python -m experiments.min_docs_sweep [--floors 10 15 20 30 40] [--cats cs.AI]

Per floor, reports:
  vocab        gated theme candidates after granularity gates
  labeled      settled cohort size (positive + hyped_died) and prevalence
  auprc        logistic OOF AUPRC at the 9-month horizon (does noise dilute signal?)
  live         live-cohort size and how many were born in the last 12 / 24 months
The ideal floor is the lowest one where AUPRC holds up — that floor sets how young a
theme can be and still be visible to the radar.

Writes artifacts/min_docs_sweep.json.
"""
from __future__ import annotations

import argparse
import json
import pathlib

from experiments.build_radar_data import HORIZONS, TERM_ARTIFACTS, dedupe_cohort  # noqa: F401 (HORIZONS re-exported)

OUT = pathlib.Path("artifacts/min_docs_sweep.json")


def main(argv=None):
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from marketbrain.corpus.load_full import load_corpus
    from marketbrain.eval import metrics, splits
    from marketbrain.eval.breakout_signature import SIGNATURE_FEATURES, build_feature_matrix
    from marketbrain.eval.corpus_inputs import corpus_term_events, cs_denominator
    from marketbrain.eval.label import EmergenceLabelConfig, _shift_month, durable_breakout_label
    from marketbrain.markets.granularity_gates import DF_CEIL, GENERIC, _is_fragment, _noun_headed, _pmi_ok

    ap = argparse.ArgumentParser()
    ap.add_argument("--floors", nargs="*", type=int, default=[10, 15, 20, 30, 40])
    ap.add_argument("--cats", nargs="*", default=["cs.AI"])
    ap.add_argument("--live-months", type=int, default=48)
    args = ap.parse_args(argv)
    cfg = EmergenceLabelConfig()

    corpus = load_corpus(args.cats, cache=False)
    denom = cs_denominator(corpus)
    grid = list(denom.index)
    corpus_end = grid[-1]
    n = len(corpus)
    denom_arr = denom.reindex(grid).to_numpy()
    records_indexed = corpus.set_index("record_id")[["month", "authors", "categories", "comment", "journal_ref"]]
    results = []

    for floor in sorted(args.floors, reverse=True):
        print(f"[sweep] min_docs={floor}: extracting terms...", flush=True)
        te = corpus_term_events(corpus, min_doc_freq=floor, ngram_max=3)
        df_count = te.groupby("term")["record_id"].nunique().to_dict()
        ceil = DF_CEIL * n
        gated = [t for t, c in df_count.items()
                 if c <= ceil and not _is_fragment(t)
                 and not (len(t.split()) == 1 and t in GENERIC)
                 and not (len(t.split()) > 1 and all(w in GENERIC for w in t.split()))
                 and _pmi_ok(t, df_count, n)]
        gated = set(_noun_headed(gated))
        te = te[te["term"].isin(gated)]

        pivot = (te.groupby(["term", "month"])["record_id"].nunique()
                 .unstack(fill_value=0).reindex(columns=grid, fill_value=0))
        rows = []
        for term, n_theme in pivot.iterrows():
            share = pd.Series(
                np.divide(n_theme.to_numpy(), denom_arr, out=np.zeros(len(grid)), where=denom_arr > 0),
                index=grid)
            res = durable_breakout_label(share, n_theme.astype(int), cfg, corpus_end=corpus_end)
            rows.append({"theme_id": term, "label": res["label"], "T0": res.get("T0"), "parent_id": term})
        labels = pd.DataFrame(rows)
        term_to_records = te.groupby("term")["record_id"].apply(set).to_dict()

        disc = labels[labels["label"].isin(("positive", "hyped_died"))].dropna(subset=["T0"]).copy()
        disc = disc[~disc["theme_id"].isin(TERM_ARTIFACTS)]
        disc = disc[disc["theme_id"].isin(dedupe_cohort(list(disc["theme_id"]), term_to_records))]
        disc["T_dec"] = disc["T0"].map(lambda t0: _shift_month(t0, 9))
        disc = disc[disc["T_dec"] <= corpus_end].reset_index(drop=True)

        feats = build_feature_matrix({t: {t} for t in disc["theme_id"]}, term_to_records, records_indexed, disc, denom)
        y = (feats["label"] == "positive").astype(int).to_numpy()
        auprc = float("nan")
        if len(feats) > 20 and 0 < y.sum() < len(y):
            X = StandardScaler().fit_transform(feats[SIGNATURE_FEATURES].fillna(0).to_numpy(float))
            oof = np.zeros(len(feats))
            for tr_df, te_df in splits.grouped_folds(feats, k=5):
                tri = feats["theme_id"].isin(tr_df["theme_id"]).to_numpy()
                tei = feats["theme_id"].isin(te_df["theme_id"]).to_numpy()
                if len(np.unique(y[tri])) < 2:
                    continue
                m = LogisticRegression(max_iter=1000).fit(X[tri], y[tri])
                oof[tei] = m.predict_proba(X[tei])[:, 1]
            auprc = float(metrics.auprc(oof, feats["label"].to_numpy()))

        lo_month = _shift_month(corpus_end, -args.live_months)
        live = labels[(labels["label"] == "censored") & labels["T0"].notna() & (labels["T0"] >= lo_month)]
        live = live[~live["theme_id"].isin(TERM_ARTIFACTS)]
        live = live[live["theme_id"].isin(dedupe_cohort(list(live["theme_id"]), term_to_records))]
        y12 = int((live["T0"] >= _shift_month(corpus_end, -12)).sum())
        y24 = int((live["T0"] >= _shift_month(corpus_end, -24)).sum())

        row = {
            "min_docs": floor,
            "vocab": len(gated),
            "labeled": len(disc),
            "prevalence": round(float(y.mean()), 3) if len(feats) else None,
            "auprc_9mo": round(auprc, 3),
            "live": len(live),
            "live_born_12mo": y12,
            "live_born_24mo": y24,
        }
        results.append(row)
        print(f"[sweep] {row}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"[sweep] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
