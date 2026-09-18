"""How much of the detector rests on the (leaky) venue feature, and how bad is the leak?

    python -m experiments.venue_sensitivity

Part A — drop-and-remeasure: logistic OOF AUPRC of the full 15-feature model vs the
same model with venue_appearance removed, at h=1 and h=9, on the concepts-v1 cohort.

Part B — leak size: for papers whose comment/journal_ref names a top venue AND whose
arXiv comment carries an acceptance year, estimate months from first submission to
that acceptance (the window during which the venue field is "known" in our data but
was NOT knowable in real time).

Writes artifacts/venue_sensitivity.json.
"""
from __future__ import annotations

import dataclasses
import json
import pathlib
import re

OUT = pathlib.Path("artifacts/venue_sensitivity.json")

# "Accepted to NeurIPS 2024", "to appear in ICML 2023", "Published at CVPR 2022"
_VENUE_YEAR = re.compile(
    r"(neurips|nips|icml|iclr|cvpr|iccv|eccv|acl|emnlp|naacl|aaai|ijcai|kdd|sigir|www)\D{0,20}(20\d{2})",
    re.I)


def acceptance_lag_months(first_month: str, comment: str, journal_ref: str) -> int | None:
    """Months from first-submission to the venue YEAR named in the metadata.

    Coarse (year granularity): assumes mid-year (June) acceptance of the named year.
    Returns None if no venue-year is present. Negative lags (metadata year before
    submission) are dropped as unreliable."""
    text = f"{comment or ''} {journal_ref or ''}"
    m = _VENUE_YEAR.search(text)
    if not m:
        return None
    acc_year = int(m.group(2))
    sy, sm = (int(x) for x in first_month.split("-"))
    lag = (acc_year - sy) * 12 + (6 - sm)  # to mid-year of acceptance year
    return lag if lag >= 0 else None


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

    corpus = load_corpus(["cs.AI"], cache=False)
    denom = cs_denominator(corpus)
    grid = list(denom.index)
    corpus_end = grid[-1]
    denom_arr = denom.reindex(grid).to_numpy()

    # -- Part B: acceptance lag over the whole corpus -----------------------------------------
    lags = []
    for r in corpus[["month", "comment", "journal_ref"]].itertuples(index=False):
        lag = acceptance_lag_months(r.month, r.comment, r.journal_ref)
        if lag is not None:
            lags.append(lag)
    lags = np.array(lags)
    partB = {
        "n_papers_with_venue_year": int(len(lags)),
        "pct_of_corpus": round(100 * len(lags) / len(corpus), 1),
        "median_lag_months": float(np.median(lags)) if len(lags) else None,
        "mean_lag_months": round(float(np.mean(lags)), 1) if len(lags) else None,
        "p25_lag": float(np.percentile(lags, 25)) if len(lags) else None,
        "p75_lag": float(np.percentile(lags, 75)) if len(lags) else None,
        "pct_lag_ge_9mo": round(100 * float((lags >= 9).mean()), 1) if len(lags) else None,
    }
    print(f"[venue] Part B (acceptance lag): {json.dumps(partB)}", flush=True)

    # -- Part A: drop-and-remeasure -----------------------------------------------------------
    te = pd.read_parquet("data/derived/concepts/concept_events.parquet")
    df_count = te.groupby("term")["record_id"].nunique()
    te = te[te["term"].isin(df_count[df_count >= 10].index)]
    pivot = (te.groupby(["term", "month"])["record_id"].nunique()
             .unstack(fill_value=0).reindex(columns=grid, fill_value=0))
    term_to_records = te.groupby("term")["record_id"].apply(set).to_dict()
    records_indexed = corpus.set_index("record_id")[["month", "authors", "categories", "comment", "journal_ref"]]
    index_set = set(records_indexed.index)

    def theme_recs(tid):
        return records_indexed.loc[list(term_to_records.get(tid, set()) & index_set)]

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

    author_index = build_author_index(corpus)
    hits = settled_hit_index([
        (r.theme_id, _shift_month(r.T_peak or r.T0, cfg.persist_hi),
         early_authors(theme_recs(r.theme_id), r.T0, as_of=corpus_end))
        for r in disc[disc["label"] == "positive"].itertuples(index=False)])

    def oof(feats, cols, y):
        X = StandardScaler().fit_transform(feats[cols].fillna(0).to_numpy(float))
        p = np.zeros(len(feats))
        for tr_df, te_df in splits.grouped_folds(feats, k=5):
            tri = feats["theme_id"].isin(tr_df["theme_id"]).to_numpy()
            tei = feats["theme_id"].isin(te_df["theme_id"]).to_numpy()
            if len(np.unique(y[tri])) < 2:
                continue
            m = LogisticRegression(max_iter=1000).fit(X[tri], y[tri])
            p[tei] = m.predict_proba(X[tei])[:, 1]
        return p

    full = SIGNATURE_FEATURES + AUTHOR_FEATURES
    no_venue = [f for f in full if f != "venue_appearance"]
    partA = {}
    for h in (1, 9):
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
        a_full = metrics.auprc(oof(feats, full, y), yl)
        a_drop = metrics.auprc(oof(feats, no_venue, y), yl)
        d, lo, hi, p1 = metrics.paired_delta_ci(oof(feats, full, y), oof(feats, no_venue, y), yl, B=2000, seed=0)
        partA[f"h{h}"] = {"full": round(float(a_full), 3), "no_venue": round(float(a_drop), 3),
                          "venue_contribution": round(float(a_full - a_drop), 3),
                          "ci": [round(float(lo), 3), round(float(hi), 3)], "p": round(float(p1), 3)}
        print(f"[venue] Part A h={h}: {json.dumps(partA[f'h{h}'])}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"drop_remeasure": partA, "acceptance_lag": partB}, indent=2))
    print(f"[venue] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
