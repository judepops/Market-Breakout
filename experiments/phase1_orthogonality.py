"""Ben's ask: is momentum ORTHOGONAL to the signature, or subsumed by it? Combine them and see if the
combination beats the signature alone.

    ~/miniconda3/envs/marketbrain/bin/python -m experiments.phase1_orthogonality [--cats cs.AI,...]

Replicates the phase1_detector pipeline on the headline chosen-field corpus (AI/ML core), then scores FOUR
scorers on identical 5-fold OOF splits:
  A  signature GBM (the 11 as-of shape features)      -- the detector
  B  B1 momentum (escore, unsupervised)               -- the baseline
  C  GBM with momentum as a 12th feature              -- "can the model extract anything extra from momentum?"
  D  per-fold logistic stack of (signature, momentum) -- "does a linear combine of the two scores beat either?"
Verdict via metrics.paired_delta_ci: C-vs-A (and D-vs-A) delta > 0 with CI excluding 0 -> momentum carries
complementary information; delta ~ 0 -> momentum is SUBSUMED by the signature (persistence/breadth already
encode what escore measures). Either answer is a result.
"""
from __future__ import annotations
import argparse
import json
import pathlib
import warnings

warnings.filterwarnings("ignore")
OUT = pathlib.Path("artifacts/phase1_orthogonality")
HEADLINE_CATS = ["cs.AI", "cs.LG", "cs.CV", "cs.CL", "stat.ML"]   # the chosen-field (deck headline) corpus


def main(argv=None):
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from marketbrain.corpus.load_full import load_corpus
    from marketbrain.eval.corpus_inputs import cs_denominator, corpus_term_events
    from marketbrain.eval.label import EmergenceLabelConfig, durable_breakout_label
    from marketbrain.eval import metrics, splits
    from marketbrain.eval.breakout_signature import build_feature_matrix, fit_predict_oof, SIGNATURE_FEATURES
    from marketbrain.markets.granularity_gates import _pmi_ok, _noun_headed, _is_fragment, GENERIC, DF_CEIL
    from marketbrain.features.emergence import EmergenceConfig, escore, month_range_ending

    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", default=",".join(HEADLINE_CATS))
    ap.add_argument("--min-docs", type=int, default=40)
    args = ap.parse_args(argv)
    cats = args.cats.split(",")
    cfg = EmergenceLabelConfig()

    # ---- identical pipeline to phase1_detector: corpus -> gates -> labels -> disc pool ----
    corpus = load_corpus(cats)
    denom = cs_denominator(corpus)
    grid = list(denom.index)
    corpus_end = grid[-1]
    n = len(corpus)
    print(f"[orth] corpus {cats}: {n:,} papers, grid {grid[0]}..{corpus_end}", flush=True)
    te = corpus_term_events(corpus, min_doc_freq=args.min_docs, ngram_max=3)
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
        share = pd.Series(np.divide(n_theme.to_numpy(), denom_arr, out=np.zeros(len(grid)), where=denom_arr > 0),
                          index=grid)
        res = durable_breakout_label(share, n_theme.astype(int), cfg, corpus_end=corpus_end)
        t0 = res.get("T0")
        t_dec = grid[grid.index(t0) + 9] if t0 in grid and grid.index(t0) + 9 < len(grid) else None
        rows.append({"theme_id": term, "label": res["label"], "T0": t0, "T_dec": t_dec, "parent_id": term})
    labels = pd.DataFrame(rows)
    disc = labels[labels["label"].isin(("positive", "hyped_died"))].dropna(subset=["T_dec"]).reset_index(drop=True)
    n_pos = int((disc["label"] == "positive").sum())
    print(f"[orth] discriminative pool: {n_pos} pos / {len(disc) - n_pos} hyped-died", flush=True)

    themes = {t: {t} for t in disc["theme_id"]}
    term_to_records = te.groupby("term")["record_id"].apply(set).to_dict()
    records_indexed = corpus.set_index("record_id")[["month", "authors", "categories", "comment", "journal_ref"]]
    print(f"[orth] building {len(disc)} as-of feature rows...", flush=True)
    feats = build_feature_matrix(themes, term_to_records, records_indexed, disc, denom)
    feats = feats.merge(disc[["theme_id"]], on="theme_id")

    # B1 momentum for the same themes, added to the feature frame as a candidate 12th feature
    ecfg = EmergenceConfig()
    def b1(tid, t_dec):
        usage = pivot.loc[tid].reindex(month_range_ending(t_dec, ecfg.window_months), fill_value=0).to_numpy()
        return escore(usage.tolist())
    b1_scores = pd.Series({r.theme_id: b1(r.theme_id, r.T_dec) for r in disc.itertuples(index=False)})
    feats["b1_momentum"] = feats["theme_id"].map(b1_scores).fillna(0.0)

    y_all = disc.set_index("theme_id")["label"]
    ids = disc["theme_id"]
    yv = (y_all.reindex(ids) == "positive").astype(int).to_numpy()
    folds = splits.grouped_folds(disc, k=5)

    # ---- the four scorers on IDENTICAL folds ----
    def oof_gbm(feature_cols):
        out = {}
        for train_df, test_df in folds:
            tr = feats[feats["theme_id"].isin(train_df["theme_id"])]
            ts = feats[feats["theme_id"].isin(test_df["theme_id"])]
            if tr.empty or ts.empty or tr["label"].nunique() < 2:
                continue
            y = (tr["label"] == "positive").astype(int).to_numpy()
            clf = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05,
                                                 min_samples_leaf=10, random_state=0)
            clf.fit(tr[feature_cols].to_numpy(), y)
            for tid, p in zip(ts["theme_id"], clf.predict_proba(ts[feature_cols].to_numpy())[:, 1]):
                out[tid] = p
        return pd.Series(out)

    def oof_stack():
        """Per fold: GBM(11) on train -> logistic on (train in-sample GBM score, momentum) -> predict test."""
        out = {}
        for train_df, test_df in folds:
            tr = feats[feats["theme_id"].isin(train_df["theme_id"])]
            ts = feats[feats["theme_id"].isin(test_df["theme_id"])]
            if tr.empty or ts.empty or tr["label"].nunique() < 2:
                continue
            y = (tr["label"] == "positive").astype(int).to_numpy()
            clf = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05,
                                                 min_samples_leaf=10, random_state=0)
            clf.fit(tr[SIGNATURE_FEATURES].to_numpy(), y)
            str_tr = clf.predict_proba(tr[SIGNATURE_FEATURES].to_numpy())[:, 1]
            str_ts = clf.predict_proba(ts[SIGNATURE_FEATURES].to_numpy())[:, 1]
            lr = LogisticRegression(max_iter=1000)
            lr.fit(np.column_stack([str_tr, tr["b1_momentum"]]), y)
            proba = lr.predict_proba(np.column_stack([str_ts, ts["b1_momentum"]]))[:, 1]
            for tid, p in zip(ts["theme_id"], proba):
                out[tid] = p
        return pd.Series(out)

    print("[orth] scoring A signature / B momentum / C +12th-feature / D logistic stack on identical folds...", flush=True)
    A = oof_gbm(SIGNATURE_FEATURES)
    B = b1_scores
    C = oof_gbm(SIGNATURE_FEATURES + ["b1_momentum"])
    D = oof_stack()

    def arr(s):
        return s.reindex(ids).fillna(0.0).to_numpy()

    summary = {"cats": cats, "n_pos": n_pos, "n_neg": len(disc) - n_pos}
    print(f"\n[orth] AUPRC on identical 5-fold OOF (n={len(disc)}, prevalence {n_pos/len(disc):.3f}):")
    for name, s in [("A signature (11)", A), ("B momentum alone", B), ("C sig+momentum GBM (12)", C), ("D logistic stack", D)]:
        ap_ = metrics.auprc(arr(s), yv)
        summary[name] = float(ap_)
        print(f"   {name:26s} AUPRC={ap_:.3f}", flush=True)

    print("\n[orth] paired deltas (bootstrap CI, B=2000) — the orthogonality verdicts:")
    for name, s in [("C - A  (momentum as 12th feature)", C), ("D - A  (logistic stack)", D)]:
        d, lo, hi, p1 = metrics.paired_delta_ci(arr(s), arr(A), yv, B=2000, seed=0)
        verdict = "COMPLEMENTARY (adds information)" if lo > 0 else ("HARMFUL" if hi < 0 else "SUBSUMED (within noise)")
        summary[name] = {"delta": float(d), "ci_lo": float(lo), "ci_hi": float(hi), "p_one_sided": float(p1)}
        print(f"   {name:36s} Δ={d:+.3f}  95% CI [{lo:+.3f},{hi:+.3f}]  -> {verdict}", flush=True)
    d, lo, hi, p1 = metrics.paired_delta_ci(arr(A), arr(B), yv, B=2000, seed=0)
    summary["A - B (context: sig vs momentum)"] = {"delta": float(d), "ci_lo": float(lo), "ci_hi": float(hi)}
    print(f"   {'A - B  (context: sig over momentum)':36s} Δ={d:+.3f}  95% CI [{lo:+.3f},{hi:+.3f}]", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"theme_id": ids.to_numpy(), "label": disc["label"].to_numpy(),
                  "signature": arr(A), "momentum": arr(B), "combined_gbm": arr(C), "stack": arr(D)}
                 ).to_parquet(OUT / "scores.parquet")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[orth] wrote {OUT}/scores.parquet + summary.json")
    print("[orth] read: SUBSUMED -> the signature already encodes what escore measures (persistence/breadth cover it);")
    print("[orth]       COMPLEMENTARY -> report the combined number as the new best and note momentum's marginal value.")


if __name__ == "__main__":
    main()
