"""Phase-1: score the breakout SIGNATURE detector through the pre-registered harness vs B1 momentum + null.

    python -m experiments.phase1_detector [--all] [--min-docs N]

The actual test of the team's thesis: does a category-agnostic, leak-free PERSISTENCE signature (11 as-of features
-> gradient-boosted trees) beat the momentum baseline at predicting DURABLE breakout -- i.e. does it separate the
durable winners from the hyped-then-died, which look identical on growth alone? Reports AUPRC + shuffle-null for
the signature vs B1 momentum vs a random placeholder, on the temporal future-holdout (Split-1) and 5-fold OOF.
"""
from __future__ import annotations
import argparse
import json
import pathlib
import warnings

warnings.filterwarnings("ignore")
OUT = pathlib.Path("artifacts/phase1")


def main(argv=None):
    import numpy as np
    import pandas as pd
    from marketbrain.corpus.load_full import load_corpus
    from marketbrain.eval.corpus_inputs import cs_denominator, corpus_term_events
    from marketbrain.eval.label import EmergenceLabelConfig, durable_breakout_label
    from marketbrain.eval import metrics, splits
    from marketbrain.eval.breakout_signature import build_feature_matrix, fit_predict_oof, SIGNATURE_FEATURES
    from marketbrain.markets.granularity_gates import _pmi_ok, _noun_headed, _is_fragment, GENERIC, DF_CEIL
    from marketbrain.features.emergence import EmergenceConfig, escore, month_range_ending

    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--min-docs", type=int, default=40)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args(argv)
    cfg = EmergenceLabelConfig()

    corpus = load_corpus(None if args.all else ["cs.AI"], cache=args.all, refresh=args.refresh)
    denom = cs_denominator(corpus)
    grid = list(denom.index)
    corpus_end = grid[-1]
    n = len(corpus)
    print(f"[phase1] {n:,} papers, grid {grid[0]}..{corpus_end}; building gated themes...", flush=True)
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

    # label every gated theme (single-phrase themes; pivot-based)
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
    print(f"[phase1] discriminative pool: {n_pos} positives / {len(disc) - n_pos} hyped-died "
          f"(prevalence {n_pos / len(disc):.3f})", flush=True)

    # FEATURES (as-of T_dec, category-agnostic) for the discriminative themes
    themes = {t: {t} for t in disc["theme_id"]}
    term_to_records = te.groupby("term")["record_id"].apply(set).to_dict()
    records_indexed = corpus.set_index("record_id")[["month", "authors", "categories", "comment", "journal_ref"]]
    print(f"[phase1] building {len(disc)} as-of feature rows...", flush=True)
    feats = build_feature_matrix(themes, term_to_records, records_indexed, disc, denom)
    feats = feats.merge(disc[["theme_id"]], on="theme_id")          # keep alignment

    # B1 momentum (unsupervised) for the same themes, via the pivot
    ecfg = EmergenceConfig()

    def b1(tid, t_dec):
        usage = pivot.loc[tid].reindex(month_range_ending(t_dec, ecfg.window_months), fill_value=0).to_numpy()
        return escore(usage.tolist())

    b1_scores = pd.Series({r.theme_id: b1(r.theme_id, r.T_dec) for r in disc.itertuples(index=False)})
    rng = np.random.RandomState(0)
    rand_scores = pd.Series(rng.rand(len(disc)), index=disc["theme_id"])
    y_all = disc.set_index("theme_id")["label"]
    summary = {"n_pos": n_pos, "n_neg": len(disc) - n_pos, "n_features": len(SIGNATURE_FEATURES)}

    def score(name, s_series, ids):
        s = s_series.reindex(ids).fillna(0.0).to_numpy()
        yy = y_all.reindex(ids).to_numpy()
        ap_ = metrics.auprc(s, yy)
        _, _, p = metrics.shuffle_null(s, yy, B=2000, seed=0)
        print(f"[phase1]   {name:20s} AUPRC={ap_:.3f}  null p={p:.3f}")
        return {"auprc": float(ap_), "null_p": float(p), "n": len(ids)}

    # Split-2: 5-fold grouped OOF (parent=theme here -> plain 5-fold CV, NOT idea-family holdout; embedding parents deferred)
    folds = splits.grouped_folds(disc, k=5)
    sig_oof = fit_predict_oof(feats, folds)
    print(f"[phase1] Split-2 (5-fold OOF, n={len(disc)}; parent=theme -> plain CV, not idea-family holdout):")
    ids = disc["theme_id"]
    summary["oof"] = {"signature": score("signature", sig_oof, ids),
                      "B1_momentum": score("B1_momentum", b1_scores, ids),
                      "random": score("random", rand_scores, ids)}
    yv = y_all.reindex(ids).to_numpy()
    sa = sig_oof.reindex(ids).fillna(0.0).to_numpy()
    ba = b1_scores.reindex(ids).fillna(0.0).to_numpy()
    d, lo, hi, p1 = metrics.paired_delta_ci(sa, ba, yv, B=2000, seed=0)
    sep = "SEPARATED from 0" if (lo > 0 or hi < 0) else "spans 0 (within noise)"
    print(f"[phase1]   signature-momentum delta = {d:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
          f"P(momentum>=sig)={p1:.3f}  -> {sep}")
    summary["oof"]["signature_vs_momentum_delta"] = {"delta": d, "ci_lo": lo, "ci_hi": hi, "p_one_sided": p1}
    pd.DataFrame({"theme_id": ids.to_numpy(), "label": disc["label"].to_numpy(),
                  "signature": sa, "momentum": ba,
                  "random": rand_scores.reindex(ids).to_numpy()}).to_parquet(OUT / "scores.parquet")

    # Split-1: temporal future holdout -- fit on train, predict test
    tr, ts = splits.temporal_split(disc)
    print(f"[phase1] Split-1 (temporal future holdout): train {len(tr)} / test {len(ts)} "
          f"({int((ts['label'] == 'positive').sum())} test positives)")
    if ts["label"].nunique() == 2 and tr["label"].nunique() == 2:
        sig_temporal = fit_predict_oof(feats, [(tr, ts)])
        summary["temporal"] = {"signature": score("signature", sig_temporal, ts["theme_id"]),
                               "B1_momentum": score("B1_momentum", b1_scores, ts["theme_id"]),
                               "random": score("random", rand_scores, ts["theme_id"])}
    else:
        print("[phase1]   temporal test fold not 2-class -- skipped (widen corpus)")

    OUT.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(OUT / "features.parquet")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[phase1] wrote {OUT}/features.parquet + summary.json")


if __name__ == "__main__":
    main()
