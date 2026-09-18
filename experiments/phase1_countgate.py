"""Count-based Gate-2 strengthening: make "got big" denominator-INVARIANT and show the Stage-A pool + edge
stop swinging with corpus choice. Pre-reg rule (fixed before AUPRC): N = 0.003 x median monthly denominator of
the REFERENCE corpus (broad-7) over 2015-2023, then N held FIXED across all corpora.

    python -m experiments.phase1_countgate --cats "cs.LG,cs.CL,cs.AI,cs.RO,cs.SE,cs.DC,stat.ML"   # reference: computes N
    python -m experiments.phase1_countgate --all --n-count <N>                                     # stress test: full-28, same N

For the loaded corpus: gate (identical to Stage A), build T_dec + features ONCE (label-independent), then for
each absolute-scale rule {share 0.3%, count 0.6N, count N, count 1.5N} relabel -> discriminative pool ->
variant-grouped OOF signature-vs-momentum delta. The win is stability of (n, delta) across corpora vs the
share-gate's swing (355->91, +0.18->+0.08).
"""
from __future__ import annotations
import argparse
import warnings

warnings.filterwarnings("ignore")


def main(argv=None):
    import numpy as np
    import pandas as pd
    from marketbrain.corpus.load_full import load_corpus
    from marketbrain.eval.corpus_inputs import cs_denominator, corpus_term_events
    from marketbrain.eval.label import EmergenceLabelConfig, durable_breakout_label
    from marketbrain.eval.breakout_signature import build_feature_matrix, fit_predict_oof, SIGNATURE_FEATURES
    from marketbrain.eval import metrics, splits
    from marketbrain.markets.granularity_gates import _pmi_ok, _noun_headed, _is_fragment, GENERIC, DF_CEIL
    from marketbrain.features.emergence import EmergenceConfig, escore, month_range_ending
    from experiments.phase1_secure import _variant_groups

    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--cats", default="cs.LG,cs.CL,cs.AI,cs.RO,cs.SE,cs.DC,stat.ML")
    ap.add_argument("--n-count", type=int, default=None, help="fixed count threshold N (omit to compute from this corpus)")
    args = ap.parse_args(argv)

    cats = None if args.all else args.cats.split(",")
    corpus = load_corpus(cats)
    denom = cs_denominator(corpus)
    grid = list(denom.index)
    corpus_end, n = grid[-1], len(corpus)
    # pre-registered N rule (reference corpus): 0.003 x median monthly denominator over 2015-2023
    ref = denom[[m for m in grid if "2015-01" <= m <= "2023-12"]]
    N = args.n_count if args.n_count is not None else int(round(0.003 * float(ref.median())))
    print(f"[countgate] {n:,} papers ({'full-28' if args.all else args.cats}); median denom(2015-2023)="
          f"{float(ref.median()):.0f} -> N={N} papers/mo (source={'given' if args.n_count else 'computed'})", flush=True)

    te = corpus_term_events(corpus, min_doc_freq=40, ngram_max=3)
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

    def label_pool(cfg):
        rows = []
        for term, n_theme in pivot.iterrows():
            share = pd.Series(np.divide(n_theme.to_numpy(), denom_arr, out=np.zeros(len(grid)), where=denom_arr > 0), index=grid)
            res = durable_breakout_label(share, n_theme.astype(int), cfg, corpus_end=corpus_end)
            t0 = res.get("T0")
            t_dec = grid[grid.index(t0) + 9] if t0 in grid and grid.index(t0) + 9 < len(grid) else None
            rows.append({"theme_id": term, "label": res["label"], "T0": t0, "T_dec": t_dec, "parent_id": term})
        return pd.DataFrame(rows)

    # build features ONCE over all themes with a valid T_dec (label-independent); reuse across rules
    base = label_pool(EmergenceLabelConfig()).dropna(subset=["T_dec"]).reset_index(drop=True)
    themes = {t: {t} for t in base["theme_id"]}
    term_to_records = te.groupby("term")["record_id"].apply(set).to_dict()
    records_indexed = corpus.set_index("record_id")[["month", "authors", "categories", "comment", "journal_ref"]]
    feats_all = build_feature_matrix(themes, term_to_records, records_indexed, base, denom).set_index("theme_id")
    ecfg = EmergenceConfig()
    mom_all = {t: escore(pivot.loc[t].reindex(month_range_ending(td, ecfg.window_months), fill_value=0).to_numpy().tolist())
               for t, td in zip(base["theme_id"], base["T_dec"])}

    rules = [("share_0.3%", EmergenceLabelConfig()),
             (f"count_{int(0.6*N)}", EmergenceLabelConfig(G_ABS_COUNT=int(0.6 * N))),
             (f"count_{N}*", EmergenceLabelConfig(G_ABS_COUNT=N)),
             (f"count_{int(1.5*N)}", EmergenceLabelConfig(G_ABS_COUNT=int(1.5 * N)))]
    print(f"\n[countgate] absolute-scale rule sweep (variant-grouped OOF; * = pre-registered primary):")
    print(f"   {'rule':14s}{'n_pos':>6s}{'n_neg':>6s}{'base':>6s}{'sig':>7s}{'mom':>7s}{'delta':>8s}   95% CI")
    for name, cfg in rules:
        lp = label_pool(cfg)
        disc = lp[lp["label"].isin(("positive", "hyped_died"))].dropna(subset=["T_dec"]).reset_index(drop=True)
        if disc["label"].nunique() < 2 or len(disc) < 30:
            print(f"   {name:14s} n={len(disc)} -- too small"); continue
        ids = [t for t in disc["theme_id"] if t in feats_all.index]
        disc = disc[disc["theme_id"].isin(ids)].reset_index(drop=True)
        feats = feats_all.loc[ids].reset_index()
        groups = _variant_groups(list(disc["theme_id"]))
        disc_g = disc.copy(); disc_g["parent_id"] = disc_g["theme_id"].map(groups)
        oof = fit_predict_oof(feats, splits.grouped_folds(disc_g, k=5, seed=0)).reindex(ids).fillna(0.0).to_numpy()
        yv = (disc.set_index("theme_id").loc[ids, "label"] == "positive").astype(int).to_numpy()
        ba = np.array([mom_all[t] for t in ids])
        d, lo, hi, p1 = metrics.paired_delta_ci(oof, ba, yv, B=2000, seed=0)
        sig, mom = metrics.auprc(oof, yv), metrics.auprc(ba, yv)
        n_pos = int(yv.sum())
        sep = "SEP" if (lo > 0 or hi < 0) else "spans0"
        print(f"   {name:14s}{n_pos:>6d}{len(disc)-n_pos:>6d}{yv.mean():>6.2f}{sig:>7.3f}{mom:>7.3f}"
              f"{d:>+8.3f}   [{lo:+.3f},{hi:+.3f}] {sep} p={p1:.3f}", flush=True)


if __name__ == "__main__":
    main()
