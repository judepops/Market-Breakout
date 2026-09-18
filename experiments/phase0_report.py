"""Phase-0 harness exercise: run the WHOLE pre-registered evaluation end-to-end against a placeholder scorer.

    python -m experiments.phase0_report [--all] [--min-docs N]

This proves the pre-registered eval (Plans/2026-06-30_phase0-preregistration.md, frozen in
artifacts/eval/PREREGISTRATION.json) actually RUNS on real data BEFORE the real breakout detector exists:
corpus -> GRANULARITY-GATED candidate themes -> durable-breakout LABELS -> prevalence/negative-controls ->
temporal + grouped SPLITS -> AUPRC + shuffle-null vs the B1 momentum baseline and a random placeholder. A random
scorer should NOT beat the baseline or the null -- that is the harness proving it rejects noise. Writes artifacts/phase0/.

Themes here are single GATED phrases (the same idea-level gates as discovery; synonym-merging into clusters is
deferred -- audit P3b). The point of Phase 0 is the machinery, not the final number.
"""
from __future__ import annotations
import argparse
import json
import pathlib
import warnings

warnings.filterwarnings("ignore")
OUT = pathlib.Path("artifacts/phase0")


def main(argv=None):
    import numpy as np
    import pandas as pd
    from marketbrain.corpus.load_full import load_corpus
    from marketbrain.eval.corpus_inputs import cs_denominator, corpus_term_events
    from marketbrain.eval.label import EmergenceLabelConfig, durable_breakout_label
    from marketbrain.eval import metrics, splits, negatives
    from marketbrain.markets.granularity_gates import _pmi_ok, _noun_headed, _is_fragment, GENERIC, DF_CEIL
    from marketbrain.features.emergence import EmergenceConfig, escore, month_range_ending

    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--min-docs", type=int, default=40, help="min distinct records for a phrase to be considered")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args(argv)
    cfg = EmergenceLabelConfig()

    corpus = load_corpus(None if args.all else ["cs.AI"], cache=args.all, refresh=args.refresh)
    denom = cs_denominator(corpus)
    grid = list(denom.index)
    corpus_end = grid[-1]
    n = len(corpus)
    print(f"[phase0] {n:,} papers, grid {grid[0]}..{corpus_end}; building term events...", flush=True)
    te = corpus_term_events(corpus, min_doc_freq=args.min_docs, ngram_max=3)

    # GRANULARITY GATES: candidate themes are idea-level gated phrases (not raw words like "future"/"years").
    df_count = te.groupby("term")["record_id"].nunique().to_dict()
    ceil = DF_CEIL * n
    gated = [t for t, c in df_count.items()
             if c <= ceil and not _is_fragment(t)
             and not (len(t.split()) == 1 and t in GENERIC)
             and not (len(t.split()) > 1 and all(w in GENERIC for w in t.split()))
             and _pmi_ok(t, df_count, n)]
    gated = set(_noun_headed(gated))
    te = te[te["term"].isin(gated)]
    print(f"[phase0] {len(gated):,} gated idea-level candidate themes; labelling...", flush=True)

    pivot = te.groupby(["term", "month"])["record_id"].nunique().unstack(fill_value=0).reindex(columns=grid, fill_value=0)
    denom_arr = denom.reindex(grid).to_numpy()
    rows = []
    for term, n_theme in pivot.iterrows():
        share = pd.Series(np.divide(n_theme.to_numpy(), denom_arr, out=np.zeros(len(grid)), where=denom_arr > 0),
                          index=grid)
        res = durable_breakout_label(share, n_theme.astype(int), cfg, corpus_end=corpus_end)
        t0 = res.get("T0")
        t_dec = grid[grid.index(t0) + 9] if t0 in grid and grid.index(t0) + 9 < len(grid) else None
        rows.append({"theme_id": term, "label": res["label"], "T0": t0, "T_dec": t_dec,
                     "T_peak": res.get("T_peak"), "parent_id": term})
    labels = pd.DataFrame(rows)
    dist = labels["label"].value_counts().to_dict()
    n_pos, n_neg = dist.get("positive", 0), dist.get("hyped_died", 0)
    print(f"[phase0] label distribution: {dist}")
    print(f"[phase0] discriminative pool: {n_pos} durable-breakout positives / {n_neg} hyped-then-died "
          f"(+{dist.get('never', 0)} never, {dist.get('censored', 0)} censored)")

    fizzle_present = negatives.map_fizzles_to_themes(te)
    print(f"[phase0] known-fizzle controls present in (gated) corpus: {sum(fizzle_present.values())}/{len(fizzle_present)}")

    OUT.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(OUT / "labels.parquet")
    summary = {"n_papers": n, "grid": [grid[0], corpus_end], "label_distribution": dist,
               "n_candidate_themes": len(gated), "fizzles_present": fizzle_present}

    disc = labels[labels["label"].isin(("positive", "hyped_died"))].reset_index(drop=True)
    if n_pos >= 3 and n_neg >= 3:
        ecfg = EmergenceConfig()

        def b1(tid, t_dec):
            if not isinstance(t_dec, str):
                return 0.0
            window = month_range_ending(t_dec, ecfg.window_months)
            usage = pivot.loc[tid].reindex(window, fill_value=0).to_numpy()
            return escore(usage.tolist())

        b1_scores = pd.Series({r.theme_id: b1(r.theme_id, r.T_dec) for r in disc.itertuples(index=False)})
        rng = np.random.RandomState(0)
        placeholder = pd.Series(rng.rand(len(disc)), index=disc["theme_id"])
        y = disc.set_index("theme_id")["label"].reindex(disc["theme_id"]).to_numpy()
        for name, sc in [("B1_momentum", b1_scores), ("random_placeholder", placeholder)]:
            s = sc.reindex(disc["theme_id"]).fillna(0.0).to_numpy()
            ap_ = metrics.auprc(s, y)
            _, _, p = metrics.shuffle_null(s, y, B=2000, seed=0)
            print(f"[phase0] {name:18s} AUPRC={ap_:.3f}  shuffle-null p={p:.3f} "
                  f"(prevalence={n_pos / (n_pos + n_neg):.3f})")
            summary.setdefault("scores", {})[name] = {"auprc": float(ap_), "null_p": float(p)}
        tr, ts = splits.temporal_split(disc)
        summary["temporal_split"] = {"train_n": len(tr), "test_n": len(ts),
                                     "train_pos": int((tr["label"] == "positive").sum()),
                                     "test_pos": int((ts["label"] == "positive").sum())}
        print(f"[phase0] temporal split (T_dec @2022-01): train {len(tr)} "
              f"({summary['temporal_split']['train_pos']} pos) / test {len(ts)} "
              f"({summary['temporal_split']['test_pos']} pos)")
    else:
        print(f"[phase0] too few discriminative themes for scoring ({n_pos} pos / {n_neg} neg) -- "
              f"harness ran; widen corpus (--all) for a powered eval.")

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[phase0] wrote {OUT}/labels.parquet + summary.json")


if __name__ == "__main__":
    main()
