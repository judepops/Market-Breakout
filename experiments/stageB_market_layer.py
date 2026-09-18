"""Stage B-B granularity fix: a permissive MARKET-LAYER theme set for the money bridge.

    python -m experiments.stageB_market_layer --cats "cs.LG,cs.CL,cs.AI,cs.RO,cs.SE,cs.DC,stat.ML"

Root cause (experiments.stageB_trace_terms): the detector's PMI gate rejects real market terms whose tail
word is ubiquitous ("diffusion MODELS", "graph neural NETWORKS", "federated LEARNING", "foundation MODELS"),
and the 8% DF ceiling rejects the very frequent ones ("language models"). Many are already labelled positive.
For the FINANCE bridge we want these markets, so this layer:
  - keeps fragment + noun-head + all-generic filters, but SKIPS PMI and raises the DF ceiling to 0.5;
  - keeps EVERY gated term with a valid emergence birth (any durable-breakout label -- censored/never included,
    since the hottest markets like LLMs are censored by the 2-yr persistence window);
  - signal = P(durable) from the Stage-A signature TRAINED on the {positive,hyped_died} subset and PREDICTED
    for every market term (leakage-safe as-of T_dec features). This tests "does predicted research-durability
    translate to capital?" for the actual markets, LLMs/diffusion/GNN included.
Output: artifacts/stageB/themes_market.parquet (drop-in for experiments.stageB_bridge via THEMES_FILE).
"""
from __future__ import annotations
import argparse
import pathlib
import warnings

warnings.filterwarnings("ignore")
OUT = pathlib.Path("artifacts/stageB")


def main(argv=None):
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier
    from marketbrain.corpus.load_full import load_corpus
    from marketbrain.eval.corpus_inputs import cs_denominator, corpus_term_events
    from marketbrain.eval.label import EmergenceLabelConfig, durable_breakout_label
    from marketbrain.eval.breakout_signature import build_feature_matrix, SIGNATURE_FEATURES
    from marketbrain.markets.granularity_gates import _noun_headed, _is_fragment, GENERIC
    from marketbrain.features.emergence import EmergenceConfig, escore, month_range_ending

    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", default="cs.LG,cs.CL,cs.AI,cs.RO,cs.SE,cs.DC,stat.ML")
    ap.add_argument("--min-docs", type=int, default=40)
    ap.add_argument("--df-ceil", type=float, default=0.5)
    ap.add_argument("--out", default="themes_market.parquet")
    args = ap.parse_args(argv)
    cfg = EmergenceLabelConfig()

    corpus = load_corpus(args.cats.split(","))
    denom = cs_denominator(corpus)
    grid = list(denom.index)
    corpus_end, n = grid[-1], len(corpus)
    print(f"[market] {n:,} papers ({args.cats}); permissive gate (skip PMI, ceil={args.df_ceil})...", flush=True)
    te = corpus_term_events(corpus, min_doc_freq=args.min_docs, ngram_max=3)
    df_count = te.groupby("term")["record_id"].nunique().to_dict()
    ceil = args.df_ceil * n
    gated = [t for t, c in df_count.items()                       # NOTE: PMI intentionally omitted
             if c <= ceil and not _is_fragment(t)
             and not (len(t.split()) == 1 and t in GENERIC)
             and not (len(t.split()) > 1 and all(w in GENERIC for w in t.split()))]
    gated = set(_noun_headed(gated))
    te = te[te["term"].isin(gated)]
    pivot = te.groupby(["term", "month"])["record_id"].nunique().unstack(fill_value=0).reindex(columns=grid, fill_value=0)
    denom_arr = denom.reindex(grid).to_numpy()

    rows = []
    for term, n_theme in pivot.iterrows():
        share = pd.Series(np.divide(n_theme.to_numpy(), denom_arr, out=np.zeros(len(grid)), where=denom_arr > 0), index=grid)
        res = durable_breakout_label(share, n_theme.astype(int), cfg, corpus_end=corpus_end)
        t0 = res.get("T0")
        if t0 is None or t0 not in grid or grid.index(t0) + 9 >= len(grid):
            continue                                              # need a valid as-of decision point T_dec
        t_dec = grid[grid.index(t0) + 9]
        rows.append({"theme_id": term, "label": res["label"], "T0": t0, "T_dec": t_dec, "parent_id": term})
    allt = pd.DataFrame(rows)
    lab_counts = allt["label"].value_counts().to_dict()
    print(f"[market] {len(allt)} market terms with a valid T_dec; labels={lab_counts}", flush=True)

    themes = {t: {t} for t in allt["theme_id"]}
    term_to_records = te.groupby("term")["record_id"].apply(set).to_dict()
    records_indexed = corpus.set_index("record_id")[["month", "authors", "categories", "comment", "journal_ref"]]
    feats = build_feature_matrix(themes, term_to_records, records_indexed, allt, denom)
    ids = list(feats["theme_id"])

    # train the durable-breakout signature on the {positive,hyped_died} subset, PREDICT P(durable) for ALL terms
    disc = feats[feats["label"].isin(("positive", "hyped_died"))]
    ytr = (disc["label"] == "positive").astype(int).to_numpy()
    clf = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05,
                                         min_samples_leaf=10, random_state=0).fit(disc[SIGNATURE_FEATURES].to_numpy(), ytr)
    p_dur = pd.Series(clf.predict_proba(feats[SIGNATURE_FEATURES].to_numpy())[:, 1], index=ids)

    ecfg = EmergenceConfig()
    tdec = allt.set_index("theme_id")["T_dec"]
    mom = pd.Series({t: escore(pivot.loc[t].reindex(
        month_range_ending(tdec[t], ecfg.window_months), fill_value=0).to_numpy().tolist()) for t in ids})

    title_by_rec = corpus.set_index("record_id")["title"].to_dict()
    month_by_rec = corpus.set_index("record_id")["month"].to_dict()
    te_by_rec = te.groupby("record_id")["term"].apply(list).to_dict()
    lab_by = allt.set_index("theme_id")["label"]
    out_rows = []
    for t in ids:
        td = tdec[t]
        recs = [r for r in term_to_records.get(t, set()) if month_by_rec.get(r, "9999") <= td]
        co = {}
        for r in recs:
            for u in te_by_rec.get(r, []):
                if u != t:
                    co[u] = co.get(u, 0) + 1
        top_terms = [w for w, _ in sorted(co.items(), key=lambda kv: kv[1], reverse=True)[:6]]
        titles = [title_by_rec.get(r, "") for r in recs if title_by_rec.get(r, "")][:3]
        theme_text = t + ". " + "; ".join(top_terms) + (". e.g. " + " | ".join(titles) if titles else "")
        out_rows.append({"theme_id": t, "label": str(lab_by[t]), "T0": allt.set_index("theme_id").loc[t, "T0"],
                         "T_dec": td, "sig_score": float(p_dur[t]), "momentum": float(mom[t]),
                         "n_papers_asof": len(recs), "top_terms": ", ".join(top_terms), "theme_text": theme_text})
    out = pd.DataFrame(out_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT / args.out)
    print(f"[market] wrote {OUT}/{args.out}: {len(out)} market themes", flush=True)
    # confirm the mega-markets are now present
    for p in ["large language", "language models", "diffusion models", "graph neural", "federated learning",
              "foundation models", "generative adversarial", "transformer", "in-context", "vision-language"]:
        m = out[out["theme_id"].str.contains(p, regex=False)]
        if len(m):
            r = m.iloc[0]
            print(f"   + {r['theme_id'][:30]:30s} label={r['label']:9s} sig={r['sig_score']:.2f}", flush=True)


if __name__ == "__main__":
    main()
