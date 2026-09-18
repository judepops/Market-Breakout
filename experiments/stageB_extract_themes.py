"""Stage B-B, step 2: extract the +0.20 candidate corpus's detector themes for the theme->money bridge.

    python -m experiments.stageB_extract_themes --cats "cs.LG,cs.CL,cs.AI,cs.RO,cs.SE,cs.DC,stat.ML"

Runs the exact detector pipeline (gate -> durable-breakout label -> as-of signature OOF + momentum) on the
chosen candidate corpus, and for each discriminative theme saves a leakage-safe SEMANTIC TEXT (theme name +
top co-occurring gated terms + a few representative paper titles, all from papers month <= T_dec) for
embedding against company descriptions. Output: artifacts/stageB/themes.parquet.
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
    from marketbrain.corpus.load_full import load_corpus
    from marketbrain.eval.corpus_inputs import cs_denominator, corpus_term_events
    from marketbrain.eval.label import EmergenceLabelConfig, durable_breakout_label
    from marketbrain.eval.breakout_signature import build_feature_matrix, fit_predict_oof, SIGNATURE_FEATURES
    from marketbrain.eval import splits
    from marketbrain.markets.granularity_gates import _pmi_ok, _noun_headed, _is_fragment, GENERIC, DF_CEIL
    from marketbrain.features.emergence import EmergenceConfig, escore, month_range_ending
    from experiments.phase1_secure import _variant_groups

    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", default="cs.LG,cs.CL,cs.AI,cs.RO,cs.SE,cs.DC,stat.ML")
    ap.add_argument("--min-docs", type=int, default=40)
    ap.add_argument("--df-ceil", type=float, default=DF_CEIL,
                    help="doc-freq ceiling (0.08 = Stage-A detector; raise ~0.35 for a coarse market layer "
                         "that admits LLMs/diffusion/RL)")
    ap.add_argument("--out", default="themes.parquet")
    args = ap.parse_args(argv)
    cfg = EmergenceLabelConfig()

    cats = args.cats.split(",")
    corpus = load_corpus(cats)
    denom = cs_denominator(corpus)
    grid = list(denom.index)
    corpus_end, n = grid[-1], len(corpus)
    print(f"[themes] {n:,} papers ({args.cats}); grid {grid[0]}..{corpus_end}; gating+labelling...", flush=True)
    te = corpus_term_events(corpus, min_doc_freq=args.min_docs, ngram_max=3)
    df_count = te.groupby("term")["record_id"].nunique().to_dict()
    ceil = args.df_ceil * n
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
        share = pd.Series(np.divide(n_theme.to_numpy(), denom_arr, out=np.zeros(len(grid)), where=denom_arr > 0), index=grid)
        res = durable_breakout_label(share, n_theme.astype(int), cfg, corpus_end=corpus_end)
        t0 = res.get("T0")
        t_dec = grid[grid.index(t0) + 9] if t0 in grid and grid.index(t0) + 9 < len(grid) else None
        rows.append({"theme_id": term, "label": res["label"], "T0": t0, "T_dec": t_dec, "parent_id": term})
    labels = pd.DataFrame(rows)
    disc = labels[labels["label"].isin(("positive", "hyped_died"))].dropna(subset=["T_dec"]).reset_index(drop=True)
    n_pos = int((disc["label"] == "positive").sum())
    print(f"[themes] discriminative pool: {n_pos} positives / {len(disc) - n_pos} hyped-died", flush=True)

    themes = {t: {t} for t in disc["theme_id"]}
    term_to_records = te.groupby("term")["record_id"].apply(set).to_dict()
    records_indexed = corpus.set_index("record_id")[["month", "authors", "categories", "comment", "journal_ref"]]
    feats = build_feature_matrix(themes, term_to_records, records_indexed, disc, denom)
    ids = list(feats["theme_id"])

    # as-of signature (variant-grouped OOF, twin-safe) + momentum
    groups = _variant_groups(list(disc["theme_id"]))
    disc_g = disc.copy(); disc_g["parent_id"] = disc_g["theme_id"].map(groups)
    sig = fit_predict_oof(feats, splits.grouped_folds(disc_g, k=5, seed=0)).reindex(ids).fillna(0.0)
    ecfg = EmergenceConfig()
    tdec = disc.set_index("theme_id")["T_dec"]
    mom = pd.Series({t: escore(pivot.loc[t].reindex(
        month_range_ending(tdec[t], ecfg.window_months), fill_value=0).to_numpy().tolist()) for t in ids})

    # leakage-safe semantic text per theme: name + top co-occurring gated terms + representative titles (month <= T_dec)
    title_by_rec = corpus.set_index("record_id")["title"].to_dict()
    month_by_rec = corpus.set_index("record_id")["month"].to_dict()
    te_by_rec = te.groupby("record_id")["term"].apply(list).to_dict()
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
        titles = [title_by_rec.get(r, "") for r in recs]
        titles = [x for x in titles if x][:3]
        theme_text = t + ". " + "; ".join(top_terms)
        if titles:
            theme_text += ". e.g. " + " | ".join(titles)
        out_rows.append({"theme_id": t, "label": disc.set_index("theme_id").loc[t, "label"],
                         "T0": disc.set_index("theme_id").loc[t, "T0"], "T_dec": td,
                         "sig_score": float(sig[t]), "momentum": float(mom[t]),
                         "n_papers_asof": len(recs), "top_terms": ", ".join(top_terms), "theme_text": theme_text})
    out = pd.DataFrame(out_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT / args.out)
    print(f"[themes] wrote {OUT}/{args.out}: {len(out)} themes (df_ceil={args.df_ceil})", flush=True)
    print("[themes] sample durable-positive theme_texts:", flush=True)
    for _, r in out[out["label"] == "positive"].sort_values("sig_score", ascending=False).head(6).iterrows():
        print(f"   [{r['sig_score']:.2f}] {r['theme_text'][:160]}", flush=True)


if __name__ == "__main__":
    main()
