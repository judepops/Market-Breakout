"""M1.2 cross-domain transfer: run the EXACT arXiv detector pipeline on bioRxiv. Pre-registered in
Plans/2026-07-06_M1.2-cross-domain-prereg.md. Primary endpoint = twin-safe paired signature-momentum Δ AUPRC
with 95% bootstrap CI. PASS if CI excludes 0 (signature transfers to biology); NULL otherwise (AI-specific).

    ~/miniconda3/envs/marketbrain/bin/python -m experiments.biorxiv_transfer [--min-docs 40]

Nothing is tuned to biology: same gates, same 11 as-of features, same label config, same momentum baseline,
same twin-safe grouping as phase1_secure. Only the corpus changes.
"""
from __future__ import annotations
import argparse
import glob
import json
import pathlib
import warnings

warnings.filterwarnings("ignore")
RAW = pathlib.Path("data/raw/biorxiv_full")
OUT = pathlib.Path("artifacts/biorxiv_transfer")


def _variant_groups(themes):
    """Union-find near-duplicate themes (token-Jaccard>=0.6 or subset containment) -- verbatim from phase1_secure."""
    toks = {t: frozenset(t.split()) for t in themes}
    tl = list(themes); uf = {t: t for t in tl}

    def find(x):
        while uf[x] != x:
            uf[x] = uf[uf[x]]; x = uf[x]
        return x
    for i in range(len(tl)):
        a = tl[i]
        for j in range(i + 1, len(tl)):
            b = tl[j]; ta, tb = toks[a], toks[b]; inter = len(ta & tb)
            if inter and (inter / len(ta | tb) >= 0.6 or ta <= tb or tb <= ta):
                ra, rb = find(a), find(b)
                if ra != rb:
                    uf[ra] = rb
    return {t: find(t) for t in tl}


def load_biorxiv():
    import pandas as pd
    files = sorted(glob.glob(str(RAW / "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no bioRxiv parquets under {RAW} -- run experiments.fetch_biorxiv first")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.drop_duplicates("record_id").reset_index(drop=True)


def load_source(source):
    """source='biorxiv' -> bioRxiv corpus; otherwise a comma-list of arXiv categories -> load_corpus."""
    if source == "biorxiv":
        return load_biorxiv(), "biorxiv"
    from marketbrain.corpus.load_full import load_corpus
    cats = source.split(",")
    return load_corpus(cats), source


def main(argv=None):
    import numpy as np
    import pandas as pd
    from marketbrain.eval.corpus_inputs import cs_denominator, corpus_term_events
    from marketbrain.eval.label import EmergenceLabelConfig, durable_breakout_label
    from marketbrain.eval import metrics, splits
    from marketbrain.eval.breakout_signature import build_feature_matrix, fit_predict_oof, SIGNATURE_FEATURES
    from marketbrain.markets.granularity_gates import _pmi_ok, _noun_headed, _is_fragment, GENERIC, DF_CEIL
    from marketbrain.features.emergence import EmergenceConfig, escore, month_range_ending

    ap = argparse.ArgumentParser()
    ap.add_argument("--min-docs", type=int, default=40)   # same as arXiv phase1 -- NOT tuned to the domain
    ap.add_argument("--source", default="biorxiv", help="'biorxiv' or a comma-list of arXiv cats, e.g. quant-ph")
    args = ap.parse_args(argv)
    cfg = EmergenceLabelConfig()

    corpus, src = load_source(args.source)
    denom = cs_denominator(corpus)
    grid = list(denom.index)
    corpus_end = grid[-1]
    n = len(corpus)
    print(f"[bx] {src} corpus: {n:,} papers, grid {grid[0]}..{corpus_end}", flush=True)

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
    print(f"[bx] gated themes: {len(gated)}", flush=True)

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
    print(f"[bx] discriminative pool: {n_pos} positive / {len(disc) - n_pos} hyped-died "
          f"(base rate {n_pos / max(len(disc),1):.3f})", flush=True)
    if len(disc) < 80:
        print(f"[bx] WARNING: n={len(disc)} < 80 -> underpowered per pre-reg; reporting as inconclusive.", flush=True)

    themes = {t: {t} for t in disc["theme_id"]}
    term_to_records = te.groupby("term")["record_id"].apply(set).to_dict()
    records_indexed = corpus.set_index("record_id")[["month", "authors", "categories", "comment", "journal_ref"]]
    print(f"[bx] building {len(disc)} as-of feature rows...", flush=True)
    feats = build_feature_matrix(themes, term_to_records, records_indexed, disc, denom).merge(disc[["theme_id"]], on="theme_id")

    ecfg = EmergenceConfig()
    def b1(tid, t_dec):
        usage = pivot.loc[tid].reindex(month_range_ending(t_dec, ecfg.window_months), fill_value=0).to_numpy()
        return escore(usage.tolist())
    b1_scores = pd.Series({r.theme_id: b1(r.theme_id, r.T_dec) for r in disc.itertuples(index=False)})
    rng = np.random.RandomState(0)
    rand_scores = pd.Series(rng.rand(len(disc)), index=disc["theme_id"])
    y_all = disc.set_index("theme_id")["label"]

    # twin-safe variant grouping (the make-or-break split)
    groups = _variant_groups(list(disc["theme_id"]))
    disc_g = disc.copy(); disc_g["parent_id"] = disc_g["theme_id"].map(groups)
    ng = len(set(groups.values()))
    print(f"[bx] twin-safe: {len(disc)} themes -> {ng} variant-groups", flush=True)

    def score(name, s_series, ids):
        s = s_series.reindex(ids).fillna(0.0).to_numpy(); yy = y_all.reindex(ids).to_numpy()
        ap_ = metrics.auprc(s, yy); _, _, p = metrics.shuffle_null(s, yy, B=2000, seed=0)
        return {"auprc": float(ap_), "null_p": float(p)}

    summary = {"corpus": src, "n_papers": n, "n_gated": len(gated), "n_disc": len(disc),
               "n_pos": n_pos, "n_neg": len(disc) - n_pos, "base_rate": n_pos / max(len(disc), 1), "n_groups": ng}

    for split_name, dd in [("ungrouped", disc), ("variant-grouped (twin-safe, PRIMARY)", disc_g)]:
        folds = splits.grouped_folds(dd, k=5)
        sig = fit_predict_oof(feats, folds)
        ids = dd["theme_id"]
        yv = y_all.reindex(ids).to_numpy()
        sa = sig.reindex(ids).fillna(0.0).to_numpy(); ba = b1_scores.reindex(ids).fillna(0.0).to_numpy()
        d, lo, hi, p1 = metrics.paired_delta_ci(sa, ba, yv, B=2000, seed=0)
        verdict = "TRANSFERS (CI excludes 0)" if lo > 0 else ("signature WORSE" if hi < 0 else "NULL (CI includes 0)")
        block = {"signature": score("signature", sig, ids), "momentum": score("momentum", b1_scores, ids),
                 "random": score("random", rand_scores, ids),
                 "paired_delta": {"delta": float(d), "ci_lo": float(lo), "ci_hi": float(hi), "p_one_sided": float(p1)},
                 "verdict": verdict}
        summary[split_name] = block
        print(f"\n[bx] {split_name}:")
        print(f"   signature AUPRC={block['signature']['auprc']:.3f}  momentum={block['momentum']['auprc']:.3f}  random={block['random']['auprc']:.3f}")
        print(f"   paired Δ = {d:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  P(mom>=sig)={p1:.3f}  -> {verdict}", flush=True)

    # diagnostic: which features carry it in biology (univariate AUPRC, variant-grouped OOF)
    print("\n[bx] mechanism (univariate AUPRC per feature — is it still persistence + author-breadth?):", flush=True)
    ids_g = disc_g["theme_id"]; yv_g = y_all.reindex(ids_g).to_numpy()
    uni = {}
    for f in SIGNATURE_FEATURES:
        col = feats.set_index("theme_id")[f].reindex(ids_g).fillna(0.0).to_numpy()
        # univariate ranking: AUPRC of the raw feature and its negation, take the stronger direction
        uni[f] = max(metrics.auprc(col, yv_g), metrics.auprc(-col, yv_g))
    for f, a in sorted(uni.items(), key=lambda kv: -kv[1])[:5]:
        print(f"   {f:22s} univariate AUPRC={a:.3f}", flush=True)
    summary["univariate_top"] = dict(sorted(uni.items(), key=lambda kv: -kv[1]))

    out_dir = OUT if src == "biorxiv" else OUT.parent / f"transfer_{src.replace(',', '_')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out_dir / "features.parquet")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    prim = summary["variant-grouped (twin-safe, PRIMARY)"]["verdict"]
    print(f"\n[bx] PRIMARY ENDPOINT: {prim}")
    print(f"[bx] (arXiv AI/ML reference: signature 0.66 vs momentum 0.43, Δ+0.23). wrote {OUT}/summary.json", flush=True)


if __name__ == "__main__":
    main()
