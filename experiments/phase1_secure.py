"""Phase-1 SECURE (Stage A3/A5): harden the +0.168 detector edge before anything is built on it.

    python -m experiments.phase1_secure [--all] [--min-docs N]

Runs, in one pass over the corpus:
  1. TWIN-LEAKAGE TEST (make-or-break): re-score the OOF with near-duplicate themes GROUPED into one fold
     (opd / on-policy distillation / ... cannot straddle train/test). Does the signature-momentum delta CI
     still exclude 0?  -> buys "generalises to unseen THEME-GROUPS", NOT forward-in-time.
  2. ABLATION: per-feature univariate AUPRC + drop-one-out from the full model. Which feature carries the edge?
  3. SPREAD SANITY (protects the cross-domain-spread money-quote): (a) category coverage is time-uniform
     (2012->2026), so spread is not a "categories I'd downloaded by T" artifact; (b) spread's signal survives
     controlling for theme volume/persistence (real diffusion, not "common phrases touch more categories").
  4. NEGATIVE CONTROLS: known hyped-then-died themes must NOT label positive (falsification).
  5. TEMPORAL (forward-prediction): reported honestly as underpowered -- family/group holdouts do NOT stand in.
"""
from __future__ import annotations
import argparse
import json
import pathlib
import warnings

warnings.filterwarnings("ignore")
OUT = pathlib.Path("artifacts/phase1_secure")


def _variant_groups(themes):
    """Union-find themes into near-duplicate groups by token-Jaccard>=0.6 or word-level subset containment."""
    toks = {t: frozenset(t.split()) for t in themes}
    tl = list(themes)
    uf = {t: t for t in tl}

    def find(x):
        while uf[x] != x:
            uf[x] = uf[uf[x]]
            x = uf[x]
        return x

    for i in range(len(tl)):
        a = tl[i]
        for j in range(i + 1, len(tl)):
            b = tl[j]
            ta, tb = toks[a], toks[b]
            inter = len(ta & tb)
            if inter and (inter / len(ta | tb) >= 0.6 or ta <= tb or tb <= ta):
                ra, rb = find(a), find(b)
                if ra != rb:
                    uf[ra] = rb
    return {t: find(t) for t in tl}


def _oof(feats, folds, cols):
    """Out-of-fold P(durable) from gradient-boosted trees over the given folds, using only `cols`."""
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier
    disc = feats[feats["label"].isin(("positive", "hyped_died"))]
    out = {}
    for tr_df, te_df in folds:
        tr = disc[disc["theme_id"].isin(tr_df["theme_id"])]
        te = disc[disc["theme_id"].isin(te_df["theme_id"])]
        if tr.empty or te.empty or tr["label"].nunique() < 2:
            continue
        yv = (tr["label"] == "positive").astype(int).to_numpy()
        clf = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05,
                                             min_samples_leaf=10, random_state=0).fit(tr[cols].to_numpy(), yv)
        for tid, p in zip(te["theme_id"], clf.predict_proba(te[cols].to_numpy())[:, 1]):
            out[tid] = p
    return pd.Series(out)


def main(argv=None):
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from marketbrain.corpus.load_full import load_corpus
    from marketbrain.eval.corpus_inputs import cs_denominator, corpus_term_events
    from marketbrain.eval.label import EmergenceLabelConfig, durable_breakout_label
    from marketbrain.eval import metrics, splits, negatives
    from marketbrain.eval.breakout_signature import build_feature_matrix, SIGNATURE_FEATURES
    from marketbrain.markets.granularity_gates import _pmi_ok, _noun_headed, _is_fragment, GENERIC, DF_CEIL
    from marketbrain.features.emergence import EmergenceConfig, escore, month_range_ending

    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--min-docs", type=int, default=40)
    args = ap.parse_args(argv)
    cfg = EmergenceLabelConfig()

    corpus = load_corpus(None if args.all else ["cs.AI"])
    denom = cs_denominator(corpus)
    grid = list(denom.index)
    corpus_end, n = grid[-1], len(corpus)
    print(f"[secure] {n:,} papers, grid {grid[0]}..{corpus_end}; gating + labelling...", flush=True)
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
        share = pd.Series(np.divide(n_theme.to_numpy(), denom_arr, out=np.zeros(len(grid)), where=denom_arr > 0), index=grid)
        res = durable_breakout_label(share, n_theme.astype(int), cfg, corpus_end=corpus_end)
        t0 = res.get("T0")
        t_dec = grid[grid.index(t0) + 9] if t0 in grid and grid.index(t0) + 9 < len(grid) else None
        rows.append({"theme_id": term, "label": res["label"], "T0": t0, "T_dec": t_dec, "parent_id": term})
    labels = pd.DataFrame(rows)
    disc = labels[labels["label"].isin(("positive", "hyped_died"))].dropna(subset=["T_dec"]).reset_index(drop=True)
    n_pos = int((disc["label"] == "positive").sum())
    print(f"[secure] discriminative pool: {n_pos} positives / {len(disc) - n_pos} hyped-died", flush=True)

    themes = {t: {t} for t in disc["theme_id"]}
    term_to_records = te.groupby("term")["record_id"].apply(set).to_dict()
    records_indexed = corpus.set_index("record_id")[["month", "authors", "categories", "comment", "journal_ref"]]
    print(f"[secure] building {len(disc)} feature rows...", flush=True)
    feats = build_feature_matrix(themes, term_to_records, records_indexed, disc, denom)
    ids = feats["theme_id"]
    y_all = feats.set_index("theme_id")["label"]
    ecfg = EmergenceConfig()
    b1 = pd.Series({t: escore(pivot.loc[t].reindex(
        month_range_ending(disc.set_index("theme_id").loc[t, "T_dec"], ecfg.window_months), fill_value=0).to_numpy().tolist())
        for t in ids})
    ba = b1.reindex(ids).fillna(0.0).to_numpy()
    yv_all = y_all.reindex(ids).to_numpy()
    summary = {"n_pos": n_pos, "n_neg": len(disc) - n_pos, "momentum_auprc": round(metrics.auprc(ba, yv_all), 4)}

    # ---- 1. TWIN-LEAKAGE TEST -------------------------------------------------
    groups = _variant_groups(list(disc["theme_id"]))
    ng = len(set(groups.values()))
    disc_g = disc.copy()
    disc_g["parent_id"] = disc_g["theme_id"].map(groups)
    print(f"\n[secure] 1. TWIN-LEAKAGE: {len(disc)} themes -> {ng} variant-groups (near-dup merged for folds)")
    folds_by = {"ungrouped (parent=theme)": splits.grouped_folds(disc, k=5),
                "variant-grouped (parent=group)": splits.grouped_folds(disc_g, k=5)}
    for tag, folds in folds_by.items():
        oof = _oof(feats, folds, SIGNATURE_FEATURES)
        sa = oof.reindex(ids).fillna(0.0).to_numpy()
        d, lo, hi, p1 = metrics.paired_delta_ci(sa, ba, yv_all, B=2000, seed=0)
        sep = "SEPARATED" if (lo > 0 or hi < 0) else "spans 0"
        print(f"[secure]   {tag:34s} sig={metrics.auprc(sa, yv_all):.3f}  mom={metrics.auprc(ba, yv_all):.3f}  "
              f"delta={d:+.3f} CI[{lo:+.3f},{hi:+.3f}] p={p1:.3f} -> {sep}")
        summary[tag.split()[0]] = {"sig_auprc": round(metrics.auprc(sa, yv_all), 4), "delta": round(d, 4),
                                   "ci": [round(lo, 4), round(hi, 4)], "separated": sep == "SEPARATED"}

    # ---- 2. ABLATION (variant-grouped) ---------------------------------------
    folds_g = folds_by["variant-grouped (parent=group)"]
    full = metrics.auprc(_oof(feats, folds_g, SIGNATURE_FEATURES).reindex(ids).fillna(0).to_numpy(), yv_all)
    print(f"\n[secure] 2. ABLATION (variant-grouped OOF, full={full:.3f}); univariate | drop-one-out:")
    abl = {}
    for f in SIGNATURE_FEATURES:
        uni = metrics.auprc(feats.set_index("theme_id")[f].reindex(ids).fillna(0).to_numpy(), yv_all)
        dropped = metrics.auprc(_oof(feats, folds_g, [c for c in SIGNATURE_FEATURES if c != f]).reindex(ids).fillna(0).to_numpy(), yv_all)
        abl[f] = {"univariate": round(uni, 3), "drop_one_out": round(dropped, 3), "hurt": round(full - dropped, 3)}
        print(f"[secure]   {f:22s} univariate={uni:.3f}  drop-one-out={dropped:.3f} (hurt {full - dropped:+.3f})")
    summary["ablation"] = abl

    # ---- 3. SPREAD SANITY (is cross_domain_spread real diffusion, or a cross-listing/era artifact?) ----
    ncats = corpus["categories"].apply(lambda c: len(c) if c is not None else 0)
    cats_by_year = ncats.groupby(corpus["month"].str[:4]).mean()
    y0, y1 = cats_by_year.get("2012", float("nan")), cats_by_year.get("2026", float("nan"))
    infl = y1 / y0 if y0 else float("nan")
    print(f"\n[secure] 3. SPREAD SANITY: avg categories/PAPER by year 2012={y0:.2f} -> 2026={y1:.2f} "
          f"(x{infl:.2f} cross-listing inflation -- the confound mechanism)")
    tdec_year = disc.set_index("theme_id")["T_dec"].str[:4].astype(float)
    X = feats.set_index("theme_id")[["cross_domain_spread", "volume_share", "persistence_months"]].reindex(ids)
    X["era"] = tdec_year.reindex(ids).to_numpy()
    Xs = (X - X.mean()) / (X.std() + 1e-9)
    yb = (y_all.reindex(ids) == "positive").astype(int).to_numpy()
    coef = LogisticRegression(max_iter=1000).fit(Xs.to_numpy(), yb).coef_[0]
    spread_c = float(coef[0])
    print("[secure]   logistic(label ~ spread + volume + persistence + era) std coefs: "
          + ", ".join(f"{c}={v:+.2f}" for c, v in zip(X.columns, coef)))
    print(f"[secure]   -> spread coef with volume+era controlled = {spread_c:+.2f} "
          f"({'positive independent diffusion signal' if spread_c > 0.15 else 'NOT a positive carrier (weak/negative)'})")
    summary["spread_sanity"] = {"crosslist_inflation": round(float(infl), 2),
                                "logit_coefs": dict(zip(list(X.columns), [round(float(c), 3) for c in coef]))}

    # ---- 4. NEGATIVE CONTROLS -------------------------------------------------
    lab_by = labels.set_index("theme_id")["label"].to_dict()
    hits = {fz.name: {t: lab_by[t] for t in fz.terms if t in lab_by} for fz in negatives.KNOWN_FIZZLE}
    hits = {k: v for k, v in hits.items() if v}
    n_pos_fizzle = sum(1 for v in hits.values() for lb in v.values() if lb == "positive")
    print(f"\n[secure] 4. NEGATIVE CONTROLS: {len(hits)} known-fizzles present as gated themes; "
          f"{n_pos_fizzle} labelled 'positive' (want 0). sample: {dict(list(hits.items())[:5])}")
    summary["negative_controls"] = {"n_fizzles_present": len(hits), "n_labelled_positive": n_pos_fizzle}

    # ---- 5. TEMPORAL (forward) ------------------------------------------------
    tr, ts = splits.temporal_split(disc)
    tp = int((ts["label"] == "positive").sum())
    print(f"\n[secure] 5. TEMPORAL (FORWARD): train {len(tr)} / test {len(ts)} ({tp} pos) -- "
          f"{'UNDERPOWERED; forward-prediction NOT established (group/family holdouts do NOT substitute)' if len(ts) < 40 else 'powered'}")
    summary["temporal"] = {"test_n": len(ts), "test_pos": tp, "powered": len(ts) >= 40}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[secure] wrote {OUT}/summary.json")


if __name__ == "__main__":
    main()
