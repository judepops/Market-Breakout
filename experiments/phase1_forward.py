"""Stage A forward (real-time) validation: rolling-origin, strict label-maturation. Pre-reg:
`Plans/2026-07-01_forward-validation.md`.

    python -m experiments.phase1_forward

At each origin C: train the signature ONLY on themes whose durable-breakout label had matured by C
(T_peak + persist_hi ≤ C, so the ≤C label == the full label) and whose decision point T_dec ≤ C; predict on
themes decided in (C, C+12mo]; evaluate against their true (observable) label. Twin-safe (no variant-group
straddles). Pool forward predictions across origins -> forward AUPRC(signature) vs momentum + paired-delta CI
+ shuffle null. Reports per-origin deltas + the weaker single-cutoff temporal holdout for contrast.
"""
from __future__ import annotations
import warnings

warnings.filterwarnings("ignore")
ORIGINS = ["2018-06", "2019-06", "2020-06", "2021-06", "2022-06"]
TEST_WINDOW = 12
CORPUS_CATS = ["cs.LG", "cs.CL", "cs.AI", "cs.RO", "cs.SE", "cs.DC", "stat.ML"]


def _shift(month, n):
    y, m = int(month[:4]), int(month[5:7])
    t = y * 12 + (m - 1) + n
    return f"{t // 12:04d}-{t % 12 + 1:02d}"


def main():
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier
    from marketbrain.corpus.load_full import load_corpus
    from marketbrain.eval.corpus_inputs import cs_denominator, corpus_term_events
    from marketbrain.eval.label import EmergenceLabelConfig, durable_breakout_label
    from marketbrain.eval.breakout_signature import build_feature_matrix, SIGNATURE_FEATURES
    from marketbrain.eval import metrics
    from marketbrain.markets.granularity_gates import _pmi_ok, _noun_headed, _is_fragment, GENERIC, DF_CEIL
    from marketbrain.features.emergence import EmergenceConfig, escore, month_range_ending
    from experiments.phase1_secure import _variant_groups

    cfg = EmergenceLabelConfig()
    corpus = load_corpus(CORPUS_CATS)
    denom = cs_denominator(corpus)
    grid = list(denom.index)
    corpus_end, n = grid[-1], len(corpus)
    print(f"[forward] {n:,} papers ({','.join(CORPUS_CATS)}); grid {grid[0]}..{corpus_end}; gating+labelling...", flush=True)
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

    rows = []
    for term, n_theme in pivot.iterrows():
        share = pd.Series(np.divide(n_theme.to_numpy(), denom_arr, out=np.zeros(len(grid)), where=denom_arr > 0), index=grid)
        res = durable_breakout_label(share, n_theme.astype(int), cfg, corpus_end=corpus_end)
        t0 = res.get("T0")
        t_dec = grid[grid.index(t0) + 9] if t0 in grid and grid.index(t0) + 9 < len(grid) else None
        rows.append({"theme_id": term, "label": res["label"], "T0": t0, "T_dec": t_dec, "T_peak": res.get("T_peak")})
    labels = pd.DataFrame(rows)
    disc = labels[labels["label"].isin(("positive", "hyped_died"))].dropna(subset=["T_dec"]).reset_index(drop=True)
    disc["parent_id"] = disc["theme_id"]
    # maturation month = end of the persistence window (T_peak + persist_hi); label is final once this <= C
    disc["mature_m"] = disc["T_peak"].apply(lambda tp: _shift(tp, cfg.persist_hi) if isinstance(tp, str) else None)
    n_pos = int((disc["label"] == "positive").sum())
    print(f"[forward] discriminative pool: {n_pos} positives / {len(disc) - n_pos} hyped-died", flush=True)

    themes = {t: {t} for t in disc["theme_id"]}
    term_to_records = te.groupby("term")["record_id"].apply(set).to_dict()
    records_indexed = corpus.set_index("record_id")[["month", "authors", "categories", "comment", "journal_ref"]]
    feats = build_feature_matrix(themes, term_to_records, records_indexed, disc, denom).set_index("theme_id")
    ecfg = EmergenceConfig()
    mom = {t: escore(pivot.loc[t].reindex(month_range_ending(td, ecfg.window_months), fill_value=0).to_numpy().tolist())
           for t, td in zip(disc["theme_id"], disc["T_dec"])}
    groups = _variant_groups(list(disc["theme_id"]))
    disc["grp"] = disc["theme_id"].map(groups)
    y = (disc.set_index("theme_id")["label"] == "positive").astype(int)

    # ---- rolling-origin forward prediction ----
    pooled = {}   # theme_id -> (sig_pred, is_pos)
    print(f"\n[forward] rolling origins (train: matured & T_dec<=C; test: T_dec in (C,C+{TEST_WINDOW}mo]):")
    for C in ORIGINS:
        Cend = _shift(C, TEST_WINDOW)
        test = disc[(disc["T_dec"] > C) & (disc["T_dec"] <= Cend)]
        test = test[~test["theme_id"].isin(pooled)]                       # each theme tested once
        train = disc[(disc["T_dec"] <= C) & (disc["mature_m"].notna()) & (disc["mature_m"] <= C)]
        test_grps = set(test["grp"])
        train = train[~train["grp"].isin(test_grps)]                      # twin-safe
        if len(test) == 0 or train["label"].nunique() < 2 or len(train) < 15:
            print(f"[forward]   C={C}: train={len(train)} test={len(test)}  -- skipped (thin)", flush=True)
            continue
        clf = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05, min_samples_leaf=10,
                                             random_state=0).fit(feats.loc[train["theme_id"], SIGNATURE_FEATURES].to_numpy(),
                                                                 y.loc[train["theme_id"]].to_numpy())
        p = clf.predict_proba(feats.loc[test["theme_id"], SIGNATURE_FEATURES].to_numpy())[:, 1]
        yv = y.loc[test["theme_id"]].to_numpy(); mv = np.array([mom[t] for t in test["theme_id"]])
        for t, pv in zip(test["theme_id"], p):
            pooled[t] = pv
        if yv.min() != yv.max():
            ds = metrics.auprc(p, yv); dm = metrics.auprc(mv, yv)
            print(f"[forward]   C={C}: train={len(train):3d} test={len(test):3d} ({int(yv.sum())} pos)  "
                  f"sig={ds:.3f} mom={dm:.3f} delta={ds - dm:+.3f}", flush=True)
        else:
            print(f"[forward]   C={C}: train={len(train):3d} test={len(test):3d} ({int(yv.sum())} pos)  -- single-class test", flush=True)

    # ---- pooled forward metrics ----
    tids = list(pooled)
    sa = np.array([pooled[t] for t in tids])
    ba = np.array([mom[t] for t in tids])
    yv = y.loc[tids].to_numpy()
    print(f"\n[forward] === POOLED FORWARD (n={len(tids)}, {int(yv.sum())} pos, base rate {yv.mean():.2f}) ===")
    if len(tids) >= 20 and yv.min() != yv.max():
        sig_ap, mom_ap = metrics.auprc(sa, yv), metrics.auprc(ba, yv)
        d, lo, hi, p1 = metrics.paired_delta_ci(sa, ba, yv, B=3000, seed=0)
        _, _, pn = metrics.shuffle_null(sa, yv, B=3000, seed=0)
        sep = "SEPARATED" if (lo > 0 or hi < 0) else "spans 0"
        powered = "POWERED" if len(tids) >= 40 else "UNDERPOWERED (n<40)"
        print(f"[forward]   signature AUPRC = {sig_ap:.3f}  (shuffle-null p={pn:.3f})")
        print(f"[forward]   momentum  AUPRC = {mom_ap:.3f}")
        print(f"[forward]   forward delta = {d:+.3f}  95% CI [{lo:+.3f},{hi:+.3f}]  p={p1:.3f} -> {sep}  [{powered}]")
        verdict = ("FORWARD PREDICTION ESTABLISHED" if (sep == "SEPARATED" and p1 < 0.05 and len(tids) >= 40)
                   else "forward prediction NOT established" + (" (underpowered)" if len(tids) < 40 else " (delta spans 0)"))
        print(f"[forward]   VERDICT: {verdict}")
    else:
        print(f"[forward]   too few pooled test themes ({len(tids)}) -- forward prediction NOT established (underpowered)")


if __name__ == "__main__":
    main()
