"""Stage A forward test, powered variant: 12-month secondary label + CONTINUOUS persistence-degree target.
Pre-reg: `Plans/2026-07-01_forward-validation.md` (12mo + continuous amendment).

    python -m experiments.phase1_forward_continuous

Why: the 24mo binary forward test was underpowered (n=29, base rate 0.79 -> AUPRC floor). Two honest fixes:
(1) shorten the persistence horizon to 12mo so more themes mature -> more forward-testable data; (2) predict the
CONTINUOUS persistence degree q/p (plateau/peak) instead of the binary durable/died -> no base-rate floor.
NOTE: the count-gate was DROPPED (its result showed it raises the base rate, and the continuous target already
removes the floor). Share-gate retained (bigger pool). Rolling-origin, strict label-maturation as-of C,
twin-safe. A pass establishes "forward-predicts 1-yr persistence DEGREE" -- the 2yr binary claim stays pending.
"""
from __future__ import annotations
import warnings

warnings.filterwarnings("ignore")
ORIGINS = ["2019-06", "2020-06", "2021-06", "2022-06", "2023-06", "2024-06"]
TEST_WINDOW = 12
CATS = ["cs.LG", "cs.CL", "cs.AI", "cs.RO", "cs.SE", "cs.DC", "stat.ML"]


def _shift(m, n):
    t = int(m[:4]) * 12 + (int(m[5:7]) - 1) + n
    return f"{t // 12:04d}-{t % 12 + 1:02d}"


def main():
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr
    from sklearn.ensemble import HistGradientBoostingClassifier
    from marketbrain.corpus.load_full import load_corpus
    from marketbrain.eval.corpus_inputs import cs_denominator, corpus_term_events
    from marketbrain.eval.label import EmergenceLabelConfig, durable_breakout_label
    from marketbrain.eval.breakout_signature import build_feature_matrix, SIGNATURE_FEATURES
    from marketbrain.markets.granularity_gates import _pmi_ok, _noun_headed, _is_fragment, GENERIC, DF_CEIL
    from marketbrain.features.emergence import EmergenceConfig, escore, month_range_ending
    from experiments.phase1_secure import _variant_groups

    cfg = EmergenceLabelConfig(persist_lo=6, persist_hi=12)      # SECONDARY 12-month durability
    corpus = load_corpus(CATS)
    denom = cs_denominator(corpus)
    grid = list(denom.index)
    corpus_end, n = grid[-1], len(corpus)
    print(f"[fwd-cont] {n:,} papers; 12mo label (persist_lo=6, persist_hi=12); gating...", flush=True)
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
        t0, tp, q, p = res.get("T0"), res.get("T_peak"), res.get("q"), res.get("p")
        t_dec = grid[grid.index(t0) + 9] if t0 in grid and grid.index(t0) + 9 < len(grid) else None
        rows.append({"theme_id": term, "label": res["label"], "T0": t0, "T_dec": t_dec, "T_peak": tp,
                     "q": q, "p": p})
    lab = pd.DataFrame(rows)
    disc = lab[lab["label"].isin(("positive", "hyped_died"))].dropna(subset=["T_dec", "q", "p"]).reset_index(drop=True)
    disc = disc[disc["p"] > 0].reset_index(drop=True)
    disc["qp"] = (disc["q"] / disc["p"]).clip(0, 2)             # continuous persistence degree
    disc["parent_id"] = disc["theme_id"]
    disc["grp"] = disc["theme_id"].map(_variant_groups(list(disc["theme_id"])))
    disc["mature_m"] = disc["T_peak"].apply(lambda tp: _shift(tp, cfg.persist_hi - 1) if isinstance(tp, str) else None)
    disc["y"] = (disc["label"] == "positive").astype(int)
    print(f"[fwd-cont] got-big pool: {len(disc)} ({int(disc['y'].sum())} durable-12mo); qp: "
          f"median={disc['qp'].median():.2f} IQR[{disc['qp'].quantile(.25):.2f},{disc['qp'].quantile(.75):.2f}]", flush=True)

    themes = {t: {t} for t in disc["theme_id"]}
    term_to_records = te.groupby("term")["record_id"].apply(set).to_dict()
    records_indexed = corpus.set_index("record_id")[["month", "authors", "categories", "comment", "journal_ref"]]
    feats = build_feature_matrix(themes, term_to_records, records_indexed, disc, denom).set_index("theme_id")
    ecfg = EmergenceConfig()
    mom = {t: escore(pivot.loc[t].reindex(month_range_ending(td, ecfg.window_months), fill_value=0).to_numpy().tolist())
           for t, td in zip(disc["theme_id"], disc["T_dec"])}

    pooled = {}
    print(f"\n[fwd-cont] rolling origins (target = continuous q/p):")
    for C in ORIGINS:
        Cend = _shift(C, TEST_WINDOW)
        test = disc[(disc["T_dec"] > C) & (disc["T_dec"] <= Cend) & (~disc["theme_id"].isin(pooled))]
        train = disc[(disc["T_dec"] <= C) & disc["mature_m"].notna() & (disc["mature_m"] <= C)]
        train = train[~train["grp"].isin(set(test["grp"]))]
        if len(test) == 0 or train["y"].nunique() < 2 or len(train) < 15:
            print(f"[fwd-cont]   C={C}: train={len(train)} test={len(test)} -- skipped (thin)", flush=True)
            continue
        clf = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05, min_samples_leaf=10,
                                             random_state=0).fit(feats.loc[train["theme_id"], SIGNATURE_FEATURES].to_numpy(),
                                                                 train["y"].to_numpy())
        p = clf.predict_proba(feats.loc[test["theme_id"], SIGNATURE_FEATURES].to_numpy())[:, 1]
        for t, pv in zip(test["theme_id"], p):
            pooled[t] = pv
        rho = spearmanr(p, test["qp"].to_numpy())[0] if len(test) >= 4 else float("nan")
        print(f"[fwd-cont]   C={C}: train={len(train):3d} test={len(test):3d}  spearman(sig,qp)={rho:+.3f}", flush=True)

    tids = list(pooled)
    sa = np.array([pooled[t] for t in tids])
    ba = np.array([mom[t] for t in tids])
    qp = disc.set_index("theme_id").loc[tids, "qp"].to_numpy()
    print(f"\n[fwd-cont] === POOLED FORWARD (continuous q/p, n={len(tids)}) ===")
    if len(tids) >= 20:
        rs = spearmanr(sa, qp)[0]; rm = spearmanr(ba, qp)[0]
        rng = np.random.RandomState(0)
        null = np.array([spearmanr(sa, rng.permutation(qp))[0] for _ in range(5000)])
        pn = (np.sum(null >= rs) + 1) / 5001
        # bootstrap CI on each and on the paired difference
        bs, bm, bd = [], [], []
        for _ in range(3000):
            i = rng.randint(0, len(tids), len(tids))
            if len(np.unique(qp[i])) < 3:
                continue
            a = spearmanr(sa[i], qp[i])[0]; b = spearmanr(ba[i], qp[i])[0]
            bs.append(a); bm.append(b); bd.append(a - b)
        ci = lambda v: (np.percentile(v, 2.5), np.percentile(v, 97.5))
        slo, shi = ci(bs); dlo, dhi = ci(bd)
        powered = "POWERED" if len(tids) >= 40 else "UNDERPOWERED (n<40)"
        print(f"[fwd-cont]   spearman(signature, qp) = {rs:+.3f}  boot95[{slo:+.3f},{shi:+.3f}]  shuffle-null p={pn:.3f}")
        print(f"[fwd-cont]   spearman(momentum , qp) = {rm:+.3f}")
        print(f"[fwd-cont]   paired diff (sig-mom)   = {rs - rm:+.3f}  boot95[{dlo:+.3f},{dhi:+.3f}]")
        clears = rs > 0 and pn < 0.05 and rs >= rm and len(tids) >= 40
        print(f"[fwd-cont]   [{powered}]  VERDICT: "
              f"{'FORWARD (1-yr persistence-degree) ESTABLISHED' if clears else 'not established'}"
              f"{'' if len(tids) >= 40 else ' -- underpowered'}{'' if pn < 0.05 else ' -- shuffle-null not cleared'}")
    else:
        print(f"[fwd-cont]   too few pooled ({len(tids)}) -- not established")


if __name__ == "__main__":
    main()
