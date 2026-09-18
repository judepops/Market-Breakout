"""Phase-1 B2: does the signature still beat the team's NAMED baseline -- momentum + LLM-categorisation?

    python -m experiments.phase1_b2 [--all]

Beating raw momentum is the easy bar. the team's actual bar is "beat momentum AND LLM-categorisation". B2 = a
logistic combine of momentum (escore) + an LLM "promise" rating per theme (gemini-2.5-flash, from the theme
phrase + 3 example titles dated <= T_dec, leakage-safe, cached). The signature must beat B2 with a paired-delta
CI that excludes 0 -- otherwise we have only cleared the easy comparator. Variant-grouped folds (no twin-leak).
"""
from __future__ import annotations
import argparse
import json
import pathlib
import warnings

warnings.filterwarnings("ignore")
OUT = pathlib.Path("artifacts/phase1_b2")


def main(argv=None):
    import numpy as np
    import pandas as pd
    from marketbrain.eval.dataset import build_discriminative_dataset
    from marketbrain.eval import metrics, splits
    from marketbrain.eval.breakout_signature import SIGNATURE_FEATURES
    from marketbrain.eval.baselines import llm_promise_scores, momentum_plus_llm
    from experiments.phase1_secure import _variant_groups, _oof

    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--model", default="gemini-2.5-flash", help="LLM for the categorisation baseline")
    args = ap.parse_args(argv)

    print("[b2] building discriminative dataset...", flush=True)
    d = build_discriminative_dataset(None if args.all else ["cs.AI"])
    disc, feats, momentum, records, t2r = d["disc"], d["feats"], d["momentum"], d["records"], d["term_to_records"]
    ids = list(feats["theme_id"])
    y = feats.set_index("theme_id")["label"]
    yv = y.reindex(ids).to_numpy()
    n_pos = int((y == "positive").sum())
    print(f"[b2] {n_pos} positives / {len(ids) - n_pos} hyped-died; fetching <=T_dec titles + LLM promise...", flush=True)

    # leakage-safe example titles (<= T_dec) per theme, for the LLM categorisation baseline
    tdec = disc.set_index("theme_id")["T_dec"]
    idx_set = set(records.index)
    titles_by_theme = {}
    for t in ids:
        recs = records.loc[list(t2r.get(t, set()) & idx_set)]
        recs = recs[recs["month"] <= tdec[t]]
        titles_by_theme[t] = list(recs["title"].head(3))
    promise = llm_promise_scores({t: {t} for t in ids}, titles_by_theme, model=args.model)

    # variant-grouped folds (no twin-leakage)
    groups = _variant_groups(list(disc["theme_id"]))
    disc_g = disc.copy()
    disc_g["parent_id"] = disc_g["theme_id"].map(groups)
    folds = splits.grouped_folds(disc_g, k=5)

    sig = _oof(feats, folds, SIGNATURE_FEATURES)
    b2 = {}
    for tr_df, te_df in folds:
        tr = disc[disc["theme_id"].isin(tr_df["theme_id"])]
        if tr["label"].nunique() < 2:
            continue
        sc = momentum_plus_llm(momentum, promise, tr)      # fit logistic(momentum, promise) on train
        for tid in te_df["theme_id"]:
            if tid in sc.index:
                b2[tid] = sc[tid]
    b2 = pd.Series(b2)
    rand = pd.Series(np.random.RandomState(0).rand(len(ids)), index=ids)

    print(f"[b2] scores (variant-grouped OOF, n={len(ids)}, {n_pos} positives):")
    summ = {"n_pos": n_pos, "n_neg": len(ids) - n_pos, "model": args.model}
    for name, s in [("signature", sig), ("B2_momentum+LLM", b2), ("B1_momentum", momentum), ("random", rand)]:
        sv = s.reindex(ids).fillna(0.0).to_numpy()
        ap_ = metrics.auprc(sv, yv)
        _, _, p = metrics.shuffle_null(sv, yv, B=2000, seed=0)
        print(f"[b2]   {name:18s} AUPRC={ap_:.3f}  null p={p:.3f}")
        summ[name] = {"auprc": round(ap_, 4), "null_p": round(p, 4)}

    sv, b2v = sig.reindex(ids).fillna(0.0).to_numpy(), b2.reindex(ids).fillna(0.0).to_numpy()
    dl, lo, hi, p1 = metrics.paired_delta_ci(sv, b2v, yv, B=2000, seed=0)
    sep = "SEPARATED (clears the team's bar)" if lo > 0 else "spans 0 (does NOT clear it)"
    print(f"[b2]   signature - B2 delta = {dl:+.3f}  95% CI [{lo:+.3f},{hi:+.3f}]  P(B2>=sig)={p1:.3f} -> {sep}")
    summ["signature_vs_B2"] = {"delta": round(dl, 4), "ci": [round(lo, 4), round(hi, 4)], "p": round(p1, 4),
                               "clears_bar": bool(lo > 0)}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2))
    print(f"[b2] wrote {OUT}/summary.json")


if __name__ == "__main__":
    main()
