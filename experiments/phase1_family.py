"""Phase-1 idea-family holdout: does the signature generalise to UNSEEN idea-families?

    python -m experiments.phase1_family [--all]

The grouped-CV so far holds out individual themes; this holds out whole IDEA-FAMILIES (themes clustered by
Gemini embedding, so no family appears in both train and test). Passing this buys "generalises to unseen
idea-families" -- the category-agnostic-signature claim -- and NOTHING about forward-in-time (that is the
temporal split, which is underpowered). It must NEVER be reported as forward-prediction.
"""
from __future__ import annotations
import argparse
import json
import pathlib
import warnings

warnings.filterwarnings("ignore")
OUT = pathlib.Path("artifacts/phase1_family")


def _embed(phrases, model="gemini-embedding-001", batch=100):
    import numpy as np
    from marketbrain.labels.gemini_grounding import make_client
    client = make_client()
    vecs = {}
    for i in range(0, len(phrases), batch):
        chunk = phrases[i:i + batch]
        r = client.models.embed_content(model=model, contents=chunk)
        for p, e in zip(chunk, r.embeddings):
            vecs[p] = np.asarray(e.values, dtype=np.float32)
    return vecs


def main(argv=None):
    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import normalize
    from marketbrain.eval.dataset import build_discriminative_dataset
    from marketbrain.eval import metrics, splits
    from marketbrain.eval.breakout_signature import SIGNATURE_FEATURES
    from experiments.phase1_secure import _oof

    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--families", type=int, default=0, help="n idea-families (0 = auto ~ n_themes/12)")
    args = ap.parse_args(argv)

    print("[family] building discriminative dataset...", flush=True)
    d = build_discriminative_dataset(None if args.all else ["cs.AI"])
    disc, feats, momentum = d["disc"], d["feats"], d["momentum"]
    ids = list(feats["theme_id"])
    y = feats.set_index("theme_id")["label"]
    yv = y.reindex(ids).to_numpy()
    n_pos = int((y == "positive").sum())

    print(f"[family] embedding {len(ids)} themes (gemini-embedding-001) + clustering into families...", flush=True)
    vecs = _embed(ids)
    X = normalize(np.vstack([vecs[t] for t in ids]))       # L2-normalise -> KMeans approximates spherical/cosine
    k = args.families or max(15, len(ids) // 8)            # balanced families (KMeans avoids the giant-blob agglomerative gives)
    fam = KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(X)
    fam_by = {t: f"fam{fam[i]}" for i, t in enumerate(ids)}
    disc_f = disc.copy()
    disc_f["parent_id"] = disc_f["theme_id"].map(fam_by)
    sizes = pd.Series(list(fam_by.values())).value_counts()
    print(f"[family] {k} idea-families (median {int(sizes.median())} themes, max {int(sizes.max())})")

    folds = splits.grouped_folds(disc_f, k=5)              # holds out whole families; none in both sides
    # assert the holdout guarantee
    for tr_df, te_df in folds:
        assert set(tr_df["parent_id"]) & set(te_df["parent_id"]) == set(), "family leaked across a fold!"

    sig = _oof(feats, folds, SIGNATURE_FEATURES)
    rand = pd.Series(np.random.RandomState(0).rand(len(ids)), index=ids)
    print(f"[family] CROSS-FAMILY holdout OOF (generalises to UNSEEN idea-families; NOT forward-in-time), "
          f"n={len(ids)}, {n_pos} positives:")
    summ = {"n_pos": n_pos, "n_neg": len(ids) - n_pos, "n_families": int(k)}
    for name, s in [("signature", sig), ("B1_momentum", momentum), ("random", rand)]:
        sv = s.reindex(ids).fillna(0.0).to_numpy()
        ap_ = metrics.auprc(sv, yv)
        _, _, p = metrics.shuffle_null(sv, yv, groups=disc_f["parent_id"].to_numpy(), B=2000, seed=0)
        print(f"[family]   {name:14s} AUPRC={ap_:.3f}  family-grouped null p={p:.3f}")
        summ[name] = {"auprc": round(ap_, 4), "null_p": round(p, 4)}

    sv = sig.reindex(ids).fillna(0.0).to_numpy()
    mv = momentum.reindex(ids).fillna(0.0).to_numpy()
    dl, lo, hi, p1 = metrics.paired_delta_ci(sv, mv, yv, B=2000, seed=0)
    sep = "SEPARATED (generalises across families)" if lo > 0 else "spans 0"
    print(f"[family]   signature - momentum delta = {dl:+.3f}  95% CI [{lo:+.3f},{hi:+.3f}]  P(mom>=sig)={p1:.3f} -> {sep}")
    summ["signature_vs_momentum_crossfamily"] = {"delta": round(dl, 4), "ci": [round(lo, 4), round(hi, 4)],
                                                 "p": round(p1, 4), "separated": bool(lo > 0)}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2))
    print(f"[family] wrote {OUT}/summary.json")


if __name__ == "__main__":
    main()
