"""Clustered Stage B-B, stage 1: cluster arXiv phrases into coherent MARKETS + market-level as-of signal.

    python -m experiments.stageB_cluster_markets --k 80

Fixes the lossy single-phrase unit: a market ("Large Language Models") is a CLUSTER of phrases
(llm, gpt, instruction tuning, rlhf...), not one. We take the permissive-gated phrases with real volume,
embed them (gemini-embedding-001), KMeans into ~k markets, then treat each market as a theme = union of its
member phrases and reuse the exact emergence machinery (share trajectory -> durable-breakout label -> 11 as-of
features -> predicted P(durable)) STRICTLY as-of the market's decision point T_dec. Output:
artifacts/stageB/clustered_markets.parquet (market name, members, sig_score, momentum, T_dec, theme_text).
"""
from __future__ import annotations
import argparse
import pathlib
import warnings

warnings.filterwarnings("ignore")
OUT = pathlib.Path("artifacts/stageB")
DF_FLOOR = 100          # phrases with >= this many papers in the corpus are clustered (real research activity)


def _embed(texts, batch=100, label="", ckpt=None):
    import time
    import numpy as np
    from marketbrain.labels.gemini_grounding import make_client
    client = make_client()
    out = []
    start = 0
    if ckpt is not None and ckpt.exists():
        out = [r for r in np.load(ckpt)]
        start = len(out)
    for i in range(start, len(texts), batch):
        chunk = [str(t)[:1500] for t in texts[i:i + batch]]
        for a in range(6):
            try:
                r = client.models.embed_content(model="gemini-embedding-001", contents=chunk)
                out.extend(np.asarray(e.values, dtype=np.float32) for e in r.embeddings)
                break
            except Exception:
                if a == 5:
                    raise
                time.sleep(2 ** a)
        if ckpt is not None and (i // batch) % 5 == 0:
            np.save(ckpt, np.vstack(out))
        if (i // batch) % 10 == 0:
            print(f"[cluster]   embed {label} {i + len(chunk)}/{len(texts)}", flush=True)
    X = np.vstack(out)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def main(argv=None):
    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.ensemble import HistGradientBoostingClassifier
    from marketbrain.corpus.load_full import load_corpus
    from marketbrain.eval.corpus_inputs import cs_denominator, corpus_term_events
    from marketbrain.eval.label import EmergenceLabelConfig, durable_breakout_label
    from marketbrain.eval.breakout_signature import build_feature_matrix, SIGNATURE_FEATURES
    from marketbrain.markets.granularity_gates import _noun_headed, _is_fragment, GENERIC
    from marketbrain.features.emergence import EmergenceConfig, escore, month_range_ending

    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", default="cs.LG,cs.CL,cs.AI,cs.RO,cs.SE,cs.DC,stat.ML")
    ap.add_argument("--k", type=int, default=80)
    ap.add_argument("--df-floor", type=int, default=DF_FLOOR)
    args = ap.parse_args(argv)
    cfg = EmergenceLabelConfig()

    corpus = load_corpus(args.cats.split(","))
    denom = cs_denominator(corpus)
    grid = list(denom.index)
    corpus_end, n = grid[-1], len(corpus)
    print(f"[cluster] {n:,} papers; permissive gate + phrases with df>={args.df_floor}...", flush=True)
    te = corpus_term_events(corpus, min_doc_freq=40, ngram_max=3)
    df_count = te.groupby("term")["record_id"].nunique().to_dict()
    ceil = 0.5 * n
    gated = [t for t, c in df_count.items()
             if args.df_floor <= c <= ceil and not _is_fragment(t)
             and not (len(t.split()) == 1 and t in GENERIC)
             and not (len(t.split()) > 1 and all(w in GENERIC for w in t.split()))]
    gated = sorted(_noun_headed(gated))
    print(f"[cluster] {len(gated)} phrases to cluster into k={args.k} markets", flush=True)
    te = te[te["term"].isin(set(gated))]
    term_to_records = te.groupby("term")["record_id"].apply(set).to_dict()

    # embed phrases + KMeans into markets
    pvec = _embed(gated, label="phrases", ckpt=OUT / "phrase_emb_ckpt.npy")
    km = KMeans(n_clusters=args.k, random_state=0, n_init=10).fit(pvec)
    lab = km.labels_
    clusters = {}
    for p, l in zip(gated, lab):
        clusters.setdefault(int(l), []).append(p)

    # market = union of member-phrase records; name = highest-df member
    markets, members_map = {}, {}
    for cid, members in clusters.items():
        members = sorted(members, key=lambda t: df_count.get(t, 0), reverse=True)
        name = members[0]
        mid = f"mkt_{cid:03d}:{name}"[:60]
        markets[mid] = set(members)
        members_map[mid] = members

    # market share trajectories -> label / T0 / T_dec (reuse the exact label machinery on the union)
    denom_arr = denom.reindex(grid).to_numpy()
    rows = []
    pivot_m = {}
    for mid, terms in markets.items():
        recs_by_month = te[te["term"].isin(terms)].groupby("month")["record_id"].nunique().reindex(grid, fill_value=0)
        pivot_m[mid] = recs_by_month
        share = pd.Series(np.divide(recs_by_month.to_numpy(), denom_arr, out=np.zeros(len(grid)), where=denom_arr > 0), index=grid)
        res = durable_breakout_label(share, recs_by_month.astype(int), cfg, corpus_end=corpus_end)
        t0 = res.get("T0")
        if t0 is None or t0 not in grid or grid.index(t0) + 9 >= len(grid):
            continue
        rows.append({"theme_id": mid, "label": res["label"], "T0": t0,
                     "T_dec": grid[grid.index(t0) + 9], "parent_id": mid})
    allm = pd.DataFrame(rows)
    print(f"[cluster] {len(allm)} markets with valid T_dec; labels={allm['label'].value_counts().to_dict()}", flush=True)

    # market as-of features (union records) + predicted durability + momentum
    records_indexed = corpus.set_index("record_id")[["month", "authors", "categories", "comment", "journal_ref"]]
    m_term_to_records = {mid: set().union(*[term_to_records.get(t, set()) for t in markets[mid]]) for mid in allm["theme_id"]}
    feats = build_feature_matrix({mid: {mid} for mid in allm["theme_id"]},
                                 {mid: m_term_to_records[mid] for mid in allm["theme_id"]},
                                 records_indexed, allm, denom)
    ids = list(feats["theme_id"])
    disc = feats[feats["label"].isin(("positive", "hyped_died"))]
    clf = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05, min_samples_leaf=8,
                                         random_state=0).fit(disc[SIGNATURE_FEATURES].to_numpy(),
                                                             (disc["label"] == "positive").astype(int).to_numpy())
    p_dur = pd.Series(clf.predict_proba(feats[SIGNATURE_FEATURES].to_numpy())[:, 1], index=ids)
    ecfg = EmergenceConfig()
    tdec = allm.set_index("theme_id")["T_dec"]
    mom = pd.Series({mid: escore(pivot_m[mid].reindex(
        month_range_ending(tdec[mid], ecfg.window_months), fill_value=0).to_numpy().tolist()) for mid in ids})

    # theme_text per market: name + top member phrases + representative titles (as-of)
    title_by_rec = corpus.set_index("record_id")["title"].to_dict()
    month_by_rec = corpus.set_index("record_id")["month"].to_dict()
    out_rows = []
    for mid in ids:
        td = tdec[mid]
        mem = members_map[mid][:8]
        recs = [r for r in m_term_to_records[mid] if month_by_rec.get(r, "9999") <= td]
        titles = [title_by_rec.get(r, "") for r in list(recs)[:400] if title_by_rec.get(r, "")]
        titles = titles[:3]
        name = mid.split(":", 1)[1] if ":" in mid else mid
        theme_text = f"{name}. " + "; ".join(mem) + (". e.g. " + " | ".join(titles) if titles else "")
        out_rows.append({"theme_id": mid, "market": name, "label": str(allm.set_index("theme_id").loc[mid, "label"]),
                         "T0": allm.set_index("theme_id").loc[mid, "T0"], "T_dec": td,
                         "sig_score": float(p_dur[mid]), "momentum": float(mom[mid]),
                         "n_members": len(members_map[mid]), "members": ", ".join(mem),
                         "n_papers_asof": len(recs), "theme_text": theme_text})
    out = pd.DataFrame(out_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT / "clustered_markets.parquet")
    print(f"[cluster] wrote {OUT}/clustered_markets.parquet: {len(out)} markets\n", flush=True)
    for _, r in out.sort_values("n_papers_asof", ascending=False).head(25).iterrows():
        print(f"   {r['market'][:26]:26s} sig={r['sig_score']:.2f} mom={r['momentum']:.2f} "
              f"[{r['label'][:8]:8s}] members: {r['members'][:70]}", flush=True)


if __name__ == "__main__":
    main()
