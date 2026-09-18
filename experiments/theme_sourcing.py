"""Who should we talk to about a theme? (the sourcing layer)

    python -m experiments.theme_sourcing --themes grpo "state space" --top 12

The business question from the 2026-07-14 meeting: "GRPO is accelerating — find the
people using that technique, maybe the professors well known in GRPO, and go chat
with them." This turns a theme into a ranked contact list.

Reuses the in-corpus author machinery (marketbrain/eval/author_features.py):
standing is an author's PRIOR papers as-of the theme's birth, so "established" means
established *before* this theme existed, not famous because of it.

Ranking = centrality on the theme (papers authored on it), tie-broken by prior standing.
Flags: founder (in the first 6 months), star (>= 30 prior papers), still-active (a paper
on the theme in the last 6 months of the corpus).

Writes artifacts/sourcing/<theme>.json.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import warnings

warnings.filterwarnings("ignore")

OUT = pathlib.Path("artifacts/sourcing")
CATS = ["cs.AI", "cs.LG", "cs.CV", "cs.CL", "cs.NE", "stat.ML"]
RECENT_MONTHS = 6


def main(argv=None):
    import numpy as np, pandas as pd
    from marketbrain.corpus.load_full import load_corpus
    from marketbrain.eval.corpus_inputs import cs_denominator, corpus_term_events
    from marketbrain.eval.label import EmergenceLabelConfig, durable_breakout_label, _shift_month
    from marketbrain.eval.author_features import (
        STAR_PRIOR, EARLY_MONTHS, build_author_index, early_authors, prior_paper_count)

    ap = argparse.ArgumentParser()
    ap.add_argument("--themes", nargs="+", required=True, help="theme terms, e.g. grpo mamba")
    ap.add_argument("--cats", nargs="*", default=CATS)
    ap.add_argument("--min-docs", type=int, default=25)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args(argv)

    corpus = load_corpus(args.cats)
    denom = cs_denominator(corpus); grid = list(denom.index); corpus_end = grid[-1]
    print(f"[sourcing] {len(corpus):,} papers, grid {grid[0]}..{corpus_end}", flush=True)

    te = corpus_term_events(corpus, min_doc_freq=args.min_docs, ngram_max=3)
    term_to_records = te.groupby("term")["record_id"].apply(set).to_dict()
    author_index = build_author_index(corpus)
    crecs = corpus.set_index("record_id")
    recent_lo = _shift_month(corpus_end, -RECENT_MONTHS)
    cfg = EmergenceLabelConfig()
    da = denom.reindex(grid).to_numpy(dtype=float)

    OUT.mkdir(parents=True, exist_ok=True)
    for want in args.themes:
        key = want.strip().lower()
        if key not in term_to_records:
            near = [t for t in term_to_records if key in t][:5]
            print(f"[sourcing] '{want}' not a gated theme. near: {near}")
            continue
        ids = list(term_to_records[key])
        recs = crecs.loc[[i for i in ids if i in crecs.index]].reset_index()

        # birth month, via the same emergence detector the pipeline uses
        nth = recs.groupby("month")["record_id"].nunique().reindex(grid, fill_value=0)
        share = pd.Series(np.divide(nth.to_numpy(dtype=float), da, out=np.zeros(len(grid)), where=da > 0), index=grid)
        res = durable_breakout_label(share, nth.astype(int), cfg, corpus_end=corpus_end)
        t0 = res.get("T0") or str(recs["month"].min())
        founders = set(early_authors(recs, t0, corpus_end, EARLY_MONTHS))

        # per-author centrality on the theme
        ex = recs[["month", "authors"]].explode("authors").dropna(subset=["authors"])
        ex = ex[ex["authors"].astype(str).str.len() > 0]
        rows = []
        for a, g in ex.groupby("authors"):
            prior = prior_paper_count(a, t0, author_index)
            rows.append({
                "author": a,
                "papers_on_theme": int(len(g)),
                "first_month": str(g["month"].min()),
                "last_month": str(g["month"].max()),
                "prior_papers_at_t0": int(prior),
                "founder": bool(a in founders),
                "star": bool(prior >= STAR_PRIOR),
                "still_active": bool(str(g["month"].max()) >= recent_lo),
            })
        rows.sort(key=lambda r: (-r["papers_on_theme"], -r["prior_papers_at_t0"]))
        top = rows[: args.top]

        payload = {"theme": key, "t0": t0, "label": res["label"], "corpus_end": corpus_end,
                   "n_papers": int(recs["record_id"].nunique()), "n_authors": len(rows),
                   "contacts": top}
        (OUT / f"{key.replace(' ', '_')}.json").write_text(json.dumps(payload, indent=1))

        print(f"\n=== {key}  (born {t0}, label={res['label']}, "
              f"{payload['n_papers']} papers, {len(rows)} authors) ===")
        print(f"{'author':32} {'papers':>6} {'prior':>6}  flags")
        for r in top:
            flags = " ".join(f for f, on in
                             (("founder", r["founder"]), ("star", r["star"]), ("active", r["still_active"])) if on)
            print(f"  {r['author'][:30]:30} {r['papers_on_theme']:6} {r['prior_papers_at_t0']:6}  {flags}")
    print(f"\n[sourcing] wrote {OUT}/", flush=True)


if __name__ == "__main__":
    main()
