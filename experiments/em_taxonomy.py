"""Phase 1 TAXONOMY demo: gated phrases -> co-occurrence-merged idea CLUSTERS (the variant-merge proof).

    python -m experiments.em_taxonomy [--all] [--refresh]

Shows the deterministic discovery pipeline end to end: corpus -> gated candidate phrases -> co-occurrence
clusters. The clusters are the emerging idea-markets; variants that share a paper collapse into one theme
(chiefly nested n-grams + acronym-glosses -- see cooccur_cluster docstring). Growth is share-based (audit P4).
"""
from __future__ import annotations
import argparse
import warnings
from collections import Counter

warnings.filterwarnings("ignore")
RECENT, PRIOR = 12, 12
MIN_RECENT = 8
MAX_PHRASES = 3000       # cluster the top emerging phrases (focus + keep co-occurrence tractable)


def main(argv=None):
    from marketbrain.corpus.load_full import load_corpus
    from marketbrain.markets.granularity_gates import (
        _pmi_ok, _noun_headed, _is_fragment, GENERIC, DF_CEIL, MIN_DOCS, share_growth)
    from marketbrain.markets.cooccur_cluster import cooccurrence, cluster
    from marketbrain.signals.arxiv_records import _terms_in

    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--refresh", action="store_true", help="force a corpus-cache rebuild (full-corpus only)")
    args = ap.parse_args(argv)

    corpus = load_corpus(None if args.all else ["cs.AI"], cache=args.all, refresh=args.refresh)
    months = sorted(corpus["month"].unique())
    rset, pset = set(months[-RECENT:]), set(months[-(RECENT + PRIOR):-RECENT])
    n = len(corpus)
    n_recent = int(corpus["month"].isin(rset).sum())
    n_prior = int(corpus["month"].isin(pset).sum())
    print(f"[tax] {n:,} papers, {len(months)} months ({months[0]}..{months[-1]}); extracting terms...", flush=True)

    paper_terms, df_count, recent, prior = [], Counter(), Counter(), Counter()
    for i, row in enumerate(corpus.itertuples(index=False)):
        terms = _terms_in(f"{row.title}. {row.abstract}", ngram_max=3)
        paper_terms.append((row.month, terms))
        in_r, in_p = row.month in rset, row.month in pset
        for t in terms:
            df_count[t] += 1
            if in_r:
                recent[t] += 1
            elif in_p:
                prior[t] += 1
        if i and i % 50000 == 0:
            print(f"  ...{i:,}/{n:,}", flush=True)

    ceil = DF_CEIL * n
    gated = [t for t, c in df_count.items()
             if MIN_DOCS <= c <= ceil
             and not _is_fragment(t)
             and not (len(t.split()) == 1 and t in GENERIC)
             and not (len(t.split()) > 1 and all(w in GENERIC for w in t.split()))
             and _pmi_ok(t, df_count, n)]
    gated = set(_noun_headed(gated))
    emerging_all = [t for t in gated if recent[t] >= MIN_RECENT]
    emerging = sorted(emerging_all, key=lambda t: -share_growth(recent[t], prior[t], n_recent, n_prior))[:MAX_PHRASES]
    eset = set(emerging)
    print(f"[tax] {len(gated):,} gated phrases | clustering top {len(emerging):,} of {len(emerging_all):,} emerging "
          f"(cap {MAX_PHRASES})", flush=True)

    cooc, df = cooccurrence(frozenset(terms & eset) for _, terms in paper_terms)
    clusters = cluster(emerging, cooc, df)

    p2c = {p: ci for ci, members in enumerate(clusters) for p in members}
    c_recent, c_prior = Counter(), Counter()
    for month, terms in paper_terms:
        hit = {p2c[p] for p in (terms & eset)}
        for ci in hit:
            if month in rset:
                c_recent[ci] += 1
            elif month in pset:
                c_prior[ci] += 1

    cg = {ci: share_growth(c_recent[ci], c_prior[ci], n_recent, n_prior) for ci in range(len(clusters))}
    ranked = sorted(range(len(clusters)), key=lambda ci: -cg[ci])
    print(f"[tax] {len(clusters):,} clusters ({sum(len(c) > 1 for c in clusters)} multi-phrase) | "
          f"top emerging idea-clusters:\n")
    for ci in ranked[:25]:
        members = sorted(clusters[ci], key=lambda p: -recent[p])
        shown = ", ".join(members[:6]) + (" ..." if len(members) > 6 else "")
        print(f"  {cg[ci]:6.1f}x  [{c_recent[ci]:4d} papers]  {members[0]}")
        print(f"            = {{{shown}}}")


if __name__ == "__main__":
    main()
