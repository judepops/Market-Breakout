"""Phase 1 NAMED TAXONOMY: corpus -> gated phrases -> clusters -> LLM-named emerging markets (the team demo).

    python -m experiments.em_markets [--all] [--top N] [--model M] [--refresh]

The full discovery pipeline end to end. The deterministic stages surface + merge idea-phrases; the LLM names
the top-N emerging clusters and rejects junk (is_market). Growth is share-based (audit P4); LLM abstentions on
transient errors are reported, not silently dropped as junk (audit P0). Cached, so re-runs are free.
"""
from __future__ import annotations
import argparse
import warnings
from collections import Counter, defaultdict

warnings.filterwarnings("ignore")
RECENT, PRIOR, MIN_RECENT, MAX_PHRASES = 12, 12, 8, 3000


def main(argv=None):
    from marketbrain.corpus.load_full import load_corpus
    from marketbrain.markets.granularity_gates import (
        _pmi_ok, _noun_headed, _is_fragment, GENERIC, DF_CEIL, MIN_DOCS, share_growth)
    from marketbrain.markets.cooccur_cluster import cooccurrence, cluster
    from marketbrain.markets.reason_cluster import name_clusters
    from marketbrain.signals.arxiv_records import _terms_in

    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--top", type=int, default=60, help="how many top emerging clusters to send to the LLM")
    ap.add_argument("--model", default="gemini-2.5-pro", help="Gemini model for the naming pass")
    ap.add_argument("--refresh", action="store_true", help="force a corpus-cache rebuild (full-corpus only)")
    args = ap.parse_args(argv)

    corpus = load_corpus(None if args.all else ["cs.AI"], cache=args.all, refresh=args.refresh)
    months = sorted(corpus["month"].unique())
    rset, pset = set(months[-RECENT:]), set(months[-(RECENT + PRIOR):-RECENT])
    n = len(corpus)
    n_recent = int(corpus["month"].isin(rset).sum())
    n_prior = int(corpus["month"].isin(pset).sum())
    print(f"[markets] {n:,} papers ({n_recent:,} recent / {n_prior:,} prior window); extracting terms...", flush=True)

    papers, df_count, recent, prior = [], Counter(), Counter(), Counter()
    for i, row in enumerate(corpus.itertuples(index=False)):
        terms = _terms_in(f"{row.title}. {row.abstract}", ngram_max=3)
        papers.append((row.month, row.title, terms))
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
    if len(emerging_all) > len(emerging):
        print(f"[markets] clustering top {len(emerging):,} of {len(emerging_all):,} emerging (capped at {MAX_PHRASES})",
              flush=True)
    cooc, df = cooccurrence(frozenset(terms & eset) for _, _, terms in papers)
    clusters = cluster(emerging, cooc, df)
    p2c = {p: ci for ci, members in enumerate(clusters) for p in members}

    c_recent, c_prior, c_titles = Counter(), Counter(), defaultdict(list)
    for month, title, terms in papers:
        hit = {p2c[p] for p in (terms & eset)}
        for ci in hit:
            if month in rset:
                c_recent[ci] += 1
                if len(c_titles[ci]) < 3:
                    c_titles[ci].append(title)
            elif month in pset:
                c_prior[ci] += 1

    cg = {ci: share_growth(c_recent[ci], c_prior[ci], n_recent, n_prior) for ci in range(len(clusters))}
    top = sorted(range(len(clusters)), key=lambda ci: -cg[ci])[:args.top]
    print(f"[markets] {len(clusters):,} clusters | naming top {len(top)} with {args.model} (cached)...", flush=True)

    named = name_clusters(((clusters[ci], c_titles[ci]) for ci in top), model=args.model)
    paired = list(zip(top, named))
    abstained = [ci for ci, res in paired if res.get("status") == "error"]
    rows = [(ci, res) for ci, res in paired
            if res.get("status", "ok") == "ok" and res["is_market"] and res["coherence"] >= 0.5]
    note = f" | {len(abstained)} abstained (transient error, retry next run)" if abstained else ""
    print(f"\n[markets] {len(rows)} real emerging markets (of {len(top)} clusters; LLM rejected the rest as junk){note}:\n")
    for ci, res in sorted(rows, key=lambda x: -cg[x[0]]):
        mem = ", ".join(res["member_phrases"][:6]) or ", ".join(clusters[ci][:6])
        print(f"  {cg[ci]:6.1f}x  [{c_recent[ci]:4d} papers]  {res['market_name']}   (coh {res['coherence']:.2f})")
        print(f"            = {{{mem}}}")


if __name__ == "__main__":
    main()
