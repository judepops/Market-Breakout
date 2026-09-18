"""Phase 1 GRANULARITY SMOKE-TEST: do the deterministic gates yield idea-level themes (the #1 de-risk)?

    python -m experiments.em_granularity            # cs.AI slice
    python -m experiments.em_granularity --all       # whole downloaded corpus

The eyeball test for the team: do the top emerging phrases read at the "RL with verifiable rewards" altitude, or
do we get "machine learning"? The gates (doc-freq ceiling + PMI + generic/all-generic drop + noun-phrase POS
filter) live in marketbrain/markets/granularity_gates.py; this is just the tuning/eyeball harness over them.
"""
from __future__ import annotations
import argparse
import warnings

warnings.filterwarnings("ignore")
RECENT, PRIOR = 12, 12
MIN_RECENT = 8


def main(argv=None):
    from marketbrain.corpus.load_full import load_corpus
    from marketbrain.markets.granularity_gates import gate

    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="use the whole downloaded corpus, not just cs.AI")
    ap.add_argument("--no-pos", action="store_true", help="skip the spaCy noun-phrase filter (see the raw noise)")
    ap.add_argument("--refresh", action="store_true", help="force a corpus-cache rebuild (full-corpus only)")
    args = ap.parse_args(argv)

    corpus = load_corpus(None if args.all else ["cs.AI"], cache=args.all, refresh=args.refresh)
    months = sorted(corpus["month"].unique())
    recent_months = months[-RECENT:]
    prior_months = months[-(RECENT + PRIOR):-RECENT]
    print(f"[smoke] {len(corpus):,} papers, {len(months)} months ({months[0]}..{months[-1]}) | "
          f"pos_filter={not args.no_pos}", flush=True)

    cand = gate(corpus, recent_months, prior_months, pos_filter=not args.no_pos)
    emerging = cand[cand["recent"] >= MIN_RECENT]
    print(f"[smoke] {len(cand):,} phrases passed the gates | {len(emerging):,} have recent signal")
    print(f"[smoke] top 45 EMERGING fine-grained themes (recent {RECENT}mo growth vs prior {PRIOR}mo):\n")
    for _, r in emerging.head(45).iterrows():
        print(f"  {r['growth']:6.1f}x  {r['phrase']:40s}  (recent {int(r['recent']):4d}, prior {int(r['prior']):4d}, "
              f"total {int(r['doc_freq'])})")


if __name__ == "__main__":
    main()
