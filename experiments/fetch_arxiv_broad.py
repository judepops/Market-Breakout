"""Kick off the BROAD AI-infra arXiv corpus pull -- the discovery corpus for the emergence method.

    python -m experiments.fetch_arxiv_broad

The existing data/raw/arxiv_records/ is a NARROW, term-filtered pull (LLM-devtools query), unusable for
bottom-up theme DISCOVERY. This pulls the whole AI-infra slice (cs.LG/CL/AI/RO/SE/DC + stat.ML), 2018-2026,
per-month, resumable, to a SEPARATE cache dir so it does not collide with the narrow pull. Titles + abstracts
+ authors -- the input to the reason-cluster taxonomy and the breakout-signature backtest.
"""
from marketbrain.signals.arxiv_records import fetch_records

QUERY = ("(cat:cs.LG OR cat:cs.CL OR cat:cs.AI OR cat:cs.RO OR cat:cs.SE OR cat:cs.DC OR cat:stat.ML)")
START, END = "2018-01", "2026-06"
CACHE = "data/raw/arxiv_broad"


def main():
    print(f"[arxiv-broad] {QUERY}\n  {START} -> {END}  cache={CACHE}", flush=True)
    df = fetch_records(START, END, query=QUERY, sleep=3.0, page_size=1000, cache_dir=CACHE, verbose=True)
    print(f"[arxiv-broad] done: {len(df):,} records in {CACHE}", flush=True)


if __name__ == "__main__":
    main()
