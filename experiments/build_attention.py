"""Phase 2 - build the ATTENTION dimension (Hacker News) across all mature markets.

Hacker News (Algolia, keyless, 2007+) is the robust all-window attention signal - Wikipedia pageviews only
go back to 2015, too late for the older deep-tech formation windows, so HN carries attention. Per market:
annual HN story counts matching the market term over [t0, t0+k], collapsed to {level, growth}, z-scored.
Run for ALL markets (measure, don't pre-mask). Resumable + backoff like the research builder.

    python -m experiments.build_attention
"""

from __future__ import annotations

import argparse
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from datetime import datetime, timezone
from json import loads
from pathlib import Path

warnings.filterwarnings("ignore", message="BigQuery Storage module not found")

HN_API = "https://hn.algolia.com/api/v1/search"
# slug -> better HN query where the market name is not a natural phrase
QUERY_OVERRIDES = {
    "nlp": "natural language processing", "llm-dev-tools": "large language model",
    "saas": "software as a service", "ml": "machine learning", "agtech": "agriculture technology",
    "cleantech": "clean technology", "fintech": "fintech",
}


def market_query(name: str, llm: "dict | None" = None) -> str:
    if llm is not None and llm.get(name):
        return llm[name]                                  # clean LLM-generated HN query
    return QUERY_OVERRIDES.get(name, name.replace("-", " "))


def fetch_year_count(year: int, query: str, *, sleep: float = 0.6, retries: int = 5) -> "int | None":
    """HN stories matching `query` in calendar `year` (Algolia nbHits). Retries on 429; None if it never works."""
    lo = int(datetime(year, 1, 1, tzinfo=timezone.utc).timestamp())
    hi = int(datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc).timestamp())
    params = {"query": query, "tags": "story",
              "numericFilters": f"created_at_i>={lo},created_at_i<={hi}", "hitsPerPage": 0}
    url = f"{HN_API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "market-brain/0.1 (research)"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                n = int(loads(resp.read()).get("nbHits", 0))
            time.sleep(sleep)
            return n
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries - 1:
                time.sleep(5 * (2 ** attempt))
                continue
            print(f"  [hn] {year} {query[:28]}: {e}")
            return None
        except Exception as e:  # noqa: BLE001
            print(f"  [hn] {year} {query[:28]}: {e}")
            return None
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/raw/labels")
    ap.add_argument("--features", default="data/raw/labels/market_features.parquet",
                    help="feature table with `market`, `t0`, `mature` (catalog OR partition)")
    ap.add_argument("--tag", default="", help="cache/output suffix (e.g. 'partition') to avoid collisions")
    ap.add_argument("--queries", default=None, help="JSON of LLM-generated {market: hn_query} (clean terms)")
    ap.add_argument("--cache", default="data/raw/rounds")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--sleep", type=float, default=0.6)
    args = ap.parse_args(argv)

    import json

    import numpy as np
    import pandas as pd

    from marketbrain.features.signal_features import _loglinear_slope, zscore

    ft = pd.read_parquet(args.features)
    markets = [(r["market"], int(r["t0"])) for _, r in ft.iterrows() if bool(r["mature"])]
    llm = json.loads(Path(args.queries).read_text()) if args.queries and Path(args.queries).exists() else None

    cache = Path(args.cache) / f"attention_years{args.tag}_k{args.k}.parquet"
    counts = {}
    if cache.exists():
        counts = {(r.market, int(r.year)): int(r.stories) for r in pd.read_parquet(cache).itertuples()}
        print(f"resuming: {len(counts)} cached")
    print(f"querying HN for {len(markets)} markets x {args.k} years ...")
    for mk, t0 in markets:
        q = market_query(mk, llm)
        for y in range(t0, t0 + args.k):
            if (mk, y) in counts:
                continue
            c = fetch_year_count(y, q, sleep=args.sleep)
            if c is not None:
                counts[(mk, y)] = c
        pd.DataFrame([(m, yy, n) for (m, yy), n in counts.items()],
                     columns=["market", "year", "stories"]).to_parquet(cache, index=False)

    rows = []
    for mk, t0 in markets:
        if not all((mk, y) in counts for y in range(t0, t0 + args.k)):
            continue
        yrs = [counts[(mk, y)] for y in range(t0, t0 + args.k)]
        rows.append({"market": mk, "t0": t0, "attention_level": int(sum(yrs)),
                     "attention_growth": round(_loglinear_slope(np.array(yrs, dtype=float)), 4),
                     "query": market_query(mk, llm)})
    if not rows:
        print("\nno markets completed (HN still failing); re-run to resume.")
        return 0
    df = pd.DataFrame(rows)
    df["attention_level_z"] = zscore([float(np.log1p(v)) for v in df["attention_level"]])
    df["attention_growth_z"] = zscore(list(df["attention_growth"]))
    df = df.sort_values("attention_level", ascending=False).reset_index(drop=True)
    out = Path(args.labels) / f"attention_dimension{args.tag}.parquet"
    df.to_parquet(out, index=False)
    print(f"\n{'market':22s} {'stories':>8s} {'growth':>7s} {'lvl-z':>6s}  query")
    for _, r in df.iterrows():
        print(f"{r['market']:22s} {int(r['attention_level']):>8,} {r['attention_growth']:>+7.3f} "
              f"{r['attention_level_z']:>6.2f}  {r['query']}")
    print(f"\n{len(df)} markets with an attention dimension. wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
