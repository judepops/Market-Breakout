"""Phase 2 - build the RESEARCH dimension (arXiv) across all mature markets.

Like innovation, run for ALL markets (measure, don't pre-mask): research-heavy markets (robotics, computer
vision, semiconductors, quantum...) get real counts, non-research software (devops, saas) correctly ~0.

Per-market query is derived from the market NAME (a phrase search across all arXiv), with a few overrides
where the slug is not a good search term. To stay within arXiv's <=1 req / 3s etiquette we count ANNUALLY
(3 calls per market for a 3-year window) and collapse to {level=sum, growth=log-linear slope}. Cached.

    python -m experiments.build_research
"""

from __future__ import annotations

import argparse
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from pathlib import Path
from xml.etree import ElementTree as ET

warnings.filterwarnings("ignore", message="BigQuery Storage module not found")

ARXIV_API = "http://export.arxiv.org/api/query"
_NS = {"os": "http://a9.com/-/spec/opensearch/1.1/"}
# slug -> better arXiv phrase where the market name is not a natural search term
QUERY_OVERRIDES = {
    "nlp": "natural language processing",
    "llm-dev-tools": "large language model",
    "devops": "devops",
    "data-analytics": "data analytics",
    "data-management": "data management",
    "data-storage": "data storage",
    "ml": "machine learning",
    "agtech": "agriculture technology",
    "cleantech": "clean technology",
    "fintech": "financial technology",
    "saas": "software as a service",
    "iot": "internet of things",
    "ar": "augmented reality",
    "vr": "virtual reality",
}


def market_query(name: str, llm: "dict | None" = None) -> "str | None":
    """LLM-generated query if available (clean, disambiguated); else a name-based phrase fallback."""
    if llm is not None and llm.get(name):
        return llm[name]
    phrase = QUERY_OVERRIDES.get(name, name.replace("-", " "))
    return f'all:"{phrase}"'


def fetch_year_count(year: int, query: str, *, sleep: float = 3.0, retries: int = 5) -> "int | None":
    """arXiv submissions matching `query` in `year`. Retries with backoff on 429; None if it never succeeds
    (so the caller can skip rather than poison the cache with a fake 0)."""
    full = f"{query} AND submittedDate:[{year}01010000 TO {year}12312359]"
    url = f"{ARXIV_API}?{urllib.parse.urlencode({'search_query': full, 'start': 0, 'max_results': 1})}"
    req = urllib.request.Request(url, headers={"User-Agent": "market-brain/0.1 (research; mailto:research@example.com)"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=40) as resp:
                xml = resp.read()
            time.sleep(sleep)
            return int(ET.fromstring(xml).findtext("os:totalResults", default="0", namespaces=_NS))
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries - 1:
                back = 8 * (2 ** attempt)
                print(f"  [arxiv] 429 on {query[:28]} {year}; backoff {back}s")
                time.sleep(back)
                continue
            print(f"  [arxiv] {year} {query[:28]}: {e}")
            return None
        except Exception as e:  # noqa: BLE001 - network
            print(f"  [arxiv] {year} {query[:28]}: {e}")
            return None
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/raw/labels")
    ap.add_argument("--cache", default="data/raw/rounds")
    ap.add_argument("--figdir", default="artifacts/figures")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--sleep", type=float, default=2.0)
    ap.add_argument("--queries", default=None, help="JSON of LLM-generated {market: arxiv_query} (clean terms)")
    args = ap.parse_args(argv)

    import json

    import numpy as np
    import pandas as pd

    from marketbrain.features.signal_features import _loglinear_slope, zscore
    from marketbrain.markets.catalog import CATALOG

    ft = pd.read_parquet(Path(args.labels) / "market_features.parquet").set_index("market")
    specs = [m for m in CATALOG if m.name in ft.index and bool(ft.loc[m.name, "mature"])]

    llm = json.loads(Path(args.queries).read_text()) if args.queries and Path(args.queries).exists() else None
    tag = "_llm" if llm else ""
    cache = Path(args.cache) / f"research_years{tag}_k{args.k}.parquet"
    counts = {}
    if cache.exists():
        ydf = pd.read_parquet(cache)
        counts = {(r.market, int(r.year)): int(r.papers) for r in ydf.itertuples()}
        print(f"resuming from cache: {len(counts)} (market, year) counts already done")
    print(f"querying arXiv for {len(specs)} markets x {args.k} years "
          f"({'LLM-clean terms' if llm else 'name-based'}; skipping cached) ...")
    for m in specs:
        q = market_query(m.name, llm)
        t0 = int(ft.loc[m.name, "t0"])
        for y in range(t0, t0 + args.k):
            if (m.name, y) in counts:
                continue
            c = fetch_year_count(y, q, sleep=args.sleep)
            if c is not None:                       # only cache real results; None (429/err) is left to retry
                counts[(m.name, y)] = c
        # persist incrementally so a kill/timeout keeps progress
        pd.DataFrame([(mk, yy, n) for (mk, yy), n in counts.items()],
                     columns=["market", "year", "papers"]).to_parquet(cache, index=False)

    rows = []
    for m in specs:
        t0 = int(ft.loc[m.name, "t0"])
        if not all((m.name, y) in counts for y in range(t0, t0 + args.k)):
            continue                                # incomplete (some calls still failing) -> skip, not fake-0
        yrs = [counts[(m.name, y)] for y in range(t0, t0 + args.k)]
        rows.append({"market": m.name, "t0": t0, "research_level": int(sum(yrs)),
                     "research_growth": round(_loglinear_slope(np.array(yrs, dtype=float)), 4),
                     "query": market_query(m.name, llm)})
    if not rows:
        print("\nno markets completed yet (arXiv still rate-limiting); re-run to resume from cache.")
        return 0
    df = pd.DataFrame(rows)
    df["research_level_z"] = zscore([float(np.log1p(v)) for v in df["research_level"]])
    df["research_growth_z"] = zscore(list(df["research_growth"]))
    df = df.sort_values("research_level", ascending=False).reset_index(drop=True)
    out = Path(args.labels) / f"research_dimension{tag}.parquet"
    df.to_parquet(out, index=False)

    print(f"\n{'market':22s} {'papers':>8s} {'growth':>7s} {'lvl-z':>6s}  query")
    for _, r in df.iterrows():
        print(f"{r['market']:22s} {int(r['research_level']):>8,} {r['research_growth']:>+7.3f} "
              f"{r['research_level_z']:>6.2f}  {r['query']}")
    print(f"\n{len(df)} markets with a research dimension. wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
