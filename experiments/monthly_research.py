"""Step 2: MONTHLY arXiv research panel per market, macro-normalised by total cs.* volume.

    python -m experiments.monthly_research

Monthly resolution (the original corpus resolution; finest for the lead-lag). One totalResults call per
(market, month) using the market's LLM-generated arXiv query, plus one per month for the cs.* denominator
(secular-growth fix). Resumable cache. arXiv's 3s etiquette makes this a multi-hour background job.
"""
from __future__ import annotations
import calendar
import json
import time
import urllib.parse
import urllib.request
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

warnings.filterwarnings("ignore")
ARXIV = "http://export.arxiv.org/api/query"
NS = {"os": "http://a9.com/-/spec/opensearch/1.1/"}
CS = "(cat:cs.LG OR cat:cs.AI OR cat:cs.CL OR cat:cs.CV OR cat:cs.SE OR cat:cs.DC OR cat:cs.CR OR cat:cs.DB)"
Y0, Y1 = 2012, 2025
OUT = Path("data/raw/timeseries")
CACHE = OUT / "monthly_research_cache.json"


def _bounds(year: int, mo: int) -> tuple[str, str]:
    last = calendar.monthrange(year, mo)[1]
    return f"{year}{mo:02d}010000", f"{year}{mo:02d}{last:02d}2359"


def _count(query: str, year: int, mo: int, sleep: float = 3.0, retries: int = 5) -> "int | None":
    lo, hi = _bounds(year, mo)
    full = f"{query} AND submittedDate:[{lo} TO {hi}]"
    url = f"{ARXIV}?{urllib.parse.urlencode({'search_query': full, 'start': 0, 'max_results': 1})}"
    req = urllib.request.Request(url, headers={"User-Agent": "market-brain/0.1 (research)"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=40) as resp:
                xml = resp.read()
            time.sleep(sleep)
            return int(ET.fromstring(xml).findtext("os:totalResults", default="0", namespaces=NS))
        except Exception:  # noqa: BLE001
            time.sleep(4 * 2 ** attempt)
    return None


def main():
    import pandas as pd
    OUT.mkdir(parents=True, exist_ok=True)
    queries = json.loads(Path("data/raw/labels/arxiv_queries.json").read_text())
    pf = pd.read_parquet("data/raw/labels/partition_features.parquet")
    markets = [m for m in pf["market"] if queries.get(m)]       # all 51
    months = [(y, mo) for y in range(Y0, Y1 + 1) for mo in range(1, 13)]
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    print(f"[research] {len(markets)} markets x {len(months)} months; {len(cache)} cached", flush=True)

    for (y, mo) in months:                                      # cs.* denominator
        k = f"_CS|{y}|{mo}"
        if k not in cache:
            cache[k] = _count(CS, y, mo)
            CACHE.write_text(json.dumps(cache))
    done = 0
    for mk in markets:
        for (y, mo) in months:
            k = f"{mk}|{y}|{mo}"
            if k not in cache:
                cache[k] = _count(queries[mk], y, mo)
                CACHE.write_text(json.dumps(cache))
            done += 1
        print(f"  {mk:24s} done ({done}/{len(markets)*len(months)})", flush=True)

    rows = []
    for mk in markets:
        for (y, mo) in months:
            c = cache.get(f"{mk}|{y}|{mo}")
            cs = cache.get(f"_CS|{y}|{mo}")
            if c is not None and cs:
                rows.append({"market": mk, "year": y, "month": mo,
                             "papers": c, "cs_total": cs, "share": c / cs})
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "monthly_research.parquet", index=False)
    print(f"\n[research] wrote {len(df)} rows across {df['market'].nunique()} markets -> "
          f"{OUT/'monthly_research.parquet'}", flush=True)


if __name__ == "__main__":
    main()
