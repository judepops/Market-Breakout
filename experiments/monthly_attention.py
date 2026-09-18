"""Step 2b: MONTHLY Hacker News attention panel per market, macro-normalised by total HN story volume.

    python -m experiments.monthly_attention

One Algolia nbHits call per (market, month) using the market's HN query, plus one per month for the total-HN
denominator. HN is far faster than arXiv (~0.5s/call). Resumable cache.
"""
from __future__ import annotations
import calendar
import json
import time
import urllib.parse
import urllib.request
import warnings
from datetime import datetime, timezone
from json import loads
from pathlib import Path

warnings.filterwarnings("ignore")
HN = "http://hn.algolia.com/api/v1/search"
Y0, Y1 = 2012, 2025
OUT = Path("data/raw/timeseries")
CACHE = OUT / "monthly_attention_cache.json"


def _bounds(year: int, mo: int) -> tuple[int, int]:
    last = calendar.monthrange(year, mo)[1]
    lo = int(datetime(year, mo, 1, tzinfo=timezone.utc).timestamp())
    hi = int(datetime(year, mo, last, 23, 59, 59, tzinfo=timezone.utc).timestamp())
    return lo, hi


def _count(query: str, year: int, mo: int, sleep: float = 0.5, retries: int = 5) -> "int | None":
    lo, hi = _bounds(year, mo)
    params = {"query": query, "tags": "story",
              "numericFilters": f"created_at_i>={lo},created_at_i<={hi}", "hitsPerPage": 0}
    url = f"{HN}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "market-brain/0.1 (research)"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                n = int(loads(resp.read()).get("nbHits", 0))
            time.sleep(sleep)
            return n
        except Exception:  # noqa: BLE001
            time.sleep(4 * 2 ** attempt)
    return None


def main():
    import pandas as pd
    OUT.mkdir(parents=True, exist_ok=True)
    queries = json.loads(Path("data/raw/labels/hn_queries_partition.json").read_text())
    pf = pd.read_parquet("data/raw/labels/partition_features.parquet")
    markets = [m for m in pf["market"] if queries.get(m)]       # all 51
    months = [(y, mo) for y in range(Y0, Y1 + 1) for mo in range(1, 13)]
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    print(f"[attention] {len(markets)} markets x {len(months)} months; {len(cache)} cached", flush=True)

    for (y, mo) in months:                                      # total-HN denominator
        k = f"_TOT|{y}|{mo}"
        if k not in cache:
            cache[k] = _count("", y, mo)
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
            tot = cache.get(f"_TOT|{y}|{mo}")
            if c is not None and tot:
                rows.append({"market": mk, "year": y, "month": mo,
                             "stories": c, "hn_total": tot, "share": c / tot})
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "monthly_attention.parquet", index=False)
    print(f"\n[attention] wrote {len(df)} rows across {df['market'].nunique()} markets -> "
          f"{OUT/'monthly_attention.parquet'}", flush=True)


if __name__ == "__main__":
    main()
