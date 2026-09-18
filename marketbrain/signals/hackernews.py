"""Hacker News story volume — a leading attention signal for dev-tools.

Monthly count of HN stories matching the market's terms, via the (keyless, fully
timestamped) Algolia HN Search API. "Show HN" / front-page volume is an early tell
for developer markets. A month's story count is complete at month-end, so
``knowledge_date = end_of(month)`` — leakage-safe, same convention as the other signals.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from ..schema import Extraction, ExtractionMethod, Observation, Provenance
from ..schema.bitemporal import SUPERSEDED_OPEN
from ..timeutil import end_of_month, month_range

HN_API = "https://hn.algolia.com/api/v1/search_by_date"
FEATURE = "hn_story_count"
LLM_DEVTOOLS_QUERY = "large language model"


def _unix_bounds(month: str) -> tuple[int, int]:
    y, m = (int(x) for x in month.split("-"))
    eo = end_of_month(month)
    lo = int(datetime(y, m, 1, tzinfo=timezone.utc).timestamp())
    hi = int(datetime(eo.year, eo.month, eo.day, 23, 59, 59, tzinfo=timezone.utc).timestamp())
    return lo, hi


def fetch_month_count(
    month: str, query: str = LLM_DEVTOOLS_QUERY, *, sleep: float = 0.5
) -> tuple[int, str]:
    """Return (story_count, request_url) for ``month`` (one Algolia call, hitsPerPage=0)."""
    lo, hi = _unix_bounds(month)
    params = {
        "query": query,
        "tags": "story",
        "numericFilters": f"created_at_i>={lo},created_at_i<={hi}",
        "hitsPerPage": 0,
    }
    url = f"{HN_API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "market-brain/0.1 (research)"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        count = int(json.loads(resp.read()).get("nbHits", 0))
    if sleep:
        time.sleep(sleep)
    return count, url


def month_count_to_observation(
    month: str, count: int, request_url: str, *, market: str, code_version: str = "v1"
) -> Observation:
    """Pure: one (month, story-count) -> a leakage-safe Observation."""
    eo = end_of_month(month)
    content_hash = hashlib.sha256(f"{request_url}|{count}".encode()).hexdigest()
    return Observation(
        observation_id=f"hn:{market}:{month}",
        feature=FEATURE,
        market=market,
        value=float(count),
        reference_date=eo,
        knowledge_date=eo,
        superseded_date=SUPERSEDED_OPEN,
        claim_text=f"{count} Hacker News stories matching the {market} query in {month}",
        provenance=Provenance(source_url=request_url, content_hash=content_hash,
                              dated_quote=f"nbHits={count}"),
        extraction=Extraction(method=ExtractionMethod.API_INGEST, source_name="hn_algolia",
                              code_version=code_version),
        confidence=1.0,
    )


def collect(
    start: str,
    end: str,
    *,
    market: str = "llm-dev-tools",
    query: str = LLM_DEVTOOLS_QUERY,
    sleep: float = 0.5,
) -> list[Observation]:
    """Fetch monthly HN story counts for [start, end] (hits the network); return Observations."""
    return [
        month_count_to_observation(m, *fetch_month_count(m, query, sleep=sleep), market=market)
        for m in month_range(start, end)
    ]
