"""arXiv research-momentum signal (deterministic, natively dated).

Monthly submission counts for the market's term set, pulled from the official arXiv
API. A month's submissions are fully known by the end of that month, so
``knowledge_date = end_of(month)`` and the series is **leakage-safe by construction**
(`reference_date == knowledge_date`, both within the month).

This is the first signal wired into the corpus. (Google Trends — the planned Signal 1
— is rate-limited/blocked from some networks, so arXiv goes first here.)

The query below is a v1 starting point for ``llm-dev-tools``; refining the precise
term set / category list is an ongoing task (see docs/DATASHEET.md §2).
"""

from __future__ import annotations

import hashlib
import time
import urllib.parse
import urllib.request
from xml.etree import ElementTree as ET

from ..schema import Extraction, ExtractionMethod, Observation, Provenance
from ..schema.bitemporal import SUPERSEDED_OPEN
from ..timeutil import end_of_month, month_range  # noqa: F401  (re-exported for back-compat)

ARXIV_API = "http://export.arxiv.org/api/query"
FEATURE = "arxiv_submissions"

# v1 llm-dev-tools query: CS categories relevant to LLM-application tooling + LLM terms.
LLM_DEVTOOLS_QUERY = (
    '(cat:cs.CL OR cat:cs.AI OR cat:cs.SE) AND '
    '(all:"large language model" OR all:"LLM" OR all:"retrieval augmented" '
    'OR all:"language model agent" OR all:"prompt engineering")'
)

_NS = {"a": "http://www.w3.org/2005/Atom", "os": "http://a9.com/-/spec/opensearch/1.1/"}


def _submitted_bounds(month: str) -> tuple[str, str]:
    y, m = (int(x) for x in month.split("-"))
    eo = end_of_month(month)
    return f"{y:04d}{m:02d}010000", f"{eo.year:04d}{eo.month:02d}{eo.day:02d}2359"


def fetch_month_count(
    month: str, query: str = LLM_DEVTOOLS_QUERY, *, sleep: float = 3.0
) -> tuple[int, str]:
    """Return (submission_count, request_url) for ``month`` via one arXiv API call.

    Reads ``opensearch:totalResults`` rather than paging results. Sleeps after the
    call to respect arXiv's "<= 1 request / 3s" etiquette.
    """
    lo, hi = _submitted_bounds(month)
    full_query = f"{query} AND submittedDate:[{lo} TO {hi}]"
    params = {"search_query": full_query, "start": 0, "max_results": 1}
    url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "market-brain/0.1 (research)"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        xml = resp.read()
    if sleep:
        time.sleep(sleep)
    total = int(ET.fromstring(xml).findtext("os:totalResults", default="0", namespaces=_NS))
    return total, url


def count_to_observation(
    month: str,
    count: int,
    request_url: str,
    *,
    market: str,
    code_version: str = "v1",
) -> Observation:
    """Pure: turn one (month, count) into a leakage-safe ``Observation``."""
    eo = end_of_month(month)
    content_hash = hashlib.sha256(f"{request_url}|{count}".encode()).hexdigest()
    return Observation(
        observation_id=f"arxiv:{market}:{month}",
        feature=FEATURE,
        market=market,
        value=float(count),
        reference_date=eo,   # the period the count is ABOUT
        knowledge_date=eo,   # fully known by end of that month (natively dated)
        superseded_date=SUPERSEDED_OPEN,
        claim_text=f"{count} arXiv submissions matching the {market} query in {month}",
        provenance=Provenance(
            source_url=request_url,
            content_hash=content_hash,
            dated_quote=f"opensearch:totalResults={count}",
        ),
        extraction=Extraction(
            method=ExtractionMethod.API_INGEST, source_name="arxiv_api", code_version=code_version
        ),
        confidence=1.0,
    )


def collect(
    start: str,
    end: str,
    *,
    market: str = "llm-dev-tools",
    query: str = LLM_DEVTOOLS_QUERY,
    sleep: float = 3.0,
) -> list[Observation]:
    """Fetch monthly counts for [start, end] (hits the network); return Observations."""
    return [
        count_to_observation(m, *fetch_month_count(m, query, sleep=sleep), market=market)
        for m in month_range(start, end)
    ]
