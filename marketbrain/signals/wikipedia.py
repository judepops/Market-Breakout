"""Wikipedia pageviews — market interest / attention.

The clean, reachable alternative to Google Trends (which is IP-blocked from some
networks). Monthly pageviews from the official Wikimedia REST API, summed over a
set of representative articles for the market. Keyless, dated, and not relative-
scaled (unlike Trends). A month's total is complete at month-end, so
``knowledge_date = end_of(month)`` — leakage-safe, same convention as the arXiv signal.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request

from ..schema import Extraction, ExtractionMethod, Observation, Provenance
from ..schema.bitemporal import SUPERSEDED_OPEN
from ..timeutil import end_of_month, month_range

WIKIMEDIA_API = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "en.wikipedia/all-access/all-agents/{article}/monthly/{start}/{end}"
)
FEATURE = "wikipedia_pageviews"
_UA = "halflife-research/0.1 (research; " + __import__("os").environ.get("CONTACT_EMAIL", "you@example.com") + ")"

# v1 llm-dev-tools article basket (representative English Wikipedia titles).
LLM_DEVTOOLS_ARTICLES = (
    "Large_language_model",
    "ChatGPT",
    "Generative_artificial_intelligence",
    "Prompt_engineering",
    "GPT-4",
    "Hugging_Face",
)


def _next_month(month: str) -> str:
    y, m = (int(x) for x in month.split("-"))
    return f"{y + 1:04d}-01" if m == 12 else f"{y:04d}-{m + 1:02d}"


def _fetch_article_monthly(
    article: str, start: str, end: str, *, sleep: float
) -> tuple[dict[str, int], str]:
    """Return ({month: views}, request_url) for one article over [start, end] (one API call).

    The API ``end`` is set to the first of the month *after* ``end`` so the end month's
    full bucket is returned — passing the end month's own first day yields a *partial*
    bucket (the cause of the Dec-2025 ~97% dip). Any partial trailing bucket the API
    returns past ``end`` is filtered out by the caller (``if month in totals``).
    """
    url = WIKIMEDIA_API.format(
        article=urllib.parse.quote(article, safe=""),
        start=f"{start.replace('-', '')}0100",
        end=f"{_next_month(end).replace('-', '')}0100",
    )
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            items = json.loads(resp.read()).get("items", [])
    except urllib.error.HTTPError as exc:
        if exc.code == 404:  # article has no data in range
            return {}, url
        raise
    finally:
        if sleep:
            time.sleep(sleep)
    monthly = {f"{it['timestamp'][:4]}-{it['timestamp'][4:6]}": int(it["views"]) for it in items}
    return monthly, url


def collect(
    start: str,
    end: str,
    *,
    market: str = "llm-dev-tools",
    articles: tuple[str, ...] = LLM_DEVTOOLS_ARTICLES,
    sleep: float = 1.0,
) -> list[Observation]:
    """Sum monthly pageviews across the article basket; return one Observation per month."""
    totals: dict[str, int] = {m: 0 for m in month_range(start, end)}
    urls: list[str] = []
    for article in articles:
        monthly, url = _fetch_article_monthly(article, start, end, sleep=sleep)
        urls.append(url)
        for month, views in monthly.items():
            if month in totals:
                totals[month] += views

    provenance_url = " ; ".join(urls)
    return [month_value_to_observation(m, totals[m], provenance_url, market=market)
            for m in month_range(start, end) if totals[m] > 0]


def month_value_to_observation(
    month: str, value: int, request_url: str, *, market: str, code_version: str = "v1"
) -> Observation:
    """Pure: one (month, total-pageviews) -> a leakage-safe Observation."""
    eo = end_of_month(month)
    content_hash = hashlib.sha256(f"{request_url}|{month}|{value}".encode()).hexdigest()
    return Observation(
        observation_id=f"wikipedia:{market}:{month}",
        feature=FEATURE,
        market=market,
        value=float(value),
        reference_date=eo,
        knowledge_date=eo,
        superseded_date=SUPERSEDED_OPEN,
        claim_text=f"{value} Wikipedia pageviews across the {market} article basket in {month}",
        provenance=Provenance(source_url=request_url, content_hash=content_hash,
                              dated_quote=f"sum(views)={value}"),
        extraction=Extraction(method=ExtractionMethod.API_INGEST, source_name="wikimedia_rest",
                              code_version=code_version),
        confidence=1.0,
    )
