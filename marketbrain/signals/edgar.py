"""SEC EDGAR Form D filings — capital / funding (the 4th dimension).

A **Form D** is the notice a US company files for an exempt (private) securities
offering — i.e. a private fundraising round. The monthly count of Form D filings
across a curated basket of llm-dev-tools companies is a funding-activity signal:
free, official, and natively dated by **filing date** (public when filed). The
monthly count is complete at month-end, so ``knowledge_date = end_of(month)`` —
leakage-safe, same convention as the other signals.

Sparse by nature (a company files a Form D only when it raises). The company basket
is the entity->market mapping (the analogue of the npm/arXiv basket) — a v1 starting
point. Counts include genuine-zero months (no funding events is a real datum).
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

EFTS_API = "https://efts.sec.gov/LATEST/search-index"
FEATURE = "edgar_formd_count"
_UA = "halflife-research " + __import__("os").environ.get("CONTACT_EMAIL", "you@example.com")  # SEC requires a UA with contact

# v1 llm-dev-tools company basket (distinctive issuer names likely to appear in a Form D).
LLM_DEVTOOLS_COMPANIES = (
    "LangChain",
    "Pinecone",
    "Hugging Face",
    "Anyscale",
    "Baseten",
    "Weaviate",
    "LlamaIndex",
    "Perplexity",
)


def _get(url: str, *, retries: int = 4) -> "dict | None":
    """GET JSON with backoff on SEC 5xx / rate-limiting. ``{}`` on 403/404 (no data);
    ``None`` on persistent failure (so the caller can skip that company, not crash)."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 404):
                return {}
            time.sleep(1.5 * (attempt + 1))  # 5xx (SEC sometimes 500s rapid/odd queries)
        except urllib.error.URLError:
            time.sleep(1.5 * (attempt + 1))
    return None


def fetch_company_formd_dates(company: str, start: str, end: str, *, sleep: float) -> list[str]:
    """All Form D filing dates for one company in [start, end] (paginated, capped at 100).

    Resilient: if SEC persistently errors for this company's query, log a warning and
    return what was collected so far rather than crashing the whole ingest.
    """
    q = urllib.parse.quote(f'"{company}"')
    eo = end_of_month(end)
    base = f"{EFTS_API}?q={q}&forms=D&startdt={start}-01&enddt={eo.isoformat()}"
    dates: list[str] = []
    frm = 0
    while True:
        data = _get(f"{base}&from={frm}")
        if data is None:
            print(f"[edgar] WARN: persistent SEC error for {company!r} (from={frm}); skipping")
            break
        hits = data.get("hits", {})
        rows = hits.get("hits", [])
        total = hits.get("total", {}).get("value", 0)
        for h in rows:
            file_date = h.get("_source", {}).get("file_date")
            if file_date:
                dates.append(file_date)
        frm += len(rows)
        if sleep:
            time.sleep(sleep)
        if not rows or frm >= total or frm >= 100:
            break
    return dates


def collect(
    start: str,
    end: str,
    *,
    market: str = "llm-dev-tools",
    companies: tuple[str, ...] = LLM_DEVTOOLS_COMPANIES,
    sleep: float = 0.7,
) -> list[Observation]:
    """Count Form D filings per month across the company basket; one Observation per month
    (including genuine-zero months — a count signal's 0 is a real datum)."""
    counts: dict[str, int] = {m: 0 for m in month_range(start, end)}
    for company in companies:
        for file_date in fetch_company_formd_dates(company, start, end, sleep=sleep):
            month = file_date[:7]
            if month in counts:
                counts[month] += 1

    request_url = f"{EFTS_API}?forms=D&q=<basket:{'|'.join(companies)}>"
    return [month_count_to_observation(m, counts[m], request_url, market=market)
            for m in month_range(start, end)]


def month_count_to_observation(
    month: str, count: int, request_url: str, *, market: str, code_version: str = "v1"
) -> Observation:
    """Pure: one (month, Form-D-count) -> a leakage-safe Observation."""
    eo = end_of_month(month)
    content_hash = hashlib.sha256(f"{request_url}|{month}|{count}".encode()).hexdigest()
    return Observation(
        observation_id=f"edgar:{market}:{month}",
        feature=FEATURE,
        market=market,
        value=float(count),
        reference_date=eo,
        knowledge_date=eo,
        superseded_date=SUPERSEDED_OPEN,
        claim_text=f"{count} SEC Form D filings across the {market} company basket in {month}",
        provenance=Provenance(source_url=request_url, content_hash=content_hash,
                              dated_quote=f"form_d_count={count}"),
        extraction=Extraction(method=ExtractionMethod.API_INGEST, source_name="sec_edgar_efts",
                              code_version=code_version),
        confidence=1.0,
    )
