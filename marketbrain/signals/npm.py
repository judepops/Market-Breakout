"""npm package downloads — developer adoption (JS/npm slice only).

Monthly download totals (npm registry downloads API) summed over a basket of canonical
JS/TS LLM-dev-tools packages, with full daily history (unlike GitHub *dependents*, which
are snapshot-forward only). A month's total is complete at month-end, so
``knowledge_date = end_of(month)`` — leakage-safe, same convention as the other signals.

**Scope caveat:** llm-dev-tools is Python-heavy (LangChain, LlamaIndex, transformers,
vLLM all live on PyPI). npm covers only the JS half — the adoption dimension is NOT
complete until the **PyPI** half is added via the BigQuery public dataset
``bigquery-public-data.pypi.file_downloads`` (the `pypistats` HTTP endpoint is rate-
limited; BigQuery is the durable path). Treat this module as the easy first half.

The package basket is a v1 starting point; refining it is an ongoing task.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from datetime import date

from ..schema import Extraction, ExtractionMethod, Observation, Provenance
from ..schema.bitemporal import SUPERSEDED_OPEN
from ..timeutil import end_of_month, month_range

NPM_API = "https://api.npmjs.org/downloads/range/{period}/{package}"
FEATURE = "npm_downloads"
_UA = "halflife-research/0.1 (research; " + __import__("os").environ.get("CONTACT_EMAIL", "you@example.com") + ")"

# v1 llm-dev-tools npm basket (JS/TS LLM-application tooling).
LLM_DEVTOOLS_PACKAGES = (
    "openai",
    "langchain",
    "ai",                       # Vercel AI SDK
    "llamaindex",
    "@langchain/core",
    "@anthropic-ai/sdk",
    "@google/generative-ai",
    "@huggingface/inference",
)


def _chunks(start: str, end: str) -> list[tuple[str, str]]:
    """<=1-year [YYYY-MM-DD, YYYY-MM-DD] windows covering start..end (npm caps a range at 18 months)."""
    sy, sm = (int(x) for x in start.split("-"))
    lo = date(sy, sm, 1)
    eo = end_of_month(end)
    out = []
    while lo <= eo:
        hi = min(date(lo.year, 12, 31), eo)
        out.append((lo.isoformat(), hi.isoformat()))
        lo = date(hi.year + 1, 1, 1)
    return out


def _fetch_package_monthly(
    package: str, start: str, end: str, *, sleep: float
) -> tuple[dict[str, int], str]:
    """Sum daily downloads into {month: total} for one package; returns (monthly, sample_url)."""
    monthly: dict[str, int] = {}
    sample_url = ""
    for lo, hi in _chunks(start, end):
        url = NPM_API.format(period=f"{lo}:{hi}", package=package)
        sample_url = sample_url or url
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                days = json.loads(resp.read()).get("downloads") or []
        except urllib.error.HTTPError as exc:
            if exc.code in (400, 404):  # package missing / no data in window
                days = []
            else:
                raise
        finally:
            if sleep:
                time.sleep(sleep)
        for d in days:
            m = d["day"][:7]
            monthly[m] = monthly.get(m, 0) + int(d["downloads"])
    return monthly, sample_url


def collect(
    start: str,
    end: str,
    *,
    market: str = "llm-dev-tools",
    packages: tuple[str, ...] = LLM_DEVTOOLS_PACKAGES,
    sleep: float = 0.3,
) -> list[Observation]:
    """Sum monthly downloads across the package basket; return one Observation per month."""
    totals: dict[str, int] = {m: 0 for m in month_range(start, end)}
    urls: list[str] = []
    for package in packages:
        monthly, url = _fetch_package_monthly(package, start, end, sleep=sleep)
        if url:
            urls.append(url)
        for month, downloads in monthly.items():
            if month in totals:
                totals[month] += downloads

    provenance_url = " ; ".join(urls)
    return [month_value_to_observation(m, totals[m], provenance_url, market=market)
            for m in month_range(start, end) if totals[m] > 0]


def month_value_to_observation(
    month: str, value: int, request_url: str, *, market: str, code_version: str = "v1"
) -> Observation:
    """Pure: one (month, total-downloads) -> a leakage-safe Observation."""
    eo = end_of_month(month)
    content_hash = hashlib.sha256(f"{request_url}|{month}|{value}".encode()).hexdigest()
    return Observation(
        observation_id=f"npm:{market}:{month}",
        feature=FEATURE,
        market=market,
        value=float(value),
        reference_date=eo,
        knowledge_date=eo,
        superseded_date=SUPERSEDED_OPEN,
        claim_text=f"{value} npm downloads across the {market} package basket in {month}",
        provenance=Provenance(source_url=request_url, content_hash=content_hash,
                              dated_quote=f"sum(downloads)={value}"),
        extraction=Extraction(method=ExtractionMethod.API_INGEST, source_name="npm_registry",
                              code_version=code_version),
        confidence=1.0,
    )
