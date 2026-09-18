"""PyPI package downloads — developer adoption (Python half — the one that matters here).

The Python complement to the npm (JS) signal. llm-dev-tools is Python-heavy (LangChain,
LlamaIndex, transformers, vLLM live on PyPI), so this is the adoption half that actually
covers the market. Monthly download totals from the BigQuery public dataset
``bigquery-public-data.pypi.file_downloads``, summed over a basket of canonical Python packages.

Leakage-safe: a month's downloads are complete at month-end → ``knowledge_date = end_of(month)``.

**Cost control (BigQuery is metered):**
- the table is partitioned by ``timestamp`` (DAY) and clustered by ``project`` — the query uses a
  partition-pruning timestamp range + a ``project IN`` filter so clustering keeps the scan small;
- a **dry-run** logs the upper-bound bytes, and ``maximum_bytes_billed`` is a runaway guard;
- the monthly result is **cached locally** (``data/raw/pypi_monthly_<start>_<end>.parquet``) so the
  query runs at most once per (start, end, basket). Delete the cache to force a refresh.

Auth via the Vela **service account**: set ``GOOGLE_APPLICATION_CREDENTIALS`` to the key JSON kept
OUTSIDE the repo (e.g. ``~/.config/gcp/``) — queries then bill to the service account's own project
(e.g. ``production-498516``). Falls back to ADC if that env var is unset. ``import google.cloud.bigquery``
is lazy so this module (and the tests) import fine without the client installed.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd

from ..schema import Extraction, ExtractionMethod, Observation, Provenance
from ..schema.bitemporal import SUPERSEDED_OPEN
from ..timeutil import end_of_month, month_range, require_month

FEATURE = "pypi_downloads"
TABLE = "bigquery-public-data.pypi.file_downloads"
MAX_BYTES_BILLED = 350 * 10**9  # runaway guard; full 7-pkg backfill bills ~315 GB actual (one-time)

# v1 llm-dev-tools Python basket (kept tight to control BigQuery cost; extend incrementally).
LLM_DEVTOOLS_PACKAGES = (
    "langchain",
    "llama-index",
    "transformers",
    "vllm",
    "openai",
    "huggingface-hub",
    "tiktoken",
)


def _cache_path(cache_dir: "str | Path", start: str, end: str, packages: tuple[str, ...]) -> Path:
    # hash the (sorted) basket into the name so changing LLM_DEVTOOLS_PACKAGES doesn't serve a stale cache
    basket = hashlib.sha256("|".join(sorted(packages)).encode()).hexdigest()[:8]
    return Path(cache_dir) / f"pypi_monthly_{start}_{end}_{basket}.parquet"


def _query_monthly(
    start: str, end: str, packages: tuple[str, ...], *, project: str, max_bytes: int
) -> pd.DataFrame:
    """Run the BigQuery monthly-download query (dry-run first, capped). Returns (month, project, downloads)."""
    from google.cloud import bigquery  # lazy: keeps the module importable without the client

    require_month(start)  # injection guard before the SQL interpolation below
    require_month(end)
    # Auth: GOOGLE_APPLICATION_CREDENTIALS (the Vela service account) if set, else ADC.
    # Billing project: the given project, else inferred from the credentials (the SA's own project).
    client = bigquery.Client(project=project) if project else bigquery.Client()
    ey, em = (int(x) for x in end.split("-"))
    upper = f"{ey + 1:04d}-01-01" if em == 12 else f"{ey:04d}-{em + 1:02d}-01"  # first day after end month
    in_list = ",".join(repr(p) for p in packages)
    sql = f"""
    SELECT TIMESTAMP_TRUNC(timestamp, MONTH) AS month, project, COUNT(*) AS downloads
    FROM `{TABLE}`
    WHERE timestamp >= TIMESTAMP('{start}-01') AND timestamp < TIMESTAMP('{upper}')
      AND project IN ({in_list})
    GROUP BY month, project
    """
    dry = client.query(sql, job_config=bigquery.QueryJobConfig(dry_run=True))
    print(f"[pypi] dry-run upper bound: {dry.total_bytes_processed / 1e9:.1f} GB")
    job = client.query(sql, job_config=bigquery.QueryJobConfig(maximum_bytes_billed=max_bytes))
    df = job.to_dataframe()
    print(f"[pypi] actual billed: {job.total_bytes_billed / 1e9:.1f} GB; {len(df)} (month,project) rows")
    return df


def collect(
    start: str,
    end: str,
    *,
    market: str = "llm-dev-tools",
    packages: tuple[str, ...] = LLM_DEVTOOLS_PACKAGES,
    project: "str | None" = None,
    cache_dir: "str | Path" = "data/raw",
    max_bytes: int = MAX_BYTES_BILLED,
    sleep: "float | None" = None,  # accepted for driver compatibility; unused
) -> list[Observation]:
    """Monthly PyPI downloads summed across the package basket; one Observation per month.

    Uses the local cache if present; otherwise runs the (capped) BigQuery query once and caches it.
    """
    project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")  # None -> infer from creds (SA project)
    cache = _cache_path(cache_dir, start, end, packages)
    if cache.exists():
        print(f"[pypi] using cached query result: {cache}")
        raw = pd.read_parquet(cache)
    else:
        raw = _query_monthly(start, end, packages, project=project, max_bytes=max_bytes)
        cache.parent.mkdir(parents=True, exist_ok=True)
        raw.to_parquet(cache, index=False)

    raw = raw.copy()
    raw["ym"] = pd.to_datetime(raw["month"]).dt.strftime("%Y-%m")
    totals = raw.groupby("ym")["downloads"].sum().to_dict()
    request_url = f"bigquery://{TABLE}?packages={'|'.join(packages)}"
    return [month_value_to_observation(m, int(totals[m]), request_url, market=market)
            for m in month_range(start, end) if totals.get(m, 0) > 0]


def month_value_to_observation(
    month: str, value: int, request_url: str, *, market: str, code_version: str = "v1"
) -> Observation:
    """Pure: one (month, total-downloads) -> a leakage-safe Observation."""
    eo = end_of_month(month)
    content_hash = hashlib.sha256(f"{request_url}|{month}|{value}".encode()).hexdigest()
    return Observation(
        observation_id=f"pypi:{market}:{month}",
        feature=FEATURE,
        market=market,
        value=float(value),
        reference_date=eo,
        knowledge_date=eo,
        superseded_date=SUPERSEDED_OPEN,
        claim_text=f"{value} PyPI downloads across the {market} package basket in {month}",
        provenance=Provenance(source_url=request_url, content_hash=content_hash,
                              dated_quote=f"sum(downloads)={value}"),
        extraction=Extraction(method=ExtractionMethod.API_INGEST, source_name="bigquery_pypi",
                              code_version=code_version),
        confidence=1.0,
    )
