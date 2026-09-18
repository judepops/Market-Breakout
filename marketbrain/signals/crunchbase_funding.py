"""Crunchbase funding — capital / funding (real dollars raised).

Monthly **dollars raised** by the ``llm-dev-tools`` company roster, from Crunchbase
``funding_rounds`` in BigQuery (Vela's private dataset). Replaces the sparse SEC EDGAR
Form-D *counts* with real funding magnitude. Dated by ``announced_on`` (when the round
became public) → ``knowledge_date = end_of(month)``, leakage-safe.

Uses the shared roster (``marketbrain.markets.llm_dev_tools``) — the same company list the
$500M label and the enrichment consume. Auth via the Vela service account
(``GOOGLE_APPLICATION_CREDENTIALS``); the query is tiny (~23 MB) but still dry-run + capped +
cached for discipline. ``import google.cloud.bigquery`` is lazy (module/tests load without it).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd

from ..markets import llm_dev_tools
from ..schema import Extraction, ExtractionMethod, Observation, Provenance
from ..schema.bitemporal import SUPERSEDED_OPEN
from ..timeutil import end_of_month, month_range, require_month

FEATURE = "crunchbase_funding_usd"
TABLE = "production-498516.Crunchbase_Dataset.funding_rounds"
MAX_BYTES_BILLED = 5 * 10**9  # runaway guard; the real query bills ~23 MB


def _cache_path(cache_dir: "str | Path", start: str, end: str, permalinks: tuple[str, ...]) -> Path:
    roster = hashlib.sha256("|".join(sorted(permalinks)).encode()).hexdigest()[:8]
    return Path(cache_dir) / f"crunchbase_funding_{start}_{end}_{roster}.parquet"


def _query_monthly(
    start: str, end: str, permalinks: tuple[str, ...], *, project: "str | None", max_bytes: int
) -> pd.DataFrame:
    from google.cloud import bigquery  # lazy

    require_month(start)  # injection guard before the SQL interpolation below
    require_month(end)
    client = bigquery.Client(project=project) if project else bigquery.Client()
    eo = end_of_month(end)
    in_list = ",".join(repr(p) for p in permalinks)
    sql = f"""
    SELECT DATE_TRUNC(announced_on, MONTH) AS month,
           COALESCE(SUM(money_raised_value_usd), 0) AS raised_usd,
           COUNT(*) AS rounds
    FROM `{TABLE}`
    WHERE funded_organization_identifier_permalink IN ({in_list})
      AND announced_on >= DATE('{start}-01') AND announced_on <= DATE('{eo.isoformat()}')
    GROUP BY month
    """
    dry = client.query(sql, job_config=bigquery.QueryJobConfig(dry_run=True))
    print(f"[crunchbase] dry-run: {dry.total_bytes_processed / 1e6:.0f} MB")
    job = client.query(sql, job_config=bigquery.QueryJobConfig(maximum_bytes_billed=max_bytes))
    df = job.to_dataframe()
    print(f"[crunchbase] actual billed: {job.total_bytes_billed / 1e6:.0f} MB; {len(df)} active months")
    return df


def collect(
    start: str,
    end: str,
    *,
    market: str = "llm-dev-tools",
    permalinks: "tuple[str, ...] | None" = None,
    project: "str | None" = None,
    cache_dir: "str | Path" = "data/raw",
    max_bytes: int = MAX_BYTES_BILLED,
    sleep: "float | None" = None,  # driver compat; unused
) -> list[Observation]:
    """Monthly $ raised by the roster; one Observation per month in range (0 when no disclosed rounds)."""
    permalinks = permalinks or llm_dev_tools.permalinks()
    project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
    cache = _cache_path(cache_dir, start, end, permalinks)
    if cache.exists():
        print(f"[crunchbase] using cached query result: {cache}")
        raw = pd.read_parquet(cache)
    else:
        raw = _query_monthly(start, end, permalinks, project=project, max_bytes=max_bytes)
        cache.parent.mkdir(parents=True, exist_ok=True)
        raw.to_parquet(cache, index=False)

    raw = raw.copy()
    raw["ym"] = pd.to_datetime(raw["month"]).dt.strftime("%Y-%m")
    raised = raw.set_index("ym")["raised_usd"].to_dict()
    rounds = raw.set_index("ym")["rounds"].to_dict()
    request_url = f"bigquery://{TABLE}?roster={market}"
    return [month_value_to_observation(m, int(raised.get(m, 0) or 0), int(rounds.get(m, 0) or 0),
                                       request_url, market=market)
            for m in month_range(start, end)]


def month_value_to_observation(
    month: str, raised_usd: int, rounds: int, request_url: str, *, market: str, code_version: str = "v1"
) -> Observation:
    """Pure: one (month, $-raised, #rounds) -> a leakage-safe Observation (value = $ raised USD)."""
    eo = end_of_month(month)
    content_hash = hashlib.sha256(f"{request_url}|{month}|{raised_usd}|{rounds}".encode()).hexdigest()
    return Observation(
        observation_id=f"crunchbase:{market}:{month}",
        feature=FEATURE,
        market=market,
        value=float(raised_usd),
        reference_date=eo,
        knowledge_date=eo,
        superseded_date=SUPERSEDED_OPEN,
        claim_text=f"${raised_usd:,} raised across {rounds} {market} funding round(s) announced in {month}",
        provenance=Provenance(source_url=request_url, content_hash=content_hash,
                              dated_quote=f"raised_usd={raised_usd};rounds={rounds}"),
        extraction=Extraction(method=ExtractionMethod.API_INGEST, source_name="crunchbase_bigquery",
                              code_version=code_version),
        confidence=1.0,
    )
