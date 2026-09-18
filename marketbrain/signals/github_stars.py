"""GitHub stars — developer adoption (the highest-value dev-tools adoption signal).

Monthly count of **stars gained** (GitHub `WatchEvent`s) across the market's anchor repos
(``markets.llm_dev_tools.github_repos()``), from the **GH Archive** BigQuery public dataset
(`githubarchive.month.*` — every public GitHub event since 2011, dated by `created_at`).

The lit review flags OSS dependency growth + HN→GitHub-stars as the single highest-value dev-tools adoption
signal — stars capture mind-share/adoption that package *downloads* miss, and they're strong exactly where
patents are ~0 (the open-source dev-tools). Leakage-safe by construction: a star event in month T is public
the moment it happens, so ``reference_date = knowledge_date = end_of(T)`` (like the other activity signals).

GH Archive solves the methodology's "GitHub = live count, no history" caveat: the dated event stream gives a
real monthly star-velocity series, not just a current snapshot. ``google.cloud.bigquery`` is lazy.

Cost note: the month tables aren't clustered on repo, so a query scans ~260-300 GB regardless of how many
repos (dry-run + ``maximum_bytes_billed`` cap + cache; a one-time pull, like the PyPI backfill).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd

from ..markets import get_market
from ..schema import Extraction, ExtractionMethod, Observation, Provenance
from ..schema.bitemporal import SUPERSEDED_OPEN
from ..timeutil import end_of_month, month_range, require_month

FEATURE = "github_stars"
DATASET = "githubarchive.month"
MAX_BYTES_BILLED = 400 * 10**9  # runaway guard; the real query bills ~260-300 GB (one-time, cached)


def _cache_path(cache_dir: "str | Path", start: str, end: str, repos: "tuple[str, ...]") -> Path:
    key = hashlib.sha256("|".join(sorted(repos)).encode()).hexdigest()[:8]
    return Path(cache_dir) / f"github_stars_{start}_{end}_{key}.parquet"


def _query_monthly(
    start: str, end: str, repos: "tuple[str, ...]", *, project: "str | None", max_bytes: int
) -> pd.DataFrame:
    from google.cloud import bigquery  # lazy

    require_month(start)
    require_month(end)
    client = bigquery.Client(project=project) if project else bigquery.Client()
    sql = f"""
    SELECT FORMAT_TIMESTAMP('%Y-%m', created_at) AS ym, COUNT(*) AS stars
    FROM `{DATASET}.*`
    WHERE _TABLE_SUFFIX BETWEEN @lo AND @hi
      AND type = 'WatchEvent'
      AND repo.name IN UNNEST(@repos)
    GROUP BY ym
    ORDER BY ym
    """
    params = [
        bigquery.ScalarQueryParameter("lo", "STRING", start.replace("-", "")),
        bigquery.ScalarQueryParameter("hi", "STRING", end.replace("-", "")),
        bigquery.ArrayQueryParameter("repos", "STRING", list(repos)),
    ]
    dry = client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params, dry_run=True))
    print(f"[github] dry-run: {dry.total_bytes_processed / 1e9:.0f} GB")
    job = client.query(sql, job_config=bigquery.QueryJobConfig(
        query_parameters=params, maximum_bytes_billed=max_bytes))
    df = job.to_dataframe()
    print(f"[github] billed: {job.total_bytes_billed / 1e9:.0f} GB; {len(df)} active months")
    return df


def collect(
    start: str,
    end: str,
    *,
    market: str = "llm-dev-tools",
    repos: "tuple[str, ...] | None" = None,
    project: "str | None" = None,
    cache_dir: "str | Path" = "data/raw",
    max_bytes: int = MAX_BYTES_BILLED,
    sleep: "float | None" = None,  # driver compat; unused
) -> list[Observation]:
    """Monthly stars gained across the market's anchor repos; one Observation per month (0 when none)."""
    repos = repos or get_market(market).github_repos()
    project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
    cache = _cache_path(cache_dir, start, end, repos)
    if cache.exists():
        print(f"[github] using cached query result: {cache}")
        raw = pd.read_parquet(cache)
    else:
        raw = _query_monthly(start, end, repos, project=project, max_bytes=max_bytes)
        cache.parent.mkdir(parents=True, exist_ok=True)
        raw.to_parquet(cache, index=False)

    counts = {ym: int(n) for ym, n in zip(raw["ym"], raw["stars"])}
    request_url = f"bigquery://{DATASET}?stars={market}"
    return [month_value_to_observation(m, counts.get(m, 0), request_url, market=market)
            for m in month_range(start, end)]


def month_value_to_observation(
    month: str, stars: int, request_url: str, *, market: str, code_version: str = "v1",
) -> Observation:
    """Pure: one (month, stars-gained) -> a leakage-safe Observation (star events are public when they happen)."""
    eo = end_of_month(month)
    content_hash = hashlib.sha256(f"{request_url}|{month}|{stars}".encode()).hexdigest()
    return Observation(
        observation_id=f"github_stars:{market}:{month}",
        feature=FEATURE,
        market=market,
        value=float(stars),
        reference_date=eo,
        knowledge_date=eo,
        superseded_date=SUPERSEDED_OPEN,
        claim_text=f"{stars} GitHub stars gained across {market} anchor repos in {month}",
        provenance=Provenance(source_url=request_url, content_hash=content_hash,
                              dated_quote=f"github_stars={stars}"),
        extraction=Extraction(method=ExtractionMethod.API_INGEST, source_name="githubarchive",
                              code_version=code_version),
        confidence=1.0,
    )
