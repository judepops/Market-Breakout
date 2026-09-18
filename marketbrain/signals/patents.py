"""Patents — IP / innovation signal (monthly patents assigned to the market's companies).

Monthly count of patents **assigned to the market's roster companies**
(``markets.llm_dev_tools.assignee_names()``) that **first became public** in each month, from
`patents-public-data.patents.publications` in BigQuery. Market-specific (generalises per-market via each
roster), and — for OSS-heavy software like llm-dev-tools — **sparse on purpose**: these companies barely
patent, so a near-zero series is real signal (low patenting culture), not missing data. Patents matter more
for database / infra / security sub-markets. One signal among many.

**Why patents are special — the two-date model earns its keep here.** Every prior signal had
``reference_date == knowledge_date`` (an arXiv submission, an npm download, a funding announcement are public
the moment they happen). A patent is NOT: it is *filed* (the innovation date) but only becomes *public*
~18 months later at **publication**, or — for applications with a non-publication request, and some
jurisdictions — only at **grant** (even later). So the leakage-correct key is the **first-public date**:

    first_public_date = publication_date if it published, else grant_date     # never the filing date

We count patents by their first-public month, so ``reference_date = knowledge_date = end_of(public_month)``
and the series is leakage-safe by construction: a patent filed in month T but published in T+k is counted in
**T+k**, never T. Keying on the filing date would inject knowledge that wasn't public yet — the exact trap
the bitemporal schema exists to prevent.

Auth via the Vela service account; ``google.cloud.bigquery`` is lazy.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import pandas as pd

from ..markets import get_market
from ..schema import Extraction, ExtractionMethod, Observation, Provenance
from ..schema.bitemporal import SUPERSEDED_OPEN
from ..timeutil import end_of_month, month_range, require_month

FEATURE = "patent_publications"
TABLE = "patents-public-data.patents.publications"
MAX_BYTES_BILLED = 20 * 10**9  # runaway guard; the assignee query bills ~5 GB

_NAME_OK = re.compile(r"^[A-Za-z0-9 .&\-]+$")  # roster names are clean; guard before the regex build
# legal-form suffixes an assignee name may carry after the company name (so "HUGGING FACE INC" matches
# "Hugging Face"), anchored to end so "COHERE" does NOT match "COHERENT" and "CHROMA" not "CHROMA TECHNOLOGY".
_SUFFIX = "INC|LLC|CORP|CORPORATION|LTD|LIMITED|CO|LP|LLP|GMBH|AG|SA|SAS|BV|OY|PLC|PTE|INCORPORATED"


def public_month(publication_date: int, grant_date: int) -> "str | None":
    """The leakage-correct month a patent counts in: its FIRST-PUBLIC month ("YYYY-MM").

    ``publication_date`` if it published (> 0), else ``grant_date`` (the non-publication-request /
    grant-only case — kept, not dropped); ``None`` if neither (excluded). The **filing date is deliberately
    not an input** — counting by filing date would leak knowledge that wasn't public yet. This mirrors the
    SQL bucketing (``IF(publication_date > 0, publication_date, grant_date)`` then by month).
    """
    d = publication_date if publication_date and publication_date > 0 else grant_date
    if not d or d <= 0:
        return None
    ym = d // 100
    return f"{ym // 100:04d}-{ym % 100:02d}"


def _assignee_regex(names: "tuple[str, ...]") -> str:
    """Anchored regex: the WHOLE assignee name is a roster name + optional legal suffix(es).

    Anchored to both ends so "Hugging Face" matches "HUGGING FACE" / "HUGGING FACE INC" but NOT
    "COHERENT" (for "Cohere") or "CHROMA TECHNOLOGY" (for "Chroma"). Names are validated before escaping.
    Applied to the assignee name with punctuation stripped (so commas/periods don't block the match).
    """
    parts = []
    for n in names:
        if not _NAME_OK.match(n):
            raise ValueError(f"invalid assignee name {n!r}")
        parts.append(re.escape(n.upper()))
    return "^(" + "|".join(parts) + r")( (" + _SUFFIX + "))*$"


def _cache_path(cache_dir: "str | Path", start: str, end: str, names: "tuple[str, ...]") -> Path:
    key = hashlib.sha256("|".join(sorted(names)).encode()).hexdigest()[:8]
    return Path(cache_dir) / f"patents_{start}_{end}_{key}.parquet"


def _query_monthly(
    start: str, end: str, names: "tuple[str, ...]", *, project: "str | None", max_bytes: int
) -> pd.DataFrame:
    from google.cloud import bigquery  # lazy

    require_month(start)
    require_month(end)
    lo = int(f"{start.replace('-', '')}01")
    hi = int(end_of_month(end).strftime("%Y%m%d"))
    pattern = _assignee_regex(names)
    client = bigquery.Client(project=project) if project else bigquery.Client()
    sql = f"""
    SELECT CAST(DIV(public_d, 100) AS INT64) AS yyyymm, COUNT(*) AS patents
    FROM (
      SELECT IF(publication_date > 0, publication_date, grant_date) AS public_d
      FROM `{TABLE}`
      WHERE EXISTS (SELECT 1 FROM UNNEST(assignee_harmonized) a
                    WHERE REGEXP_CONTAINS(REGEXP_REPLACE(UPPER(a.name), r'[.,]', ''), @pat))
    )
    WHERE public_d BETWEEN {lo} AND {hi}
    GROUP BY yyyymm
    ORDER BY yyyymm
    """
    cfg = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("pat", "STRING", pattern)],
        maximum_bytes_billed=max_bytes,
    )
    dry = client.query(sql, job_config=bigquery.QueryJobConfig(
        query_parameters=cfg.query_parameters, dry_run=True))
    print(f"[patents] dry-run: {dry.total_bytes_processed / 1e9:.1f} GB")
    job = client.query(sql, job_config=cfg)
    df = job.to_dataframe()
    print(f"[patents] billed: {job.total_bytes_billed / 1e9:.1f} GB; {len(df)} active months")
    return df


def collect(
    start: str,
    end: str,
    *,
    market: str = "llm-dev-tools",
    assignees: "tuple[str, ...] | None" = None,
    project: "str | None" = None,
    cache_dir: "str | Path" = "data/raw",
    max_bytes: int = MAX_BYTES_BILLED,
    sleep: "float | None" = None,  # driver compat; unused
) -> list[Observation]:
    """Monthly count of patents assigned to the market's companies, first made public per month; 0 when none."""
    names = assignees or get_market(market).assignee_names()
    project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
    cache = _cache_path(cache_dir, start, end, names)
    if cache.exists():
        print(f"[patents] using cached query result: {cache}")
        raw = pd.read_parquet(cache)
    else:
        raw = _query_monthly(start, end, names, project=project, max_bytes=max_bytes)
        cache.parent.mkdir(parents=True, exist_ok=True)
        raw.to_parquet(cache, index=False)

    counts = {f"{int(ym) // 100:04d}-{int(ym) % 100:02d}": int(n)
              for ym, n in zip(raw["yyyymm"], raw["patents"])}
    request_url = f"bigquery://{TABLE}?assignees={market}"
    return [month_value_to_observation(m, counts.get(m, 0), request_url, market=market)
            for m in month_range(start, end)]


def month_value_to_observation(
    month: str, count: int, request_url: str, *, market: str, code_version: str = "v1",
) -> Observation:
    """Pure: one (month, patent-count) -> a leakage-safe Observation, keyed on the first-public month."""
    eo = end_of_month(month)
    content_hash = hashlib.sha256(f"{request_url}|{month}|{count}".encode()).hexdigest()
    return Observation(
        observation_id=f"patents:{market}:{month}",
        feature=FEATURE,
        market=market,
        value=float(count),
        reference_date=eo,   # the publication-activity month
        knowledge_date=eo,   # public by end of that month (keyed on first-public, NOT filing)
        superseded_date=SUPERSEDED_OPEN,
        claim_text=f"{count} patents assigned to {market} roster companies first made public in {month}",
        provenance=Provenance(source_url=request_url, content_hash=content_hash,
                              dated_quote=f"patent_publications={count}"),
        extraction=Extraction(method=ExtractionMethod.API_INGEST, source_name="bigquery_patents",
                              code_version=code_version),
        confidence=1.0,
    )
