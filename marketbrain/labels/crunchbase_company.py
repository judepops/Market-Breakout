"""Per-company funding milestones from Crunchbase — the structured base for the success label.

LABEL infrastructure, NOT a feature: allowed to use the future, kept out of the feature corpus + the
leakage gate. For each roster company it pulls the facts needed to decide a >=$500M outcome (total raised,
post-money valuation, M&A price, IPO valuation), each with the date it occurred. Auth via the Vela service
account (``GOOGLE_APPLICATION_CREDENTIALS``); ``google.cloud.bigquery`` is imported lazily so the module
(and offline tests of the label logic) load without the SDK.

The BigQuery schema was confirmed live (2026-06): ``organizations`` carries a curated
``valuation_value_usd`` + ``valuation_date``; ``funding_rounds`` carries per-round raise + post-money
valuation dated by ``announced_on``; ``acquisitions`` carries ``price_value_usd`` + ``announced_on_value``
joined on ``acquiree_identifier_permalink``; ``ipos`` carries ``valuation_value_usd`` + ``went_public_on``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date

DATASET = "production-498516.Crunchbase_Dataset"
MAX_BYTES_BILLED = 5 * 10**9  # runaway guard


@dataclass(frozen=True)
class CompanyFunding:
    """Raw Crunchbase funding facts for one company (label decides the outcome from these)."""

    permalink: str
    name: "str | None"
    valuation_usd: "int | None"        # max(curated org valuation, max post-money round valuation)
    valuation_date: "date | None"      # date of the curated org valuation
    total_raised_usd: "int | None"
    raise_cross_date: "date | None"    # date cumulative raise first reached the threshold
    ma_price_usd: "int | None"         # acquisition price (if acquired)
    ma_date: "date | None"
    ipo_valuation_usd: "int | None"
    ipo_date: "date | None"


_SQL = f"""
WITH orgs AS (
  SELECT identifier_permalink AS permalink, name,
         valuation_value_usd AS org_valuation_usd, valuation_date
  FROM `{DATASET}.organizations`
  WHERE identifier_permalink IN UNNEST(@permalinks)
),
rounds AS (
  SELECT funded_organization_identifier_permalink AS permalink,
         SUM(money_raised_value_usd) AS total_raised_usd,
         MAX(post_money_valuation_value_usd) AS max_round_valuation_usd
  FROM `{DATASET}.funding_rounds`
  WHERE funded_organization_identifier_permalink IN UNNEST(@permalinks)
  GROUP BY permalink
),
raise_cross AS (
  SELECT permalink, MIN(announced_on) AS raise_cross_date FROM (
    SELECT funded_organization_identifier_permalink AS permalink, announced_on,
           SUM(money_raised_value_usd) OVER (
             PARTITION BY funded_organization_identifier_permalink ORDER BY announced_on) AS cum
    FROM `{DATASET}.funding_rounds`
    WHERE funded_organization_identifier_permalink IN UNNEST(@permalinks)
  ) WHERE cum >= @threshold GROUP BY permalink
),
ma AS (
  SELECT acquiree_identifier_permalink AS permalink,
         MAX(price_value_usd) AS ma_price_usd, MIN(announced_on_value) AS ma_date
  FROM `{DATASET}.acquisitions`
  WHERE acquiree_identifier_permalink IN UNNEST(@permalinks)
  GROUP BY permalink
),
ipo AS (
  SELECT organization_identifier_permalink AS permalink,
         MAX(valuation_value_usd) AS ipo_valuation_usd, MIN(went_public_on) AS ipo_date
  FROM `{DATASET}.ipos`
  WHERE organization_identifier_permalink IN UNNEST(@permalinks)
  GROUP BY permalink
)
SELECT o.permalink, o.name,
       NULLIF(GREATEST(COALESCE(o.org_valuation_usd, 0),
                       COALESCE(r.max_round_valuation_usd, 0)), 0) AS valuation_usd,
       o.valuation_date,
       r.total_raised_usd, rc.raise_cross_date,
       m.ma_price_usd, m.ma_date, i.ipo_valuation_usd, i.ipo_date
FROM orgs o
LEFT JOIN rounds r USING (permalink)
LEFT JOIN raise_cross rc USING (permalink)
LEFT JOIN ma m USING (permalink)
LEFT JOIN ipo i USING (permalink)
"""


def pull(
    permalinks: "tuple[str, ...] | list[str]",
    *,
    threshold_usd: int = 500_000_000,
    project: "str | None" = None,
    max_bytes: int = MAX_BYTES_BILLED,
    verbose: bool = True,
) -> "dict[str, CompanyFunding]":
    """Query Crunchbase for the per-company funding facts; returns {permalink: CompanyFunding}.

    Parameterized (ArrayQueryParameter) — no string interpolation of the roster. The ``@threshold`` only
    affects the cumulative-raise crossing date, not which companies are returned.
    """
    from google.cloud import bigquery  # lazy

    project = project or os.environ.get("GOOGLE_CLOUD_PROJECT") or "production-498516"
    client = bigquery.Client(project=project)
    params = [
        bigquery.ArrayQueryParameter("permalinks", "STRING", list(permalinks)),
        bigquery.ScalarQueryParameter("threshold", "INT64", int(threshold_usd)),
    ]
    cfg = bigquery.QueryJobConfig(query_parameters=params, maximum_bytes_billed=max_bytes)
    job = client.query(_SQL, job_config=cfg)
    rows = list(job.result())
    if verbose:
        print(f"[cb-company] billed {job.total_bytes_billed / 1e6:.0f} MB; {len(rows)} companies")
    return {
        r.permalink: CompanyFunding(
            permalink=r.permalink, name=r.name,
            valuation_usd=r.valuation_usd, valuation_date=r.valuation_date,
            total_raised_usd=r.total_raised_usd, raise_cross_date=r.raise_cross_date,
            ma_price_usd=r.ma_price_usd, ma_date=r.ma_date,
            ipo_valuation_usd=r.ipo_valuation_usd, ipo_date=r.ipo_date,
        )
        for r in rows
    }
