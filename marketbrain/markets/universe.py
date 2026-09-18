"""Full-universe roster resolver — the saturation fix (2026-06-16 meeting).

The curated rosters (``llm_dev_tools.ROSTER`` etc.) are hand-picked *prominent* companies, so both PoC
markets came back ~all-winners (57%, 100%) — no negatives, nothing learnable. To get outcome variation and a
real **outlier-rate** target, the LABEL/rate must be computed over the **full Crunchbase universe** of a
market: every fundable company in the market's category, winners and non-winners alike.

A market's universe is a **tag spec** resolved against ``Crunchbase_Dataset.organizations``:
  * ``tag_groups`` — a list of OR-groups, AND'd together. An org qualifies if, for EVERY group, it carries at
    least one of the group's tags. So ``[["Artificial Intelligence (AI)"], ["AI Infrastructure",
    "Generative AI", "Developer Tools"]]`` = "an AI company that is also infra/genai/dev-tools".
  * ``floor_usd`` — the $5M fundable floor (the team): only companies that raised >= $5M (cleaner denominator
    than all-companies, most of which never raised and carry no signal).
  * ``founded_from`` — optional founded-year floor (e.g. 2018 for the LLM era).

Returns one ``UniverseCompany`` per org with the facts the label needs (total raised, valuation, founded),
so the per-company outcome + the market rate are a single cost-capped scan — NO Gemini at universe scale
(grounding stays a spot-check on the curated headline companies).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

FUNDABLE_FLOOR_USD = 5_000_000
THRESHOLD_USD = 500_000_000
_TAG_OK = re.compile(r"^[A-Za-z0-9 /+&()\-.]+$")


@dataclass(frozen=True)
class UniverseCompany:
    permalink: str
    name: "str | None"
    funding_total_usd: "int | None"
    valuation_usd: "int | None"
    founded: "str | None"

    @property
    def outcome_usd(self) -> int:
        """Best realised outcome = max(total raised, post-money valuation). The continuous target."""
        return max(self.funding_total_usd or 0, self.valuation_usd or 0)

    def is_outlier(self, threshold_usd: int = THRESHOLD_USD) -> bool:
        return self.outcome_usd >= threshold_usd


def _where(tag_groups, *, floor_usd, founded_from):
    """Build the parameterised WHERE clause. Tags are validated (no SQL metachars) then LIKE-matched."""
    clauses = ["funding_total_value_usd >= @floor"]
    for group in tag_groups:
        ors = []
        for t in group:
            if not _TAG_OK.match(t):
                raise ValueError(f"invalid category tag {t!r}")
            ors.append(f"categories_value LIKE '%{t}%'")
        clauses.append("(" + " OR ".join(ors) + ")")
    if founded_from is not None:
        clauses.append("founded_on_value >= @founded_from")
    return " AND ".join(clauses)


def fetch_universe(tag_groups, *, floor_usd=FUNDABLE_FLOOR_USD, founded_from=None,
                   project=None, max_bytes=5 * 10**9):
    """Pull every fundable (>= floor) org matching the tag spec. One cost-capped scan of organizations."""
    from google.cloud import bigquery

    if not tag_groups:
        raise ValueError("tag_groups must be non-empty")
    client = bigquery.Client(project=project or "production-498516")
    where = _where(tag_groups, floor_usd=floor_usd, founded_from=founded_from)
    sql = f"""SELECT identifier_permalink AS permalink, name,
                     funding_total_value_usd AS funding, valuation_value_usd AS valuation,
                     founded_on_value AS founded
              FROM `production-498516.Crunchbase_Dataset.organizations`
              WHERE {where}"""
    params = [bigquery.ScalarQueryParameter("floor", "INT64", int(floor_usd))]
    if founded_from is not None:
        params.append(bigquery.ScalarQueryParameter("founded_from", "STRING", f"{int(founded_from)}-01-01"))
    cfg = bigquery.QueryJobConfig(query_parameters=params, maximum_bytes_billed=max_bytes)
    job = client.query(sql, job_config=cfg)
    rows = list(job.result())
    out = [UniverseCompany(r.permalink, r.name,
                           int(r.funding) if r.funding else None,
                           int(r.valuation) if r.valuation else None,
                           str(r.founded) if r.founded else None) for r in rows]
    return out, (job.total_bytes_billed or 0)


def outlier_rate(companies, *, threshold_usd=THRESHOLD_USD):
    """(rate, n_outliers, n_universe) over a resolved universe."""
    n = len(companies)
    o = sum(1 for c in companies if c.is_outlier(threshold_usd))
    return (o / n if n else 0.0), o, n
