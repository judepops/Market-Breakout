"""Market-structure features from Crunchbase funding rounds — Phase 1 (no per-market curation).

The universal features the whole N=10 spine can get at once, with zero curation: built straight from
``funding_rounds`` filtered by the market's category tag spec, dated by ``announced_on``. These feed the
first crude features->rate predictor before the curation-heavy signal stacks exist.

**Event-time, vintage-aligned (the methodological must-get-right).** Markets formed at different calendar
times, so everything is measured relative to each market's **formation year t0** (a founding-date marker, no
hindsight): a **formation cohort** = companies whose first round falls in ``[t0, t0+k]``; features are
funding activity over that same window; the **target** is the fraction of that cohort that later reached
$500M (measured by the maturity horizon ``t0+k+w``). Same ``k`` (feature lead) and ``w`` (outcome window)
for every market, so old and young markets are comparable. A market whose horizon exceeds "now" is **flagged
immature**, not given a truncated window.

**Leakage discipline.** Features use ONLY rounds with ``announced_on`` inside the feature window — never the
company's cumulative/lifetime funding total (that is future info and is nearly the label). Early-stage
formation metrics are the load-bearing features; mega-rounds are excluded from the leading features and kept
only as context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

EARLY_TYPES = ("pre_seed", "seed", "angel", "series_a")   # the formation / leading-edge round types
THRESHOLD_USD = 500_000_000
MEGA_USD = 100_000_000
_TAG_OK = re.compile(r"^[A-Za-z0-9 /+&()\-.]+$")


def _tag_where(tag_groups):
    clauses = []
    for group in tag_groups:
        ors = []
        for t in group:
            if not _TAG_OK.match(t):
                raise ValueError(f"invalid category tag {t!r}")
            ors.append(f"funded_organization_categories_value LIKE '%{t}%'")
        clauses.append("(" + " OR ".join(ors) + ")")
    return " AND ".join(clauses)


def fetch_rounds(tag_groups, *, project=None, max_bytes=5 * 10**9):
    """All funding rounds in the market's category, dated. NO funding-total floor (that would leak)."""
    from google.cloud import bigquery

    if not tag_groups:
        raise ValueError("tag_groups must be non-empty")
    where = _tag_where(tag_groups)
    sql = f"""SELECT funded_organization_identifier_permalink AS permalink,
                     announced_on, investment_type,
                     money_raised_value_usd AS raised_usd,
                     post_money_valuation_value_usd AS post_usd
              FROM `production-498516.Crunchbase_Dataset.funding_rounds`
              WHERE announced_on IS NOT NULL AND funded_organization_identifier_permalink IS NOT NULL
                AND {where}"""
    client = bigquery.Client(project=project or "production-498516")
    job = client.query(sql, job_config=bigquery.QueryJobConfig(maximum_bytes_billed=max_bytes))
    df = job.result().to_dataframe()
    df["announced_on"] = pd.to_datetime(df["announced_on"])
    df["year"] = df["announced_on"].dt.year
    return df, (job.total_bytes_billed or 0)


def formation_year(df: pd.DataFrame, q: float = 0.25) -> int:
    """t0 = the q-quantile of companies' FIRST-round year — when the market's early movers appeared.
    A low quantile (default 0.25) marks formation, not the recent-startup median."""
    first = df.groupby("permalink")["year"].min()
    return int(first.quantile(q))


def _hhi(values) -> float:
    """Herfindahl index (0..1) of a list of non-negative shares; 0 if total is 0."""
    tot = float(sum(values))
    if tot <= 0:
        return 0.0
    return float(sum((v / tot) ** 2 for v in values))


@dataclass(frozen=True)
class MarketFeatures:
    market: str
    t0: int
    k: int
    w: int
    mature: bool                 # t0 + k + w <= as_of_year
    cohort_size: int             # companies first-funded in [t0, t0+k]
    early_round_count: int       # early-stage rounds in the window
    early_usd: int               # early-stage $ in the window
    median_first_round_usd: int
    entrant_growth: float        # entrants in 2nd half of window / 1st half (formation momentum)
    seed_hhi: float              # concentration of early-stage $ across the cohort
    mega_round_count: int        # >=$100M rounds in the window (context)
    target_rate: "float | None"  # fraction of cohort reaching $500M (None if immature)
    n_outliers: int


def build_features(df: pd.DataFrame, market: str, *, k: int = 3, w: int = 7, q: float = 0.25,
                   as_of_year: int = 2026, threshold_usd: int = THRESHOLD_USD) -> MarketFeatures:
    """Pure: a market's rounds -> event-time formation-cohort features + the later outlier-rate target."""
    t0 = formation_year(df, q=q)
    lo, hi = t0, t0 + k
    win = df[(df["year"] >= lo) & (df["year"] < hi)]
    first_year = df.groupby("permalink")["year"].min()
    cohort = set(first_year[(first_year >= lo) & (first_year < hi)].index)

    early = win[win["investment_type"].isin(EARLY_TYPES)]
    early_by_co = early.groupby("permalink")["raised_usd"].sum()
    half = lo + k / 2
    g1 = first_year[(first_year >= lo) & (first_year < half)].count()
    g2 = first_year[(first_year >= half) & (first_year < hi)].count()
    entrant_growth = float(g2 / g1) if g1 else 0.0

    mature = (t0 + k + w) <= as_of_year
    # target: of the cohort, fraction that EVER reached $500M (cumulative raised or any post-money).
    # leakage-safe for a prediction example because the cohort + features are fixed by [t0, t0+k] and the
    # outcomes accrue afterwards; only trusted when the market is mature (horizon t0+k+w has passed).
    co_rounds = df[df["permalink"].isin(cohort)]
    raised = co_rounds.groupby("permalink")["raised_usd"].sum().fillna(0)
    post = co_rounds.groupby("permalink")["post_usd"].max().fillna(0)
    outliers = {p for p in cohort
                if raised.get(p, 0) >= threshold_usd or post.get(p, 0) >= threshold_usd}
    rate = (len(outliers) / len(cohort)) if cohort else 0.0

    return MarketFeatures(
        market=market, t0=t0, k=k, w=w, mature=mature, cohort_size=len(cohort),
        early_round_count=len(early), early_usd=int(early["raised_usd"].fillna(0).sum()),
        median_first_round_usd=int(early.groupby("permalink")["raised_usd"].first().median() or 0)
        if len(early) else 0,
        entrant_growth=round(entrant_growth, 3),
        seed_hhi=round(_hhi(list(early_by_co.values)), 4),
        mega_round_count=int((win["raised_usd"].fillna(0) >= MEGA_USD).sum()),
        target_rate=(round(rate, 4) if mature else None), n_outliers=len(outliers),
    )
