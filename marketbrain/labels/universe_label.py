"""Continuous / percentile-per-segment label over the full market universe (the saturation fix).

The binary $500M label saturates on curated rosters (57% / 100% winners — no negatives, nothing learnable).
Computed over the FULL fundable universe (``markets.<m>.universe()``), the SAME threshold yields a real RATE
with negatives, and each company gets:
  * a CONTINUOUS outcome = max(total raised, post-money valuation) USD, and
  * its PERCENTILE within the market segment (0..1).
The percentile-per-segment target replaces the flat binary: a market's quality is its outlier RATE (a
continuous, cross-market-comparable number), and a company's target is its within-segment percentile.

Pure Crunchbase — NO Gemini at universe scale (grounding stays a spot-check on the curated headliners).
Leakage note: this is a LABEL (the outcome we predict), hindsight by definition; it never touches the
feature panel or ``check_no_future_leakage``.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

from ..markets import get_market
from ..markets.universe import FUNDABLE_FLOOR_USD, THRESHOLD_USD, outlier_rate


@dataclass(frozen=True)
class CompanyPercentile:
    permalink: str
    name: "str | None"
    outcome_usd: int
    percentile: float        # rank within the market universe, 0..1 (fraction of universe with outcome <= this)
    is_outlier: bool


@dataclass(frozen=True)
class MarketRateLabel:
    market: str
    n_universe: int
    n_outliers: int
    rate: float                  # the continuous, cross-market-comparable target (replaces the binary)
    total_funding_usd: int       # the team's market-potential proxy
    threshold_usd: int
    floor_usd: int
    p50_outcome_usd: int
    p90_outcome_usd: int
    p99_outcome_usd: int


def _percentile_ranks(values: "list[int]") -> "list[float]":
    """For each value, the fraction of values <= it (so the max is 1.0). Stable under ties."""
    srt = sorted(values)
    n = len(srt) or 1
    return [bisect.bisect_right(srt, v) / n for v in values]


def percentile_value(values: "list[int]", q: float) -> int:
    """The q-quantile outcome value (q in [0,1]); 0 on an empty universe."""
    if not values:
        return 0
    srt = sorted(values)
    return srt[min(len(srt) - 1, int(q * len(srt)))]


def build_from_universe(
    companies, market: str, *, threshold_usd: int = THRESHOLD_USD, floor_usd: int = FUNDABLE_FLOOR_USD,
) -> "tuple[MarketRateLabel, list[CompanyPercentile]]":
    """Pure: a resolved universe -> the market rate label + per-company percentiles."""
    outs = [c.outcome_usd for c in companies]
    pct = _percentile_ranks(outs)
    per_company = [
        CompanyPercentile(c.permalink, c.name, c.outcome_usd, p, c.is_outlier(threshold_usd))
        for c, p in zip(companies, pct)
    ]
    rate, o, n = outlier_rate(companies, threshold_usd=threshold_usd)
    label = MarketRateLabel(
        market=market, n_universe=n, n_outliers=o, rate=rate, total_funding_usd=sum(outs),
        threshold_usd=threshold_usd, floor_usd=floor_usd,
        p50_outcome_usd=percentile_value(outs, 0.5), p90_outcome_usd=percentile_value(outs, 0.9),
        p99_outcome_usd=percentile_value(outs, 0.99),
    )
    return label, per_company


def run(market: str, *, project: "str | None" = None):
    """End-to-end: resolve the market universe (BigQuery) -> rate label + per-company percentiles."""
    comps = get_market(market).universe(project=project)
    return build_from_universe(comps, market)
