"""The typed feature schema and the outlier-market label.

A ``FeatureValue`` is the normalized panel cell (Stage 6) derived from one or more
``Observation`` rows. It keeps ``knowledge_date_max`` so the no-future-leakage
check can be applied directly to the panel, and ``source_observation_ids`` so the
panel stays joinable back to the immutable bitemporal log.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Optional

from ..timeutil import end_of_month
from .enums import ConfidenceSource, FeatureType


@dataclass(frozen=True)
class MarketMonth:
    """A panel cell coordinate. ``month`` is the period field ("YYYY-MM")."""

    market: str
    month: str

    def end_of(self) -> date:
        """The leakage cutoff for this cell's column."""
        return end_of_month(self.month)


@dataclass(frozen=True)
class FeatureValue:
    """Typed feature schema from ``corpus_methodology.md`` Stage 0:
    ``(feature, market, month, value, type, confidence)``.
    """

    feature: str
    market: str
    month: str  # "YYYY-MM"
    value: Any
    type: FeatureType
    confidence: float
    knowledge_date_max: date  # max(knowledge_date) over source observations — the leakage key
    confidence_source: ConfidenceSource = ConfidenceSource.DETERMINISTIC
    source_observation_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class OutlierLabel:
    """The prediction target (resolved with the team, 2026-06).

    An outlier MARKET produced an outlier COMPANY, where outlier (from VCBench) = ANY of
    ``>= $500M raised``, ``>= $500M valuation``, or ``>= $500M M&A`` — **not ARR** (the earlier
    "$100M ARR" was a mistaken placeholder). Newer Vela techniques also use a percentile-driven
    target (e.g. 90th percentile on a chosen metric) instead of a fixed cutoff.

    Bitemporal: ``label_knowledge_date`` (when the milestone became public) must never precede
    the months it would inform — this prevents outcome leakage.
    """

    market: str
    is_outlier: bool
    metric: str = "raise_or_valuation_or_ma_usd"   # any of: total raised | valuation | M&A value (USD)
    threshold_usd: float = 500_000_000.0
    horizon_years: float = 5.0
    percentile: Optional[float] = None             # alternative: percentile-driven target (e.g. 90.0)
    label_reference_date: Optional[date] = None   # when the outlier milestone occurred
    label_knowledge_date: Optional[date] = None   # when it became public (never < a feature's month)
    evidence_url: Optional[str] = None
