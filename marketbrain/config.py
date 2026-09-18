"""Typed configuration (plain dataclasses, not YAML).

Holds the things a run needs to agree on: where the corpus lives, the integrity-
check thresholds, and the outlier-label definition (a project decision — see
``docs/DATASHEET.md`` §3 — resolved: outlier = $500M raised / valuation / M&A, not ARR).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class LabelConfig:
    """The outlier-market label definition (see OutlierLabel). Resolved: $500M, not ARR."""

    metric: str = "raise_or_valuation_or_ma_usd"   # any of: raised | valuation | M&A (USD)
    threshold_usd: float = 500_000_000.0
    horizon_years: float = 5.0
    percentile: "float | None" = None              # alternative percentile-driven target (e.g. 90.0)


@dataclass(frozen=True)
class IntegrityConfig:
    max_backreport_days: int = 3650   # ~10y; larger gaps are flagged for review
    capture_stale_days: int = 90      # archive captures older than this are low-confidence


@dataclass(frozen=True)
class Config:
    corpus_root: Path = Path("data/raw")
    label: LabelConfig = field(default_factory=LabelConfig)
    integrity: IntegrityConfig = field(default_factory=IntegrityConfig)

    # PoC market (v1): LLM developer tools, narrowly scoped (see docs/DATASHEET.md §2).
    # Start with ONE market to validate the pipeline; humanoid-robotics is the deferred second.
    markets: tuple[str, ...] = ("llm-dev-tools",)
