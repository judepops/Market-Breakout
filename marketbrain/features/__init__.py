"""Feature layer — normalize the raw signals into a comparable, combined view.

This is NOT prediction. It turns the 6 hand-built signals into:
  * standardized **levels** (z-scores) and **growth rates** (YoY) — the level-vs-flow distinction
    (research/adoption compound; attention/funding spike), so we read each on the right scale;
  * one combined series **per dimension** (research / attention / adoption / funding), where npm + PyPI
    are the two halves of a single *adoption* dimension so it isn't double-weighted;
  * a **correlation matrix** across the signals (6 hand-built signals → a correlation, NOT embed-dedup).

The funding dimension uses the Crunchbase $ signal (not the sparse EDGAR counts). See ``panel.py``.
"""

from __future__ import annotations

from .emergence import (
    EmergenceConfig,
    calibrate_cutoff,
    emergence_trajectory,
    emergent_terms,
    escore,
    gate_escores,
    market_emergence_score,
)
from .panel import (
    CORE_SIGNALS,
    DIMENSIONS,
    build_feature_panel,
    correlation_matrix,
    signal_levels,
)

__all__ = [
    "DIMENSIONS",
    "CORE_SIGNALS",
    "signal_levels",
    "build_feature_panel",
    "correlation_matrix",
    "EmergenceConfig",
    "escore",
    "emergent_terms",
    "market_emergence_score",
    "emergence_trajectory",
    "gate_escores",
    "calibrate_cutoff",
]
