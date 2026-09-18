"""Build-failing integrity checks for the market-brain corpus."""

from __future__ import annotations

from .checks import (
    ERROR,
    WARNING,
    IntegrityError,
    IntegrityFailure,
    check_capture_staleness,
    check_coverage_completeness,
    check_event_vs_knowledge_skew,
    check_hindsight_spotcheck,
    check_no_future_leakage,
    check_provenance_completeness,
    check_survivorship,
)
from .runner import run_all_checks

__all__ = [
    "ERROR",
    "WARNING",
    "IntegrityError",
    "IntegrityFailure",
    "check_no_future_leakage",
    "check_event_vs_knowledge_skew",
    "check_capture_staleness",
    "check_provenance_completeness",
    "check_hindsight_spotcheck",
    "check_coverage_completeness",
    "check_survivorship",
    "run_all_checks",
]
