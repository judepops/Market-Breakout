"""Aggregate the integrity checks and fail the build on any error.

This is what CI and ``experiments/validate_cut.py`` call. Observation-level checks
run when an observations frame is supplied; panel-level checks run when a panel is
supplied; survivorship additionally needs a universe.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from . import checks as C
from .checks import ERROR, WARNING, IntegrityError, IntegrityFailure


def run_all_checks(
    observations: Optional[pd.DataFrame] = None,
    panel: Optional[pd.DataFrame] = None,
    universe: Optional[pd.DataFrame] = None,
    *,
    raise_on_error: bool = True,
    verbose: bool = True,
) -> list[IntegrityFailure]:
    failures: list[IntegrityFailure] = []

    if observations is not None:
        failures += C.check_event_vs_knowledge_skew(observations)
        failures += C.check_capture_staleness(observations)
        failures += C.check_provenance_completeness(observations)
        failures += C.check_hindsight_spotcheck(observations)
        failures += C.check_coverage_completeness(observations)
    if panel is not None:
        failures += C.check_no_future_leakage(panel)
        if universe is not None:
            failures += C.check_survivorship(panel, universe)

    if verbose:
        _print_report(failures)
    if raise_on_error and any(f.severity == ERROR for f in failures):
        raise IntegrityError(failures)
    return failures


def _print_report(failures: list[IntegrityFailure]) -> None:
    errors = [f for f in failures if f.severity == ERROR]
    warnings = [f for f in failures if f.severity == WARNING]
    print(f"[integrity] {len(errors)} error(s), {len(warnings)} warning(s)")
    for group, label in ((errors, "ERROR"), (warnings, "WARN")):
        counts: dict[str, int] = {}
        for f in group:
            counts[f.check] = counts.get(f.check, 0) + 1
        for f in group[:50]:
            loc = " ".join(x for x in (f.market, f.month, f.observation_id) if x)
            print(f"  [{label}] {f.check}: {f.detail}" + (f"  ({loc})" if loc else ""))
        if len(group) > 50:
            print(f"  ... and {len(group) - 50} more")
        if counts:
            print(f"  by check: {counts}")
