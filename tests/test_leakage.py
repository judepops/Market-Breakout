"""THE canary: the no-future-leakage assertion must catch a planted future-dated cell."""

from __future__ import annotations

from datetime import date

import pytest

from marketbrain.validation import IntegrityError, check_no_future_leakage, run_all_checks
from tests.conftest import make_fv, panel_frame


def test_clean_cut_passes():
    panel = panel_frame([
        make_fv(month="2024-01", knowledge_date_max=date(2024, 1, 20)),
        make_fv(feature="gh_stars", month="2024-01", knowledge_date_max=date(2024, 1, 31)),
    ])
    assert check_no_future_leakage(panel) == []


def test_planted_future_dated_cell_fails():
    panel = panel_frame([
        make_fv(month="2024-01", knowledge_date_max=date(2024, 1, 20)),
        # this cell was "known" in February but claims to be in the January column
        make_fv(feature="gh_stars", month="2024-01", knowledge_date_max=date(2024, 2, 1)),
    ])
    failures = check_no_future_leakage(panel)
    assert len(failures) == 1
    assert failures[0].severity == "error"
    assert failures[0].check == "no_future_leakage"
    assert failures[0].month == "2024-01"
    with pytest.raises(IntegrityError):
        run_all_checks(panel=panel, raise_on_error=True, verbose=False)


def test_leakage_boundary_is_inclusive():
    # exactly end-of-month is allowed; one day later is not
    ok = panel_frame([make_fv(month="2024-01", knowledge_date_max=date(2024, 1, 31))])
    assert check_no_future_leakage(ok) == []
    bad = panel_frame([make_fv(month="2024-01", knowledge_date_max=date(2024, 2, 1))])
    assert len(check_no_future_leakage(bad)) == 1


def test_leap_year_month_end():
    # Feb 2024 has 29 days — the cutoff must respect it
    ok = panel_frame([make_fv(month="2024-02", knowledge_date_max=date(2024, 2, 29))])
    assert check_no_future_leakage(ok) == []
    bad = panel_frame([make_fv(month="2024-02", knowledge_date_max=date(2024, 3, 1))])
    assert len(check_no_future_leakage(bad)) == 1


def test_missing_knowledge_date_is_error():
    panel = panel_frame([make_fv(month="2024-01", knowledge_date_max=None)])
    failures = check_no_future_leakage(panel)
    assert len(failures) == 1 and failures[0].severity == "error"
