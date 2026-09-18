"""Vintage / as-of reads: a revised fact must return the value known at each date
(Croushore-Stark "read only column V"), and build_panel must be leakage-safe.
"""

from __future__ import annotations

from datetime import date

from marketbrain.corpus import as_of, build_panel
from marketbrain.validation import check_no_future_leakage
from tests.conftest import make_obs, obs_frame


def _three_vintages():
    """One fact about 2024-01, revised twice. Each revision end-dates the prior."""
    return obs_frame([
        make_obs(value=10.0, reference_date=date(2024, 1, 15),
                 knowledge_date=date(2024, 1, 31), superseded_date=date(2024, 2, 29), obs_id="v0"),
        make_obs(value=12.0, reference_date=date(2024, 1, 15),
                 knowledge_date=date(2024, 2, 29), superseded_date=date(2024, 3, 31), obs_id="v1"),
        make_obs(value=15.0, reference_date=date(2024, 1, 15),
                 knowledge_date=date(2024, 3, 31), obs_id="v2"),
    ])


def test_as_of_returns_value_known_at_each_date():
    df = _three_vintages()
    assert as_of(df, date(2024, 1, 31))["value"].tolist() == [10.0]
    assert as_of(df, date(2024, 2, 15))["value"].tolist() == [10.0]   # revision not yet known
    assert as_of(df, date(2024, 2, 29))["value"].tolist() == [12.0]   # supersede boundary
    assert as_of(df, date(2025, 1, 1))["value"].tolist() == [15.0]    # latest vintage


def test_build_panel_is_leakage_safe_and_uses_known_vintage():
    df = _three_vintages()
    panel = build_panel(df, "2024-01")
    # as-of end of Jan, only the first vintage was known
    assert panel["value"].tolist() == [10.0]
    assert check_no_future_leakage(panel) == []


def test_build_panel_excludes_future_knowledge():
    # a value about Jan but only knowable in Feb must not appear in the Jan cut
    df = obs_frame([
        make_obs(value=99.0, reference_date=date(2024, 1, 10),
                 knowledge_date=date(2024, 2, 5), obs_id="late"),
    ])
    assert build_panel(df, "2024-01").empty
