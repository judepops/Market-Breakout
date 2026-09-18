"""Each non-leakage integrity check catches its planted violation."""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from marketbrain.schema import QCStatus
from marketbrain.validation import (
    check_capture_staleness,
    check_coverage_completeness,
    check_event_vs_knowledge_skew,
    check_hindsight_spotcheck,
    check_provenance_completeness,
    check_survivorship,
)
from tests.conftest import make_fv, make_obs, obs_frame, panel_frame


def test_skew_error_when_known_before_it_happened():
    df = obs_frame([make_obs(reference_date=date(2024, 3, 1), knowledge_date=date(2024, 1, 1))])
    failures = check_event_vs_knowledge_skew(df)
    assert len(failures) == 1 and failures[0].severity == "error"


def test_skew_warns_on_large_backreport_gap():
    df = obs_frame([make_obs(reference_date=date(2004, 1, 1), knowledge_date=date(2024, 1, 1))])
    failures = check_event_vs_knowledge_skew(df, max_backreport_days=3650)
    assert any(f.severity == "warning" for f in failures)


def test_capture_staleness_warns():
    df = obs_frame([make_obs(
        reference_date=date(2024, 3, 15),
        knowledge_date=date(2024, 3, 31),
        archive_capture_timestamp=datetime(2023, 11, 1),  # ~120 days before end of March
    )])
    failures = check_capture_staleness(df, stale_days=90)
    assert len(failures) == 1 and failures[0].severity == "warning"


def test_provenance_completeness_error_and_abstain_exempt():
    bad = obs_frame([make_obs(content_hash="")])              # value-bearing, no hash
    assert any(f.severity == "error" for f in check_provenance_completeness(bad))

    abstained = obs_frame([make_obs(source_url="", content_hash="", value=None,
                                    qc_status=QCStatus.ABSTAINED)])
    assert check_provenance_completeness(abstained) == []


def test_hindsight_phrase_warns():
    df = obs_frame([make_obs(claim_text="By 2026 this became the market leader.")])
    failures = check_hindsight_spotcheck(df)
    assert len(failures) == 1 and failures[0].severity == "warning"


def test_survivorship_flags_missing_active_market():
    # universe says a dead market was active in 2024-01, but the panel omits it
    panel = panel_frame([make_fv(market="llm-dev-tools", month="2024-01")])
    universe = pd.DataFrame([
        {"market": "llm-dev-tools", "active_from": "2023-01", "active_to": None},
        {"market": "crypto-wallets", "active_from": "2023-06", "active_to": "2024-06"},
    ])
    failures = check_survivorship(panel, universe)
    assert any(f.market == "crypto-wallets" and f.severity == "error" for f in failures)


def test_coverage_completeness_flags_partial_month():
    # smooth high-volume series with one ~3% "partial" month (the Wikipedia Dec-2025 bug class)
    obs = [make_obs(feature="wikipedia_pageviews", value=v, reference_date=date(2024, m, 15),
                    knowledge_date=date(2024, m, 28), obs_id=f"w{m}")
           for m, v in [(1, 4_000_000), (2, 4_200_000), (3, 130_000), (4, 4_100_000), (5, 3_900_000)]]
    failures = check_coverage_completeness(obs_frame(obs))
    assert any(f.month == "2024-03" and f.severity == "warning" for f in failures)


def test_coverage_completeness_ignores_sparse_counts():
    # a sparse funding-like count series: a 0 among 1s/2s must NOT flag (median < min_median)
    obs = [make_obs(feature="edgar_formd_count", value=v, reference_date=date(2024, m, 15),
                    knowledge_date=date(2024, m, 28), obs_id=f"e{m}")
           for m, v in [(1, 1), (2, 1), (3, 0), (4, 2), (5, 1)]]
    assert check_coverage_completeness(obs_frame(obs)) == []


def test_survivorship_flags_cell_outside_universe():
    panel = panel_frame([make_fv(market="ghost-market", month="2024-01")])
    universe = pd.DataFrame([{"market": "llm-dev-tools", "active_from": "2023-01", "active_to": None}])
    failures = check_survivorship(panel, universe)
    assert any(f.market == "ghost-market" and f.severity == "error" for f in failures)
