"""Offline tests for the arXiv signal (no network — the network call is not exercised here)."""

from __future__ import annotations

from datetime import date

from marketbrain.corpus import build_panel
from marketbrain.schema import observations_to_frame
from marketbrain.signals.arxiv import FEATURE, count_to_observation, month_range
from marketbrain.validation import check_no_future_leakage
from marketbrain.timeutil import end_of_month


def test_month_range_inclusive():
    assert month_range("2024-11", "2025-02") == ["2024-11", "2024-12", "2025-01", "2025-02"]


def test_count_to_observation_is_leakage_safe_and_dated():
    o = count_to_observation("2024-03", 123, "http://export.arxiv.org/api/query?x=1",
                             market="llm-dev-tools")
    assert o.value == 123.0
    assert o.feature == FEATURE and o.market == "llm-dev-tools"
    # natively dated: a month's submissions are fully known by month-end
    assert o.knowledge_date == end_of_month("2024-03") == date(2024, 3, 31)
    assert o.reference_date == o.knowledge_date
    assert o.provenance.source_url.startswith("http") and o.provenance.content_hash


def test_panel_from_arxiv_counts_passes_leakage():
    obs = [count_to_observation(m, c, "http://x", market="llm-dev-tools")
           for m, c in [("2024-01", 10), ("2024-02", 14), ("2024-03", 21)]]
    df = observations_to_frame(obs)
    for m in ["2024-01", "2024-02", "2024-03"]:
        panel = build_panel(df, m, feature_types={FEATURE: "count"})
        assert len(panel) == 1
        assert panel["value"].iloc[0] in (10.0, 14.0, 21.0)
        assert check_no_future_leakage(panel) == []
