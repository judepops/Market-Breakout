"""Offline tests for the GitHub stars signal (no BigQuery)."""

from __future__ import annotations

from marketbrain.signals.github_stars import FEATURE, month_value_to_observation
from marketbrain.timeutil import end_of_month


def test_observation_is_leakage_safe():
    o = month_value_to_observation("2024-03", 5000, "bq://x", market="llm-dev-tools")
    assert o.feature == FEATURE and o.value == 5000.0
    # star events are public the moment they happen -> knowledge_date == reference_date == end of month
    assert o.knowledge_date == o.reference_date == end_of_month("2024-03")
    assert o.provenance.content_hash


def test_observation_zero_is_a_real_datum():
    o = month_value_to_observation("2022-06", 0, "bq://x", market="llm-dev-tools")
    assert o.value == 0.0 and o.knowledge_date == end_of_month("2022-06")
