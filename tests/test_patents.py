"""Offline tests for the patents signal — the leakage-correct first-public dating (no BigQuery)."""

from __future__ import annotations

from marketbrain.signals.patents import FEATURE, month_value_to_observation, public_month
from marketbrain.timeutil import end_of_month


def test_public_month_keys_on_publication_not_filing():
    # A patent FILED 2015-03 but PUBLISHED 2016-09 counts in 2016-09 — the filing date is not even an
    # input, so it cannot leak into an earlier month. This is the patents leakage guarantee.
    assert public_month(20160901, 0) == "2016-09"


def test_public_month_grant_only_is_kept_not_dropped():
    # Non-publication-request / grant-only patents: no publication date -> use the grant date.
    assert public_month(0, 20170615) == "2017-06"


def test_public_month_none_when_never_public():
    assert public_month(0, 0) is None


def test_observation_is_leakage_safe():
    o = month_value_to_observation("2018-04", 7, "bq://x", market="llm-dev-tools")
    assert o.feature == FEATURE and o.value == 7.0
    assert o.knowledge_date == o.reference_date == end_of_month("2018-04")
    assert o.provenance.content_hash


def test_observation_zero_is_a_real_datum():
    # A month with no patents is real signal (low patenting), not missing data.
    o = month_value_to_observation("2012-01", 0, "bq://x", market="llm-dev-tools")
    assert o.value == 0.0 and o.knowledge_date == end_of_month("2012-01")
