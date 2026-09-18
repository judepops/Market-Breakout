"""Tests for the full-universe percentile label + the universe resolver's pure parts (no network)."""

from __future__ import annotations

import pytest

from marketbrain.labels.universe_label import (
    build_from_universe,
    percentile_value,
    _percentile_ranks,
)
from marketbrain.markets.universe import UniverseCompany, _where, outlier_rate


def _uni(funding, valuation=None, permalink="p", name="n"):
    return UniverseCompany(permalink, name, funding, valuation, None)


def test_outcome_is_max_of_raise_and_valuation():
    assert _uni(3, 9).outcome_usd == 9
    assert _uni(9, 3).outcome_usd == 9
    assert _uni(None, None).outcome_usd == 0


def test_is_outlier_threshold_boundary():
    assert _uni(500_000_000).is_outlier() is True
    assert _uni(499_999_999).is_outlier() is False
    assert _uni(None, 600_000_000).is_outlier() is True   # valuation path


def test_percentile_ranks_are_fraction_leq_and_max_is_one():
    ranks = _percentile_ranks([10, 20, 30, 40])
    assert ranks == [0.25, 0.5, 0.75, 1.0]
    # ties share the higher rank (fraction <= value)
    assert _percentile_ranks([5, 5, 9]) == [2 / 3, 2 / 3, 1.0]


def test_percentile_value_quantiles():
    vals = list(range(1, 101))  # 1..100
    assert percentile_value(vals, 0.5) == 51
    assert percentile_value(vals, 0.9) == 91
    assert percentile_value(vals, 0.99) == 100
    assert percentile_value([], 0.5) == 0


def test_build_from_universe_rate_and_percentiles():
    comps = [
        _uni(6_000_000, permalink="a"),                 # fundable, not outlier
        _uni(10_000_000, permalink="b"),                # not outlier
        _uni(700_000_000, permalink="c"),               # outlier (raise)
        _uni(20_000_000, valuation=900_000_000, permalink="d"),  # outlier (valuation)
    ]
    label, per = build_from_universe(comps, "test-market")
    assert label.n_universe == 4 and label.n_outliers == 2
    assert label.rate == 0.5
    assert label.total_funding_usd == 900_000_000 + 700_000_000 + 10_000_000 + 6_000_000
    by = {p.permalink: p for p in per}
    assert by["a"].percentile == 0.25 and not by["a"].is_outlier
    assert by["d"].percentile == 1.0 and by["d"].is_outlier   # 900M is the top outcome


def test_outlier_rate_helper_empty():
    assert outlier_rate([]) == (0.0, 0, 0)


def test_where_builds_and_of_or_and_rejects_bad_tags():
    where = _where([["A", "B"], ["C"]], floor_usd=5_000_000, founded_from=2018)
    assert "funding_total_value_usd >= @floor" in where
    assert "(categories_value LIKE '%A%' OR categories_value LIKE '%B%')" in where
    assert "(categories_value LIKE '%C%')" in where
    assert "founded_on_value >= @founded_from" in where
    with pytest.raises(ValueError):
        _where([["bad'; DROP"]], floor_usd=5_000_000, founded_from=None)
