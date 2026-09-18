"""Tests for the Phase 1 market-structure features (synthetic rounds, no network)."""

from __future__ import annotations

import pandas as pd
import pytest

from marketbrain.features.market_structure import (
    _hhi,
    _tag_where,
    build_features,
    formation_year,
)


def _rounds(rows):
    """rows: (permalink, 'YYYY-MM-DD', investment_type, raised_usd, post_usd)."""
    df = pd.DataFrame(rows, columns=["permalink", "announced_on", "investment_type", "raised_usd", "post_usd"])
    df["announced_on"] = pd.to_datetime(df["announced_on"])
    df["year"] = df["announced_on"].dt.year
    return df


def test_hhi_concentrated_vs_uniform():
    assert _hhi([10, 0, 0]) == 1.0                       # one player takes all
    assert abs(_hhi([1, 1, 1, 1]) - 0.25) < 1e-9         # four equal -> 1/4
    assert _hhi([0, 0]) == 0.0                           # empty


def test_formation_year_is_quantile_of_first_round_years():
    df = _rounds([
        ("a", "2010-01-01", "seed", 1, None), ("a", "2014-01-01", "series_a", 2, None),
        ("b", "2012-01-01", "seed", 1, None), ("c", "2016-01-01", "seed", 1, None),
        ("d", "2018-01-01", "seed", 1, None),
    ])
    # first-round years = [2010, 2012, 2016, 2018]; 25th percentile -> 2011/2012 region
    assert formation_year(df, q=0.0) == 2010
    assert formation_year(df, q=1.0) == 2018


def test_tag_where_builds_and_of_or_and_rejects_bad_tags():
    where = _tag_where([["A", "B"], ["C"]])
    assert "(funded_organization_categories_value LIKE '%A%' OR " \
           "funded_organization_categories_value LIKE '%B%')" in where
    assert "(funded_organization_categories_value LIKE '%C%')" in where
    with pytest.raises(ValueError):
        _tag_where([["x'; DROP TABLE"]])


def test_build_features_cohort_window_and_target():
    # t0 (q=0) = 2010. k=3 -> window [2010, 2013). cohort = first-round in window = {a, b}.
    # a crosses $500M (a later round), b does not. c is outside the window.
    df = _rounds([
        ("a", "2010-02-01", "seed", 2_000_000, None),
        ("a", "2012-06-01", "series_a", 8_000_000, None),
        ("a", "2016-01-01", "series_c", 600_000_000, 1_200_000_000),   # outlier (post-window outcome)
        ("b", "2011-01-01", "seed", 1_000_000, None),
        ("c", "2015-01-01", "seed", 3_000_000, None),                  # not in cohort (first round 2015)
    ])
    f = build_features(df, "m", k=3, w=7, q=0.0, as_of_year=2026)
    assert f.t0 == 2010 and f.k == 3 and f.w == 7
    assert f.cohort_size == 2                       # a, b
    assert f.mature is True                         # 2010+3+7 = 2020 <= 2026
    assert f.n_outliers == 1                        # a crossed $500M
    assert f.target_rate == 0.5                     # 1/2
    assert f.early_round_count == 3                 # a:seed+series_a, b:seed (all in window, early types)


def test_immature_market_has_null_target():
    df = _rounds([("a", "2022-01-01", "seed", 1_000_000, None),
                  ("b", "2023-01-01", "seed", 1_000_000, None)])
    f = build_features(df, "young", k=3, w=7, q=0.0, as_of_year=2026)   # 2022+10 = 2032 > 2026
    assert f.mature is False and f.target_rate is None
