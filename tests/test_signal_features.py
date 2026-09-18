"""Tests for the signal->feature collapse layer (window -> level/growth -> normalize)."""

from __future__ import annotations

from marketbrain.features.signal_features import (
    reduce_series,
    window_months,
    zscore,
)


def test_window_months_is_the_formation_window():
    w = window_months(2015, 3)
    assert len(w) == 36 and w[0] == "2015-01" and w[-1] == "2017-12"
    assert "2018-01" not in w                                   # strictly [t0, t0+k)


def test_level_is_total_only_within_window():
    series = {"2014-12": 99, "2015-06": 10, "2016-06": 20, "2017-12": 5, "2018-01": 100}
    f = reduce_series(series, 2015, 3)
    assert f.level == 35                                        # 10+20+5; 2014-12 and 2018-01 excluded (leakage)
    assert f.n_months == 36


def test_growth_positive_for_accelerating_flat_for_constant():
    months = window_months(2015, 3)
    accel = {m: float(i * i) for i, m in enumerate(months)}     # rising
    flat = {m: 50.0 for m in months}
    assert reduce_series(accel, 2015, 3).growth > 0.05
    assert abs(reduce_series(flat, 2015, 3).growth) < 1e-9      # constant -> ~0 slope
    assert reduce_series({}, 2015, 3).growth == 0.0             # empty -> 0


def test_growth_is_approximately_scale_free():
    # log-linear slope is ~scale-free for non-trivial levels; the log1p +1 only matters near zero
    months = window_months(2015, 2)
    small = {m: float(100 * 2 ** i) for i, m in enumerate(months)}
    big = {m: float(100_000 * 2 ** i) for i, m in enumerate(months)}  # same growth, 1000x level
    assert abs(reduce_series(small, 2015, 2).growth - reduce_series(big, 2015, 2).growth) < 0.02


def test_zscore_normalizes_and_preserves_masked():
    out = zscore([0.0, 10.0, 20.0, None])
    assert out[3] is None                                       # masked dimension stays masked
    assert abs(sum(x for x in out if x is not None)) < 1e-9     # zero-mean over the present values
    assert out[2] > out[1] > out[0]                            # order preserved
