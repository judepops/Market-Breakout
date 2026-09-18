import numpy as np
import pandas as pd
import pytest

from marketbrain.eval.trajectory import theme_trajectory


def _series(values, start="2023-01"):
    y, m = (int(x) for x in start.split("-"))
    idx = []
    for i in range(len(values)):
        idx.append(f"{y + (m - 1 + i) // 12:04d}-{(m - 1 + i) % 12 + 1:02d}")
    return pd.Series(values, index=idx, dtype=float)


def test_accelerating_rising_series():
    s = _series([1, 2, 3, 5, 8, 12, 18, 26, 38, 55])
    t = theme_trajectory(s, as_of="2023-10")
    assert t["state"] == "accelerating"
    assert t["recent_slope"] > 0.10
    assert t["peak_month"] == "2023-10"


def test_sustained_plateau_after_peak():
    s = _series([2, 10, 40, 42, 41, 40, 41, 42, 40, 41, 40, 41])
    t = theme_trajectory(s, as_of="2023-12")
    assert t["state"] == "sustained"
    assert t["retention"] >= 0.5
    assert abs(t["recent_slope"]) <= 0.10


def test_cooling_declining_but_alive():
    s = _series([5, 20, 60, 55, 48, 40, 34, 29, 25, 21])
    t = theme_trajectory(s, as_of="2023-10")
    assert t["state"] == "cooling"
    assert t["recent_slope"] < -0.10
    assert t["retention"] >= 0.25


def test_faded_collapsed_from_peak():
    s = _series([10, 60, 80, 40, 15, 6, 3, 2, 1, 1, 1, 1])
    t = theme_trajectory(s, as_of="2023-12")
    assert t["state"] == "faded"
    assert t["retention"] < 0.25
    assert t["months_since_peak"] >= 8


def test_none_for_thin_series():
    s = _series([0, 0, 1, 0, 0, 0])
    assert theme_trajectory(s, as_of="2023-06") is None


def test_as_of_truncates_future():
    s = _series([1, 2, 4, 8, 16, 32, 1, 1, 1, 1])
    t = theme_trajectory(s, as_of="2023-06")  # future collapse invisible as of June
    assert t["state"] == "accelerating"
