"""Tests for the lead-lag primitives (synthetic, no corpus)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from marketbrain.features.leadlag import best_lag, lagged_corr, permutation_pvalue, yoy_growth


def test_yoy_growth():
    s = pd.Series(range(1, 25), dtype=float)  # 1..24
    g = yoy_growth(s, periods=12)
    assert np.isnan(g.iloc[0])
    assert abs(g.iloc[12] - (13 / 1 - 1)) < 1e-9   # month 13 (=13) vs month 1 (=1)


def test_best_lag_recovers_a_known_lead():
    # target_t = driver_{t-2}: driver leads target by exactly 2 months
    driver = np.random.default_rng(1).standard_normal(30)
    target = np.full(30, np.nan)
    target[2:] = driver[:-2]
    k, c = best_lag(driver, target, max_lag=6)
    assert k == 2 and c > 0.99


def test_lagged_corr_zero_lag_is_plain_correlation():
    x = np.arange(20, dtype=float)
    assert abs(lagged_corr(x, 2 * x + 1, 0) - 1.0) < 1e-9


def test_permutation_pvalue_flags_a_real_lead_over_noise():
    # white noise -> only the true alignment correlates; circular-shift null is conservative over the lag
    # window, so test a tight window + the meaningful relative property (real lead << unrelated noise).
    rng = np.random.default_rng(2)
    driver = rng.standard_normal(40)
    target = np.full(40, np.nan)
    target[2:] = driver[:-2]                      # a clean lead of 2 months
    noise = rng.standard_normal(40)
    p_real = permutation_pvalue(driver, target, max_lag=3)
    p_noise = permutation_pvalue(driver, noise, max_lag=3)
    assert p_real < 0.15 and p_real < p_noise
