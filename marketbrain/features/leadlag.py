"""Lead-lag (time-lag) analysis primitives — does one signal lead another, and by how many months?

The lead-lag structure is not a side-analysis: it **is** the predictive setup (features at T -> outcome at
T+k). These primitives quantify it, exploratorily.

Two disciplines, both essential at this sample size (~30 monthly points, one market — hypothesis-generating,
NOT inference):
  * **Operate on a STATIONARY transform (YoY growth), never levels.** Every signal here trends hard
    (everything ramped post-ChatGPT); correlating two trending *levels* gives a spuriously high number that
    just reflects the shared trend. YoY growth de-trends and removes annual seasonality.
  * **Significance via a circular-shift permutation** that preserves the target's autocorrelation — a plain
    shuffle would understate the null and over-state significance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def yoy_growth(series: pd.Series, periods: int = 12) -> pd.Series:
    """Year-on-year growth — de-trends + removes annual seasonality. (Callers drop the leading NaNs.)"""
    return series.pct_change(periods)


def lagged_corr(driver: np.ndarray, target: np.ndarray, k: int) -> float:
    """Pearson corr of ``driver_t`` vs ``target_{t+k}`` — driver leads target by ``k >= 0`` months.

    Aligns the overlapping window, drops non-finite pairs; returns NaN if < 8 usable points or a series is
    constant over the window.
    """
    a = driver if k == 0 else driver[:-k]
    b = target if k == 0 else target[k:]
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 8:
        return np.nan
    a, b = a[mask], b[mask]
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def ccf(driver: np.ndarray, target: np.ndarray, max_lag: int = 6) -> "dict[int, float]":
    """Cross-correlation function over lags 0..max_lag (driver leading)."""
    return {k: lagged_corr(driver, target, k) for k in range(max_lag + 1)}


def best_lag(driver: np.ndarray, target: np.ndarray, max_lag: int = 6) -> "tuple[int | None, float]":
    """The lag in 0..max_lag with the largest |corr| (driver leading target), and that corr."""
    valid = {k: v for k, v in ccf(driver, target, max_lag).items() if np.isfinite(v)}
    if not valid:
        return None, float("nan")
    k = max(valid, key=lambda j: abs(valid[j]))
    return k, valid[k]


def permutation_pvalue(driver: np.ndarray, target: np.ndarray, max_lag: int = 6) -> float:
    """Exact circular-shift permutation p-value for the observed peak |corr| (driver leading target).

    Enumerates all non-trivial circular shifts of the target (preserving its autocorrelation) and asks how
    often a random alignment matches/beats the observed peak |corr| over lags 0..max_lag. Deterministic.
    """
    _, obs = best_lag(driver, target, max_lag)
    if not np.isfinite(obs):
        return float("nan")
    n = len(target)
    null = []
    for shift in range(1, n):
        shifted = np.concatenate([target[-shift:], target[:-shift]])
        _, peak = best_lag(driver, shifted, max_lag)
        if np.isfinite(peak):
            null.append(abs(peak))
    if not null:
        return float("nan")
    ge = sum(1 for x in null if x >= abs(obs))
    return (ge + 1) / (len(null) + 1)
