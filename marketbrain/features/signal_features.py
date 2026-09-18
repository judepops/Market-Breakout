"""Collapse a signal's windowed monthly series into normalized {level, growth} features (the red-arrow step).

A raw monthly time-series (patents/mo, stars/mo, arXiv/mo) over a market's formation window [t0, t0+k] becomes
two scalars:
  * LEVEL  - how much total activity (raw; normalized across markets before modelling, since a 250k-patent
             market and a 200-paper market are not directly comparable).
  * GROWTH - the log-linear slope = the exponential growth rate over the window. Scale-free already, and the
             LEADING tell (is this signal accelerating as the market forms?), which is the heart of the thesis.

Pure + leakage-safe: only the window's months are read, so the feature uses no post-window information.
Normalization across markets (`zscore`) is separate, applied once all markets' raw values are in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SignalFeature:
    level: float       # total activity in the window (raw; z-score across markets before modelling)
    growth: float      # log-linear slope over the window = exponential growth rate (already scale-free)
    n_months: int


def window_months(t0: int, k: int) -> list[str]:
    """The 'YYYY-MM' months in the formation window [t0, t0+k)."""
    return [f"{y:04d}-{m:02d}" for y in range(t0, t0 + k) for m in range(1, 13)]


def _loglinear_slope(vals: np.ndarray) -> float:
    """OLS slope of log1p(value) vs month index = exponential growth rate; 0 if flat/empty."""
    if len(vals) < 3 or vals.sum() <= 0:
        return 0.0
    y = np.log1p(vals)
    x = np.arange(len(vals), dtype=float)
    xm = x.mean()
    denom = float(((x - xm) ** 2).sum())
    if denom == 0:
        return 0.0
    return float(((x - xm) * (y - y.mean())).sum() / denom)


def reduce_series(series: "dict[str, float]", t0: int, k: int) -> SignalFeature:
    """Window a monthly {('YYYY-MM') -> value} series to [t0, t0+k] and reduce to (level, growth)."""
    months = window_months(t0, k)
    vals = np.array([float(series.get(m, 0.0)) for m in months], dtype=float)
    return SignalFeature(level=float(vals.sum()), growth=_loglinear_slope(vals), n_months=len(months))


def series_from_observations(observations) -> "dict[str, float]":
    """Build a {('YYYY-MM') -> value} series from a collector's Observations (keyed on reference month)."""
    return {o.reference_date.strftime("%Y-%m"): float(o.value) for o in observations}


def zscore(xs: "list[float | None]") -> "list[float | None]":
    """Cross-market normalization. ``None`` (a masked / absent dimension) stays ``None``, never imputed to 0."""
    vals = [x for x in xs if x is not None]
    if not vals:
        return list(xs)
    mu = sum(vals) / len(vals)
    var = sum((x - mu) ** 2 for x in vals) / len(vals)
    sd = var ** 0.5 or 1.0
    return [None if x is None else (x - mu) / sd for x in xs]
