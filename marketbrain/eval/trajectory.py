"""Per-theme trajectory state as of a month — what the theme is doing NOW.

Computed from the raw monthly distinct-record counts (not the share), 3-month
smoothed. Distinct from the durable-breakout *label* (label.py): the label is a
settled outcome; the trajectory is the current read, defined for live themes too.

v1 state rules (thresholds are module constants, documented in the radar spec —
market-brain-app/docs/superpowers/specs/2026-07-14-time-series-radar-design.md):
  faded         retention < faded_ret
  accelerating  recent_slope > accel_slope
  cooling       recent_slope < cool_slope  (and not faded)
  sustained     retention >= sustain_ret and |recent_slope| <= accel_slope
  cooling       everything else (mid retention, flat slope)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrajectoryConfig:
    smooth_months: int = 3      # rolling-mean window
    slope_window: int = 6       # trailing months for the slope fit
    recent_window: int = 3      # trailing months for retention numerator
    accel_slope: float = 0.10   # +10%/month
    cool_slope: float = -0.10   # -10%/month
    faded_ret: float = 0.25
    sustain_ret: float = 0.50
    min_active_months: int = 3  # below this the trajectory is undefined


DEFAULT = TrajectoryConfig()


def _month_diff(a: str, b: str) -> int:
    ay, am = (int(x) for x in a.split("-"))
    by, bm = (int(x) for x in b.split("-"))
    return (by - ay) * 12 + (bm - am)


def theme_trajectory(
    n_theme: pd.Series, as_of: str, cfg: TrajectoryConfig = DEFAULT
) -> dict | None:
    """Trajectory dict for one theme as of ``as_of`` (inclusive), or None if too thin."""
    s = n_theme.loc[:as_of].astype(float)
    if int((s > 0).sum()) < cfg.min_active_months:
        return None
    sm = s.rolling(cfg.smooth_months, min_periods=1).mean()

    peak = float(sm.max())
    peak_month = str(sm.idxmax())
    recent = float(sm.iloc[-cfg.recent_window :].mean())
    retention = recent / peak if peak > 0 else 0.0

    w = sm.iloc[-cfg.slope_window :]
    if len(w) >= 4 and float(w.mean()) > 0:
        slope = float(np.polyfit(np.arange(len(w)), w.to_numpy(), 1)[0]) / float(w.mean())
    else:
        slope = 0.0

    if retention < cfg.faded_ret:
        state = "faded"
    elif slope > cfg.accel_slope:
        state = "accelerating"
    elif slope < cfg.cool_slope:
        state = "cooling"
    elif retention >= cfg.sustain_ret:
        state = "sustained"
    else:
        state = "cooling"

    return {
        "state": state,
        "retention": round(retention, 3),
        "peak_month": peak_month,
        "months_since_peak": _month_diff(peak_month, as_of),
        "recent_slope": round(slope, 3),
    }
