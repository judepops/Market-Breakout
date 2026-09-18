"""Durable-breakout LABEL for the Phase-0 arXiv backtest (pre-registration §2).

This is the *ground truth* the detector must predict from early data only. It is computed
retrospectively over the FULL share series and is allowed to use the future, because it is the
target, not a feature (the leakage firewall in §4 constrains the detector, not this label).

The label, frozen verbatim from ``Plans/2026-06-30_phase0-preregistration.md`` §2 /
``artifacts/eval/PREREGISTRATION.json`` (``label_durable_breakout``):

  Work on the smoothed share ``s(T) = smooth_months centred rolling mean of share_theme(T)``.
  - Birth ``T0``  = first month with ``n_theme >= birth_min_records`` in a 3-month window.
  - Baseline ``b`` = median(s) over ``[T0, T0+base_months-1]``.
  - Peak ``p``     = max(s) over ``[T0, T0+persist_hi]`` at ``T_peak``.
  - Plateau ``q``  = median(s) over ``[T_peak+persist_lo, T_peak+persist_hi-1]`` (12-mo window).

  GROWTH gate:      ``p/b >= G_REL`` AND ``p >= G_ABS``.
  PERSISTENCE gate: ``q/p >= P_RET`` AND ``q >= P_FLOOR_MULT * b``.

  POSITIVE  iff growth AND persistence.   HYPED_DIED iff growth AND NOT persistence.
  NEVER     iff NOT growth.               CENSORED   iff ``corpus_end < T_peak+persist_hi-1``
  (window unobservable) — takes precedence over the gate verdict.

``theme_series`` reuses the ``groupby(...)["record_id"].nunique()`` aggregation pattern from
``features/emergence.py``. The macro-normaliser ``cs_denominator`` (total monthly cs.* doc count)
is passed in verbatim — it is NOT recomputed from any buggy API total (preregistration normaliser_note).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..timeutil import month_range


# ---------------------------------------------------------------------------
# month arithmetic (small local helper; timeutil has month_range but not "add N")
# ---------------------------------------------------------------------------
def _shift_month(month: str, n: int) -> str:
    """The month ``n`` calendar months after ``month`` ("YYYY-MM"); ``n`` may be negative."""
    y, m = (int(x) for x in month.split("-"))
    total = y * 12 + (m - 1) + n
    sy, sm = divmod(total, 12)
    return f"{sy:04d}-{sm + 1:02d}"


@dataclass(frozen=True)
class EmergenceLabelConfig:
    """Frozen label thresholds (§2.5 / PREREGISTRATION.json ``label_durable_breakout``)."""

    G_REL: float = 4.0            # relative growth: peak >= 4x baseline
    G_ABS: float = 0.003          # absolute scale: peak >= 0.3% of monthly cs.* docs
    P_RET: float = 0.5            # retention: plateau >= 50% of peak
    P_FLOOR_MULT: float = 2.0     # floor: plateau >= 2x baseline
    persist_lo: int = 12          # plateau window starts T_peak + 12
    persist_hi: int = 24          # peak search window length; plateau window ends T_peak + 23
    birth_min_records: int = 5    # >= 5 records in a birth_window-month window -> birth
    birth_window: int = 3         # trailing-month window for the birth threshold
    base_months: int = 6          # baseline = median over [T0, T0+5]
    smooth_months: int = 3        # centred rolling-mean smoothing of share


def theme_series(
    term_events: pd.DataFrame, theme_terms: set[str], cs_denominator: pd.Series
) -> pd.Series:
    """Monthly macro-normalised SHARE series for a theme, over the full month grid.

    ``n_theme(T)`` = count of DISTINCT ``record_id`` in month ``T`` whose ``term`` is in
    ``theme_terms`` (the ``groupby(...).nunique()`` pattern from ``features/emergence.py``).
    ``share(T) = n_theme(T) / cs_denominator[T]``. The series is reindexed over the full month
    grid spanned by ``cs_denominator`` (sorted ascending), 0 where the theme is absent.

    ``cs_denominator`` is the total monthly cs.* document count, passed in verbatim (NOT recomputed
    from a buggy API total — preregistration normaliser_note).
    """
    grid = list(cs_denominator.index)
    terms = set(theme_terms)
    ev = term_events[term_events["term"].isin(terms)]
    if len(ev):
        n_theme = ev.groupby("month")["record_id"].nunique().reindex(grid, fill_value=0).astype(float)
    else:
        n_theme = pd.Series(0.0, index=grid)
    denom = cs_denominator.reindex(grid).astype(float)
    share = n_theme / denom.where(denom > 0, other=np.nan)
    return share.fillna(0.0)


def _n_theme_counts(
    term_events: pd.DataFrame, theme_terms: set[str], grid: list[str]
) -> pd.Series:
    """Raw monthly distinct-record counts ``n_theme(T)`` over ``grid`` (the birth-threshold input)."""
    terms = set(theme_terms)
    ev = term_events[term_events["term"].isin(terms)]
    if len(ev):
        return ev.groupby("month")["record_id"].nunique().reindex(grid, fill_value=0).astype(int)
    return pd.Series(0, index=grid, dtype=int)


def _birth_month(n_theme: pd.Series, min_records: int, window: int = 3) -> "str | None":
    """First month whose trailing ``window``-month sum has ``>= min_records`` records (§2.1)."""
    grid = list(n_theme.index)
    rolled = n_theme.rolling(window=window, min_periods=1).sum()
    for m in grid:
        if rolled.loc[m] >= min_records:
            return m
    return None


def durable_breakout_label(
    series: pd.Series,
    n_theme_series: pd.Series,
    config: EmergenceLabelConfig,
    corpus_end: str,
) -> dict:
    """Compute the durable-breakout label for one theme (§2, frozen).

    Parameters
    ----------
    series : the macro-normalised SHARE series (``theme_series`` output), indexed by "YYYY-MM".
    n_theme_series : the raw distinct-record count series (birth-threshold input), same index.
    config : frozen :class:`EmergenceLabelConfig`.
    corpus_end : last observable month "YYYY-MM"; controls right-censoring (§2.4).

    Returns
    -------
    dict with keys ``label`` ("positive"/"hyped_died"/"never"/"censored"), ``T0``, ``T_peak``,
    ``b`` (baseline), ``p`` (peak), ``q`` (plateau), and the four gate booleans
    ``growth_rel``, ``growth_abs``, ``persist_ret``, ``persist_floor``.
    """
    out = {
        "label": "never", "T0": None, "T_peak": None,
        "b": float("nan"), "p": float("nan"), "q": float("nan"),
        "growth_rel": False, "growth_abs": False,
        "persist_ret": False, "persist_floor": False,
    }

    # s = centred rolling mean of the share (kills single-month spikes, §2.1).
    s = series.rolling(window=config.smooth_months, center=True, min_periods=1).mean()

    # Birth T0 from the RAW record counts (§2.1).
    t0 = _birth_month(n_theme_series, config.birth_min_records, config.birth_window)
    if t0 is None:
        out["label"] = "never"  # too small to bud (§2.4) -> background
        return out
    out["T0"] = t0

    grid = list(s.index)
    grid_set = set(grid)

    # Baseline b = median(s) over [T0, T0+base_months-1].
    base_window = [m for m in month_range(t0, _shift_month(t0, config.base_months - 1)) if m in grid_set]
    b = float(np.median(s.loc[base_window].to_numpy())) if base_window else float("nan")
    out["b"] = b

    # Peak p = max(s) over [T0, T0+persist_hi] at T_peak.
    peak_window = [m for m in month_range(t0, _shift_month(t0, config.persist_hi)) if m in grid_set]
    if not peak_window:
        out["label"] = "never"
        return out
    peak_vals = s.loc[peak_window]
    p = float(peak_vals.max())
    t_peak = str(peak_vals.idxmax())
    out["p"] = p
    out["T_peak"] = t_peak

    # Growth gate (did it get big?).
    growth_rel = (b > 0) and (p / b >= config.G_REL)
    growth_abs = p >= config.G_ABS
    out["growth_rel"] = bool(growth_rel)
    out["growth_abs"] = bool(growth_abs)
    growth = bool(growth_rel and growth_abs)

    # A growth-gate failure is fully observed (b, p over [T0, T0+24]) and does NOT depend on the plateau
    # window, so the theme is 'never emerged' background (§3.3 Pool C) even if its plateau runs past the
    # corpus end. Decide this BEFORE censoring (audit fix), else late never-themes inflate the censored count.
    if not growth:
        out["label"] = "never"
        return out

    # Plateau window [T_peak+persist_lo, T_peak+persist_hi-1] (the 12-month persistence window).
    plateau_lo = _shift_month(t_peak, config.persist_lo)
    plateau_hi = _shift_month(t_peak, config.persist_hi - 1)

    # Right-censoring (§2.4): grew big, but we cannot yet observe the end of the plateau window.
    if corpus_end < plateau_hi:
        out["label"] = "censored"
        return out

    plateau_window = [m for m in month_range(plateau_lo, plateau_hi) if m in grid_set]
    q = float(np.median(s.loc[plateau_window].to_numpy())) if plateau_window else float("nan")
    out["q"] = q

    # Persistence gate (did it stay big?).
    persist_ret = (p > 0) and (q / p >= config.P_RET)
    persist_floor = q >= config.P_FLOOR_MULT * b
    out["persist_ret"] = bool(persist_ret)
    out["persist_floor"] = bool(persist_floor)
    persistence = bool(persist_ret and persist_floor)

    # Verdict (§2.3): growth passed + plateau observed -> positive if it stayed big, else hyped_died.
    out["label"] = "positive" if persistence else "hyped_died"
    return out


def build_labels(
    themes: dict,
    term_events: pd.DataFrame,
    cs_denominator: pd.Series,
    config: EmergenceLabelConfig,
    corpus_end: str,
) -> pd.DataFrame:
    """Label every theme (§2) into a tidy frame.

    Parameters
    ----------
    themes : ``{theme_id: set_of_terms}`` (the granularity layer's theme -> terms map).
    term_events : ``(record_id, month, term, authors)`` from
        ``signals.arxiv_records.extract_term_events``.
    cs_denominator : total monthly cs.* doc count, indexed by "YYYY-MM" (sorted ascending).
    config : frozen :class:`EmergenceLabelConfig`.
    corpus_end : last observable month "YYYY-MM".

    Returns
    -------
    DataFrame with columns ``theme_id, label, T0, T_dec, T_peak, b, p, q`` and the four gate
    booleans. ``T_dec = T0 + 9`` (the detector decision point, §4); ``None`` when ``T0`` is undefined.
    """
    grid = list(cs_denominator.index)
    rows = []
    for theme_id, terms in themes.items():
        share = theme_series(term_events, terms, cs_denominator)
        n_theme = _n_theme_counts(term_events, terms, grid)
        res = durable_breakout_label(share, n_theme, config, corpus_end)
        t0 = res["T0"]
        rows.append({
            "theme_id": str(theme_id),
            "label": res["label"],
            "T0": t0,
            "T_dec": _shift_month(t0, 9) if t0 is not None else None,
            "T_peak": res["T_peak"],
            "b": res["b"],
            "p": res["p"],
            "q": res["q"],
            "growth_rel": res["growth_rel"],
            "growth_abs": res["growth_abs"],
            "persist_ret": res["persist_ret"],
            "persist_floor": res["persist_floor"],
        })
    cols = ["theme_id", "label", "T0", "T_dec", "T_peak", "b", "p", "q",
            "growth_rel", "growth_abs", "persist_ret", "persist_floor"]
    return pd.DataFrame(rows, columns=cols)
