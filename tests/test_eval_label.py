"""Tests for the durable-breakout label (preregistration §2).

Four synthetic trajectories, each engineered to land in one labelled bucket:
  * ``durable``  — grows ~40x past the absolute floor and HOLDS the plateau -> positive
  * ``hyped``    — grows ~40x then COLLAPSES to < 0.5 * peak -> hyped_died
  * ``flat``     — stays at its birth baseline (never quadruples) -> never
  * ``censored`` — a clean durable breakout whose peak is < persist_hi months before corpus_end

Fixtures are tiny in-code DataFrames (house style): one term per theme, a constant cs.* denominator,
and per-month record counts chosen so the macro-normalised share hits the engineered levels.
No network, no file I/O.
"""

from __future__ import annotations

import pandas as pd

from marketbrain.eval.label import (
    EmergenceLabelConfig,
    build_labels,
    durable_breakout_label,
    theme_series,
)
from marketbrain.timeutil import month_range

CFG = EmergenceLabelConfig()
DENOM = 100_000               # constant monthly cs.* doc count -> share = n_theme / DENOM
GRID_START = "2018-01"
GRID_END = "2024-12"          # 84 months: room for T0 + peak(+24) + plateau(+24)


def _month_at(offset: int, start: str = GRID_START) -> str:
    return month_range(start, "2099-12")[offset]


def _records_for(term: str, counts: dict[str, int], rid_prefix: str) -> pd.DataFrame:
    """Build term_events rows: ``count`` distinct record_ids in each month using ``term``."""
    rows = []
    for month, n in counts.items():
        for i in range(n):
            rows.append({"record_id": f"{rid_prefix}-{month}-{i}", "month": month,
                         "term": term, "authors": frozenset({f"a{i}"})})
    return pd.DataFrame(rows, columns=["record_id", "month", "term", "authors"])


def _cs_denominator(grid: list[str]) -> pd.Series:
    return pd.Series(DENOM, index=grid, dtype=float)


# --- trajectory builders (per-month n_theme counts) -------------------------
def _durable_counts() -> dict[str, int]:
    """Baseline ~10 (share 1e-4) for 12mo, ramp to peak ~400 (share 4e-3), hold plateau ~300."""
    counts: dict[str, int] = {}
    for k in range(12):                       # baseline / birth region: T0 = month 0
        counts[_month_at(k)] = 10
    for k in range(12, 18):                    # ramp up
        counts[_month_at(k)] = 10 + (k - 11) * 65
    for k in range(18, 26):                     # peak plateau ~400 (peak within [T0, T0+24])
        counts[_month_at(k)] = 400
    for k in range(26, 60):                     # durable plateau ~300 (>= 0.5*peak, >= 2*baseline)
        counts[_month_at(k)] = 300
    return counts


def _hyped_counts() -> dict[str, int]:
    """Same growth as durable, but the plateau COLLAPSES back to baseline ~10."""
    counts = _durable_counts()
    for k in range(26, 60):                     # collapse after the peak
        counts[_month_at(k)] = 10
    return counts


def _flat_counts() -> dict[str, int]:
    """Births (>=5 in a 3-mo window) but never quadruples: stays ~12 throughout."""
    return {_month_at(k): 12 for k in range(60)}


def _flat_late_counts() -> dict[str, int]:
    """Flat (never quadruples) but born LATE so its plateau window runs PAST corpus_end."""
    return {_month_at(k): 12 for k in range(66, 84)}    # births ~index 66, flat to the corpus end


def _censored_counts() -> dict[str, int]:
    """A clean durable breakout, but T0 is late so the plateau window runs past corpus_end.

    Grid ends 2024-12 (index 83). T0 lands ~2023-07 (index 66); the peak ~2024-03 (index 74)
    so the plateau window end T_peak+23 (~2026) is well past corpus_end -> right-censored (§2.4).
    """
    counts: dict[str, int] = {}
    for k in range(66, 72):                     # baseline region births late (T0 ~ index 66)
        counts[_month_at(k)] = 10
    for k in range(72, 80):
        counts[_month_at(k)] = 400              # peaks near the end of the corpus
    return counts


def _series_pair(counts: dict[str, int], term: str, prefix: str):
    grid = month_range(GRID_START, GRID_END)
    denom = _cs_denominator(grid)
    events = _records_for(term, counts, prefix)
    share = theme_series(events, {term}, denom)
    n_theme = (events.groupby("month")["record_id"].nunique()
               .reindex(grid, fill_value=0).astype(int))
    return share, n_theme, denom, events, grid


# --- direct durable_breakout_label tests ------------------------------------
def test_durable_is_positive_and_passes_all_gates():
    share, n_theme, _, _, _ = _series_pair(_durable_counts(), "durable", "D")
    res = durable_breakout_label(share, n_theme, CFG, corpus_end=GRID_END)
    assert res["label"] == "positive"
    assert res["growth_rel"] and res["growth_abs"]
    assert res["persist_ret"] and res["persist_floor"]
    assert res["p"] / res["b"] >= CFG.G_REL
    assert res["p"] >= CFG.G_ABS
    assert res["q"] / res["p"] >= CFG.P_RET
    assert res["q"] >= CFG.P_FLOOR_MULT * res["b"]


def test_hyped_died_passes_growth_fails_persistence():
    share, n_theme, _, _, _ = _series_pair(_hyped_counts(), "hyped", "H")
    res = durable_breakout_label(share, n_theme, CFG, corpus_end=GRID_END)
    assert res["label"] == "hyped_died"
    assert res["growth_rel"] and res["growth_abs"]          # it DID get big
    assert not (res["persist_ret"] and res["persist_floor"])  # but did NOT hold
    assert res["q"] / res["p"] < CFG.P_RET


def test_flat_is_never_and_fails_growth():
    share, n_theme, _, _, _ = _series_pair(_flat_counts(), "flat", "F")
    res = durable_breakout_label(share, n_theme, CFG, corpus_end=GRID_END)
    assert res["label"] == "never"
    assert not res["growth_rel"]                  # never quadrupled its baseline
    assert res["T0"] is not None                  # it did birth (>=5 in a 3-mo window)


def test_censored_when_plateau_window_runs_past_corpus_end():
    share, n_theme, _, _, _ = _series_pair(_censored_counts(), "censored", "C")
    res = durable_breakout_label(share, n_theme, CFG, corpus_end=GRID_END)
    assert res["label"] == "censored"
    # censoring takes precedence: the verdict is set before the gates are scored to completion
    assert res["T_peak"] is not None


def test_flat_born_late_is_never_not_censored():
    """Audit regression: a growth-gate FAILURE is fully observed, so it is decided BEFORE censoring.
    A never-emerged theme born late (plateau runs past corpus_end) must be 'never' (Pool C), not 'censored'."""
    share, n_theme, _, _, _ = _series_pair(_flat_late_counts(), "flatlate", "FL")
    res = durable_breakout_label(share, n_theme, CFG, corpus_end=GRID_END)
    assert res["label"] == "never"
    assert not res["growth_rel"]


def test_never_when_no_birth():
    grid = month_range(GRID_START, GRID_END)
    denom = _cs_denominator(grid)
    # only 2 records ever -> never reaches birth_min_records in any 3-month window
    events = _records_for("tiny", {_month_at(3): 2}, "T")
    share = theme_series(events, {"tiny"}, denom)
    n_theme = pd.Series(0, index=grid, dtype=int)
    n_theme.loc[_month_at(3)] = 2
    res = durable_breakout_label(share, n_theme, CFG, corpus_end=GRID_END)
    assert res["label"] == "never"
    assert res["T0"] is None


# --- theme_series unit -------------------------------------------------------
def test_theme_series_is_macro_normalised_share():
    grid = month_range(GRID_START, GRID_END)
    denom = _cs_denominator(grid)
    events = _records_for("x", {_month_at(0): 200, _month_at(1): 0}, "X")
    share = theme_series(events, {"x"}, denom)
    assert share.loc[_month_at(0)] == 200 / DENOM
    assert share.loc[_month_at(1)] == 0.0
    assert list(share.index) == grid            # reindexed over the full grid, 0 where absent


def test_theme_series_counts_distinct_records_not_rows():
    grid = month_range(GRID_START, GRID_END)
    denom = _cs_denominator(grid)
    # same record_id appears under two terms in the theme -> counted ONCE
    events = pd.DataFrame([
        {"record_id": "r1", "month": _month_at(0), "term": "a", "authors": frozenset()},
        {"record_id": "r1", "month": _month_at(0), "term": "b", "authors": frozenset()},
        {"record_id": "r2", "month": _month_at(0), "term": "a", "authors": frozenset()},
    ])
    share = theme_series(events, {"a", "b"}, denom)
    assert share.loc[_month_at(0)] == 2 / DENOM


# --- build_labels end-to-end -------------------------------------------------
def test_build_labels_assigns_each_trajectory_its_label():
    grid = month_range(GRID_START, GRID_END)
    denom = _cs_denominator(grid)
    frames = [
        _records_for("durable", _durable_counts(), "D"),
        _records_for("hyped", _hyped_counts(), "H"),
        _records_for("flat", _flat_counts(), "F"),
        _records_for("censored", _censored_counts(), "C"),
    ]
    events = pd.concat(frames, ignore_index=True)
    themes = {
        "th_durable": {"durable"},
        "th_hyped": {"hyped"},
        "th_flat": {"flat"},
        "th_censored": {"censored"},
    }
    labels = build_labels(themes, events, denom, CFG, corpus_end=GRID_END)
    by_id = labels.set_index("theme_id")["label"].to_dict()
    assert by_id["th_durable"] == "positive"
    assert by_id["th_hyped"] == "hyped_died"
    assert by_id["th_flat"] == "never"
    assert by_id["th_censored"] == "censored"

    # T_dec = T0 + 9 for a born theme
    row = labels.set_index("theme_id").loc["th_durable"]
    t0_idx = grid.index(row["T0"])
    assert grid[t0_idx + 9] == row["T_dec"]
