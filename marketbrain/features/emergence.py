"""Carley/Garner technical-emergence scoring (over arXiv term usage).

Implements the gates (Carley et al. 2018) + the composite emergence score (Garner, Carley, Porter
& Newman 2017) — NOT a guess; the score structure is from Garner 2017:

  EScore (additive, "absolute" — not dataset-normalized) = active-period trend + recent trend +
  active-period slope, with cutoff >= 1.77 (~= sqrt(pi)). Aggregate market/player score =
  sum( sqrt(EScore_term) * record_count_term ); normalized score divides by sqrt(total records).

The four gates a term must pass to be "emergent" (Carley 2018):
  * novelty     — appears in < ``novelty_ceiling`` of BASE-period records
  * persistence — appears in >= ``min_periods`` distinct months AND >= ``min_records`` records
  * community   — used by > 1 *non-co-authoring* author
  * growth      — active-period usage >= ``growth_ratio`` x base-period usage

Four tweaks for monthly arXiv (vs the paper's decade-scale WoS corpora), all configurable:
  1. ``novelty_ceiling`` (0.15) and ``escore_cutoff`` (1.77) are the paper's WoS values — **re-tune
     for arXiv** (exposed as config; calibrate against the corpus).
  2. **Sliding window** keeps the ~3:7 base:active ratio but in MONTHS (default 9 + 21 = 30), not years.
  3. **No citation/co-citation features** — usage is submission/co-word only (citations leak the future).
  4. **Community** uses arXiv author identities (the ">1 non-co-authoring author" rule).

The EScore sub-formulas are implemented as growth measures (flat usage -> ~0; accelerating -> large),
which matches the >=1.77 cutoff semantics; exact normalization latitude is noted where it applies.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from ..timeutil import month_range


@dataclass(frozen=True)
class EmergenceConfig:
    base_months: int = 9          # ~3/10 of a 30-month window (keeps the 3:7 base:active ratio)
    active_months: int = 21       # ~7/10
    novelty_ceiling: float = 0.15  # paper (WoS) value — RE-TUNE for arXiv
    min_periods: int = 3          # >= 3 distinct months
    min_records: int = 7          # >= 7 records
    growth_ratio: float = 2.0     # active >= 2x base
    escore_cutoff: float = 1.77   # paper (WoS) value — RE-TUNE for arXiv

    @property
    def window_months(self) -> int:
        return self.base_months + self.active_months


def _growth(recent: float, earlier: float) -> float:
    """Relative change (0 for flat, >0 accelerating). Guarded against a zero base."""
    if earlier <= 0:
        return float(recent > 0)  # 0 -> something is emergence; cap the singularity at 1.0
    return (recent - earlier) / earlier


def escore(active_usage: list[float]) -> float:
    """Garner composite EScore over the active-period monthly usage series (additive, absolute).

    Components (each a growth measure, summed):
      * active-period trend: recent 3 months vs first 3 months of the active window
      * recent trend:        recent 2 months vs the prior 2 months
      * active-period slope: OLS slope of the back half, scaled by the active mean (dimensionless)
    """
    u = list(active_usage)
    n = len(u)
    if n < 6:
        return 0.0
    active_trend = _growth(sum(u[-3:]), sum(u[:3]))
    recent_trend = _growth(sum(u[-2:]), sum(u[-4:-2]))
    half = u[n // 2:]
    mean = np.mean(u) or 1.0
    slope = float(np.polyfit(range(len(half)), half, 1)[0]) / mean if len(half) >= 2 else 0.0
    return active_trend + recent_trend + slope


def _is_emergent(
    usage: pd.Series, authors: list[frozenset], base_record_total: int, cfg: EmergenceConfig
) -> "float | None":
    """Return the EScore if the term passes all four gates, else None.

    ``usage`` is the term's monthly record-count over the window (base months first, then active);
    ``authors`` is one author-set per record using the term; ``base_record_total`` is the count of
    ALL records in the base period (the novelty denominator).
    """
    base = usage.iloc[:cfg.base_months]
    active = usage.iloc[cfg.base_months:]
    base_records, active_records = base.sum(), active.sum()

    # persistence: enough distinct months + records
    if (usage > 0).sum() < cfg.min_periods or (base_records + active_records) < cfg.min_records:
        return None
    # novelty: rare among base-period records (< ceiling of ALL base records)
    if base_record_total > 0 and base_records >= cfg.novelty_ceiling * base_record_total:
        return None
    # growth: active usage >= growth_ratio x base usage
    if active_records < cfg.growth_ratio * max(base_records, 1):
        return None
    # community: > 1 non-co-authoring author (>= 2 distinct, non-identical author-sets)
    sets = {a for a in authors if a}
    if len(sets) < 2 or len(set().union(*sets)) < 2:
        return None

    score = escore(active.tolist())
    return score if score >= cfg.escore_cutoff else None


def emergent_terms(
    term_events: pd.DataFrame, as_of_month: str, cfg: EmergenceConfig = EmergenceConfig()
) -> dict[str, float]:
    """Terms emergent as-of ``as_of_month`` (leakage-safe: only months in the window ending at as_of).

    ``term_events`` columns: ``record_id``, ``month`` ("YYYY-MM"), ``term``, ``authors`` (frozenset).
    Returns {term: EScore} for terms passing all gates with EScore >= cutoff.
    """
    window = month_range_ending(as_of_month, cfg.window_months)
    ev = term_events[term_events["month"].isin(window)]
    base_window = window[:cfg.base_months]
    base_record_total = ev[ev["month"].isin(base_window)]["record_id"].nunique()
    out: dict[str, float] = {}
    for term, grp in ev.groupby("term"):
        usage = (grp.groupby("month")["record_id"].nunique()
                 .reindex(window, fill_value=0).astype(float))
        score = _is_emergent(usage, list(grp["authors"]), base_record_total, cfg)
        if score is not None:
            out[term] = score
    return out


def market_emergence_score(
    term_events: pd.DataFrame, as_of_month: str, cfg: EmergenceConfig = EmergenceConfig()
) -> float:
    """Garner aggregate as-of ``as_of_month``: sum( sqrt(EScore) * record_count ) over emergent terms."""
    window = month_range_ending(as_of_month, cfg.window_months)
    ev = term_events[term_events["month"].isin(window)]
    terms = emergent_terms(term_events, as_of_month, cfg)
    total = 0.0
    for term, score in terms.items():
        records = ev[ev["term"] == term]["record_id"].nunique()
        total += float(np.sqrt(score)) * records
    return total


def emergence_trajectory(
    term_events: pd.DataFrame, months: list[str], cfg: EmergenceConfig = EmergenceConfig(),
    *, normalize: bool = False,
) -> dict[str, float]:
    """Fast batch emergence score for each month in ``months`` (each leakage-safe).

    Equivalent to calling ``market_emergence_score`` per month, but precomputes a term x month
    usage pivot ONCE instead of re-scanning the events for every month and every emergent term
    (the per-month path is O(terms x rows) and does not scale to a full trajectory). The community
    gate is assessed over the full pull rather than per-window — a monotone ">= 2 distinct
    non-co-authoring author-sets" check, which does not change which terms aggregate at market scale.

    **Cross-market comparability:** the raw aggregate (sum sqrt(EScore)*record_count) scales with corpus
    VOLUME, so a high-throughput market (LLM dev-tools, ~51k records) outscores a smaller one (data-infra)
    on size alone — not emergence intensity. Pass ``normalize=True`` for Garner's normalized score
    (raw / sqrt(total in-window records)), which divides the volume out and is the metric to use when
    comparing emergence ACROSS markets. (Pair it with a calibrated ``escore_cutoff`` — see
    ``calibrate_cutoff`` — rather than the paper's WoS 1.77.)
    """
    counts = term_events.groupby(["term", "month"])["record_id"].nunique().unstack(fill_value=0)
    recs_per_month = term_events.groupby("month")["record_id"].nunique()
    nonempty = term_events[term_events["authors"].astype(bool)]
    community_ok = (nonempty.groupby("term")["authors"].nunique() >= 2).reindex(
        counts.index, fill_value=False)

    out: dict[str, float] = {}
    for as_of in months:
        window = month_range_ending(as_of, cfg.window_months)
        base_cols = [m for m in window[:cfg.base_months] if m in counts.columns]
        win_cols = [m for m in window if m in counts.columns]
        active_cols = [m for m in window[cfg.base_months:] if m in counts.columns]
        if not active_cols:
            out[as_of] = 0.0
            continue
        base = counts[base_cols].sum(axis=1) if base_cols else pd.Series(0.0, index=counts.index)
        active = counts[active_cols].sum(axis=1)
        periods = (counts[win_cols] > 0).sum(axis=1)
        base_total = float(recs_per_month.reindex(base_cols).fillna(0).sum()) if base_cols else 0.0

        mask = ((periods >= cfg.min_periods) & ((base + active) >= cfg.min_records)
                & (active >= cfg.growth_ratio * base.clip(lower=1)) & community_ok)
        if base_total > 0:
            mask &= base < cfg.novelty_ceiling * base_total

        cand = counts.index[mask]
        if len(cand) == 0:
            out[as_of] = 0.0
            continue
        active_full = counts.reindex(columns=window[cfg.base_months:], fill_value=0).loc[cand]
        recs_in_win = counts.reindex(columns=window, fill_value=0).loc[cand].sum(axis=1)
        total = 0.0
        for term in cand:
            es = escore(active_full.loc[term].tolist())
            if es >= cfg.escore_cutoff:
                total += float(np.sqrt(es)) * int(recs_in_win.loc[term])
        if normalize:
            win_total = float(recs_per_month.reindex(win_cols).fillna(0).sum())
            out[as_of] = total / np.sqrt(win_total) if win_total > 0 else 0.0
        else:
            out[as_of] = total
    return out


def gate_escores(
    term_events: pd.DataFrame, as_of_month: str, cfg: EmergenceConfig = EmergenceConfig()
) -> list[float]:
    """EScores of every term that passes the four STRUCTURAL gates (novelty / persistence / community /
    growth) as-of ``as_of_month`` — i.e. *before* the EScore cutoff is applied.

    This is the distribution to calibrate ``escore_cutoff`` against per corpus (``calibrate_cutoff``),
    rather than inheriting the paper's WoS value (1.77), which was tuned to decade-scale WoS corpora.
    """
    cfg_open = replace(cfg, escore_cutoff=float("-inf"))  # keep the gates, drop the cutoff
    return [escore_  # the dict values are EScores of gate-passing terms
            for escore_ in emergent_terms(term_events, as_of_month, cfg_open).values()]


def calibrate_cutoff(
    term_events: pd.DataFrame, as_of_month: str, *, percentile: float = 90.0,
    cfg: EmergenceConfig = EmergenceConfig(),
) -> float:
    """A data-driven ``escore_cutoff``: the ``percentile`` of the gate-passing EScore distribution as-of
    ``as_of_month``. Setting the cutoff to e.g. the 90th percentile keeps the top-decile of emergent terms
    — a self-calibrating, cross-market-consistent selectivity (vs the WoS-inherited 1.77)."""
    vals = gate_escores(term_events, as_of_month, cfg)
    return float(np.percentile(vals, percentile)) if vals else float(cfg.escore_cutoff)


def month_range_ending(as_of_month: str, n_months: int) -> list[str]:
    """The ``n_months`` months ending at (and including) ``as_of_month``."""
    y, m = (int(x) for x in as_of_month.split("-"))
    start_total = y * 12 + (m - 1) - (n_months - 1)
    sy, sm = divmod(start_total, 12)
    return month_range(f"{sy:04d}-{sm + 1:02d}", as_of_month)
