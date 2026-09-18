"""Normalize signals into a combined feature panel + a correlation matrix.

Design (per the feature-engineering checklist):
  * **npm + PyPI are one *adoption* dimension** (two halves) — combined so adoption isn't
    double-weighted against research / attention / funding.
  * **Level-vs-flow:** each signal gets a standardized *level* (z-score) AND a *growth rate*
    (year-over-year log change). Research/adoption compound (read growth); attention/funding
    spike (read level). Both are produced; the consumer picks the right one per signal.
  * **Funding = Crunchbase $** (not the sparse EDGAR counts).
  * Combining 6 hand-built signals is a **correlation matrix**, not embedding-dedup.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 4 dimensions; npm + PyPI collapse into one adoption dimension.
DIMENSIONS: dict[str, list[str]] = {
    "research": ["arxiv_submissions"],
    "attention": ["wikipedia_pageviews", "hn_story_count"],
    "adoption": ["npm_downloads", "pypi_downloads"],
    "funding": ["crunchbase_funding_usd"],
}

# Signals for the correlation matrix — the funding signal of record is Crunchbase, not EDGAR.
CORE_SIGNALS: list[str] = [
    "arxiv_submissions", "wikipedia_pageviews", "hn_story_count",
    "npm_downloads", "pypi_downloads", "crunchbase_funding_usd",
]


def signal_levels(observations: pd.DataFrame, market: str = "llm-dev-tools") -> pd.DataFrame:
    """Wide panel: rows = month ("YYYY-MM", sorted), columns = signal, values = raw level."""
    df = observations[observations["market"] == market].copy()
    df["month"] = pd.to_datetime(df["reference_date"]).dt.strftime("%Y-%m")
    return df.pivot_table(index="month", columns="feature", values="value", aggfunc="last").sort_index()


def _zscore(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd and sd > 0 else s * 0.0


def _yoy_log_growth(s: pd.Series, periods: int = 12) -> pd.Series:
    """Year-over-year log growth; undefined (NaN) where either endpoint is <= 0 (e.g. lumpy funding)."""
    prev = s.shift(periods)
    out = np.log(s.where(s > 0) / prev.where(prev > 0))
    return out.replace([np.inf, -np.inf], np.nan)


def build_feature_panel(observations: pd.DataFrame, market: str = "llm-dev-tools") -> pd.DataFrame:
    """Per-signal level_z + yoy_growth, plus one combined level_z + yoy_growth per dimension."""
    wide = signal_levels(observations, market)
    out = pd.DataFrame(index=wide.index)
    for feat in wide.columns:
        out[f"{feat}__level_z"] = _zscore(wide[feat])
        out[f"{feat}__yoy_growth"] = _yoy_log_growth(wide[feat])
    for dim, feats in DIMENSIONS.items():
        present = [f for f in feats if f in wide.columns]
        if not present:
            continue
        # combine the dimension's signals on a comparable scale (mean of z-scores / of growths)
        out[f"dim_{dim}__level_z"] = pd.concat([_zscore(wide[f]) for f in present], axis=1).mean(axis=1)
        out[f"dim_{dim}__yoy_growth"] = pd.concat(
            [_yoy_log_growth(wide[f]) for f in present], axis=1).mean(axis=1)
    return out


def correlation_matrix(
    observations: pd.DataFrame, market: str = "llm-dev-tools", signals: "list[str] | None" = None
) -> pd.DataFrame:
    """Pearson correlation across signal *levels* (6 hand-built signals — a correlation, not dedup)."""
    wide = signal_levels(observations, market)
    cols = [c for c in (signals or CORE_SIGNALS) if c in wide.columns]
    return wide[cols].corr()
