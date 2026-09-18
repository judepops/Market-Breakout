"""Vintage / as-of reads and panel construction.

``as_of`` implements the Croushore-Stark "read only column V" semantics: what was
known on date D. ``build_panel`` derives a leakage-safe monthly panel column from
the bitemporal log — it is the place that *enforces* the no-future-leakage rule at
construction time (the integrity check then guards against mistakes).
"""

from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd

from ..schema.enums import FeatureType
from ..timeutil import end_of_month, to_month


def as_of(observations: pd.DataFrame, when: date) -> pd.DataFrame:
    """Rows whose vintage was current on ``when``: knowledge_date <= when < superseded_date."""
    if observations.empty:
        return observations
    ts = pd.Timestamp(when)
    kd = pd.to_datetime(observations["knowledge_date"])
    sd = pd.to_datetime(observations["superseded_date"])
    return observations[(kd <= ts) & (ts < sd)]


def build_panel(
    observations: pd.DataFrame,
    month: str,
    feature_types: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """Build the as-of-``month`` panel column from the bitemporal log.

    For each (feature, market) whose *reference* period is ``month``: keep only
    rows known by end-of-month and still current then, then take the latest
    ``knowledge_date`` (the most recent vintage). The resulting cell's
    ``knowledge_date_max`` is <= end_of(month) by construction.
    """
    feature_types = feature_types or {}
    cutoff = pd.Timestamp(end_of_month(month))
    if observations.empty:
        return _empty_panel()

    df = observations.copy()
    df["knowledge_date"] = pd.to_datetime(df["knowledge_date"])
    df["reference_date"] = pd.to_datetime(df["reference_date"])
    df["superseded_date"] = pd.to_datetime(df["superseded_date"])

    ref_month = df["reference_date"].apply(lambda d: to_month(d.date()))
    current = df[
        (ref_month == month)
        & (df["knowledge_date"] <= cutoff)
        & (cutoff < df["superseded_date"])
    ]
    if current.empty:
        return _empty_panel()

    rows = []
    for (feature, market), grp in current.groupby(["feature", "market"]):
        latest = grp.loc[grp["knowledge_date"].idxmax()]
        rows.append({
            "feature": feature,
            "market": market,
            "month": month,
            "value": latest["value"],
            "type": feature_types.get(feature, FeatureType.COUNT.value),
            "confidence": latest.get("confidence"),
            "confidence_source": latest.get("confidence_source"),
            "knowledge_date_max": latest["knowledge_date"],
            "source_observation_ids": [latest["observation_id"]],
        })
    return pd.DataFrame(rows)


def _empty_panel() -> pd.DataFrame:
    from ..schema.table import PANEL_COLUMNS

    return pd.DataFrame(columns=list(PANEL_COLUMNS))
