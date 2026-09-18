"""Persist success labels to parquet under ``data/raw/labels/`` — SEPARATE from the feature corpus.

Labels never go through the bitemporal ``CorpusStore`` / ``build_panel`` / leakage gate (they are allowed
to use the future). One file per market: per-company outcome rows + a market aggregate row, distinguished
by a ``level`` column.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .build_label import CompanyOutcome
from ..schema.features import OutlierLabel

DEFAULT_ROOT = "data/raw/labels"


def _iso(d) -> "str | None":
    return d.isoformat() if d else None


def to_frame(label: OutlierLabel, outcomes: "list[CompanyOutcome]", market: str) -> pd.DataFrame:
    rows = [
        {
            "level": "company", "market": market, "permalink": o.permalink, "name": o.name,
            "is_outlier": o.is_outlier, "metric": o.metric, "value_usd": o.value_usd,
            "milestone_date": _iso(o.milestone_date), "knowledge_date": _iso(o.knowledge_date),
            "evidence_url": o.evidence_url, "source": o.source, "verified": o.verified,
            "cb_value_usd": o.cb_value_usd, "grounded_value_usd": o.grounded_value_usd,
        }
        for o in outcomes
    ]
    rows.append({
        "level": "market", "market": market, "permalink": None, "name": None,
        "is_outlier": label.is_outlier, "metric": label.metric, "value_usd": None,
        "milestone_date": _iso(label.label_reference_date),
        "knowledge_date": _iso(label.label_knowledge_date),
        "evidence_url": label.evidence_url, "source": None, "verified": None,
        "cb_value_usd": None, "grounded_value_usd": None,
        "threshold_usd": label.threshold_usd, "horizon_years": label.horizon_years,
        "n_outliers": sum(1 for o in outcomes if o.is_outlier),
    })
    return pd.DataFrame(rows)


def write(
    label: OutlierLabel, outcomes: "list[CompanyOutcome]", *,
    market: str = "llm-dev-tools", root: "str | Path" = DEFAULT_ROOT,
) -> Path:
    out = Path(root)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{market}.parquet"
    to_frame(label, outcomes, market).to_parquet(path, index=False)
    return path


def load(market: str = "llm-dev-tools", root: "str | Path" = DEFAULT_ROOT) -> pd.DataFrame:
    return pd.read_parquet(Path(root) / f"{market}.parquet")
