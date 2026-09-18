"""Flat column contracts + dataclass<->DataFrame conversion.

The dataclasses in this package are the typed contract; the corpus on disk is a
columnar table. These two must never drift, so the column lists live next to the
conversion helpers and are checked against the dataclass fields in tests.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Iterable

import pandas as pd

from .bitemporal import Extraction, Observation, Provenance
from .features import FeatureValue

# --- the raw bitemporal observation log (Stage 3 output) ---
OBSERVATION_COLUMNS: dict[str, str] = {
    "observation_id": "string",
    "feature": "string",
    "market": "string",
    "value": "object",
    "claim_text": "string",
    "reference_date": "datetime64[ns]",
    "knowledge_date": "datetime64[ns]",   # the leakage / as-of key
    "superseded_date": "datetime64[ns]",
    "source_url": "string",
    "content_hash": "string",
    "archive_capture_timestamp": "datetime64[ns]",
    "dated_quote": "string",
    "earliest_verifiable_date": "datetime64[ns]",
    "extraction_method": "string",
    "source_name": "string",
    "code_version": "string",
    "prompt_version": "string",
    "run_id": "string",
    "confidence": "float64",
    "confidence_source": "string",
    "qc_status": "string",
}

# --- the normalized panel (Stage 6 output): one row per (feature, market, month) ---
PANEL_COLUMNS: dict[str, str] = {
    "feature": "string",
    "market": "string",
    "month": "string",                    # "YYYY-MM"
    "value": "object",
    "type": "string",
    "confidence": "float64",
    "confidence_source": "string",
    "knowledge_date_max": "datetime64[ns]",  # max knowledge_date over sources — leakage key
    "source_observation_ids": "object",
}


def _observation_to_row(obs: Observation) -> dict:
    p, e = obs.provenance, obs.extraction
    return {
        "observation_id": obs.observation_id,
        "feature": obs.feature,
        "market": obs.market,
        "value": obs.value,
        "claim_text": obs.claim_text,
        "reference_date": obs.reference_date,
        "knowledge_date": obs.knowledge_date,
        "superseded_date": obs.superseded_date,
        "source_url": p.source_url,
        "content_hash": p.content_hash,
        "archive_capture_timestamp": p.archive_capture_timestamp,
        "dated_quote": p.dated_quote,
        "earliest_verifiable_date": p.earliest_verifiable_date,
        "extraction_method": e.method.value,
        "source_name": e.source_name,
        "code_version": e.code_version,
        "prompt_version": e.prompt_version,
        "run_id": e.run_id,
        "confidence": obs.confidence,
        "confidence_source": obs.confidence_source.value,
        "qc_status": obs.qc_status.value,
    }


def _feature_value_to_row(fv: FeatureValue) -> dict:
    return {
        "feature": fv.feature,
        "market": fv.market,
        "month": fv.month,
        "value": fv.value,
        "type": fv.type.value,
        "confidence": fv.confidence,
        "confidence_source": fv.confidence_source.value,
        "knowledge_date_max": fv.knowledge_date_max,
        "source_observation_ids": list(fv.source_observation_ids),
    }


def _coerce(df: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    """Order columns and apply the declared dtypes (datetime columns via to_datetime)."""
    df = df.reindex(columns=list(columns))
    for col, dtype in columns.items():
        if dtype.startswith("datetime64"):
            df[col] = pd.to_datetime(df[col], errors="coerce")
        elif dtype == "object":
            continue
        else:
            df[col] = df[col].astype(dtype)
    return df


def observations_to_frame(observations: Iterable[Observation]) -> pd.DataFrame:
    rows = [_observation_to_row(o) for o in observations]
    return _coerce(pd.DataFrame(rows), OBSERVATION_COLUMNS)


def panel_to_frame(values: Iterable[FeatureValue]) -> pd.DataFrame:
    rows = [_feature_value_to_row(v) for v in values]
    return _coerce(pd.DataFrame(rows), PANEL_COLUMNS)


def observation_field_names() -> set[str]:
    """Dataclass-side field names (flattening provenance/extraction) for drift tests.

    ``Extraction.method`` is stored flattened as the ``extraction_method`` column.
    """
    base = {f.name for f in fields(Observation)} - {"provenance", "extraction"}
    prov = {f.name for f in fields(Provenance)}
    extr = {f.name for f in fields(Extraction)} - {"method"} | {"extraction_method"}
    return base | prov | extr
