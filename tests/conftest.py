"""Shared builders for small in-code fixtures (no network, no real data)."""

from __future__ import annotations

from datetime import date

import pandas as pd

from marketbrain.schema import (
    SUPERSEDED_OPEN,
    ConfidenceSource,
    Extraction,
    ExtractionMethod,
    FeatureType,
    FeatureValue,
    Observation,
    Provenance,
    QCStatus,
    observations_to_frame,
    panel_to_frame,
)

_counter = {"n": 0}


def make_obs(
    *,
    feature: str = "arxiv_count",
    market: str = "llm-dev-tools",
    value=10.0,
    reference_date: date = date(2024, 1, 15),
    knowledge_date: date = date(2024, 1, 31),
    superseded_date: date = SUPERSEDED_OPEN,
    source_url: str = "https://example.org/page",
    content_hash: str = "deadbeef",
    archive_capture_timestamp=None,
    claim_text=None,
    qc_status: QCStatus = QCStatus.PASS,
    obs_id: str | None = None,
) -> Observation:
    _counter["n"] += 1
    return Observation(
        observation_id=obs_id or f"obs-{_counter['n']}",
        feature=feature,
        market=market,
        value=value,
        reference_date=reference_date,
        knowledge_date=knowledge_date,
        superseded_date=superseded_date,
        claim_text=claim_text,
        provenance=Provenance(
            source_url=source_url,
            content_hash=content_hash,
            archive_capture_timestamp=archive_capture_timestamp,
        ),
        extraction=Extraction(method=ExtractionMethod.API_INGEST, source_name="arxiv_api"),
        confidence=1.0,
        confidence_source=ConfidenceSource.DETERMINISTIC,
        qc_status=qc_status,
    )


def obs_frame(observations) -> pd.DataFrame:
    return observations_to_frame(observations)


def make_fv(
    *,
    feature: str = "arxiv_count",
    market: str = "llm-dev-tools",
    month: str = "2024-01",
    value=10.0,
    knowledge_date_max: date = date(2024, 1, 31),
    confidence: float = 1.0,
    obs_ids=("obs-1",),
) -> FeatureValue:
    return FeatureValue(
        feature=feature,
        market=market,
        month=month,
        value=value,
        type=FeatureType.COUNT,
        confidence=confidence,
        knowledge_date_max=knowledge_date_max,
        source_observation_ids=tuple(obs_ids),
    )


def panel_frame(values) -> pd.DataFrame:
    return panel_to_frame(values)
