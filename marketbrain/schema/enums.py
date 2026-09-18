"""Controlled vocabularies for the market-brain corpus.

These are intentionally small for v1 (deterministic signals only). LLM-extraction
and LLM-verification vocabularies (FActScore "supported/not-supported", etc.) are
deferred to a later version together with the think-reason-learn feature layer.
"""

from __future__ import annotations

from enum import Enum


class ExtractionMethod(str, Enum):
    """How a value entered the corpus. v1 is deterministic / API-ingested only."""

    API_INGEST = "api_ingest"        # pulled from a natively-dated API (arXiv, GitHub, Trends, ...)
    DETERMINISTIC = "deterministic"  # derived by a deterministic transform of other corpus values
    ARCHIVE = "archive"              # read from a dated web-archive capture (Wayback / Common Crawl)
    HUMAN = "human"                  # hand-coded during the audit


class FeatureType(str, Enum):
    """Semantics of a feature value (the ``type`` field of the feature schema)."""

    COUNT = "count"                # e.g. arXiv submissions in the month
    LEVEL = "level"                # standardized index level
    RATE = "rate"                  # normalized-by-volume rate
    MOMENTUM = "momentum"          # growth / first difference
    ACCELERATION = "acceleration"  # second difference
    CURVATURE = "curvature"        # inflection-proximity (Bass/loglet parameter)
    BOOLEAN = "boolean"            # YES/NO feature
    CATEGORICAL = "categorical"


class ConfidenceSource(str, Enum):
    """How the ``confidence`` on a value was derived."""

    DETERMINISTIC = "deterministic"  # 1.0 for a directly-ingested or derived value
    CAPTURE_QUALITY = "capture_quality"  # downgraded by capture-staleness check
    MULTI_SOURCE = "multi_source"    # raised/lowered by deterministic multi-source agreement


class QCStatus(str, Enum):
    """Deterministic quality-control state for an observation (v1)."""

    UNCHECKED = "unchecked"
    PASS = "pass"
    FAIL = "fail"
    ABSTAINED = "abstained"  # no evidence found — a valid outcome, never fabricate a value
