"""The bitemporal data model — the foundation everything rides on.

Implements the data-model table in ``Docs/corpus_methodology.md`` §3:

    value/claim | reference_date | knowledge_date | superseded_date(∞) |
    source_url + archive_capture_timestamp + content_hash | extraction_method + version

The single most important invariant in the whole project lives here: a datum is
keyed by TWO dates — ``reference_date`` (the period the fact is *about*) and
``knowledge_date`` (when it first became public). A query for month T must never
return anything whose ``knowledge_date`` is after end-of-T. See
``marketbrain.validation.checks.check_no_future_leakage``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Optional

from .enums import ConfidenceSource, ExtractionMethod, QCStatus

# ALFRED ``realtime_end`` = +infinity. A far-future sentinel (not None) keeps the
# as-of query a single branch-free comparison: realtime_start <= D < realtime_end.
SUPERSEDED_OPEN: date = date(9999, 12, 31)


@dataclass(frozen=True)
class Provenance:
    """Where a claim came from. ``check_provenance_completeness`` rejects any
    value-bearing observation missing ``source_url`` or ``content_hash``.
    """

    source_url: str
    content_hash: str  # sha256 of the captured bytes the value was read from
    archive_capture_timestamp: Optional[datetime] = None  # the real "as-of" for an archived page
    dated_quote: Optional[str] = None
    earliest_verifiable_date: Optional[date] = None  # earliest provable date of the source


@dataclass(frozen=True)
class Extraction:
    """Reproducibility metadata: how the value was produced."""

    method: ExtractionMethod
    source_name: Optional[str] = None    # e.g. "arxiv_api", "github_api", "google_trends", "wayback"
    code_version: Optional[str] = None   # pipeline git sha / version that produced the value
    prompt_version: Optional[str] = None  # None in v1 (reserved for the future LLM extractor)
    run_id: Optional[str] = None         # links back to a checkpoint row


@dataclass(frozen=True)
class Observation:
    """One bitemporal datum (Stage 3 raw output).

    Invariants (enforced by ``marketbrain.validation`` and asserted in tests):
      * ``knowledge_date >= reference_date``                (event-vs-knowledge skew)
      * provenance present for any value-bearing observation (provenance completeness)
      * ``realtime_start <= D < realtime_end``              (as-of / vintage reads)
    """

    observation_id: str
    feature: str
    market: str
    value: Any
    reference_date: date   # the period/event the fact is ABOUT
    knowledge_date: date   # when it first became public — integrity-critical
    provenance: Provenance
    extraction: Extraction
    superseded_date: date = SUPERSEDED_OPEN  # ALFRED realtime_end; ∞ until a revision replaces it
    claim_text: Optional[str] = None
    confidence: Optional[float] = None
    confidence_source: ConfidenceSource = ConfidenceSource.DETERMINISTIC
    qc_status: QCStatus = QCStatus.UNCHECKED

    # --- ALFRED vocabulary aliases (no duplicate storage) ---
    @property
    def realtime_start(self) -> date:
        return self.knowledge_date

    @property
    def realtime_end(self) -> date:
        return self.superseded_date
