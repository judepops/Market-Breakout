"""arXiv technical-emergence signal — the flagship, methodology-grounded "research is heating up" feature.

Runs the validated Carley/Garner scorer (``features.emergence``) over the full arXiv records
(``arxiv_records``) to produce a monthly **emergence score** for the market:

    score(T) = sum over emergent terms of  sqrt(EScore_term) * record_count_term

computed on the 30-month window ending at T. Leakage-safe by construction: the window for month T
contains only papers submitted by end-of-T, so ``knowledge_date = reference_date = end_of(T)``.

This is a *derived* signal — it pulls records once (cached), extracts terms once, then scores each
month. Exposes the ``FEATURE`` / ``collect`` contract so the generic ``collect_signal`` driver and
the ``REGISTRY`` can dispatch it by name.
"""

from __future__ import annotations

import hashlib

from ..features.emergence import EmergenceConfig, emergence_trajectory, month_range_ending
from ..schema import Extraction, ExtractionMethod, Observation, Provenance
from ..schema.bitemporal import SUPERSEDED_OPEN
from ..timeutil import end_of_month, month_range
from .arxiv import LLM_DEVTOOLS_QUERY
from .arxiv_records import extract_term_events, fetch_records

FEATURE = "arxiv_emergence"


def _to_observation(month: str, score: float, *, market: str, records_hash: str,
                    code_version: str = "v1") -> Observation:
    eo = end_of_month(month)
    return Observation(
        observation_id=f"arxiv_emergence:{market}:{month}",
        feature=FEATURE,
        market=market,
        value=float(score),
        reference_date=eo,   # the window ENDS at this month
        knowledge_date=eo,   # every paper in the window is public by end-of-month (natively dated)
        superseded_date=SUPERSEDED_OPEN,
        claim_text=f"Carley/Garner arXiv emergence score for {market} as-of {month} = {score:.2f}",
        provenance=Provenance(
            source_url="http://export.arxiv.org/api/query?search_query=" + LLM_DEVTOOLS_QUERY,
            content_hash=records_hash,
            dated_quote=f"emergence_score={score:.4f}",
        ),
        extraction=Extraction(
            method=ExtractionMethod.API_INGEST, source_name="arxiv_api", code_version=code_version
        ),
        confidence=1.0,
    )


def collect(
    start: str, end: str, *, market: str = "llm-dev-tools", sleep: float = 3.0,
    cfg: EmergenceConfig = EmergenceConfig(), min_doc_freq: int = 10,
) -> list[Observation]:
    """Fetch records over the trailing window, extract terms, score each month in [start, end]."""
    window_start = month_range_ending(start, cfg.window_months)[0]
    records = fetch_records(window_start, end, sleep=sleep)
    events = extract_term_events(records, min_doc_freq=min_doc_freq)
    records_hash = hashlib.sha256(
        f"{LLM_DEVTOOLS_QUERY}|{window_start}|{end}|{len(records)}".encode()
    ).hexdigest()
    months = month_range(start, end)
    scores = emergence_trajectory(events, months, cfg)
    return [
        _to_observation(m, scores[m], market=market, records_hash=records_hash)
        for m in months
    ]
