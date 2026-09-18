"""Success-LABEL construction (separate from the deterministic feature corpus).

The label is the **outcome we predict** — "did a market produce a >=$500M company, and when" — so unlike
features it MAY use the future (it is hindsight by definition) and MAY use an LLM (Gemini web-grounding).
To keep that freedom from ever contaminating the leakage-safe feature corpus, labels live in this separate
package + a separate store and NEVER pass through ``build_panel`` / ``check_no_future_leakage``.

Pipeline: ``crunchbase_company`` (structured base) -> ``gemini_grounding`` (verify + cite + gap-fill) ->
``build_label`` (reconcile -> ``OutlierLabel`` per company -> market aggregate) -> ``store`` (parquet).
"""

from __future__ import annotations

from .build_label import CompanyOutcome, build_market_label, decide_company, run
from .crunchbase_company import CompanyFunding
from .gemini_grounding import GroundedResult, parse_response

__all__ = [
    "CompanyFunding",
    "GroundedResult",
    "parse_response",
    "CompanyOutcome",
    "decide_company",
    "build_market_label",
    "run",
]
