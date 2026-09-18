"""Deterministic, dated signal ingestion (Stages 2-3, v1).

Each signal module pulls raw dated values from a natively-dated source and emits
leakage-safe ``Observation`` rows (``knowledge_date`` set to when the value became
public). NO LLMs in v1. See ``docs/signals_v1.md`` and ``../../Docs/data_acquisition.md``.

Every signal module exposes ``FEATURE`` (str) and ``collect(start, end, *, market,
sleep)`` so the generic driver ``experiments/collect_signal.py`` can dispatch by name.
"""

from __future__ import annotations

from . import (
    arxiv,
    arxiv_emergence,
    crunchbase_funding,
    edgar,
    github_stars,
    hackernews,
    npm,
    patents,
    pypi,
    wikipedia,
)

REGISTRY = {
    "arxiv": arxiv,             # research momentum (reachable)
    "wikipedia": wikipedia,     # market interest / attention — the Google-Trends replacement
    "hackernews": hackernews,   # leading dev-tools attention
    "npm": npm,                 # developer adoption — JS slice
    "pypi": pypi,               # developer adoption — Python slice (BigQuery; the one that matters)
    "github": github_stars,     # developer adoption — GitHub stars (GH Archive; highest-value dev signal)
    "edgar": edgar,             # capital / funding — SEC Form D filing counts (sparse)
    "crunchbase": crunchbase_funding,  # capital / funding — real $ raised (Crunchbase BigQuery)
    "emergence": arxiv_emergence,  # research emergence — Carley/Garner EScore over arXiv terms
    "patents": patents,         # IP / innovation — patent publications by CPC (BigQuery; sparse for software)
}

__all__ = ["REGISTRY", "arxiv", "arxiv_emergence", "wikipedia", "hackernews", "npm", "pypi",
           "github_stars", "edgar", "crunchbase_funding", "patents"]
