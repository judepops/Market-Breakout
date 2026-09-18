"""Tests for arXiv term extraction (deterministic, no network)."""

from __future__ import annotations

import pandas as pd

from marketbrain.signals.arxiv_records import _terms_in, extract_term_events


def test_terms_drops_boundary_stopwords_and_short_tokens():
    terms = _terms_in("We propose a retrieval augmented generation model")
    assert "retrieval augmented generation" in terms
    assert "retrieval augmented" in terms
    assert "we propose" not in terms          # boundary stopword
    assert "a retrieval" not in terms          # boundary stopword + short token
    assert all(len(tok) >= 3 for t in terms for tok in t.split())


def test_extract_prunes_by_doc_freq_and_keeps_authors():
    # "agentic workflow" appears in 3 records; "one off phrase" in 1 -> pruned at min_doc_freq=2
    rows = [
        ("r1", "2024-01", "Agentic workflow", "an agentic workflow study", ["Alice"]),
        ("r2", "2024-02", "Agentic workflow", "another agentic workflow", ["Bob"]),
        ("r3", "2024-03", "Agentic workflow", "more agentic workflow here", ["Carol", "Dan"]),
        ("r4", "2024-03", "One off phrase", "one off phrase only", ["Eve"]),
    ]
    records = pd.DataFrame(rows, columns=["record_id", "month", "title", "abstract", "authors"])
    ev = extract_term_events(records, min_doc_freq=2)

    assert set(ev.columns) == {"record_id", "month", "term", "authors"}
    kept = set(ev["term"])
    assert "agentic workflow" in kept
    assert "one off phrase" not in kept                 # doc-freq 1 < 2 -> pruned
    assert all(isinstance(a, frozenset) for a in ev["authors"])
    r3_authors = ev[ev["record_id"] == "r3"]["authors"].iloc[0]
    assert r3_authors == frozenset({"Carol", "Dan"})
