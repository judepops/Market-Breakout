"""Corpus -> Phase-0 eval inputs: the macro-normaliser, term-events, and theme term-sets.

Leak-free by construction: arXiv submission month = knowledge month, so a theme's monthly series is
contemporaneous. The macro-normaliser N_cs(T) (total docs per month) is computed from OUR corpus, NOT the
monthly_research.py totalResults series, which the audit confirmed is a buggy API `totalResults` artifact.
"""
from __future__ import annotations


def cs_denominator(corpus):
    """N_cs(T): distinct documents per month in the loaded corpus -- the share denominator (audit P4 fix)."""
    return corpus.groupby("month")["record_id"].nunique().sort_index()


def corpus_term_events(corpus, *, min_doc_freq=25, ngram_max=3):
    """(record_id, month, term, authors) over the corpus -- reuses the production extractor verbatim."""
    from marketbrain.signals.arxiv_records import extract_term_events
    return extract_term_events(corpus, min_doc_freq=min_doc_freq, ngram_max=ngram_max)


def themes_from_clusters(clusters, member_to_terms=None):
    """Map discovery clusters -> {theme_id: frozenset(terms)}. A theme's term-set is its member phrases
    (optionally expanded via member_to_terms). theme_id is a stable index-based id."""
    out = {}
    for i, members in enumerate(clusters):
        terms = set()
        for m in members:
            terms |= set(member_to_terms.get(m, [m])) if member_to_terms else {m}
        out[f"theme_{i:04d}"] = frozenset(terms)
    return out
