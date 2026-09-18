"""Enrichment - the LLM auto-seeder for per-market signal anchors (the v1.1 LLM-as-tool layer).

Generates the per-market configuration the signal collectors need (arXiv terms + categories now; GitHub
repos / packages next) using Gemini, so the signals measure the right thing per market without hand
curation. LLM output is always DETERMINISTICALLY checked (the generated query must return real arXiv
papers) before it is trusted - same discipline as the grounded label.
"""
