"""Tests for the disjoint-partition assignment + the LLM classifier parser (no network)."""

from __future__ import annotations

from marketbrain.enrichment.classify import parse_assignments
from marketbrain.markets.partition import assign_deterministic, tags_present


def test_assign_single_tag_is_deterministic():
    assert assign_deterministic("Developer Tools, Software") == "developer-tools"   # one taxonomy tag
    assert assign_deterministic("Restaurants, Food") is None                        # no taxonomy tag -> dropped


def test_assign_multi_tag_is_ambiguous():
    a = assign_deterministic("Generative AI, Developer Tools, SaaS")
    assert a == "AMBIGUOUS"
    assert tags_present("Generative AI, Developer Tools, SaaS") == ["generative-ai", "developer-tools", "saas"]


def test_specific_beats_general_in_tags_present_order():
    # priority order: the specific market precedes the catch-all
    pres = tags_present("Artificial Intelligence (AI), Generative AI")
    assert pres.index("generative-ai") < pres.index("artificial-intelligence")


def test_parse_assignments_keeps_only_valid_candidates():
    batch = [("Acme", ["database", "analytics"], "..."), ("Beta", ["devops", "saas"], "...")]
    out = parse_assignments('{"0": "database", "1": "not-a-candidate"}', batch)
    assert out == {0: "database"}        # 1's answer not in its candidates -> dropped
    assert parse_assignments("no json", batch) == {}
