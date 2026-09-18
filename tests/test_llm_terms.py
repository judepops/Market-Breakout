"""Tests for the LLM term-generation pure logic (parse + query build); no network."""

from __future__ import annotations

from marketbrain.enrichment.llm_terms import build_arxiv_query, parse_repos, parse_terms


def test_parse_repos_validates_and_dedups():
    text = ('{"repos": ["kubernetes/kubernetes", "https://github.com/redis/redis/", '
            '"redis/redis", "not a repo", "owner/with.dots-ok"]}')
    repos = parse_repos(text)
    assert repos == ("kubernetes/kubernetes", "redis/redis", "owner/with.dots-ok")  # url-stripped, deduped
    assert parse_repos("no json") == ()
    assert parse_repos('{"repos": []}') == ()                # market with no OSS adoption


def test_parse_terms_extracts_and_validates():
    text = ('prose... {"terms": ["solar cell", "\\"photovoltaic\\"", ""], '
            '"categories": ["physics.app-ph", "cond-mat.mtrl-sci", "not a cat!"]}')
    terms, cats = parse_terms(text)
    assert terms == ("solar cell", "photovoltaic")            # blanks dropped, quotes stripped
    assert cats == ("physics.app-ph", "cond-mat.mtrl-sci")    # invalid category code filtered out


def test_parse_terms_handles_no_json_and_caps():
    assert parse_terms("no json here") == ((), ())
    big = '{"terms": %s, "categories": []}' % str([f"t{i}" for i in range(10)]).replace("'", '"')
    terms, _ = parse_terms(big)
    assert len(terms) == 6                                    # capped at 6


def test_build_arxiv_query_disambiguates_with_categories():
    q = build_arxiv_query(("solar cell", "photovoltaic"), ("physics.app-ph", "cond-mat.mtrl-sci"))
    assert q == '(cat:physics.app-ph OR cat:cond-mat.mtrl-sci) AND (all:"solar cell" OR all:"photovoltaic")'


def test_build_arxiv_query_terms_only_and_empty():
    assert build_arxiv_query(("devops",), ()) == '(all:"devops")'
    assert build_arxiv_query((), ()) is None                  # no terms -> no query (market has no research base)


def test_parse_hn_query():
    from marketbrain.enrichment.llm_terms import parse_hn_query
    assert parse_hn_query('blah {"query": "modern data stack"}') == "modern data stack"
    assert parse_hn_query('{"query": ""}') is None
    assert parse_hn_query("no json") is None
