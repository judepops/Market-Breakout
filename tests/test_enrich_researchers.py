from experiments.enrich_researchers import collect_researchers, parse_enrichment


def test_parse_valid_with_founder():
    text = '{"affiliation": "UC Berkeley", "profile_url": "https://scholar.google.com/x", "is_founder": true, "founder_company": "Physical Intelligence", "confidence": "high"}'
    d = parse_enrichment(text)
    assert d["affiliation"] == "UC Berkeley"
    assert d["profile_url"] == "https://scholar.google.com/x"
    assert d["is_founder"] is True and d["founder_company"] == "Physical Intelligence"


def test_founder_flag_requires_company():
    d = parse_enrichment('{"is_founder": true, "founder_company": null, "affiliation": "MIT"}')
    assert d["is_founder"] is False and d["founder_company"] is None


def test_bad_url_dropped():
    d = parse_enrichment('{"profile_url": "scholar.google.com/x", "affiliation": "MIT"}')
    assert d["profile_url"] is None  # not http(s)


def test_json_in_markdown_fence():
    d = parse_enrichment('```json\n{"affiliation": "Stanford", "is_founder": false}\n```')
    assert d["affiliation"] == "Stanford" and d["is_founder"] is False


def test_garbage_returns_none():
    assert parse_enrichment("no json here") is None
    assert parse_enrichment("") is None


def test_collect_researchers_dedupes_and_contexts():
    snap = {
        "live": [{"id": "rl post-training", "researchers": [{"name": "Ada"}, {"name": "Bob"}]}],
        "fads": [{"id": "gait recognition", "researchers": [{"name": "Ada"}]}],
    }
    out = collect_researchers(snap)
    assert set(out) == {"Ada", "Bob"}
    assert set(out["Ada"]) == {"rl post-training", "gait recognition"}
