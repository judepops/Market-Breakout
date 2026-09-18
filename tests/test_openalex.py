import json

from experiments.fetch_openalex import arxiv_doi, parse_works


def test_arxiv_doi_strips_version_and_prefix():
    assert arxiv_doi("http://arxiv.org/abs/2312.00752v2") == "10.48550/arXiv.2312.00752"
    assert arxiv_doi("http://arxiv.org/abs/2005.14165") == "10.48550/arXiv.2005.14165"
    assert arxiv_doi("not a record id") is None


def test_parse_works_maps_doi_back_to_record():
    payload = {"results": [{
        "doi": "https://doi.org/10.48550/arxiv.2312.00752",
        "cited_by_count": 1019,
        "publication_year": 2023,
        "counts_by_year": [{"year": 2024, "cited_by_count": 236}, {"year": 2025, "cited_by_count": 693}],
    }]}
    doi_map = {"10.48550/arxiv.2312.00752": "http://arxiv.org/abs/2312.00752v1"}
    rows = parse_works(payload, doi_map)
    assert len(rows) == 1
    assert rows[0]["record_id"] == "http://arxiv.org/abs/2312.00752v1"
    assert json.loads(rows[0]["counts_by_year"]) == {"2024": 236, "2025": 693}
    assert rows[0]["cited_by_count"] == 1019


def test_parse_works_skips_unknown_dois():
    payload = {"results": [{"doi": "https://doi.org/10.1234/other", "cited_by_count": 5,
                            "publication_year": 2020, "counts_by_year": []}]}
    assert parse_works(payload, {}) == []
