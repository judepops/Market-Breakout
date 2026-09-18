"""Tests for the LLM-funding parser (dated round history + verification); no network."""

from __future__ import annotations

from datetime import date

from marketbrain.signals.llm_funding import parse_funding


def test_parse_funding_dated_rounds_and_verify():
    text = ('Cohere raised a $125M Series A in 2021 and a $500M Series D on 2024-07-22; ~$970M total. '
            '{"total_raised_usd": 970000000, "rounds": ['
            '{"date": "2021-09-09", "amount_usd": 125000000, "stage": "series_a"}, '
            '{"date": "2024-07-22", "amount_usd": 500000000, "stage": "series_d"}], '
            '"source_url": "https://example.com/x"}')
    three = ("https://a.com", "https://b.com", "https://c.com")
    f = parse_funding("Cohere", text, three)
    assert f.total_raised_usd == 970_000_000 and f.verified is True       # total + >= 3 sources
    assert len(f.rounds) == 2 and f.rounds[0].date == date(2021, 9, 9) and f.rounds[0].stage == "series_a"
    assert f.latest_round_date == date(2024, 7, 22) and f.latest_round_usd == 500_000_000  # derived = most recent
    assert f.evidence_url == "https://a.com"


def test_parse_funding_unverified_below_3_sources():
    text = '{"total_raised_usd": 970000000, "rounds": [], "source_url": null}'
    assert parse_funding("X", text, ("https://a.com", "https://b.com")).verified is False   # only 2 sources


def test_parse_funding_handles_nulls_and_no_json():
    f = parse_funding("X", '{"total_raised_usd": null, "rounds": [], "source_url": null}',
                      ("https://a.com", "https://b.com", "https://c.com"))
    assert f.total_raised_usd is None and f.verified is False and f.rounds == ()   # no total -> unverified
    g = parse_funding("Y", "no json at all", ())
    assert g.total_raised_usd is None and g.evidence_url is None and g.rounds == ()
