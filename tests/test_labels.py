"""Offline tests for the success-label logic (no BigQuery, no Gemini)."""

from __future__ import annotations

from datetime import date

from marketbrain.labels import store
from marketbrain.labels.build_label import build_market_label, decide_company
from marketbrain.labels.crunchbase_company import CompanyFunding
from marketbrain.labels.gemini_grounding import GroundedResult, parse_response

THRESH = 500_000_000


def _cf(**kw) -> CompanyFunding:
    base = dict(permalink="x", name="X", valuation_usd=None, valuation_date=None,
                total_raised_usd=None, raise_cross_date=None, ma_price_usd=None, ma_date=None,
                ipo_valuation_usd=None, ipo_date=None)
    base.update(kw)
    return CompanyFunding(**base)


# --- decide_company ---

def test_outlier_first_crossing_is_earliest_qualifying_metric():
    # raise crossed $500M in 2024-01; valuation $13B dated 2025-09 -> the raise is the FIRST crossing
    cf = _cf(valuation_usd=13_000_000_000, valuation_date=date(2025, 9, 9),
             total_raised_usd=1_000_000_000, raise_cross_date=date(2024, 1, 1))
    o = decide_company(cf, None, threshold_usd=THRESH)
    assert o.is_outlier and o.metric == "raise" and o.milestone_date == date(2024, 1, 1)


def test_below_threshold_is_not_outlier():
    cf = _cf(total_raised_usd=400_000_000, valuation_usd=450_000_000, valuation_date=date(2023, 1, 1))
    o = decide_company(cf, None, threshold_usd=THRESH)
    assert not o.is_outlier and o.cb_value_usd == 450_000_000 and o.milestone_date is None


def test_grounding_gap_fills_when_crunchbase_empty():
    g = GroundedResult("x", True, "valuation", 600_000_000, date(2025, 1, 1), date(2025, 1, 2),
                       "https://news/x", ("https://news/x",), True, "{}")
    o = decide_company(_cf(), g, threshold_usd=THRESH)
    assert o.is_outlier and o.source == "gemini" and o.verified
    assert o.evidence_url == "https://news/x" and o.knowledge_date == date(2025, 1, 2)


def test_crunchbase_outlier_unverified_when_grounding_cannot_confirm():
    cf = _cf(valuation_usd=2_000_000_000, valuation_date=date(2024, 5, 1))
    g = GroundedResult("x", None, None, None, None, None, None, (), False, "{}")
    o = decide_company(cf, g, threshold_usd=THRESH)
    assert o.is_outlier and o.source == "crunchbase" and not o.verified


def test_knowledge_date_clamped_to_milestone():
    # grounding's public_date precedes the Crunchbase milestone -> clamped up (can't be public before it happened)
    cf = _cf(valuation_usd=2_000_000_000, valuation_date=date(2024, 8, 1))
    g = GroundedResult("x", True, "valuation", 2_000_000_000, date(2023, 1, 1), date(2021, 1, 1),
                       "https://a", ("https://a",), True, "{}")
    o = decide_company(cf, g, threshold_usd=THRESH)
    assert o.milestone_date == date(2024, 8, 1) and o.knowledge_date == date(2024, 8, 1)


# --- build_market_label ---

def test_market_label_uses_earliest_company_crossing():
    a = decide_company(_cf(permalink="a", valuation_usd=1_000_000_000, valuation_date=date(2025, 3, 1)),
                       None, threshold_usd=THRESH)
    b = decide_company(_cf(permalink="b", ma_price_usd=900_000_000, ma_date=date(2023, 7, 1)),
                       None, threshold_usd=THRESH)
    c = decide_company(_cf(permalink="c", total_raised_usd=10_000_000), None, threshold_usd=THRESH)
    label = build_market_label([a, b, c], "m", threshold_usd=THRESH)
    assert label.is_outlier and label.label_reference_date == date(2023, 7, 1)


def test_market_label_false_when_no_outliers():
    c = decide_company(_cf(total_raised_usd=1_000_000), None, threshold_usd=THRESH)
    assert build_market_label([c], "m", threshold_usd=THRESH).is_outlier is False


# --- gemini parse + deterministic re-check ---

def test_parse_verified_uses_grounding_url_not_prose():
    text = ('{"is_outlier": true, "metric":"valuation", "value_usd": 600000000, '
            '"milestone_date":"2025-01-01","public_date":"2025-01-02","source_url":"https://model-says"}')
    cites = ("https://real-citation", "https://b", "https://c")  # >= 3 sources
    r = parse_response("x", text, cites, threshold_usd=THRESH)
    assert r.verified and r.value_usd == 600_000_000 and r.metric == "valuation"
    assert r.evidence_url == "https://real-citation"   # grounding URI, not the model's source_url


def test_parse_requires_min_corroborating_sources():
    j = '{"is_outlier": true, "value_usd": 600000000}'
    assert not parse_response("x", j, (), threshold_usd=THRESH).verified                       # 0
    assert not parse_response("x", j, ("https://a", "https://b"), threshold_usd=THRESH).verified  # 2 < 3
    assert parse_response("x", j, ("https://a", "https://b", "https://c"), threshold_usd=THRESH).verified  # 3


def test_parse_below_threshold_is_unverified():
    r = parse_response("x", '{"value_usd": 100000000}', ("https://a", "https://b", "https://c"),
                       threshold_usd=THRESH)
    assert not r.verified


def test_parse_prose_wrapped_json():
    text = 'Here is the result:\n```json\n{"is_outlier": false, "value_usd": null}\n```\nThanks.'
    r = parse_response("x", text, (), threshold_usd=THRESH)
    assert r.is_outlier is False and r.value_usd is None


# --- store round-trip (separate from the feature corpus) ---

def test_store_roundtrip(tmp_path):
    o = decide_company(_cf(permalink="a", valuation_usd=1_000_000_000, valuation_date=date(2025, 1, 1)),
                       None, threshold_usd=THRESH)
    label = build_market_label([o], "m", threshold_usd=THRESH)
    store.write(label, [o], market="m", root=tmp_path)
    df = store.load("m", root=tmp_path)
    assert (df.level == "company").sum() == 1 and (df.level == "market").sum() == 1
    assert bool(df[df.level == "market"].iloc[0].is_outlier) is True
