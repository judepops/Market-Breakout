"""Offline tests for the Wikipedia and Hacker News signals + the registry (no network)."""

from __future__ import annotations

from marketbrain.corpus import build_panel
from marketbrain.schema import observations_to_frame
from marketbrain.markets import llm_dev_tools
from marketbrain.signals import REGISTRY
from marketbrain.signals.crunchbase_funding import FEATURE as CB_FEATURE
from marketbrain.signals.crunchbase_funding import month_value_to_observation as cb_obs
from marketbrain.signals.edgar import FEATURE as EDGAR_FEATURE
from marketbrain.signals.edgar import month_count_to_observation as edgar_obs
from marketbrain.signals.hackernews import FEATURE as HN_FEATURE
from marketbrain.signals.hackernews import _unix_bounds, month_count_to_observation
from marketbrain.signals.npm import FEATURE as NPM_FEATURE
from marketbrain.signals.npm import _chunks
from marketbrain.signals.npm import month_value_to_observation as npm_month_obs
from marketbrain.signals.pypi import FEATURE as PYPI_FEATURE
from marketbrain.signals.pypi import month_value_to_observation as pypi_month_obs
from marketbrain.signals.wikipedia import FEATURE as WIKI_FEATURE
from marketbrain.signals.wikipedia import month_value_to_observation
from marketbrain.timeutil import end_of_month
from marketbrain.validation import check_no_future_leakage


def test_registry_modules_expose_the_contract():
    assert set(REGISTRY) == {"arxiv", "wikipedia", "hackernews", "npm", "pypi", "github", "edgar",
                             "crunchbase", "emergence", "patents"}
    for mod in REGISTRY.values():
        assert isinstance(mod.FEATURE, str) and callable(mod.collect)


def test_crunchbase_observation_is_leakage_safe():
    o = cb_obs("2024-03", 500_000_000, 2, "bq://x", market="llm-dev-tools")
    assert o.feature == CB_FEATURE and o.value == 500_000_000.0
    assert o.knowledge_date == o.reference_date == end_of_month("2024-03")
    assert o.provenance.content_hash


def test_llm_dev_tools_roster_is_well_formed():
    perms = llm_dev_tools.permalinks()
    assert len(perms) == len(set(perms)) >= 20         # unique permalinks, a real roster
    assert "langchain" in perms and "together-1a7e" in perms  # disambiguated permalinks
    p2c = llm_dev_tools.package_to_company()
    assert p2c.get("langchain") == "LangChain" and p2c.get("transformers") == "Hugging Face"


def test_pypi_observation_is_leakage_safe():
    o = pypi_month_obs("2024-03", 50_000_000, "bigquery://x", market="llm-dev-tools")
    assert o.feature == PYPI_FEATURE and o.value == 50_000_000.0
    assert o.knowledge_date == o.reference_date == end_of_month("2024-03")
    assert o.provenance.content_hash


def test_edgar_observation_is_leakage_safe_incl_zero():
    o = edgar_obs("2024-03", 0, "http://x", market="llm-dev-tools")  # zero is a real datum
    assert o.feature == EDGAR_FEATURE and o.value == 0.0
    assert o.knowledge_date == o.reference_date == end_of_month("2024-03")
    assert o.provenance.content_hash


def test_npm_observation_is_leakage_safe():
    o = npm_month_obs("2024-03", 5_000_000, "http://x", market="llm-dev-tools")
    assert o.feature == NPM_FEATURE and o.value == 5_000_000.0
    assert o.knowledge_date == o.reference_date == end_of_month("2024-03")


def test_npm_chunks_cover_range_and_respect_cap():
    from datetime import date
    chunks = _chunks("2022-06", "2025-12")
    assert chunks[0][0] == "2022-06-01" and chunks[-1][1] == "2025-12-31"
    for lo, hi in chunks:
        assert (date.fromisoformat(hi) - date.fromisoformat(lo)).days <= 548  # npm 18-month cap


def test_wikipedia_observation_is_leakage_safe():
    o = month_value_to_observation("2024-03", 12345, "http://x", market="llm-dev-tools")
    assert o.feature == WIKI_FEATURE and o.value == 12345.0
    assert o.knowledge_date == o.reference_date == end_of_month("2024-03")
    assert o.provenance.content_hash


def test_hackernews_observation_is_leakage_safe():
    o = month_count_to_observation("2024-03", 42, "http://x", market="llm-dev-tools")
    assert o.feature == HN_FEATURE and o.value == 42.0
    assert o.knowledge_date == o.reference_date == end_of_month("2024-03")


def test_hackernews_unix_bounds_cover_the_month():
    lo, hi = _unix_bounds("2024-02")  # leap February
    assert hi - lo > 28 * 24 * 3600   # at least 28 days of seconds


def test_mixed_feature_panel_passes_leakage():
    obs = [
        month_value_to_observation("2024-01", 1000, "http://w", market="llm-dev-tools"),
        month_count_to_observation("2024-01", 20, "http://h", market="llm-dev-tools"),
    ]
    panel = build_panel(observations_to_frame(obs), "2024-01")
    assert len(panel) == 2  # two features share the same monthly cut
    assert check_no_future_leakage(panel) == []
