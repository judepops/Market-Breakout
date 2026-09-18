"""Tests for the market registry + the data-infra roster (offline)."""

from __future__ import annotations

import pytest

from marketbrain.markets import data_infra, get_market, llm_dev_tools


def test_registry_resolves_known_markets():
    assert get_market("llm-dev-tools") is llm_dev_tools
    assert get_market("data-infra") is data_infra


def test_unknown_market_raises():
    with pytest.raises(KeyError):
        get_market("nope")


def test_data_infra_roster_is_well_formed():
    perms = data_infra.permalinks()
    assert len(perms) == len(set(perms)) >= 18           # unique permalinks, a real roster
    assert "databricks" in perms and "snowflake-computing" in perms
    assert len(data_infra.names()) == len(perms) == len(data_infra.assignee_names())
    assert all("/" in r for r in data_infra.github_repos())  # owner/repo form


def test_both_markets_share_the_interface():
    # every signal + the label dispatch on these four accessors, so both markets must expose all of them
    for m in (llm_dev_tools, data_infra):
        assert m.permalinks() and m.names() and m.assignee_names() and m.github_repos()
