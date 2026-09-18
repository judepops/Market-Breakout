"""Catalog integrity — specs are well-formed and resolve (no network)."""

from __future__ import annotations

import pytest

from marketbrain.markets.catalog import CATALOG, clean_assignee, get_spec
from marketbrain.markets.universe import _where


def test_clean_assignee_normalises_and_filters():
    assert clean_assignee("NVIDIA") == "NVIDIA"
    assert clean_assignee("Yahoo!") == "Yahoo"                       # strips the rejected char
    assert clean_assignee("AT&T") == "AT&T"                          # & is allowed
    assert clean_assignee("Société Générale") == "Soci t G n rale"   # accents -> spaces, collapsed
    assert clean_assignee("X") is None                               # too short
    assert clean_assignee(None) is None


def test_catalog_has_positives_and_negatives():
    kinds = {m.kind for m in CATALOG}
    assert kinds == {"positive", "negative"}
    assert sum(m.kind == "negative" for m in CATALOG) >= 2   # deliberate low-rate spread
    assert len(CATALOG) >= 8                                 # N≈10 for a real cross-market target


def test_every_spec_builds_valid_sql_and_unique_names():
    names = [m.name for m in CATALOG]
    assert len(names) == len(set(names))                     # no dup market names
    for m in CATALOG:
        assert m.tag_groups and all(g for g in m.tag_groups)  # non-empty groups
        where = _where([list(g) for g in m.tag_groups], floor_usd=5_000_000,
                       founded_from=m.founded_from)
        assert "funding_total_value_usd >= @floor" in where   # the $5M fundable floor is always applied


def test_get_spec_roundtrip_and_unknown():
    assert get_spec("llm-dev-tools").kind == "positive"
    with pytest.raises(KeyError):
        get_spec("not-a-market")


def test_footprint_rich_flags_corpus_targets():
    # coverage tokens drawn from the full toolkit; footprint_rich = >= 2 DEPTH signals (gh/pkg/ax/pat)
    assert all(set(m.coverage) <= {"gh", "pkg", "ax", "pat", "wiki", "hn"} for m in CATALOG)
    assert get_spec("llm-dev-tools").footprint_rich          # gh,pkg,ax -> 3 depth
    assert get_spec("biotechnology").footprint_rich          # ax,pat -> carried by research + patents, not OSS
    assert not get_spec("saas").footprint_rich               # attention only, no depth signal
    rich = [m for m in CATALOG if m.footprint_rich]
    assert len(rich) >= 20                                   # the full toolkit reaches far beyond dev-tools
