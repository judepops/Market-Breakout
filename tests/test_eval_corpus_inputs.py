"""Tests for the corpus -> eval-inputs bridge (macro-normaliser + theme term-sets)."""
import pandas as pd

from marketbrain.eval.corpus_inputs import cs_denominator, themes_from_clusters


def test_cs_denominator_counts_distinct_docs_per_month():
    corpus = pd.DataFrame({"record_id": ["a", "b", "c"], "month": ["2020-01", "2020-01", "2020-02"]})
    den = cs_denominator(corpus)
    assert den["2020-01"] == 2 and den["2020-02"] == 1
    assert list(den.index) == ["2020-01", "2020-02"]            # sorted ascending


def test_themes_from_clusters_builds_term_sets():
    th = themes_from_clusters([["x", "y"], ["z"]])
    assert th["theme_0000"] == frozenset({"x", "y"})
    assert th["theme_0001"] == frozenset({"z"})


def test_themes_from_clusters_expands_via_member_map():
    th = themes_from_clusters([["opd"]], member_to_terms={"opd": ["opd", "on-policy distillation"]})
    assert th["theme_0000"] == frozenset({"opd", "on-policy distillation"})
