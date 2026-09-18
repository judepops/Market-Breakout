"""Tests for the hand-curated negative-control set (pre-registration §3.2)."""

from __future__ import annotations

import pandas as pd

from marketbrain.eval.negatives import (
    FLAGGED_AMBIGUOUS,
    KNOWN_FIZZLE,
    FizzleControl,
    map_fizzles_to_themes,
    present_in_corpus,
)


def test_known_fizzle_has_about_twenty_entries():
    assert 18 <= len(KNOWN_FIZZLE) <= 22


def test_every_entry_has_nonempty_terms_and_a_name():
    for f in KNOWN_FIZZLE:
        assert isinstance(f, FizzleControl)
        assert f.name and f.name.strip()
        assert len(f.terms) >= 1
        assert all(t and t.strip() for t in f.terms)


def test_exactly_three_flagged_ambiguous_and_they_are_the_named_three():
    flagged = {f.name for f in FLAGGED_AMBIGUOUS}
    assert flagged == {"quantum machine learning", "federated learning", "prompt engineering"}
    assert all(f.flagged_ambiguous for f in FLAGGED_AMBIGUOUS)
    # And every other entry is NOT flagged.
    assert sum(f.flagged_ambiguous for f in KNOWN_FIZZLE) == 3


def test_names_are_unique():
    names = [f.name for f in KNOWN_FIZZLE]
    assert len(names) == len(set(names))


def _events(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    """Build a tiny term_events frame from (record_id, month, term) rows (authors unused here)."""
    df = pd.DataFrame(rows, columns=["record_id", "month", "term"])
    df["authors"] = [frozenset()] * len(df)
    return df


def test_present_in_corpus_true_when_window_clears_threshold():
    metaverse = next(f for f in KNOWN_FIZZLE if f.name == "metaverse")
    # 5 distinct records mentioning "metaverse" within a single month -> clears birth (>=5 in 3mo).
    rows = [(f"r{i}", "2021-03", "metaverse") for i in range(5)]
    assert present_in_corpus(metaverse, _events(rows)) is True


def test_present_in_corpus_true_when_threshold_spread_across_three_month_window():
    metaverse = next(f for f in KNOWN_FIZZLE if f.name == "metaverse")
    rows = [("r1", "2021-01", "metaverse"), ("r2", "2021-01", "metaverse"),
            ("r3", "2021-02", "metaverse"), ("r4", "2021-03", "metaverse"),
            ("r5", "2021-03", "metaverse")]
    assert present_in_corpus(metaverse, _events(rows)) is True


def test_present_in_corpus_false_when_below_threshold():
    metaverse = next(f for f in KNOWN_FIZZLE if f.name == "metaverse")
    rows = [("r1", "2021-01", "metaverse"), ("r2", "2021-05", "metaverse")]
    # Only 2 records, and they are far apart -> no 3-month window reaches 5.
    assert present_in_corpus(metaverse, _events(rows)) is False


def test_present_in_corpus_false_when_records_too_sparse_across_gap():
    metaverse = next(f for f in KNOWN_FIZZLE if f.name == "metaverse")
    # 6 records but spread one-per-quarter so no 3-month window holds >=5.
    rows = [(f"r{i}", m, "metaverse") for i, m in
            enumerate(["2020-01", "2020-04", "2020-07", "2020-10", "2021-01", "2021-04"])]
    assert present_in_corpus(metaverse, _events(rows)) is False


def test_present_in_corpus_unique_records_not_double_counted():
    metaverse = next(f for f in KNOWN_FIZZLE if f.name == "metaverse")
    # Same 3 records repeated across rows (e.g. multiple matching terms) -> only 3 distinct records.
    rows = [("r1", "2021-03", "metaverse"), ("r1", "2021-03", "metaverse"),
            ("r2", "2021-03", "metaverse"), ("r3", "2021-03", "metaverse")]
    assert present_in_corpus(metaverse, _events(rows)) is False


def test_present_in_corpus_case_insensitive_term_match():
    metaverse = next(f for f in KNOWN_FIZZLE if f.name == "metaverse")
    rows = [(f"r{i}", "2021-03", "Metaverse") for i in range(5)]
    assert present_in_corpus(metaverse, _events(rows)) is True


def test_present_in_corpus_empty_events():
    metaverse = next(f for f in KNOWN_FIZZLE if f.name == "metaverse")
    assert present_in_corpus(metaverse, _events([])) is False


def test_map_fizzles_returns_present_only_for_clearing_fizzles():
    # web3 clears (5 records in one month); spiking neural networks does not (2 records).
    rows = [(f"w{i}", "2021-06", "web3") for i in range(5)]
    rows += [("s1", "2021-06", "spiking neural network"),
             ("s2", "2021-06", "spiking neural network")]
    result = map_fizzles_to_themes(_events(rows))
    assert set(result) == {f.name for f in KNOWN_FIZZLE}
    assert result["web3"] is True
    assert result["spiking neural networks"] is False
    # Fizzles with no matching events at all are absent from the corpus.
    assert result["metaverse"] is False


def test_map_fizzles_matches_any_of_a_fizzles_terms():
    # no-code/low-code has several surface forms; "low code" alone should suffice.
    rows = [(f"r{i}", "2022-01", "low code") for i in range(5)]
    result = map_fizzles_to_themes(_events(rows))
    assert result["no-code/low-code"] is True
