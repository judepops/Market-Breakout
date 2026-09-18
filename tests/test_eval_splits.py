"""Tests for marketbrain.eval.splits (pre-registration section 5).

Small in-code labels_df: 6 parents p1..p6, themes scattered across months 2019..2024. No I/O, no network.
The core assertion is the anti-leakage guarantee: grouped_folds NEVER shares a parent across train/test.
"""

from __future__ import annotations

import pandas as pd

from marketbrain.eval.splits import (
    assign_parents_by_hash,
    grouped_folds,
    rolling_origins,
    temporal_split,
)


def _labels() -> pd.DataFrame:
    # (theme_id, label, T0, T_dec, parent_id). T_dec spans 2019..2024 so temporal/rolling cuts bite.
    rows = [
        ("t01", "positive", "2018-06", "2019-03", "p1"),
        ("t02", "hyped_died", "2018-09", "2019-09", "p1"),
        ("t03", "never", "2019-04", "2020-01", "p2"),
        ("t04", "positive", "2019-09", "2020-06", "p2"),
        ("t05", "hyped_died", "2020-02", "2021-03", "p3"),
        ("t06", "positive", "2020-08", "2021-06", "p3"),
        ("t07", "censored", "2021-01", "2021-12", "p4"),
        ("t08", "positive", "2021-04", "2022-01", "p4"),
        ("t09", "hyped_died", "2021-09", "2022-06", "p5"),
        ("t10", "positive", "2022-03", "2022-12", "p5"),
        ("t11", "never", "2022-10", "2023-06", "p6"),
        ("t12", "positive", "2023-05", "2024-02", "p6"),
    ]
    return pd.DataFrame(rows, columns=["theme_id", "label", "T0", "T_dec", "parent_id"])


def test_temporal_split_partitions_on_t_dec() -> None:
    df = _labels()
    train, test = temporal_split(df, cutoff="2022-01")

    # train = T_dec <= 2021-12; test = T_dec >= 2022-01.
    assert set(train["theme_id"]) == {"t01", "t02", "t03", "t04", "t05", "t06", "t07"}
    assert set(test["theme_id"]) == {"t08", "t09", "t10", "t11", "t12"}
    # Disjoint and exhaustive.
    assert set(train["theme_id"]) & set(test["theme_id"]) == set()
    assert len(train) + len(test) == len(df)
    # The 2022-01 boundary theme lands in test, not train.
    assert "t08" in set(test["theme_id"])


def test_temporal_split_default_cutoff() -> None:
    df = _labels()
    train, test = temporal_split(df)  # default 2022-01
    assert (train["T_dec"] <= "2021-12").all()
    assert (test["T_dec"] >= "2022-01").all()


def test_temporal_split_does_not_mutate_input() -> None:
    df = _labels()
    before = df.copy()
    temporal_split(df)
    pd.testing.assert_frame_equal(df, before)


def test_rolling_origins_windows_are_correct() -> None:
    df = _labels()
    origins = ["2020-06", "2021-06", "2022-06", "2023-06"]
    results = rolling_origins(df, origins)

    assert [o for o, _, _ in results] == origins

    by_origin = {o: (tr, te) for o, tr, te in results}

    # origin 2020-06: train T_dec < 2020-06; test T_dec in [2020-06, 2021-05].
    tr, te = by_origin["2020-06"]
    assert set(tr["theme_id"]) == {"t01", "t02", "t03"}
    assert set(te["theme_id"]) == {"t04", "t05"}  # 2020-06 and 2021-03; t06 (2021-06) is out of window

    # origin 2021-06: train T_dec < 2021-06; test T_dec in [2021-06, 2022-05].
    tr, te = by_origin["2021-06"]
    assert set(tr["theme_id"]) == {"t01", "t02", "t03", "t04", "t05"}
    assert set(te["theme_id"]) == {"t06", "t07", "t08"}  # 2021-06, 2021-12, 2022-01

    # origin 2022-06: test in [2022-06, 2023-05].
    tr, te = by_origin["2022-06"]
    assert set(te["theme_id"]) == {"t09", "t10"}  # 2022-06, 2022-12

    # origin 2023-06: test in [2023-06, 2024-05].
    tr, te = by_origin["2023-06"]
    assert set(te["theme_id"]) == {"t11", "t12"}  # 2023-06, 2024-02


def test_rolling_origins_window_is_exactly_12_months() -> None:
    # A theme exactly 12 months after the origin must fall OUTSIDE the [origin, origin+12mo) window.
    df = pd.DataFrame(
        [
            ("a", "positive", "2019-01", "2020-06", "p1"),  # = origin -> in
            ("b", "positive", "2019-01", "2021-05", "p2"),  # origin+11 -> in
            ("c", "positive", "2019-01", "2021-06", "p3"),  # origin+12 -> OUT
        ],
        columns=["theme_id", "label", "T0", "T_dec", "parent_id"],
    )
    (_, _, test) = rolling_origins(df, ["2020-06"])[0]
    assert set(test["theme_id"]) == {"a", "b"}


def test_grouped_folds_never_shares_a_parent() -> None:
    df = _labels()
    folds = grouped_folds(df, k=5, seed=0)

    assert len(folds) == 5
    for train, test in folds:
        # The core anti-leakage guarantee.
        assert set(train["parent_id"]) & set(test["parent_id"]) == set()


def test_grouped_folds_each_parent_in_exactly_one_test_fold() -> None:
    df = _labels()
    folds = grouped_folds(df, k=5, seed=0)

    all_parents = set(df["parent_id"])
    seen_in_test: list[str] = []
    for _, test in folds:
        seen_in_test.extend(test["parent_id"].unique())
    # Each parent appears in exactly one fold's test set -> clean OOF pooling.
    assert sorted(seen_in_test) == sorted(all_parents)
    assert len(seen_in_test) == len(set(seen_in_test))


def test_grouped_folds_train_test_disjoint_and_exhaustive_per_fold() -> None:
    df = _labels()
    for train, test in grouped_folds(df, k=5, seed=0):
        assert set(train["theme_id"]) & set(test["theme_id"]) == set()
        assert len(train) + len(test) == len(df)


def test_grouped_folds_guarantee_holds_with_many_themes_per_parent() -> None:
    # A parent with several fine themes (the realistic case) must still land wholly on one side per fold.
    rows = []
    for p in range(8):
        for t in range(3):  # 3 themes per parent
            rows.append((f"t{p}_{t}", "positive", "2020-01", "2021-01", f"p{p}"))
    df = pd.DataFrame(rows, columns=["theme_id", "label", "T0", "T_dec", "parent_id"])

    for train, test in grouped_folds(df, k=5, seed=0):
        # No parent family straddles the fold boundary, even with multiple themes each.
        assert set(train["parent_id"]) & set(test["parent_id"]) == set()
    # Every theme of a test parent moves together.
    folds = grouped_folds(df, k=5, seed=0)
    for _, test in folds:
        for parent in test["parent_id"].unique():
            assert (df[df["parent_id"] == parent]["theme_id"].isin(test["theme_id"])).all()


def test_assign_parents_by_hash_is_deterministic_and_in_range() -> None:
    df = _labels()
    a = assign_parents_by_hash(df, k=5, seed=0)
    b = assign_parents_by_hash(df, k=5, seed=0)
    pd.testing.assert_series_equal(a, b)
    assert set(a.index) == set(df["parent_id"])
    assert a.between(0, 4).all()
    # A different seed can produce a different partition (not guaranteed different, but valid range).
    c = assign_parents_by_hash(df, k=5, seed=1)
    assert c.between(0, 4).all()
