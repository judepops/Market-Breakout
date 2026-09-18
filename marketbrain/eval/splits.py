"""Generalisation SPLITS for the Phase-0 evaluation (pre-registration section 5).

Two orthogonal held-out axes, both frozen by ``Plans/2026-06-30_phase0-preregistration.md``:

- Split 1 -- held-out FUTURE periods (temporal generalisation): :func:`temporal_split` and the
  :func:`rolling_origins` variant. Train on early decision points, score the post-cutoff regime once.
- Split 2 -- held-out unseen IDEA-TYPES (semantic generalisation): :func:`grouped_folds`, a GroupKFold
  on ``parent_id`` so no idea-family appears in both train and test of a fold. The real partition is
  embedding clustering of theme names+terms; :func:`assign_parents_by_hash` is the reproducible stub.

Pure functions over the frozen LABELS contract: a ``labels_df`` with columns
``theme_id``, ``label`` ("positive"/"hyped_died"/"never"/"censored"), ``T0``, ``T_dec`` (both "YYYY-MM"),
and ``parent_id`` (the coarse semantic parent / idea-family). String month compare is exact because
"YYYY-MM" sorts lexically. No I/O, no network, no mutation of the input frame.
"""

from __future__ import annotations

import hashlib

import pandas as pd

__all__ = [
    "temporal_split",
    "rolling_origins",
    "grouped_folds",
    "assign_parents_by_hash",
]


def _parse_month(month: str) -> tuple[int, int]:
    year_s, mon_s = month.split("-")
    return int(year_s), int(mon_s)


def _shift_month(month: str, delta: int) -> str:
    """Return ``month`` ("YYYY-MM") shifted by ``delta`` calendar months (may be negative)."""
    year, mon = _parse_month(month)
    idx = year * 12 + (mon - 1) + delta
    return f"{idx // 12:04d}-{idx % 12 + 1:02d}"


def temporal_split(labels_df: pd.DataFrame, cutoff: str = "2022-01") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split 1 (section 5.1): held-out future periods by ``T_dec``.

    Train = themes decided strictly before ``cutoff`` (``T_dec <= cutoff - 1 month``);
    test = themes decided at or after ``cutoff`` (``T_dec >= cutoff``, the post-ChatGPT regime).
    Lexical "YYYY-MM" comparison is exact. Returns ``(train_df, test_df)`` as copies preserving row order.
    """
    train_max = _shift_month(cutoff, -1)
    train = labels_df[labels_df["T_dec"] <= train_max].copy()
    test = labels_df[labels_df["T_dec"] >= cutoff].copy()
    return train, test


def rolling_origins(
    labels_df: pd.DataFrame, origins: list[str]
) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Rolling-origin variant of Split 1 (section 5.1).

    For each ``origin`` month: train = themes with ``T_dec < origin``; test = themes with
    ``T_dec`` in the 12-month window ``[origin, origin + 12 months)`` (i.e. ``origin <= T_dec <= origin+11``).
    Returns a list of ``(origin, train_df, test_df)`` in the order ``origins`` was given.
    """
    out: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    for origin in origins:
        window_end = _shift_month(origin, 11)  # inclusive last month of the 12-month test window
        train = labels_df[labels_df["T_dec"] < origin].copy()
        test = labels_df[
            (labels_df["T_dec"] >= origin) & (labels_df["T_dec"] <= window_end)
        ].copy()
        out.append((origin, train, test))
    return out


def grouped_folds(
    labels_df: pd.DataFrame, k: int = 5, seed: int = 0
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Split 2 (section 5.2): GroupKFold by ``parent_id`` so no parent crosses train/test.

    The decisive anti-leakage guarantee: ``set(train.parent_id) & set(test.parent_id) == set()`` for
    every fold (asserted below, raising ``AssertionError`` if ever violated). Parents are assigned to
    folds deterministically by hashed name (:func:`assign_parents_by_hash`, controlled by ``seed``);
    fold ``f`` uses the parents hashed to bucket ``f`` as TEST and all other parents as TRAIN. Pooling
    the out-of-fold scores over all ``k`` folds yields one grouped-OOF AUPRC (section 5.2), which is why
    every parent is in exactly one test fold. Returns ``k`` ``(train_df, test_df)`` tuples.
    """
    parent_fold = assign_parents_by_hash(labels_df, k=k, seed=seed)
    fold_of_row = labels_df["parent_id"].map(parent_fold)

    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for f in range(k):
        is_test = fold_of_row == f
        test = labels_df[is_test].copy()
        train = labels_df[~is_test].copy()
        # Core leakage guarantee: a parent family is never split across train and test of a fold.
        assert set(train["parent_id"]) & set(test["parent_id"]) == set(), (
            f"parent leak across fold {f}: a parent_id appears in both train and test"
        )
        folds.append((train, test))
    return folds


def assign_parents_by_hash(labels_df: pd.DataFrame, k: int = 5, seed: int = 0) -> pd.Series:
    """Deterministic fallback parent -> fold map via ``hash(parent_name) % k`` (section 5.2).

    The REAL idea-type partition is embedding clustering of theme names + centroid terms into ~20-40
    semantic parents; this is the reproducible STUB used when that embedding partition is unavailable.
    Hashing is done with a stable digest (md5 of ``f"{seed}:{parent_id}"``) rather than Python's
    salted ``hash`` so the assignment is reproducible across processes. Returns a Series indexed by the
    unique ``parent_id`` values, with integer fold labels in ``[0, k)``.
    """
    parents = pd.Index(labels_df["parent_id"].unique())
    folds = [
        int(hashlib.md5(f"{seed}:{p}".encode()).hexdigest(), 16) % k for p in parents
    ]
    return pd.Series(folds, index=parents, name="fold")
