"""Tests for marketbrain.eval.metrics (Phase-0 imbalance metrics + permutation nulls).

In-code numpy fixtures only -- NO network, NO file I/O. Checks the frozen contract of section 6/9:
a perfect scorer -> AUPRC 1.0; a random scorer's shuffle null is not tiny; precision_at_k on a hand
example; bootstrap_ci brackets the point estimate; grouped shuffle_null is small only when the signal
is real.
"""

from __future__ import annotations

import numpy as np

from marketbrain.eval.metrics import (
    auprc,
    block_permutation_null,
    bootstrap_ci,
    precision_at_k,
    roc_auc,
    shuffle_null,
)


def test_auprc_perfect_scorer_is_one():
    # scores rank all positives strictly above all negatives -> perfect PR curve.
    labels = np.array(["hyped_died", "hyped_died", "positive", "positive"])
    scores = np.array([0.1, 0.2, 0.9, 0.8])
    assert auprc(scores, labels) == 1.0
    assert roc_auc(scores, labels) == 1.0


def test_auprc_accepts_numeric_labels():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.9, 0.8])
    assert auprc(scores, labels) == 1.0


def test_auprc_random_scorer_near_prevalence():
    # AUPRC of an uninformative scorer ~ positive prevalence.
    rng = np.random.RandomState(7)
    n = 400
    labels = (rng.rand(n) < 0.25).astype(int)
    scores = rng.rand(n)  # independent of labels
    val = auprc(scores, labels)
    prevalence = labels.mean()
    assert abs(val - prevalence) < 0.08


def test_precision_at_k_hand_example():
    # top-3 by score are indices 4,3,2 with labels 1,0,1 -> 2/3 positive.
    scores = np.array([0.1, 0.2, 0.7, 0.8, 0.9])
    labels = np.array([0, 0, 1, 0, 1])
    assert precision_at_k(scores, labels, 3) == 2.0 / 3.0
    # top-2 are indices 4,3 -> labels 1,0 -> 1/2.
    assert precision_at_k(scores, labels, 2) == 0.5
    # k clamps to n.
    assert precision_at_k(scores, labels, 99) == labels.mean()


def test_precision_at_k_perfect_top_k():
    scores = np.array([0.9, 0.8, 0.1, 0.2])
    labels = np.array([1, 1, 0, 0])
    assert precision_at_k(scores, labels, 2) == 1.0


def test_bootstrap_ci_brackets_point():
    rng = np.random.RandomState(0)
    n = 120
    labels = (rng.rand(n) < 0.3).astype(int)
    # informative-but-noisy scorer so the bootstrap distribution is non-degenerate.
    scores = labels + rng.normal(0, 1.0, size=n)
    point, lo, hi = bootstrap_ci(scores, labels, B=500, seed=1)
    assert lo <= point <= hi
    assert lo < hi  # a real interval, not a collapsed point
    assert 0.0 <= lo and hi <= 1.0


def test_bootstrap_ci_is_reproducible():
    rng = np.random.RandomState(3)
    labels = (rng.rand(60) < 0.4).astype(int)
    scores = labels + rng.normal(0, 0.8, size=60)
    a = bootstrap_ci(scores, labels, B=300, seed=42)
    b = bootstrap_ci(scores, labels, B=300, seed=42)
    assert a == b


def test_shuffle_null_random_scorer_not_tiny():
    # A scorer independent of labels should NOT beat the shuffle null -> p well above 0.01.
    rng = np.random.RandomState(11)
    n = 150
    labels = (rng.rand(n) < 0.25).astype(int)
    scores = rng.rand(n)
    observed, null_mean, p = shuffle_null(scores, labels, B=2000, seed=0)
    assert p > 0.01
    # observed should sit near the null mean (no real signal).
    assert abs(observed - null_mean) < 0.12


def test_shuffle_null_perfect_scorer_is_significant():
    rng = np.random.RandomState(5)
    n = 100
    labels = (rng.rand(n) < 0.3).astype(int)
    scores = labels + rng.normal(0, 0.05, size=n)  # near-perfect separation
    observed, null_mean, p = shuffle_null(scores, labels, B=2000, seed=0)
    assert observed > null_mean
    assert p < 0.05


def test_grouped_shuffle_null_small_only_when_signal_real():
    # 12 parents, 3 children each. Real case: score correlates with the per-parent label.
    n_parents = 12
    children = 3
    rng = np.random.RandomState(2)
    parent_label = (rng.rand(n_parents) < 0.4).astype(int)
    groups = np.repeat(np.arange(n_parents), children)
    labels = np.repeat(parent_label, children)
    # real signal: score = parent label + small noise (group-correlated).
    real_scores = labels + rng.normal(0, 0.05, size=n_parents * children)
    obs_r, _, p_real = shuffle_null(real_scores, labels, groups=groups, B=3000, seed=0)
    assert p_real < 0.05

    # null signal: score independent of the parent label -> grouped p should not be tiny.
    fake_scores = rng.rand(n_parents * children)
    _, _, p_fake = shuffle_null(fake_scores, labels, groups=groups, B=3000, seed=0)
    assert p_fake > 0.05


def test_grouped_null_preserves_clustering_vs_flat():
    # With a group-correlated signal, the GROUPED null is harder (larger p) than the naive flat null,
    # because it preserves family clustering instead of breaking it.
    n_parents = 10
    children = 4
    rng = np.random.RandomState(9)
    parent_label = (rng.rand(n_parents) < 0.5).astype(int)
    groups = np.repeat(np.arange(n_parents), children)
    labels = np.repeat(parent_label, children)
    scores = labels + rng.normal(0, 0.3, size=n_parents * children)
    _, _, p_flat = shuffle_null(scores, labels, B=3000, seed=0)
    _, _, p_grouped = shuffle_null(scores, labels, groups=groups, B=3000, seed=0)
    assert p_grouped >= p_flat


def test_block_permutation_null_real_signal_significant():
    # Two 6-month-era blocks; within each block the score separates the classes.
    rng = np.random.RandomState(4)
    blocks = np.array([0] * 30 + [1] * 30)
    labels = np.zeros(60, dtype=int)
    # positives scattered across BOTH blocks so the signal is not "later = winner".
    labels[5:15] = 1
    labels[35:45] = 1
    scores = labels + rng.normal(0, 0.1, size=60)
    observed, null_mean, p = block_permutation_null(scores, labels, blocks, B=2000, seed=0)
    assert observed > null_mean
    assert p < 0.05


def test_block_permutation_null_era_confound_not_significant():
    # All positives live in the later block; a scorer that is just "later" should NOT beat the
    # within-block null, because the block permutation holds the era composition fixed.
    rng = np.random.RandomState(6)
    blocks = np.array([0] * 40 + [1] * 40)
    labels = np.array([0] * 40 + [1] * 8 + [0] * 32)  # all positives in block 1
    era_score = blocks.astype(float) + rng.normal(0, 0.01, size=80)  # "later = higher"
    _, _, p = block_permutation_null(era_score, labels, blocks, B=2000, seed=0)
    assert p > 0.05


def test_block_permutation_reproducible():
    rng = np.random.RandomState(8)
    blocks = np.array([0] * 20 + [1] * 20)
    labels = (rng.rand(40) < 0.3).astype(int)
    scores = rng.rand(40)
    a = block_permutation_null(scores, labels, blocks, B=500, seed=0)
    b = block_permutation_null(scores, labels, blocks, B=500, seed=0)
    assert a == b
