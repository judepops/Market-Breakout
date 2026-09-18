"""Tests for the paired-delta bootstrap CI that qualifies the signature-vs-momentum comparison."""
import numpy as np

from marketbrain.eval.metrics import paired_delta_ci


def test_delta_ci_excludes_zero_when_a_clearly_better():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0] * 6)
    a = y + 0.01                                   # near-perfect scorer (AUPRC ~ 1)
    b = np.random.RandomState(0).rand(len(y))      # random scorer (AUPRC ~ prevalence)
    d, lo, hi, p = paired_delta_ci(a, b, y, B=500, seed=0)
    assert d > 0 and lo > 0 and p < 0.05           # A beats B -> CI excludes 0


def test_delta_ci_spans_zero_when_equivalent():
    rng = np.random.RandomState(1)
    y = (rng.rand(80) > 0.5).astype(int)
    a, b = rng.rand(80), rng.rand(80)              # two equivalent random scorers
    d, lo, hi, p = paired_delta_ci(a, b, y, B=500, seed=0)
    assert lo < 0 < hi                             # CI spans 0 -> not separated
