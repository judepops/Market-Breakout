"""Imbalance-appropriate metrics + permutation nulls for the Phase-0 evaluation.

Frozen by Plans/2026-06-30_phase0-preregistration.md sections 6 and 9. PURE numpy/sklearn: every
function takes arrays and returns values, with NO I/O and NO side effects. The discriminative eval
runs on the two label-symmetric classes only -- durable-breakout POSITIVES ("positive" -> 1) and the
data-derived hyped-then-died NEGATIVES ("hyped_died" -> 0); the "never"/"censored" themes are not part
of this contrast (see preregistration sections 2-3).

Metrics (section 6):
  * ``auprc``           -- area under precision-recall (the PRIMARY metric; positives are rare).
  * ``roc_auc``         -- area under ROC, reported alongside.
  * ``precision_at_k``  -- of the top-k highest scores, the fraction that are positive.
  * ``bootstrap_ci``    -- 2000-resample percentile CI, resampling THEMES with replacement.

Permutation nulls (section 9):
  * ``shuffle_null``            -- permute the label vector (optionally at the parent-GROUP level so the
                                  null preserves family clustering).
  * ``block_permutation_null``  -- permute labels WITHIN time blocks, preserving era structure.

All randomness flows through ``numpy.random.RandomState(seed)`` for reproducibility (matching
``s_curve_lag.py`` / ``features/leadlag.py::permutation_pvalue``).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

# Frozen label->binary mapping for the discriminative eval (preregistration sections 2-3).
LABEL_TO_BINARY = {"positive": 1, "hyped_died": 0}


def _as_binary(labels: np.ndarray) -> np.ndarray:
    """Coerce a label vector to {0,1} int. Accepts already-numeric {0,1} or the frozen string labels."""
    arr = np.asarray(labels)
    if arr.dtype.kind in ("U", "S", "O"):
        try:
            return np.array([LABEL_TO_BINARY[str(v)] for v in arr], dtype=int)
        except KeyError as exc:  # pragma: no cover - defensive: only the two eval classes are valid here
            raise ValueError(
                f"label {exc.args[0]!r} is not part of the discriminative eval; "
                "expected one of {'positive','hyped_died'} or numeric {0,1}"
            ) from None
    out = arr.astype(int)
    if not np.isin(out, (0, 1)).all():
        raise ValueError("numeric labels must be in {0, 1}")
    return out


def auprc(scores, labels) -> float:
    """Area under the precision-recall curve (sklearn ``average_precision_score``).

    PRIMARY metric: positives are rare, so AUPRC is the imbalance-appropriate summary. A random scorer
    scores ~= positive prevalence; a perfect scorer scores 1.0.
    """
    y = _as_binary(labels)
    return float(average_precision_score(y, np.asarray(scores, dtype=float)))


def roc_auc(scores, labels) -> float:
    """Area under the ROC curve."""
    y = _as_binary(labels)
    return float(roc_auc_score(y, np.asarray(scores, dtype=float)))


def precision_at_k(scores, labels, k: int) -> float:
    """Of the ``k`` highest-scoring themes, the fraction that are positive.

    Decision-relevant and robust to small ``n`` ("of the top-10 you flagged, 6 broke out"). Ties are
    broken by a stable descending sort; ``k`` is clamped to the number of available themes.
    """
    y = _as_binary(labels)
    s = np.asarray(scores, dtype=float)
    if k <= 0:
        raise ValueError("k must be a positive integer")
    k = min(int(k), s.shape[0])
    if k == 0:
        return float("nan")
    top = np.argsort(-s, kind="stable")[:k]
    return float(y[top].mean())


def bootstrap_ci(
    scores,
    labels,
    stat: Callable = auprc,
    B: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for ``stat``, resampling THEMES with replacement.

    Returns ``(point, lo, hi)`` where ``point`` is ``stat`` on the full sample and ``(lo, hi)`` are the
    ``[alpha/2, 1-alpha/2]`` percentiles over ``B`` resamples. Degenerate resamples (all one class) are
    skipped, since AUPRC/ROC are undefined there; the CI is taken over the surviving resamples.
    """
    s = np.asarray(scores, dtype=float)
    y = _as_binary(labels)
    n = s.shape[0]
    point = float(stat(s, y))
    rng = np.random.RandomState(seed)
    boot: list[float] = []
    for _ in range(B):
        idx = rng.randint(0, n, size=n)
        yb = y[idx]
        if yb.min() == yb.max():  # one-class resample -> stat undefined, skip
            continue
        boot.append(float(stat(s[idx], yb)))
    if not boot:  # pragma: no cover - only when essentially every resample is degenerate
        return point, float("nan"), float("nan")
    lo = float(np.percentile(boot, 100 * (alpha / 2.0)))
    hi = float(np.percentile(boot, 100 * (1.0 - alpha / 2.0)))
    return point, lo, hi


def paired_delta_ci(
    scores_a,
    scores_b,
    labels,
    stat: Callable = auprc,
    B: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float, float]:
    """Percentile bootstrap CI for the PAIRED delta ``stat(a) - stat(b)`` on the same themes.

    Resamples themes with replacement and recomputes BOTH statistics on each resample, so the (positive)
    correlation between the two scorers is accounted for -- the appropriate, more powerful paired test for
    "does A beat B". Returns ``(delta_point, lo, hi, p_one_sided)`` where ``p_one_sided`` is the fraction of
    resamples with delta <= 0 (the bootstrap probability that B is at least as good as A). The comparative
    claim "A beats B" is supported when the CI excludes 0 (equivalently p_one_sided < alpha).
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    y = _as_binary(labels)
    n = y.shape[0]
    point = float(stat(a, y) - stat(b, y))
    rng = np.random.RandomState(seed)
    deltas: list[float] = []
    for _ in range(B):
        idx = rng.randint(0, n, size=n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        deltas.append(float(stat(a[idx], yb) - stat(b[idx], yb)))
    if not deltas:  # pragma: no cover
        return point, float("nan"), float("nan"), float("nan")
    lo = float(np.percentile(deltas, 100 * (alpha / 2.0)))
    hi = float(np.percentile(deltas, 100 * (1.0 - alpha / 2.0)))
    p_one = float(np.mean([d <= 0 for d in deltas]))
    return point, lo, hi, p_one


def _grouped_permutation(y: np.ndarray, groups: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Permute labels at the GROUP level: shuffle one label per group, broadcast back to members.

    Preserves family clustering (each parent keeps a single label, as the real labels do under the
    grouped-CV structure) so the null is not artificially easy. Requires labels to be constant within a
    group (one label per parent), which the data contract guarantees.
    """
    uniq = np.unique(groups)
    # one representative label per group, in ``uniq`` order
    group_labels = np.array([y[groups == g][0] for g in uniq])
    permuted = rng.permutation(group_labels)
    mapping = {g: permuted[i] for i, g in enumerate(uniq)}
    return np.array([mapping[g] for g in groups])


def shuffle_null(
    scores,
    labels,
    stat: Callable = auprc,
    groups=None,
    B: int = 5000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Permutation null: hold scores fixed, permute the LABEL vector ``B`` times.

    Returns ``(observed, null_mean, p_value)`` with ``p = (#{permuted >= observed} + 1)/(B + 1)`` (the
    standard +1 correction so a real signal never reports an impossible p=0). If ``groups`` (parent ids)
    is given, labels are permuted AT THE GROUP LEVEL so the null preserves family clustering
    (preregistration section 9, Split-2 grouped permutation). Permuted resamples that collapse to a
    single class are skipped (their stat is undefined).
    """
    s = np.asarray(scores, dtype=float)
    y = _as_binary(labels)
    observed = float(stat(s, y))
    rng = np.random.RandomState(seed)
    grp = None if groups is None else np.asarray(groups)
    null_stats: list[float] = []
    for _ in range(B):
        yp = _grouped_permutation(y, grp, rng) if grp is not None else rng.permutation(y)
        if yp.min() == yp.max():  # single-class permutation -> undefined, skip
            continue
        null_stats.append(float(stat(s, yp)))
    null_arr = np.asarray(null_stats, dtype=float)
    null_mean = float(null_arr.mean()) if null_arr.size else float("nan")
    n_eff = null_arr.size
    ge = int(np.sum(null_arr >= observed))
    p_value = (ge + 1) / (n_eff + 1) if n_eff else float("nan")
    return observed, null_mean, float(p_value)


def block_permutation_null(
    scores,
    labels,
    blocks,
    stat: Callable = auprc,
    B: int = 5000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Block-permutation null: permute labels WITHIN time blocks, preserving era structure.

    ``blocks`` is an array of block ids (e.g. 6-month era bins). Labels are shuffled only among themes
    sharing a block, so the null cannot be beaten merely by "later periods are winners"
    (preregistration section 9, temporal null for Split-1). Returns ``(observed, null_mean, p_value)``
    with the same ``(ge+1)/(n+1)`` correction as ``shuffle_null``. Permutations that collapse to a
    single class are skipped.
    """
    s = np.asarray(scores, dtype=float)
    y = _as_binary(labels)
    blk = np.asarray(blocks)
    observed = float(stat(s, y))
    rng = np.random.RandomState(seed)
    # precompute the member indices of each block once
    block_index = {b: np.where(blk == b)[0] for b in np.unique(blk)}
    null_stats: list[float] = []
    for _ in range(B):
        yp = y.copy()
        for idx in block_index.values():
            if idx.size > 1:
                yp[idx] = y[idx][rng.permutation(idx.size)]
        if yp.min() == yp.max():  # single-class permutation -> undefined, skip
            continue
        null_stats.append(float(stat(s, yp)))
    null_arr = np.asarray(null_stats, dtype=float)
    null_mean = float(null_arr.mean()) if null_arr.size else float("nan")
    n_eff = null_arr.size
    ge = int(np.sum(null_arr >= observed))
    p_value = (ge + 1) / (n_eff + 1) if n_eff else float("nan")
    return observed, null_mean, float(p_value)
