import numpy as np
import pandas as pd

from experiments.build_radar_data import (
    DIM_META,
    HORIZONS,
    TERM_ARTIFACTS,
    TIER_CUTS,
    calibration_bins,
    confidence_from_bins,
    dedupe_cohort,
    dims_payload,
    durability_pct,
    pick_horizon,
    series_payload,
    tier_from_pct,
)
from marketbrain.eval.breakout_signature import SIGNATURE_FEATURES


def test_dim_meta_covers_all_model_features():
    from marketbrain.eval.author_features import AUTHOR_FEATURES

    assert set(DIM_META) == set(SIGNATURE_FEATURES) | set(AUTHOR_FEATURES)


def test_tier_cuts_monotone_and_match_v1_distribution():
    assert [t for _, t in TIER_CUTS] == ["L5", "L4", "L3", "L2", "L1"]
    pcts = np.linspace(0, 99.9, 1000)
    tiers = [tier_from_pct(p) for p in pcts]
    # v1 distribution ~10/20/25/20/25% (cuts at 90/70/45/25)
    assert abs(tiers.count("L5") - 100) <= 20
    assert abs(tiers.count("L4") - 200) <= 20
    assert abs(tiers.count("L3") - 250) <= 20
    assert abs(tiers.count("L2") - 200) <= 20
    assert abs(tiers.count("L1") - 250) <= 20
    assert tier_from_pct(95) == "L5" and tier_from_pct(5) == "L1"


def test_durability_pct_is_rank_among_reference():
    ref = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    assert durability_pct(0.95, ref) == 100.0
    assert durability_pct(0.05, ref) == 0.0
    assert 40.0 <= durability_pct(0.45, ref) <= 50.0


def test_series_payload_trims_leading_zeros_and_is_ints():
    s = pd.Series(
        [0, 0, 0, 2, 5, 9],
        index=["2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06"],
    )
    out = series_payload(s)
    assert out == {"start": "2024-04", "monthly": [2, 5, 9]}
    assert all(isinstance(v, int) for v in out["monthly"])


def test_series_payload_none_for_empty():
    s = pd.Series([0, 0], index=["2024-01", "2024-02"])
    assert series_payload(s) is None


def test_dedupe_cohort_drops_token_subset_with_shared_records():
    recs = {
        "cot": set(range(100)),
        "chain-of-thought cot reasoning": set(range(40)),  # subset of cot's records
        "graph neural networks": set(range(200, 300)),
    }
    kept = dedupe_cohort(list(recs), recs)
    assert "cot" in kept  # higher df wins
    assert "chain-of-thought cot reasoning" not in kept
    assert "graph neural networks" in kept


def test_dedupe_cohort_keeps_distinct_themes_with_low_overlap():
    recs = {
        "state space models": set(range(100)),
        "mamba": set(range(90, 200)),  # ~5% Jaccard — distinct
    }
    kept = dedupe_cohort(list(recs), recs)
    assert set(kept) == {"state space models", "mamba"}


def test_dedupe_cohort_merges_singular_plural():
    recs = {
        "vlms": set(range(120)),
        "vlm": set(range(200, 280)),  # different papers, same concept
    }
    kept = dedupe_cohort(list(recs), recs)
    assert kept == ["vlms"]  # higher df spelling wins


def test_term_artifacts_are_lowercase_terms():
    assert all(t == t.lower() for t in TERM_ARTIFACTS)
    assert "bench" in TERM_ARTIFACTS and "b-instruct" in TERM_ARTIFACTS


def test_pick_horizon_largest_leq_age():
    assert HORIZONS == [1, 2, 3, 4, 6, 9]
    assert pick_horizon(0) == 1  # too young: earliest model, better than nothing
    assert pick_horizon(1) == 1
    assert pick_horizon(2) == 2
    assert pick_horizon(5) == 4
    assert pick_horizon(8) == 6
    assert pick_horizon(9) == 9
    assert pick_horizon(40) == 9


def test_calibration_bins_monotone_ns_and_rates():
    rng = np.random.RandomState(0)
    probs = rng.rand(200)
    y = (probs + rng.normal(0, 0.2, 200) > 0.6).astype(int)  # higher prob -> more durable
    bins = calibration_bins(probs, y, k=5)
    assert len(bins) == 5
    assert sum(b["n"] for b in bins) == 200
    assert [b["lo"] for b in bins] == sorted(b["lo"] for b in bins)
    assert bins[-1]["rate"] > bins[0]["rate"]  # calibration is informative


def test_confidence_from_bins_lookup():
    bins = [
        {"lo": 0.0, "rate": 0.1, "n": 40},
        {"lo": 0.3, "rate": 0.4, "n": 40},
        {"lo": 0.7, "rate": 0.8, "n": 40},
    ]
    assert confidence_from_bins(0.05, bins)["rate"] == 0.1
    assert confidence_from_bins(0.5, bins)["rate"] == 0.4
    assert confidence_from_bins(0.95, bins)["rate"] == 0.8


def test_dims_payload_signed_contributions():
    all_feats = list(DIM_META)
    feats_std = {f: 1.0 for f in all_feats}
    coefs = {f: (0.5 if i % 2 == 0 else -0.5) for i, f in enumerate(all_feats)}
    scores = {f: 80.0 for f in all_feats}
    dims = dims_payload(feats_std=feats_std, coefs=coefs, pct_scores=scores)
    assert len(dims) == len(all_feats)
    assert {d["label"] for d in dims} == {label for label, _ in DIM_META.values()}
    for d in dims:
        assert set(d) == {"label", "score", "contrib", "desc"}
        assert abs(d["contrib"]) == 0.5
    signs = {d["contrib"] > 0 for d in dims}
    assert signs == {True, False}
