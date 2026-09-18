"""Tests for Carley/Garner emergence scoring (synthetic term usage, no network)."""

from __future__ import annotations

import pandas as pd

from marketbrain.features.emergence import (
    EmergenceConfig,
    calibrate_cutoff,
    emergence_trajectory,
    emergent_terms,
    escore,
    gate_escores,
    market_emergence_score,
    month_range_ending,
)

CFG = EmergenceConfig()  # 9 base + 21 active = 30-month window


def _events() -> pd.DataFrame:
    window = month_range_ending("2025-12", CFG.window_months)
    active = window[CFG.base_months:]
    rows = []
    rid = 0
    # established term: present every month incl. base (NOT novel) and flat (low EScore)
    for mo in window:
        for k in range(4):
            rid += 1
            rows.append((f"r{rid}", mo, "transformer", frozenset({f"auth{k}"})))
    # emerging term: ~0 in base, accelerating across the active window, many distinct authors
    for i, mo in enumerate(active):
        for k in range(i + 1):  # 1, 2, 3, ... records per active month
            rid += 1
            rows.append((f"r{rid}", mo, "agentic", frozenset({f"writer{(i + k) % 7}"})))
    return pd.DataFrame(rows, columns=["record_id", "month", "term", "authors"])


def test_escore_flat_is_near_zero_accelerating_is_high():
    assert abs(escore([3.0] * 21)) < 0.2
    assert escore([1, 1, 2, 2, 3, 4, 6, 9, 13, 18] + list(range(20, 32))) > 1.77


def test_emerging_term_passes_established_term_fails_novelty():
    ev = _events()
    terms = emergent_terms(ev, "2025-12", CFG)
    assert "agentic" in terms and terms["agentic"] >= CFG.escore_cutoff
    assert "transformer" not in terms                      # too common in the base period
    assert market_emergence_score(ev, "2025-12", CFG) > 0


def test_trajectory_batch_matches_per_month_spec():
    ev = _events()
    months = ["2025-10", "2025-11", "2025-12"]
    traj = emergence_trajectory(ev, months, CFG)
    for m in months:
        assert abs(traj[m] - market_emergence_score(ev, m, CFG)) < 1e-6


def test_normalize_divides_out_corpus_volume():
    ev = _events()
    raw = emergence_trajectory(ev, ["2025-12"], CFG)["2025-12"]
    norm = emergence_trajectory(ev, ["2025-12"], CFG, normalize=True)["2025-12"]
    window = month_range_ending("2025-12", CFG.window_months)
    win_recs = ev[ev["month"].isin(window)]["record_id"].nunique()
    assert raw > 0 and abs(norm - raw / (win_recs ** 0.5)) < 1e-9 and norm < raw


def test_calibrate_cutoff_is_a_percentile_of_gate_escores():
    ev = _events()
    gs = gate_escores(ev, "2025-12", CFG)          # gate-passers' EScores (agentic; transformer fails novelty)
    assert gs and "agentic" in emergent_terms(ev, "2025-12", EmergenceConfig(escore_cutoff=float("-inf")))
    cut = calibrate_cutoff(ev, "2025-12", percentile=50, cfg=CFG)
    assert min(gs) <= cut <= max(gs)


def test_window_is_30_months_and_ends_at_as_of():
    w = month_range_ending("2024-06", CFG.window_months)
    assert len(w) == 30 and w[-1] == "2024-06" and w[0] == "2022-01"
