"""Tests for the breakout signature -- focus on the two correctness guarantees that matter most:
leakage-safety (future records don't change the as-of features) and category-identity invariance
(the same trajectory in different categories yields the SAME features -- the time-biased-category trap)."""
import pandas as pd

from marketbrain.eval.breakout_signature import SIGNATURE_FEATURES, theme_features
from marketbrain.timeutil import month_range

GRID = month_range("2019-01", "2022-12")
TOTAL = pd.Series(1000.0, index=GRID)


def _recs(specs):
    """specs: list of (record_id, month, author, categories, comment)."""
    return pd.DataFrame(
        [{"record_id": r, "month": m, "authors": frozenset({a}), "categories": list(c),
          "comment": cm, "journal_ref": ""} for r, m, a, c, cm in specs],
        columns=["record_id", "month", "authors", "categories", "comment", "journal_ref"])


def _ramp(prefix, cats, comment=""):
    """A growing theme: a few records/month ramping up over 2020."""
    specs = []
    for i, m in enumerate(month_range("2020-01", "2020-12")):
        for k in range((i // 3) + 1):
            specs.append((f"{prefix}{i}-{k}", m, f"{prefix}auth{i}-{k}", cats, comment))
    return specs


def test_as_of_leakage_firewall():
    """Records AFTER as_of must not change the features (the firewall is enforced inside the function)."""
    past = _recs(_ramp("p", ["cs.AI"]))
    future = _recs([(f"f{i}", m, f"fa{i}", ["cs.AI"], "") for i, m in enumerate(month_range("2021-01", "2021-12"))])
    f_clean = theme_features(past, "2020-12", TOTAL)
    f_contaminated = theme_features(pd.concat([past, future], ignore_index=True), "2020-12", TOTAL)
    assert f_clean == f_contaminated


def test_category_identity_invariance():
    """Same trajectory, DIFFERENT category identities but same spread -> identical feature vector."""
    a = theme_features(_recs(_ramp("x", ["cs.AI", "cs.LG"])), "2020-12", TOTAL)
    b = theme_features(_recs(_ramp("x", ["cs.CV", "cs.RO"])), "2020-12", TOTAL)   # different cats, same count/spread
    assert a == b
    assert "primary_category" not in SIGNATURE_FEATURES and "category" not in SIGNATURE_FEATURES


def test_features_are_sane_on_a_growing_theme():
    f = theme_features(_recs(_ramp("g", ["cs.AI", "cs.LG"], comment="Accepted at NeurIPS 2020")), "2020-12", TOTAL)
    assert set(f) == set(SIGNATURE_FEATURES)
    assert f["growth_ratio"] > 1.0                 # it grew
    assert f["persistence_months"] == 12           # active every month of 2020
    assert f["cross_listing_rate"] == 1.0          # every record carries 2 categories
    assert f["cross_domain_spread"] > 0.0          # spans >1 category
    assert f["venue_appearance"] > 0.0             # NeurIPS in the comment
    assert 0.0 <= f["author_inflow"] <= 1.0


def test_handles_array_and_missing_author_category_cells():
    """The real corpus round-trips list cells to numpy arrays (and some are None) -- must not crash."""
    import numpy as np
    recs = pd.DataFrame([
        {"record_id": "r1", "month": "2020-03", "authors": np.array(["a", "b"]),
         "categories": np.array(["cs.AI", "cs.LG"]), "comment": "", "journal_ref": ""},
        {"record_id": "r2", "month": "2020-04", "authors": ["c"], "categories": ["cs.AI"],
         "comment": "", "journal_ref": ""},
        {"record_id": "r3", "month": "2020-05", "authors": None, "categories": None,
         "comment": "", "journal_ref": ""},
    ])
    f = theme_features(recs, "2020-12", TOTAL)
    assert set(f) == set(SIGNATURE_FEATURES)
    assert f["persistence_months"] == 3
    assert f["cross_listing_rate"] == 1 / 3            # only r1 has >=2 categories


def test_empty_theme_returns_zeros():
    empty = _recs([]).iloc[0:0]
    f = theme_features(empty, "2020-12", TOTAL)
    assert f == {k: 0.0 for k in SIGNATURE_FEATURES}
