"""Offline tests for the feature layer (normalize + correlation)."""

from __future__ import annotations

from datetime import date

import numpy as np

from marketbrain.features import build_feature_panel, correlation_matrix
from tests.conftest import make_obs, obs_frame


def _corpus():
    # research + the two adoption halves over 14 months (so YoY is defined at month 13+)
    rows = []
    for i, m in enumerate(range(1, 15)):
        ref = date(2024, m, 28) if m <= 12 else date(2025, m - 12, 28)
        rows += [
            make_obs(feature="arxiv_submissions", value=100 + i * 10, reference_date=ref,
                     knowledge_date=ref, obs_id=f"a{i}"),
            make_obs(feature="npm_downloads", value=1e6 + i * 1e5, reference_date=ref,
                     knowledge_date=ref, obs_id=f"n{i}"),
            make_obs(feature="pypi_downloads", value=5e6 + i * 5e5, reference_date=ref,
                     knowledge_date=ref, obs_id=f"p{i}"),
        ]
    return obs_frame(rows)


def test_panel_has_levels_growth_and_dimension_combine():
    panel = build_feature_panel(_corpus())
    assert "arxiv_submissions__level_z" in panel
    assert "arxiv_submissions__yoy_growth" in panel
    assert "dim_adoption__level_z" in panel            # npm+pypi collapsed into one adoption dim
    assert abs(panel["arxiv_submissions__level_z"].mean()) < 1e-9   # standardized -> mean 0
    g = panel["arxiv_submissions__yoy_growth"]
    assert g.iloc[:12].isna().all() and not np.isnan(g.iloc[12])    # YoY undefined until month 13


def test_correlation_matrix_square_symmetric_and_adoption_halves_correlate():
    corr = correlation_matrix(_corpus())
    assert list(corr.index) == list(corr.columns)
    assert (corr.values.diagonal() == 1).all()
    # npm and pypi both rise monotonically -> near-perfectly correlated (they're one dimension)
    assert corr.loc["npm_downloads", "pypi_downloads"] > 0.99
