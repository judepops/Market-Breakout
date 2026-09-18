from experiments.venue_sensitivity import acceptance_lag_months


def test_lag_from_comment_year():
    # submitted 2023-01, accepted NeurIPS 2023 -> ~5 months to mid-year
    assert acceptance_lag_months("2023-01", "Accepted at NeurIPS 2023", "") == 5


def test_lag_next_year_acceptance():
    # submitted 2023-10, accepted ICML 2024 -> to 2024-06 = 8 months
    assert acceptance_lag_months("2023-10", "To appear in ICML 2024", "") == 8


def test_no_venue_year_returns_none():
    assert acceptance_lag_months("2023-01", "Under review", "") is None
    assert acceptance_lag_months("2023-01", "", "") is None


def test_negative_lag_dropped():
    # metadata year before submission (data noise) -> None
    assert acceptance_lag_months("2024-01", "CVPR 2022", "") is None


def test_journal_ref_counts():
    assert acceptance_lag_months("2022-03", "", "Published at CVPR 2022") == 3
