import pandas as pd

from marketbrain.eval.author_features import (
    AUTHOR_FEATURES,
    build_author_index,
    early_authors,
    prior_paper_count,
    settled_hit_index,
    theme_author_features,
)


def _corpus():
    return pd.DataFrame({
        "month": ["2020-01", "2020-05", "2021-01", "2021-01", "2021-02"],
        "authors": [["Ada Lovelace"], ["Ada Lovelace", "Bob New"],
                    ["Ada Lovelace"], ["Carol Fresh"], ["Bob New"]],
    })


def test_prior_paper_count_is_strictly_before():
    idx = build_author_index(_corpus())
    assert prior_paper_count("Ada Lovelace", "2021-01", idx) == 2  # 2021-01 paper excluded
    assert prior_paper_count("Ada Lovelace", "2021-02", idx) == 3
    assert prior_paper_count("Carol Fresh", "2021-01", idx) == 0
    assert prior_paper_count("Nobody", "2021-01", idx) == 0


def test_early_authors_window_and_asof():
    recs = pd.DataFrame({
        "month": ["2021-01", "2021-02", "2021-09"],
        "authors": [["Ada Lovelace"], ["Bob New"], ["Late Joiner"]],
    })
    # 6-month window from t0 excludes the 2021-09 paper
    assert early_authors(recs, t0="2021-01", as_of="2021-12") == ["Ada Lovelace", "Bob New"]
    # as_of caps the window for very young themes
    assert early_authors(recs, t0="2021-01", as_of="2021-01") == ["Ada Lovelace"]


def test_theme_author_features_values():
    idx = build_author_index(_corpus())
    hits = settled_hit_index([("old theme", "2020-12", ["Ada Lovelace"])])
    f = theme_author_features(["Ada Lovelace", "Carol Fresh"], t0="2021-01", index=idx, hits=hits)
    assert set(f) == set(AUTHOR_FEATURES)
    assert f["auth_newcomer_ratio"] == 0.5           # Carol has no prior papers
    assert f["auth_prior_hit"] == 0.5                # Ada was on a theme settled 2020-12 < 2021-01
    assert f["auth_star"] == 0.0
    assert f["auth_prior_papers"] > 0


def test_prior_hit_requires_settlement_before_t0():
    hits = settled_hit_index([("t", "2021-06", ["Ada Lovelace"])])  # settles AFTER t0
    idx = build_author_index(_corpus())
    f = theme_author_features(["Ada Lovelace"], t0="2021-01", index=idx, hits=hits)
    assert f["auth_prior_hit"] == 0.0


def test_empty_authors():
    f = theme_author_features([], t0="2021-01", index={}, hits={})
    assert all(v == 0.0 for v in f.values())


def test_citation_prestige_is_dated_and_prior_only():
    import numpy as np

    from marketbrain.eval.author_features import (
        build_author_paper_index, citations_before_year, theme_citation_features)

    corpus = pd.DataFrame({
        "month": ["2018-01", "2020-06", "2021-01"],
        "record_id": ["r_old", "r_recent", "r_new"],
        "authors": [["Ada Lovelace"], ["Ada Lovelace"], ["Ada Lovelace"]],
    })
    idx = build_author_paper_index(corpus)
    # r_old earned 100 cites in 2019 + 50 in 2021; only pre-2021 years count for a 2021-01 theme
    cit = {"r_old": {"2019": 100, "2021": 50}, "r_recent": {"2021": 10}}
    assert citations_before_year("r_old", 2021, cit) == 100
    f = theme_citation_features(["Ada Lovelace"], t0="2021-01", paper_index=idx, cit_lookup=cit)
    # prior papers of Ada before 2021-01: r_old + r_recent; dated cites: 100 + 0
    assert abs(f["auth_cit_max"] - np.log1p(100)) < 1e-9
    f2 = theme_citation_features(["Nobody"], t0="2021-01", paper_index=idx, cit_lookup=cit)
    assert f2["auth_cit_max"] == 0.0
