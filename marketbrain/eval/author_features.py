"""In-corpus author-quality features — leakage-free by construction.

Everything is computed from the corpus itself as-of a given month: an author's
history is their earlier papers in OUR corpus (exact full-name match — a known,
documented approximation until external author IDs arrive), and "track record"
counts only themes whose durable outcome had already SETTLED before the new
theme was born.

Features (all over the theme's EARLY authors — authors of its papers in the
first ``early_months`` of life, up to the as-of month):

    auth_prior_papers    median prior-paper count of early authors, log1p
    auth_newcomer_ratio  fraction of early authors with zero prior papers
    auth_star            1 if any early author has >= STAR_PRIOR prior papers
    auth_prior_hit       fraction of early authors who were early authors of a
                         theme that had already proven durable before this t0
"""
from __future__ import annotations

import numpy as np
import pandas as pd

AUTHOR_FEATURES = ["auth_prior_papers", "auth_newcomer_ratio", "auth_star", "auth_prior_hit"]
CITATION_FEATURES = ["auth_cit_max", "auth_cit_med"]
STAR_PRIOR = 30      # prior papers marking an established, prolific author
EARLY_MONTHS = 6     # a theme's "founding team" window after t0


def build_author_index(corpus: pd.DataFrame) -> dict[str, np.ndarray]:
    """author -> sorted array of months ("YYYY-MM") of their papers in the corpus."""
    ex = corpus[["month", "authors"]].explode("authors").dropna(subset=["authors"])
    ex = ex[ex["authors"].str.len() > 0]
    return {a: np.sort(g.to_numpy()) for a, g in ex.groupby("authors")["month"]}


def prior_paper_count(author: str, before_month: str, index: dict[str, np.ndarray]) -> int:
    """Papers by ``author`` strictly before ``before_month`` (as-of discipline)."""
    months = index.get(author)
    if months is None:
        return 0
    return int(np.searchsorted(months, before_month, side="left"))


def early_authors(recs: pd.DataFrame, t0: str, as_of: str, early_months: int = EARLY_MONTHS) -> list[str]:
    """Distinct authors of the theme's papers in [t0, t0+early_months), up to as_of."""
    from .label import _shift_month

    hi = min(_shift_month(t0, early_months), as_of)
    window = recs[(recs["month"] >= t0) & (recs["month"] <= hi)]
    out: list[str] = []
    for authors in window["authors"]:
        if authors is None:
            continue
        for a in list(authors):  # tolerate list or numpy array from parquet
            if a and a not in out:
                out.append(a)
    return out


def settled_hit_index(
    positives: list[tuple[str, str, list[str]]],
) -> dict[str, str]:
    """author -> earliest month at which they were on a theme that had PROVEN durable.

    ``positives``: (theme_id, settle_month, early_author_list) for each durable theme;
    settle_month is when the outcome became knowable (T_peak + persist window).
    """
    out: dict[str, str] = {}
    for _, settle, authors in positives:
        for a in authors:
            if a not in out or settle < out[a]:
                out[a] = settle
    return out


def build_author_paper_index(corpus: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """author -> (months sorted asc, record_ids aligned) for citation-prestige lookups."""
    ex = corpus[["month", "record_id", "authors"]].explode("authors").dropna(subset=["authors"])
    ex = ex[ex["authors"].str.len() > 0].sort_values("month")
    return {a: (g["month"].to_numpy(), g["record_id"].to_numpy())
            for a, g in ex.groupby("authors")}


def citations_before_year(record_id: str, year: int, cit_lookup: dict[str, dict]) -> int:
    """Citations this paper accumulated in years strictly before ``year`` (as-of, dated)."""
    cby = cit_lookup.get(record_id)
    if not cby:
        return 0
    return sum(c for y, c in cby.items() if int(y) < year)


def theme_citation_features(
    authors: list[str],
    t0: str,
    paper_index: dict[str, tuple[np.ndarray, np.ndarray]],
    cit_lookup: dict[str, dict],
) -> dict[str, float]:
    """Founding-author citation prestige as-of birth: citations their PRIOR papers had
    accumulated in years strictly before the birth year. Leakage-safe at year granularity."""
    if not authors:
        return {f: 0.0 for f in CITATION_FEATURES}
    year = int(t0[:4])
    prestige = []
    for a in authors:
        entry = paper_index.get(a)
        if entry is None:
            prestige.append(0.0)
            continue
        months, rids = entry
        n_prior = int(np.searchsorted(months, t0, side="left"))
        prestige.append(float(sum(citations_before_year(r, year, cit_lookup) for r in rids[:n_prior])))
    arr = np.array(prestige)
    return {
        "auth_cit_max": float(np.log1p(arr.max())),
        "auth_cit_med": float(np.log1p(np.median(arr))),
    }


def theme_author_features(
    authors: list[str],
    t0: str,
    index: dict[str, np.ndarray],
    hits: dict[str, str],
) -> dict[str, float]:
    """The four author features for one theme, as-of its birth cohort."""
    if not authors:
        return {f: 0.0 for f in AUTHOR_FEATURES}
    priors = np.array([prior_paper_count(a, t0, index) for a in authors], dtype=float)
    # an author's paper in the theme's own birth month counts toward "prior" via
    # side="left" exclusion of t0 itself — strictly earlier months only.
    hit = sum(1 for a in authors if a in hits and hits[a] < t0)
    return {
        "auth_prior_papers": float(np.log1p(np.median(priors))),
        "auth_newcomer_ratio": float((priors == 0).mean()),
        "auth_star": float((priors >= STAR_PRIOR).any()),
        "auth_prior_hit": hit / len(authors),
    }
