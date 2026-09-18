"""Who is behind a theme — top researchers and their key papers (in-corpus, no LLM).

For a research-focused VC: given a theme's papers, surface the people driving it
(ranked by early involvement + track record + volume on the theme) and the papers
worth reading first. Everything is derived from the corpus we already have; author
background / Google Scholar enrichment is a separate optional layer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _prior_count(author: str, before_month: str, author_index: dict) -> int:
    months = author_index.get(author)
    if months is None:
        return 0
    return int(np.searchsorted(months, before_month, side="left"))


def theme_researchers(
    recs: pd.DataFrame,
    t0: str,
    author_index: dict,
    top_n: int = 5,
) -> list[dict]:
    """Top researchers on a theme.

    ``recs`` columns: month, authors (list). Score each author by
    theme_papers (their count on this theme) + log1p(prior track record before t0)
    + an early-involvement bonus (on the theme in its first 6 months). Returns
    ``[{name, theme_papers, prior_papers, early}]`` sorted by score.
    """
    from .label import _shift_month

    early_hi = _shift_month(t0, 6)
    theme_papers: dict[str, int] = {}
    early: dict[str, bool] = {}
    for r in recs.itertuples(index=False):
        for a in ([] if r.authors is None else list(r.authors)):
            if not a:
                continue
            theme_papers[a] = theme_papers.get(a, 0) + 1
            if r.month <= early_hi:
                early[a] = True
    scored = []
    for a, tp in theme_papers.items():
        prior = _prior_count(a, t0, author_index)
        score = tp + np.log1p(prior) + (1.5 if early.get(a) else 0.0)
        scored.append((score, {"name": a, "theme_papers": tp,
                               "prior_papers": prior, "early": bool(early.get(a))}))
    scored.sort(key=lambda x: -x[0])
    return [d for _, d in scored[:top_n]]


def theme_key_papers(
    recs: pd.DataFrame,
    author_index: dict,
    t0: str,
    top_n: int = 5,
) -> list[dict]:
    """The theme's papers worth reading first.

    ``recs`` columns: record_id, month, title, authors. Ranked by founding-author
    prestige (sum of authors' prior-paper counts as-of the paper's month), with
    early papers favored on ties. record_id is the arXiv abs URL (the link).
    Returns ``[{title, authors, month, url}]``.
    """
    if "record_id" not in recs.columns:  # record_id may arrive as the index
        recs = recs.reset_index()
    rows = []
    for r in recs.itertuples(index=False):
        authors = [] if r.authors is None else [a for a in list(r.authors) if a]
        prestige = sum(_prior_count(a, r.month, author_index) for a in authors)
        early = 1 if r.month <= _short_shift(t0) else 0
        rows.append((prestige, early, {
            "title": r.title or r.record_id,
            "authors": authors[:4],
            "month": r.month,
            "url": r.record_id,
        }))
    rows.sort(key=lambda x: (-x[0], -x[1], x[2]["month"]))
    return [d for _, _, d in rows[:top_n]]


def _short_shift(t0: str) -> str:
    """t0 + 6 months (early-paper cutoff), inlined to avoid a label import cycle."""
    y, m = (int(x) for x in t0.split("-"))
    total = y * 12 + (m - 1) + 6
    return f"{total // 12:04d}-{total % 12 + 1:02d}"
