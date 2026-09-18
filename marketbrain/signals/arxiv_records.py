"""arXiv full-record pull + deterministic term extraction (feeds the emergence signal).

The count signal (``arxiv.py``) only needs ``opensearch:totalResults``. The Carley/Garner
emergence feature needs the *content* — title, abstract, authors — to see which research
*terms* are accelerating. This module does the heavier pull (tens of thousands of full
records) and turns each into the ``term_events`` table the scorer consumes.

Everything here is **deterministic** (v1 rule: no LLMs): terms are noun-phrase-ish 1-3 grams
from title+abstract with a stopword/boilerplate filter, pruned to a minimum global document
frequency; "authors" are the arXiv author names (for the community gate). Records are pulled
**per month** (each cached to parquet, so the long pull is resumable) and are leakage-safe by
construction — a paper submitted in month T is public by end-of-T.
"""

from __future__ import annotations

import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd

from ..timeutil import month_range
from .arxiv import ARXIV_API, LLM_DEVTOOLS_QUERY, _NS, _submitted_bounds

CACHE_DIR = "data/raw/arxiv_records"

# English stopwords + academic-paper boilerplate. A term is dropped if it *starts or ends*
# with any of these (the noun-phrase heuristic), so "language model" survives but
# "the model" / "we propose" / "based model" do not.
_STOP = frozenset("""
a an the this that these those of in on at to for from by with as is are was were be been being
and or but not no nor so than then thus also we our us they their it its he she his her you your
i me my mine ours yours them which who whom whose what when where why how all any both each few more
most other some such only own same too very can will just should now into over under again further
do does did doing have has had having would could may might must shall about above below up down out
off between against during before after while because if else via per use used using uses based
propose proposed proposing method methods approach approaches result results paper papers study
studies work works show shows shown present presents presented introduce introduces novel new recent
existing different various several many one two three first second new given due across within without
""".split())

_WORD = re.compile(r"[a-z][a-z\-]+")


def _get(url: str, *, sleep: float, retries: int = 4) -> bytes:
    """GET with arXiv etiquette + exponential backoff (the API flakes under load)."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "market-brain/0.1 (research)"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            if sleep:
                time.sleep(sleep)
            return data
        except (urllib.error.URLError, TimeoutError) as exc:  # pragma: no cover - network
            last = exc
            time.sleep(sleep * (2 ** attempt))
    raise RuntimeError(f"arXiv GET failed after {retries} tries: {url}") from last


def _parse_entries(xml: bytes) -> tuple[list[dict], int]:
    root = ET.fromstring(xml)
    total = int(root.findtext("os:totalResults", default="0", namespaces=_NS))
    rows = []
    for e in root.findall("a:entry", _NS):
        rid = (e.findtext("a:id", default="", namespaces=_NS) or "").strip()
        title = " ".join((e.findtext("a:title", default="", namespaces=_NS) or "").split())
        summary = " ".join((e.findtext("a:summary", default="", namespaces=_NS) or "").split())
        authors = [
            (a.findtext("a:name", default="", namespaces=_NS) or "").strip()
            for a in e.findall("a:author", _NS)
        ]
        if rid:
            rows.append({"record_id": rid, "title": title, "abstract": summary,
                         "authors": [x for x in authors if x]})
    return rows, total


def _fetch_month(month: str, query: str, *, sleep: float, page_size: int) -> pd.DataFrame:
    lo, hi = _submitted_bounds(month)
    fq = f"{query} AND submittedDate:[{lo} TO {hi}]"
    seen: dict[str, dict] = {}
    start = 0
    while True:
        params = {"search_query": fq, "start": start, "max_results": page_size,
                  "sortBy": "submittedDate", "sortOrder": "ascending"}
        rows, total = _parse_entries(_get(f"{ARXIV_API}?{urllib.parse.urlencode(params)}", sleep=sleep))
        for r in rows:
            seen[r["record_id"]] = r
        start += page_size
        if not rows or start >= total or start >= 30000:
            break
    out = pd.DataFrame(list(seen.values()), columns=["record_id", "title", "abstract", "authors"])
    out.insert(1, "month", month)
    return out


def fetch_records(
    start: str, end: str, *, query: str = LLM_DEVTOOLS_QUERY, sleep: float = 3.0,
    page_size: int = 1000, cache_dir: str = CACHE_DIR, verbose: bool = True,
) -> pd.DataFrame:
    """Full arXiv records for [start, end], one cached parquet per month (resumable)."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    frames = []
    for m in month_range(start, end):
        fp = cache / f"{m}.parquet"
        if fp.exists():
            frames.append(pd.read_parquet(fp))
            continue
        dfm = _fetch_month(m, query, sleep=sleep, page_size=page_size)
        dfm.to_parquet(fp)
        frames.append(dfm)
        if verbose:
            print(f"[arxiv-records] {m}: {len(dfm)} records")
    cols = ["record_id", "month", "title", "abstract", "authors"]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)


def _terms_in(text: str, ngram_max: int = 3) -> set[str]:
    """Noun-phrase-ish 1..``ngram_max`` grams: no boundary stopword, every token >= 3 chars."""
    toks = _WORD.findall(text.lower())
    out: set[str] = set()
    for n in range(1, ngram_max + 1):
        for i in range(len(toks) - n + 1):
            gram = toks[i:i + n]
            if gram[0] in _STOP or gram[-1] in _STOP or any(len(t) < 3 for t in gram):
                continue
            out.add(" ".join(gram))
    return out


def extract_term_events(
    records: pd.DataFrame, *, min_doc_freq: int = 10, ngram_max: int = 2
) -> pd.DataFrame:
    """Records -> ``term_events`` (record_id, month, term, authors) for the emergence scorer.

    Terms are extracted from title+abstract, pruned to those appearing in >= ``min_doc_freq``
    distinct records over the whole pull (cuts the typo/rare tail; the in-window emergence gates
    do the real filtering). ``authors`` is one frozenset per record (shared across its terms).

    ``ngram_max`` defaults to 2 (uni+bigrams) for the production pull: trigrams roughly quadruple
    the distinct-term count (memory) for little gain — multiword terms like "retrieval augmented
    generation" are still captured via their bigrams ("retrieval augmented", "augmented generation").
    """
    per_record: list[tuple[str, str, frozenset, set]] = []
    df_counter: Counter = Counter()
    for row in records.itertuples(index=False):
        terms = _terms_in(f"{row.title}. {row.abstract}", ngram_max=ngram_max)
        if not terms:
            continue
        authors = frozenset(row.authors) if row.authors is not None else frozenset()
        per_record.append((row.record_id, row.month, authors, terms))
        df_counter.update(terms)

    keep = {t for t, c in df_counter.items() if c >= min_doc_freq}
    rec_ids, months, term_col, author_col = [], [], [], []
    for rid, month, authors, terms in per_record:
        for t in terms & keep:
            rec_ids.append(rid)
            months.append(month)
            term_col.append(t)
            author_col.append(authors)
    return pd.DataFrame({"record_id": rec_ids, "month": months,
                         "term": term_col, "authors": author_col})
