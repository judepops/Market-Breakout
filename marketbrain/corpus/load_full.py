"""Load and dedup the full arXiv corpus into one table -- the substrate for the emergence harness.

    from marketbrain.corpus.load_full import load_corpus
    df = load_corpus()             # every downloaded category, deduped (cached, auto-invalidated)
    df = load_corpus(["cs.AI"])    # one category (not cached)

Each paper is fetched once per category it is listed in (data/raw/arxiv_full/<cat>/<month>.parquet), but every
row already carries the paper's FULL category list (arXiv returns all <category> tags per entry), so dedup is a
plain drop on record_id. Columns: record_id, month (submission YYYY-MM), published, title, abstract, authors,
categories, primary_category, comment, journal_ref, doi.

Cache invalidation (audit P1): the full-corpus cache is keyed on a manifest of (source path, mtime) over the raw
month-parquets, so a newly downloaded month rebuilds the cache automatically instead of being silently ignored
(which would freeze the live-radar's "recent" window in the past). Pass refresh=True to force a rebuild.
"""
from __future__ import annotations
import glob
import hashlib
import os
import pathlib

RAW = pathlib.Path("data/raw/arxiv_full")
DERIVED = pathlib.Path("data/derived")


def _manifest(files):
    """A hash of the source set + their mtimes -- changes whenever a month-parquet is added/updated."""
    blob = "|".join(f"{f}:{os.path.getmtime(f):.0f}" for f in files)
    return hashlib.sha1(blob.encode()).hexdigest()


def load_corpus(cats=None, *, cache=True, refresh=False):
    import pandas as pd
    out = DERIVED / "corpus.parquet"
    man = DERIVED / "corpus.manifest"
    full = cats is None
    if full:
        files = sorted(glob.glob(str(RAW / "*" / "*.parquet")))
    else:
        files = sorted(f for c in cats for f in glob.glob(str(RAW / c / "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no arxiv_full parquets under {RAW} for cats={cats}")
    cur = _manifest(files)
    if full and cache and not refresh and out.exists() and man.exists() and man.read_text().strip() == cur:
        return pd.read_parquet(out)
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.drop_duplicates("record_id").reset_index(drop=True)
    if full and cache:
        DERIVED.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out)
        man.write_text(cur)
    return df
