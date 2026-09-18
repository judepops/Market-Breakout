"""Comprehensive arXiv corpus pull (MAXIMUM paper info) -- the discovery corpus for the emergence method.

    python -m experiments.fetch_arxiv_full

Fetches EVERY paper in a broad AI/CS/EE/quantum category set, 2012-2026, PER CATEGORY (so it dodges arXiv's
~30k-per-query cap that truncates high-volume months), resumable (one parquet per category-month). Captures
the full record: id, exact submission date, title, abstract, authors, ALL category tags (cross-domain signal),
primary category, the comment field (often the venue, e.g. "Accepted to NeurIPS 2024" -> the validation
signal), journal_ref and doi. Cross-listed papers appear under multiple categories; dedup by record_id at
load time. Distinct cache dir (data/raw/arxiv_full) so it never collides with the narrow term-pull.
"""
from __future__ import annotations
import pathlib
import time
import urllib.error
import urllib.parse
import urllib.request
from xml.etree import ElementTree as ET

import pandas as pd

from marketbrain.timeutil import month_range
from marketbrain.signals.arxiv import ARXIV_API, _submitted_bounds

NS = {"a": "http://www.w3.org/2005/Atom", "os": "http://a9.com/-/spec/opensearch/1.1/",
      "arxiv": "http://arxiv.org/schemas/atom"}
CATS = ["cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.RO", "cs.NE", "cs.IR", "cs.MA", "cs.DB", "cs.SE",
        "cs.DC", "cs.CR", "cs.AR", "cs.PL", "cs.HC", "cs.SD", "cs.SY", "cs.NI", "cs.OS", "cs.DS",
        "cs.GR", "cs.CY", "stat.ML", "eess.AS", "eess.SY", "eess.IV", "eess.SP", "quant-ph"]
START, END = "2012-01", "2026-06"
CACHE = pathlib.Path("data/raw/arxiv_full")
SLEEP, PAGE = 3.0, 1000


def _get(url: str) -> bytes:
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "market-brain/0.1 (research)"})
            data = urllib.request.urlopen(req, timeout=60).read()
            time.sleep(SLEEP)
            return data
        except (urllib.error.URLError, TimeoutError):
            time.sleep(SLEEP * (2 ** attempt))
    return b""


def _parse(xml: bytes):
    """Returns (rows, total), or None if the response was empty/malformed (the caller retries the page)."""
    if not xml:
        return None
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return None
    total = int(root.findtext("os:totalResults", default="0", namespaces=NS))
    rows = []
    for e in root.findall("a:entry", NS):
        rid = (e.findtext("a:id", default="", namespaces=NS) or "").strip()
        if not rid:
            continue
        prim = e.find("arxiv:primary_category", NS)
        rows.append({
            "record_id": rid,
            "published": (e.findtext("a:published", default="", namespaces=NS) or "")[:10],
            "title": " ".join((e.findtext("a:title", default="", namespaces=NS) or "").split()),
            "abstract": " ".join((e.findtext("a:summary", default="", namespaces=NS) or "").split()),
            "authors": [(a.findtext("a:name", default="", namespaces=NS) or "").strip()
                        for a in e.findall("a:author", NS)],
            "categories": [c.get("term") for c in e.findall("a:category", NS) if c.get("term")],
            "primary_category": prim.get("term") if prim is not None else "",
            "comment": " ".join((e.findtext("arxiv:comment", default="", namespaces=NS) or "").split()),
            "journal_ref": " ".join((e.findtext("arxiv:journal_ref", default="", namespaces=NS) or "").split()),
            "doi": (e.findtext("arxiv:doi", default="", namespaces=NS) or "").strip(),
        })
    return rows, total


def _fetch(cat: str, month: str) -> pd.DataFrame:
    lo, hi = _submitted_bounds(month)
    fq = f"cat:{cat} AND submittedDate:[{lo} TO {hi}]"
    seen: dict[str, dict] = {}
    start = 0
    while True:
        p = {"search_query": fq, "start": start, "max_results": PAGE,
             "sortBy": "submittedDate", "sortOrder": "ascending"}
        parsed = None
        for _ in range(6):                          # retry a flaky/empty page instead of crashing
            parsed = _parse(_get(f"{ARXIV_API}?{urllib.parse.urlencode(p)}"))
            if parsed is not None:
                break
            time.sleep(10)
        if parsed is None:
            print(f"[arxiv-full] WARN {cat} {month} @start={start}: page failed repeatedly, skipping rest of month",
                  flush=True)
            break
        rows, total = parsed
        for r in rows:
            seen[r["record_id"]] = r
        start += PAGE
        if not rows or start >= total or start >= 30000:
            break
    cols = ["record_id", "published", "title", "abstract", "authors", "categories",
            "primary_category", "comment", "journal_ref", "doi"]
    df = pd.DataFrame(list(seen.values()), columns=cols)
    df.insert(1, "month", month)
    return df


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    months = list(month_range(START, END))
    print(f"[arxiv-full] {len(CATS)} categories x {len(months)} months ({START}..{END}) -> {CACHE}", flush=True)
    for ci, cat in enumerate(CATS):
        d = CACHE / cat
        d.mkdir(exist_ok=True)
        n = 0
        for m in months:
            fp = d / f"{m}.parquet"
            if fp.exists():
                continue
            df = _fetch(cat, m)
            df.to_parquet(fp)
            n += len(df)
        print(f"[arxiv-full] {cat:10s} ({ci + 1}/{len(CATS)}) done | +{n:,} new records", flush=True)
    print("[arxiv-full] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
