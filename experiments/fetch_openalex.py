"""Fetch OpenAlex citation histories for the arXiv corpus (Tier-2 enrichment).

    python -m experiments.fetch_openalex [--cats cs.AI] [--rps 5]

Batch lookup by the papers' auto-assigned arXiv DOIs (10.48550/arXiv.<id>), 50 per
request, polite pool via mailto. Stores resumable parquet chunks under
data/derived/openalex/ with: record_id, cited_by_count, publication_year, and
counts_by_year as a JSON string ({year: cites}) — the dated citation record that
makes leakage-free "citations as-of" features possible.

Free API, no key. ~3.7k requests for the full corpus.
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import re
import time
import urllib.parse
import urllib.request

STORE = pathlib.Path("data/derived/openalex")
MAILTO = __import__("os").environ.get("CONTACT_EMAIL", "you@example.com")
BATCH = 50

_ARXIV_ID = re.compile(r"abs/([^v]+(?:v\d+)?)$")


def arxiv_doi(record_id: str) -> str | None:
    """arXiv record URL -> auto-assigned DOI (version suffix stripped)."""
    m = _ARXIV_ID.search(record_id)
    if not m:
        return None
    bare = re.sub(r"v\d+$", "", m.group(1))
    return f"10.48550/arXiv.{bare}"


def parse_works(payload: dict, doi_to_record: dict[str, str]) -> list[dict]:
    """OpenAlex results -> store rows (record_id keyed)."""
    rows = []
    for w in payload.get("results", []):
        doi = (w.get("doi") or "").replace("https://doi.org/", "").lower()
        rid = doi_to_record.get(doi)
        if not rid:
            continue
        cby = {int(e["year"]): int(e["cited_by_count"]) for e in w.get("counts_by_year", [])}
        rows.append({
            "record_id": rid,
            "cited_by_count": int(w.get("cited_by_count") or 0),
            "publication_year": int(w.get("publication_year") or 0),
            "counts_by_year": json.dumps(cby),
        })
    return rows


def main(argv=None):
    import pandas as pd

    from marketbrain.corpus.load_full import load_corpus

    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", nargs="*", default=["cs.AI"])
    ap.add_argument("--rps", type=float, default=5.0)
    args = ap.parse_args(argv)
    STORE.mkdir(parents=True, exist_ok=True)

    corpus = load_corpus(args.cats, cache=False)[["record_id"]]
    done: set[str] = set()
    for f in glob.glob(str(STORE / "chunk-*.parquet")):
        done.update(pd.read_parquet(f, columns=["record_id"])["record_id"])
    # a separate misses file records DOIs OpenAlex doesn't know, so we don't refetch
    misses_path = STORE / "misses.json"
    misses: set[str] = set(json.loads(misses_path.read_text())) if misses_path.exists() else set()
    pend = corpus[~corpus["record_id"].isin(done | misses)]["record_id"].tolist()
    print(f"[openalex] {len(done):,} fetched, {len(misses):,} known misses, {len(pend):,} pending", flush=True)

    chunk_idx = 1 + max([int(pathlib.Path(f).stem.split("-")[1])
                         for f in glob.glob(str(STORE / "chunk-*.parquet"))], default=-1)
    buf: list[dict] = []
    for start in range(0, len(pend), BATCH):
        part = pend[start : start + BATCH]
        doi_to_record = {arxiv_doi(r).lower(): r for r in part if arxiv_doi(r)}
        flt = "doi:" + "|".join(doi_to_record)
        url = ("https://api.openalex.org/works?" + urllib.parse.urlencode({
            "filter": flt, "per-page": BATCH, "mailto": MAILTO,
            "select": "doi,cited_by_count,counts_by_year,publication_year"}))
        for attempt in range(8):
            try:
                with urllib.request.urlopen(urllib.request.Request(url), timeout=60) as resp:
                    payload = json.loads(resp.read())
                break
            except urllib.error.HTTPError as e:
                if attempt == 7:
                    raise
                # 429 = rate limit: back off for minutes, not seconds
                wait = 60 * (attempt + 1) if e.code == 429 else 10 * (attempt + 1)
                print(f"[openalex] HTTP {e.code}, sleeping {wait}s", flush=True)
                time.sleep(wait)
            except Exception as e:  # noqa: BLE001 — network flake: back off and retry
                if attempt == 7:
                    raise
                print(f"[openalex] retry after {type(e).__name__}", flush=True)
                time.sleep(10 * (attempt + 1))
        rows = parse_works(payload, doi_to_record)
        found = {r["record_id"] for r in rows}
        misses.update(set(part) - found)
        buf.extend(rows)
        if len(buf) >= 10_000:
            out = STORE / f"chunk-{chunk_idx}.parquet"
            pd.DataFrame(buf).to_parquet(out)
            misses_path.write_text(json.dumps(sorted(misses)))
            print(f"[openalex] wrote {out} ({len(buf):,} rows); "
                  f"{start + len(part):,}/{len(pend):,} processed", flush=True)
            buf, chunk_idx = [], chunk_idx + 1
        time.sleep(1.0 / args.rps)
    if buf:
        pd.DataFrame(buf).to_parquet(STORE / f"chunk-{chunk_idx}.parquet")
    misses_path.write_text(json.dumps(sorted(misses)))
    print(f"[openalex] DONE — {len(misses):,} DOIs not in OpenAlex", flush=True)


if __name__ == "__main__":
    main()
