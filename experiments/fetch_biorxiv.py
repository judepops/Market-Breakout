"""Pull the bioRxiv preprint corpus into the SAME schema as the arXiv corpus, so the emergence harness
(gates -> label -> signature -> momentum) runs unchanged on biology. Cross-domain transfer test M1.2.

    ~/miniconda3/envs/marketbrain/bin/python -m experiments.fetch_biorxiv [--start 2013-11 --end 2024-12]

Source: the public bioRxiv details API (https://api.biorxiv.org/details/biorxiv/<from>/<to>/<cursor>), 30
records/page. We keep VERSION 1 only (the original posting) so `month` = the true knowledge/emergence month.
One parquet per month at data/raw/biorxiv_full/<YYYY-MM>.parquet (resumable: existing months are skipped).
Columns mirror load_corpus: record_id, month, published, title, abstract, authors(list), categories(list),
primary_category, comment, journal_ref, doi.

Default end 2024-12 leaves >=~18mo for the durability label to resolve (corpus 'now' is mid-2026).
"""
from __future__ import annotations
import argparse
import json
import pathlib
import time
import urllib.request

RAW = pathlib.Path("data/raw/biorxiv_full")
API = "https://api.biorxiv.org/details/biorxiv/{a}/{b}/{cur}"


def _get(url, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=45) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))
    return None


def _month_bounds(ym):
    y, m = int(ym[:4]), int(ym[5:7])
    last = [31, 29 if y % 4 == 0 and (y % 100 or y % 400 == 0) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1]
    return f"{ym}-01", f"{ym}-{last:02d}"


def _months(start, end):
    y, m = int(start[:4]), int(start[5:7]); ey, em = int(end[:4]), int(end[5:7])
    out = []
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m = 1; y += 1
    return out


def fetch_month(ym):
    import pandas as pd
    a, b = _month_bounds(ym)
    rows = {}
    cursor = 0
    total = None
    while True:
        d = _get(API.format(a=a, b=b, cur=cursor))
        msg = (d.get("messages") or [{}])[0]
        if total is None:
            total = int(msg.get("total", 0))
        coll = d.get("collection", [])
        if not coll:
            break
        for r in coll:
            if str(r.get("version", "1")) != "1":       # original posting only -> true knowledge month
                continue
            doi = r.get("doi")
            if not doi or doi in rows:
                continue
            auth = [x.strip() for x in str(r.get("authors", "")).split(";") if x.strip()]
            cat = (r.get("category") or "").strip().lower()
            pub = r.get("published", "")
            rows[doi] = {
                "record_id": doi, "month": str(r.get("date", ""))[:7], "published": pub,
                "title": r.get("title", ""), "abstract": r.get("abstract", ""),
                "authors": auth, "categories": [cat] if cat else [],
                "primary_category": cat, "comment": "",
                "journal_ref": pub if pub and pub != "NA" else "", "doi": doi,
            }
        cursor += len(coll)
        if total and cursor >= total:
            break
        time.sleep(0.15)
    df = pd.DataFrame(list(rows.values()))
    if not df.empty:
        df = df[df["month"] == ym].reset_index(drop=True)   # guard against boundary bleed
    return df


def _one(ym):
    """Fetch+write a single month; returns (ym, n) or (ym, -1) on failure. Skips if already on disk."""
    import pandas as pd
    out = RAW / f"{ym}.parquet"
    if out.exists():
        try:
            return ym, len(pd.read_parquet(out))
        except Exception:
            pass
    try:
        df = fetch_month(ym)
    except Exception as e:
        print(f"[biorxiv] {ym} FAILED: {e}", flush=True)
        return ym, -1
    df.to_parquet(out)
    return ym, len(df)


def main(argv=None):
    from concurrent.futures import ThreadPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2013-11")     # bioRxiv launch
    ap.add_argument("--end", default="2024-12")        # leave >=18mo for the durability label
    ap.add_argument("--workers", type=int, default=6)  # months are independent + I/O-bound -> safe to parallelise
    args = ap.parse_args(argv)
    RAW.mkdir(parents=True, exist_ok=True)
    months = _months(args.start, args.end)
    grand = done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for ym, nrec in ex.map(_one, months):
            if nrec < 0:
                continue
            grand += nrec; done += 1
            print(f"[biorxiv] {ym}: {nrec:>5d} preprints  (done {done}/{len(months)} months, cumulative {grand:>7d})", flush=True)
    print(f"[biorxiv] DONE: {grand} preprints across {done} months -> {RAW}", flush=True)


if __name__ == "__main__":
    main()
