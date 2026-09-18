"""Fetch NEW arXiv fields for the multi-field translational transfer test (pre-reg
Plans/2026-07-06_multifield-translational-prereg.md). Reuses fetch_arxiv_full._fetch verbatim (same corpus
schema, resumable per category-month), just over a different category list + modest parallelism.

    ~/miniconda3/envs/marketbrain/bin/python -m experiments.fetch_arxiv_fields [--workers 4]

Predicted-FIRE: q-fin.*, econ.*   ·   Predicted-NULL: hep-th, gr-qc, math.PR   (eess.* already on disk).
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor

from experiments.fetch_arxiv_full import _fetch, CACHE, START, END
from marketbrain.timeutil import month_range

FIELDS = [
    # q-fin (predicted FIRE) — whole archive
    "q-fin.CP", "q-fin.EC", "q-fin.GN", "q-fin.MF", "q-fin.PM", "q-fin.PR", "q-fin.RM", "q-fin.ST", "q-fin.TR",
    # econ (predicted FIRE)
    "econ.EM", "econ.GN", "econ.TH",
    # theoretical (predicted NULL)
    "hep-th", "gr-qc", "math.PR",
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)   # modest — arXiv likes ~1 req / few sec; _get already sleeps 3s
    ap.add_argument("--cats", default=",".join(FIELDS))
    args = ap.parse_args(argv)
    cats = args.cats.split(",")
    months = list(month_range(START, END))
    CACHE.mkdir(parents=True, exist_ok=True)

    def one(job):
        cat, m = job
        d = CACHE / cat
        d.mkdir(exist_ok=True)
        fp = d / f"{m}.parquet"
        if fp.exists():
            return (cat, 0)
        try:
            df = _fetch(cat, m)
        except Exception as e:
            print(f"[fields] {cat} {m} FAILED: {str(e)[:80]}", flush=True)
            return (cat, -1)
        df.to_parquet(fp)
        return (cat, len(df))

    jobs = [(c, m) for c in cats for m in months]
    print(f"[fields] {len(cats)} cats x {len(months)} months = {len(jobs)} category-months ({args.workers} workers)", flush=True)
    tally = {c: 0 for c in cats}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        done = 0
        for cat, n in ex.map(one, jobs):
            done += 1
            if n > 0:
                tally[cat] += n
            if done % 100 == 0:
                print(f"[fields] {done}/{len(jobs)} category-months done | cumulative {sum(tally.values()):,}", flush=True)
    for c in cats:
        print(f"[fields] {c:12s} total {tally[c]:,}", flush=True)
    print(f"[fields] DONE: {sum(tally.values()):,} records across {len(cats)} cats -> {CACHE}", flush=True)


if __name__ == "__main__":
    main()
