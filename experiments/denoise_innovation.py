"""De-noise the innovation dimension: is the raw patent SUM just mega-incumbents (Apple/IBM), and does a
robust per-company measure recover real signal?

One scan returns per-(assignee, year) patent counts for our markets' companies; we map each assignee back to
its market(s), window to each market's formation years, and compute several robust measures per market:
  sum                - the raw (incumbent-dominated) measure
  sum_ex_top         - sum excluding the single biggest patenter (drops the dominant incumbent)
  median             - median patents per patenting company (one giant cannot move it)
  breadth            - how many companies patent at all (innovation is widespread, not one firm)
  top_share          - top firm's share of the market's patents (a concentration / noise gauge)
Then test which (z-scored) correlates with the outlier rate. If none do, innovation is genuinely weak.

    python -m experiments.denoise_innovation
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="BigQuery Storage module not found")

from marketbrain.signals.patents import _SUFFIX  # noqa: E402

_STRIP_SUFFIX = re.compile(rf"( ({_SUFFIX}))+$")


def _core(aname: str) -> str:
    return _STRIP_SUFFIX.sub("", aname).strip()


def _query(global_regex, lo_year, hi_year, *, project, max_bytes=30 * 10**9):
    from google.cloud import bigquery
    client = bigquery.Client(project=project or "production-498516")
    sql = """
    SELECT aname, DIV(public_d, 10000) AS year, COUNT(*) AS patents
    FROM (
      SELECT IF(publication_date > 0, publication_date, grant_date) AS public_d,
             REGEXP_REPLACE(UPPER(a.name), r'[.,]', '') AS aname
      FROM `patents-public-data.patents.publications`, UNNEST(assignee_harmonized) a
    )
    WHERE public_d BETWEEN @lo AND @hi AND REGEXP_CONTAINS(aname, @g)
    GROUP BY aname, year
    """
    params = [bigquery.ScalarQueryParameter("lo", "INT64", lo_year * 10000 + 101),
              bigquery.ScalarQueryParameter("hi", "INT64", hi_year * 10000 + 1231),
              bigquery.ScalarQueryParameter("g", "STRING", global_regex)]
    dry = client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params, dry_run=True))
    print(f"[denoise] dry-run {dry.total_bytes_processed / 1e9:.1f} GB")
    job = client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params, maximum_bytes_billed=max_bytes))
    rows = [(r.aname, int(r.year), int(r.patents)) for r in job.result()]
    print(f"[denoise] billed {job.total_bytes_billed / 1e9:.1f} GB; {len(rows)} (assignee, year) rows")
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/raw/labels")
    ap.add_argument("--cache", default="data/raw/rounds")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--top-n", type=int, default=40)
    ap.add_argument("--project", default=None)
    args = ap.parse_args(argv)

    import numpy as np
    import pandas as pd

    from marketbrain.features.signal_features import zscore
    from marketbrain.markets.catalog import CATALOG
    from marketbrain.signals.patents import _assignee_regex

    ft = pd.read_parquet(Path(args.labels) / "market_features.parquet").set_index("market")
    specs = [m for m in CATALOG if m.name in ft.index and bool(ft.loc[m.name, "mature"])]

    # per-market cleaned names + name(core, upper) -> markets ; global union regex
    name2mkt: dict[str, set] = {}
    all_names: set[str] = set()
    t0 = {}
    for m in specs:
        t0[m.name] = int(ft.loc[m.name, "t0"])
        for nm in m.assignees(project=args.project, top_n=args.top_n):
            all_names.add(nm)
            name2mkt.setdefault(_core(nm.upper()), set()).add(m.name)
    global_regex = _assignee_regex(tuple(sorted(all_names)))

    cache = Path(args.cache) / f"denoise_assignee_years_n{args.top_n}.parquet"
    if cache.exists():
        rows = list(pd.read_parquet(cache).itertuples(index=False, name=None))
        print(f"using cached: {len(rows)} rows")
    else:
        lo, hi = min(t0.values()), max(t0.values()) + args.k
        rows = _query(global_regex, lo, hi, project=args.project)
        pd.DataFrame(rows, columns=["aname", "year", "patents"]).to_parquet(cache, index=False)

    # per market: {company_core -> patents within its window}
    per = {m.name: {} for m in specs}
    for aname, year, pats in rows:
        core = _core(aname)
        for mk in name2mkt.get(core, ()):
            if t0[mk] <= year < t0[mk] + args.k:
                per[mk][core] = per[mk].get(core, 0) + pats

    recs = []
    for m in specs:
        vals = sorted(per[m.name].values(), reverse=True)
        s = sum(vals)
        recs.append({"market": m.name, "rate": float(ft.loc[m.name, "target_rate"]),
                     "sum": s, "sum_ex_top": s - (vals[0] if vals else 0),
                     "median": float(np.median(vals)) if vals else 0.0,
                     "breadth": len(vals), "top_share": (vals[0] / s if s else 0.0)})
    df = pd.DataFrame(recs)

    def sp(a, b):
        a, b = pd.Series(a).rank(), pd.Series(b).rank()
        return float(np.corrcoef(a, b)[0, 1])

    print(f"\nN={len(df)}.  Spearman of each innovation measure vs outlier rate:")
    for col in ["sum", "sum_ex_top", "median", "breadth", "top_share"]:
        z = [v if v is not None else None for v in zscore([float(np.log1p(x)) if col != "top_share" else float(x)
                                                           for x in df[col]])]
        print(f"  {col:12s} r = {sp(z, df['rate']):+.2f}")
    print(f"\nmedian top-firm share of market patents: {df['top_share'].median():.0%} "
          f"(high -> raw sum is incumbent-dominated)")
    df.to_parquet(Path(args.labels) / "innovation_denoised.parquet", index=False)
    print(f"wrote {Path(args.labels) / 'innovation_denoised.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
