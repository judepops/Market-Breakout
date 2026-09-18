"""Phase 2 - build the ADOPTION dimension (GitHub stars) across all mature markets.

Auto-seeded repos (build_market_repos.py) -> ONE combined GH Archive scan for monthly stars (WatchEvent) per
repo over the global window -> each market's repos summed per month, windowed to its formation window
[t0, t0+k], collapsed to {level, growth} (signal_features), z-scored. Run for ALL markets (measure, don't
pre-mask): markets with no OSS repos / pre-GitHub windows get ~0.

    python -m experiments.build_adoption
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="BigQuery Storage module not found")

DATASET = "githubarchive.month"
MAX_BYTES = 300 * 10**9


def _query_stars(repos, lo_ym: str, hi_ym: str, *, project, max_bytes=MAX_BYTES):
    """{repo -> {'YYYY-MM' -> stars}} across the window. One scan over all repos."""
    from google.cloud import bigquery
    client = bigquery.Client(project=project or "production-498516")
    sql = f"""
    SELECT repo.name AS repo, FORMAT_TIMESTAMP('%Y-%m', created_at) AS ym, COUNT(*) AS stars
    FROM `{DATASET}.*`
    WHERE _TABLE_SUFFIX BETWEEN @lo AND @hi AND type = 'WatchEvent' AND repo.name IN UNNEST(@repos)
    GROUP BY repo, ym
    """
    params = [bigquery.ScalarQueryParameter("lo", "STRING", lo_ym),
              bigquery.ScalarQueryParameter("hi", "STRING", hi_ym),
              bigquery.ArrayQueryParameter("repos", "STRING", list(repos))]
    dry = client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params, dry_run=True))
    print(f"[adoption] dry-run {dry.total_bytes_processed / 1e9:.0f} GB ({len(repos)} repos)")
    job = client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params, maximum_bytes_billed=max_bytes))
    out = {}
    for r in job.result():
        out.setdefault(r.repo, {})[r.ym] = int(r.stars)
    print(f"[adoption] billed {job.total_bytes_billed / 1e9:.0f} GB; {len(out)} repos with stars")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/raw/labels")
    ap.add_argument("--cache", default="data/raw/rounds")
    ap.add_argument("--repos", default="data/raw/labels/github_repos.json")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--project", default=None)
    args = ap.parse_args(argv)

    import numpy as np
    import pandas as pd

    from marketbrain.features.signal_features import reduce_series, zscore

    ft = pd.read_parquet(Path(args.labels) / "market_features.parquet").set_index("market")
    repos_by_market = {k: v for k, v in json.loads(Path(args.repos).read_text()).items()
                       if k in ft.index and bool(ft.loc[k, "mature"])}
    all_repos = sorted({r for v in repos_by_market.values() for r in v})
    t0 = {k: int(ft.loc[k, "t0"]) for k in repos_by_market}
    lo_y, hi_y = max(2012, min(t0.values())), max(t0.values()) + args.k

    cache = Path(args.cache) / f"adoption_stars_{lo_y}_{hi_y}.parquet"
    if cache.exists():
        sdf = pd.read_parquet(cache)
        stars = {r: dict(zip(g["ym"], g["stars"])) for r, g in sdf.groupby("repo")}
        print(f"using cached stars: {len(stars)} repos")
    else:
        stars = _query_stars(all_repos, f"{lo_y}01", f"{hi_y}12", project=args.project)
        pd.DataFrame([(r, ym, n) for r, s in stars.items() for ym, n in s.items()],
                     columns=["repo", "ym", "stars"]).to_parquet(cache, index=False)

    rows = []
    for mk, repos in repos_by_market.items():
        series = {}
        for r in repos:
            for ym, n in stars.get(r, {}).items():
                series[ym] = series.get(ym, 0) + n
        f = reduce_series(series, t0[mk], args.k)
        rows.append({"market": mk, "t0": t0[mk], "n_repos": len(repos),
                     "adoption_level": int(f.level), "adoption_growth": round(f.growth, 4)})
    df = pd.DataFrame(rows)
    df["adoption_level_z"] = zscore([float(np.log1p(v)) for v in df["adoption_level"]])
    df["adoption_growth_z"] = zscore(list(df["adoption_growth"]))
    df = df.sort_values("adoption_level", ascending=False).reset_index(drop=True)
    out = Path(args.labels) / "adoption_dimension.parquet"
    df.to_parquet(out, index=False)
    print(f"\n{'market':22s} {'repos':>5s} {'stars':>9s} {'growth':>7s} {'lvl-z':>6s}")
    for _, r in df.iterrows():
        print(f"{r['market']:22s} {int(r['n_repos']):>5d} {int(r['adoption_level']):>9,} "
              f"{r['adoption_growth']:>+7.3f} {r['adoption_level_z']:>6.2f}")
    print(f"\n{len(df)} markets with an adoption dimension. wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
