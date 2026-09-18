"""Step 2c: MONTHLY GitHub adoption panel via GH Archive (point-in-time stars), the third leading signal.

    python -m experiments.monthly_adoption

GH Archive WatchEvent = a star, natively timestamped (no stale-star bug). One capped BigQuery scan of
WatchEvents for the markets' repos -> stars per (repo, month) -> per (market, month), macro-normalised by the
corpus total that month. Each repo assigned to one partition market (first occurrence). Saves the raw
repo-month so re-aggregation is free.
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
OUT = Path("data/raw/timeseries")


def main():
    import pandas as pd
    from google.cloud import bigquery
    OUT.mkdir(parents=True, exist_ok=True)

    partition = set(pd.read_parquet("data/raw/labels/partition_features.parquet")["market"])
    repo_market: dict[str, str] = {}
    for mk, repos in json.loads(Path("data/raw/labels/github_repos.json").read_text()).items():
        if mk not in partition:
            continue
        for r in (repos or []):
            repo_market.setdefault(r, mk)
    repos = sorted(repo_market)
    print(f"[adoption] {len(repos)} repos across {len(set(repo_market.values()))} partition markets", flush=True)

    sql = """
    SELECT repo.name AS repo, DATE_TRUNC(DATE(created_at), MONTH) AS m, COUNT(*) AS stars
    FROM `githubarchive.month.20*`
    WHERE type = 'WatchEvent' AND repo.name IN UNNEST(@repos)
    GROUP BY repo, m
    """
    c = bigquery.Client(project="production-498516")
    jc = bigquery.QueryJobConfig(
        maximum_bytes_billed=1_000_000_000_000,
        query_parameters=[bigquery.ArrayQueryParameter("repos", "STRING", repos)])
    df = c.query(sql, job_config=jc).result().to_dataframe()
    df.to_parquet(OUT / "monthly_adoption_raw.parquet", index=False)
    print(f"[adoption] {len(df):,} (repo, month) rows", flush=True)

    df["market"] = df["repo"].map(repo_market)
    df["m"] = pd.to_datetime(df["m"])
    df["year"], df["month"] = df["m"].dt.year, df["m"].dt.month
    panel = df.groupby(["market", "year", "month"], as_index=False)["stars"].sum()
    panel = panel[(panel["year"] >= 2012) & (panel["year"] <= 2025)]
    tot = panel.groupby(["year", "month"])["stars"].transform("sum")
    panel["share"] = panel["stars"] / tot
    panel.to_parquet(OUT / "monthly_adoption.parquet", index=False)
    print(f"[adoption] wrote {len(panel)} rows across {panel['market'].nunique()} markets -> "
          f"{OUT/'monthly_adoption.parquet'}", flush=True)


if __name__ == "__main__":
    main()
