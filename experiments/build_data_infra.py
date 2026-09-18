"""Feasibility spike: build the data-infra label and show the feature -> outcome TIME GAP.

    python -m experiments.build_data_infra

Runs the $500M success label (Crunchbase + Gemini) for the `data-infra` market through the SAME machinery as
llm-dev-tools, then reports, per outlier company, founded-date -> first >=$500M-crossing date. The headline:
a real multi-year gap (outcomes years after the emergence/adoption window) that llm-dev-tools couldn't show.
Needs GOOGLE_APPLICATION_CREDENTIALS + GEMINI_API_KEY.
"""

from __future__ import annotations

import argparse

from marketbrain.labels import build_label, store
from marketbrain.markets import data_infra


def _founded_dates(permalinks, project):
    from google.cloud import bigquery
    c = bigquery.Client(project=project or "production-498516")
    sql = """SELECT identifier_permalink AS permalink, founded_on_value AS founded
             FROM `production-498516.Crunchbase_Dataset.organizations`
             WHERE identifier_permalink IN UNNEST(@p)"""
    job = c.query(sql, job_config=bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("p", "STRING", list(permalinks))],
        maximum_bytes_billed=2 * 10**9))
    return {r.permalink: r.founded for r in job.result()}


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description="Data-infra feasibility spike: label + feature->outcome gap.")
    ap.add_argument("--root", default="data/raw/labels")
    ap.add_argument("--no-ground", action="store_true")
    ap.add_argument("--project", default=None)
    args = ap.parse_args(argv)

    market = "data-infra"
    label, outcomes = build_label.run(market, ground=not args.no_ground, project=args.project)
    store.write(label, outcomes, market=market, root=args.root)

    founded = _founded_dates(data_infra.permalinks(), args.project)
    print(f"\n=== TIME GAP: founded -> first >=$500M crossing ({market}) ===")
    gaps = []
    for o in sorted(outcomes, key=lambda o: o.milestone_date or __import__("datetime").date.max):
        if not (o.is_outlier and o.milestone_date):
            continue
        f = founded.get(o.permalink)
        gy = f"{(o.milestone_date - f).days / 365.25:4.1f}y" if f else "  n/a"
        if f:
            gaps.append((o.milestone_date - f).days / 365.25)
        val = f"${o.value_usd / 1e9:.1f}B" if o.value_usd else "n/a"
        print(f"  {str(o.name)[:16]:16s} founded {str(f):>10s} -> crossed {o.milestone_date} "
              f"({o.metric:9s} {val:7s})  gap {gy}")
    n = sum(1 for o in outcomes if o.is_outlier)
    if gaps:
        med = sorted(gaps)[len(gaps) // 2]
        print(f"\n  {n}/{len(outcomes)} outliers; median feature->outcome gap = {med:.1f} years "
              f"(vs llm-dev-tools ~0: outcomes at the START of the data window).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
