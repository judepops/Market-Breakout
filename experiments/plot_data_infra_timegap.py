"""Plot the data-infra feature->outcome TIME GAP: founded -> first >=$500M crossing, per company.

    python -m experiments.plot_data_infra_timegap --out artifacts/figures

Reads the stored data-infra label (`data/raw/labels/data-infra.parquet`) for the crossing dates and queries
Crunchbase for founded dates. The headline of the pivot: a real multi-year gap that llm-dev-tools (outcomes
at the start of its data window) structurally couldn't show.
"""

from __future__ import annotations

import argparse

import pandas as pd


def _founded(permalinks, project):
    from google.cloud import bigquery
    c = bigquery.Client(project=project or "production-498516")
    sql = """SELECT identifier_permalink AS permalink, founded_on_value AS founded
             FROM `production-498516.Crunchbase_Dataset.organizations` WHERE identifier_permalink IN UNNEST(@p)"""
    job = c.query(sql, job_config=bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("p", "STRING", list(permalinks))],
        maximum_bytes_billed=2 * 10**9))
    return {r.permalink: r.founded for r in job.result()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/raw/labels/data-infra.parquet")
    ap.add_argument("--out", default="artifacts/figures")
    ap.add_argument("--project", default=None)
    args = ap.parse_args(argv)

    df = pd.read_parquet(args.labels)
    co = df[(df["level"] == "company") & (df["is_outlier"])].copy()
    founded = _founded(co["permalink"], args.project)

    rows = []
    for _, r in co.iterrows():
        f, m = founded.get(r["permalink"]), r["milestone_date"]
        if f and m:
            rows.append((str(r["name"])[:16], pd.Timestamp(f), pd.Timestamp(m)))
    rows.sort(key=lambda x: x[2])
    gaps = [(m - f).days / 365.25 for _, f, m in rows]
    med = sorted(gaps)[len(gaps) // 2]

    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    from marketbrain.figures.style import C_BASELINE_STRONG, C_GNN, apply_style, save
    apply_style()
    fig, ax = plt.subplots(figsize=(7.1, 0.34 * len(rows) + 1.2))
    for i, (name, f, m) in enumerate(rows):
        ax.plot([f, m], [i, i], color="#cfcfcf", lw=1.4, zorder=1)
        ax.scatter([f], [i], color=C_BASELINE_STRONG, s=16, zorder=2)
        ax.scatter([m], [i], color=C_GNN, s=24, zorder=3)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_title(r"data-infra: founded $\to$ first \$500M crossing "
                 f"(median gap {med:.1f} yrs)")
    ax.scatter([], [], color=C_BASELINE_STRONG, s=16, label="founded")
    ax.scatter([], [], color=C_GNN, s=24, label=r"crossed \$500M")
    ax.legend(loc="lower right", fontsize=6)
    png, _ = save(fig, args.out, "data_infra_timegap")
    print(f"saved {png}  (median gap {med:.1f}y over {len(rows)} outliers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
