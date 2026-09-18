"""Outlier-RATE across software sub-markets — the predictable target for the multi-market build.

    python -m experiments.outlier_rates --out artifacts/figures

The pivot's core claim: a *market's* quality is its **propensity to produce >=$500M outliers** — a RATE, not
a saturated yes/no. This computes, per developer-facing-software Crunchbase category tag, the outlier rate.

**Denominator (2026-06-16 meeting — Ben + the team):** the *fundable* universe, i.e. companies that raised
>=$5M, NOT literally every company in the tag. Ben wanted the raw outlier rate (needs the non-winners);
the team wanted a $5M minimum-funding floor (seed data is too sparse to source cleanly) + total funding $ as a
market-potential proxy. These reconcile:
    outlier rate = (companies >= $500M) / (companies that raised >= $5M)          [Ben]
    total funding $ per market                                                     [the team]
The $5M floor is a cleaner, more meaningful denominator than all-companies (most of which never raised and
carry no signal). An outlier = >=$500M via total raised OR post-money valuation (M&A omitted — needs the
acquisitions join; this is a full-universe proxy, the per-company label adds M&A + grounding).

Uses the full Crunchbase universe per tag (one cost-capped scan of `organizations`). NOT a per-market
pipeline — a target table. For contrast the script also prints the old all-companies rate.
"""

from __future__ import annotations

import argparse
import re

THRESHOLD_USD = 500_000_000      # >=$500M = outlier
FUNDABLE_FLOOR_USD = 5_000_000   # >=$5M raised = "fundable" universe (the meeting's denominator)
# Real Crunchbase category tags (verified to return data — avoids the No-Code/Vector-Search zero trap).
# Developer-facing / data / infrastructure software, where the signal stack has power.
CATEGORIES = [
    "Developer Tools", "Database", "DevOps", "Cloud Infrastructure", "Open Source", "Data Storage",
    "Machine Learning", "Analytics", "Cyber Security", "Internet of Things", "Blockchain",
    "Virtual Reality",
]
_TAG_OK = re.compile(r"^[A-Za-z0-9 /+&-]+$")


def fetch_rates(categories, *, threshold_usd=THRESHOLD_USD, floor_usd=FUNDABLE_FLOOR_USD,
                project=None, max_bytes=5 * 10**9):
    """Per category: (tag, n_all, n_fundable, n_outlier, rate_all, rate_fundable, total_funding_usd).

    rate_fundable = n_outlier / n_fundable is the headline (the meeting's agreed metric); rate_all keeps the
    old all-companies number for contrast. total_funding_usd = Σ funding over the fundable universe.
    """
    from google.cloud import bigquery

    for cat in categories:
        if not _TAG_OK.match(cat):
            raise ValueError(f"invalid category tag {cat!r}")
    client = bigquery.Client(project=project or "production-498516")
    is_out = "(funding_total_value_usd >= @thr OR valuation_value_usd >= @thr)"
    in_universe = "funding_total_value_usd >= @floor"
    sel = ",\n".join(
        f"COUNTIF(categories_value LIKE '%{c}%') AS c{i}_n_all, "
        f"COUNTIF(categories_value LIKE '%{c}%' AND {in_universe}) AS c{i}_n_fund, "
        f"COUNTIF(categories_value LIKE '%{c}%' AND {in_universe} AND {is_out}) AS c{i}_o, "
        f"SUM(IF(categories_value LIKE '%{c}%' AND {in_universe}, funding_total_value_usd, 0)) AS c{i}_fund"
        for i, c in enumerate(categories)
    )
    sql = f"SELECT {sel} FROM `production-498516.Crunchbase_Dataset.organizations`"
    cfg = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("thr", "INT64", int(threshold_usd)),
            bigquery.ScalarQueryParameter("floor", "INT64", int(floor_usd)),
        ],
        maximum_bytes_billed=max_bytes)
    job = client.query(sql, job_config=cfg)
    r = list(job.result())[0]
    print(f"[rates] billed {job.total_bytes_billed / 1e6:.0f} MB")
    rows = []
    for i, c in enumerate(categories):
        n_all = int(getattr(r, f"c{i}_n_all"))
        n_fund = int(getattr(r, f"c{i}_n_fund"))
        o = int(getattr(r, f"c{i}_o"))
        fund = int(getattr(r, f"c{i}_fund") or 0)
        rate_all = o / n_all if n_all else 0.0          # old all-companies rate (numerator over fundable)
        rate_fund = o / n_fund if n_fund else 0.0       # the meeting's metric
        rows.append((c, n_all, n_fund, o, rate_all, rate_fund, fund))
    return sorted(rows, key=lambda x: x[5], reverse=True)  # sort by the fundable rate


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Outlier-rate per software sub-market (>=$5M fundable universe).")
    ap.add_argument("--out", default="artifacts/figures")
    ap.add_argument("--project", default=None)
    ap.add_argument("--floor", type=int, default=FUNDABLE_FLOOR_USD, help="fundable-universe funding floor USD")
    args = ap.parse_args(argv)

    rows = fetch_rates(CATEGORIES, floor_usd=args.floor, project=args.project)
    print(f"\nfundable universe = raised >= ${args.floor/1e6:.0f}M  |  outlier = >= $500M raised or valuation\n")
    print(f"{'sub-market (Crunchbase tag)':28s} {'all':>8s} {'>=$5M':>8s} {'>=$500M':>8s} "
          f"{'rate/all':>9s} {'rate/$5M':>9s} {'total $':>9s}")
    for c, n_all, n_fund, o, rate_all, rate_fund, fund in rows:
        print(f"{c:28s} {n_all:>8,} {n_fund:>8,} {o:>8,} {rate_all:>8.2%} {rate_fund:>8.2%} "
              f"{fund/1e9:>7.1f}B")
    lo, hi = rows[-1][5], rows[0][5]
    if lo:
        print(f"\nfundable outlier rate spans {lo:.2%} ({rows[-1][0]}) -> {hi:.2%} ({rows[0][0]}) "
              f"= {hi / lo:.1f}x variation — the predictable target (vs the saturated curated rosters).")
    # the two PoC markets used CURATED winner-rosters and saturated — shown for contrast (off any linear scale)
    curated = [("llm-dev-tools (curated)", 21, 12, 12 / 21),
               ("data-infra (curated)", 21, 21, 1.0)]
    _plot(rows, curated, args.out)
    return 0


def _plot(universe, curated, out):
    """Log-scale lollipop: per-category outlier rate over the >=$5M FUNDABLE universe (the meeting's metric),
    with total funding $ annotated, AND the saturated curated PoC rosters for contrast on the same axis."""
    import matplotlib.pyplot as plt

    from marketbrain.figures.style import C_BASELINE_STRONG, C_GNN, apply_style, save

    apply_style()
    # universe rows: (cat, n_all, n_fund, o, rate_all, rate_fund, fund)
    uni = [(c, rate_fund, fund) for (c, _na, _nf, _o, _ra, rate_fund, fund) in universe]
    items = ([(c, r, f, "universe") for (c, r, f) in uni]
             + [(c, r, None, "curated") for (c, _n, _o, r) in curated])
    items.sort(key=lambda x: x[1])  # ascending rate -> lowest at bottom
    fig, ax = plt.subplots(figsize=(7.4, 0.4 * len(items) + 1.3))
    for i, (_, rate, fund, kind) in enumerate(items):
        color = C_BASELINE_STRONG if kind == "curated" else C_GNN
        ax.plot([0.01, rate * 100], [i, i], color="#e2e2e6", lw=0.9, zorder=1)
        ax.scatter([rate * 100], [i], color=color, s=34, zorder=3)
        if kind == "curated":
            lab = f"  {rate:.0%}"
        else:
            lab = f"  {rate:.2%}  (\\${fund/1e9:.0f}B)" if fund else f"  {rate:.2%}"
        ax.text(rate * 100, i, lab, va="center", fontsize=6.5, color=color,
                fontweight="bold" if kind == "curated" else "normal")
    ax.set_xscale("log")
    ax.set_xlim(0.01, 300)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([x[0] for x in items])
    ax.set_xlabel(r"\$500M outlier rate over the \$5M-fundable universe (log scale, %)")
    ax.set_title(r"Outlier rate per sub-market: \$500M / (raised $\geq$ \$5M)  ·  total funding in (  )")
    ax.scatter([], [], color=C_GNN, s=34, label="fundable universe (the target; total funding annotated)")
    ax.scatter([], [], color=C_BASELINE_STRONG, s=34, label="curated PoC roster (saturated, useless)")
    ax.legend(loc="upper left", fontsize=6.5)
    png, _ = save(fig, out, "outlier_rates_software")
    print(f"saved {png}")


if __name__ == "__main__":
    raise SystemExit(main())
