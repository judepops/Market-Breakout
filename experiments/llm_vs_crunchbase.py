"""Head-to-head: LLM-scraped funding (Gemini grounding) vs Crunchbase, on a known company roster.

    python -m experiments.llm_vs_crunchbase

For each roster company, pull Crunchbase (total raised + latest round date) and the grounded-Gemini funding,
and compare: do the totals agree (LLM as a reliable source?), and does the LLM find a MORE RECENT round than
Crunchbase has (the freshness/lag the production system cares about)?
"""

from __future__ import annotations

import argparse
import warnings

warnings.filterwarnings("ignore", message="BigQuery Storage module not found")


def _crunchbase(permalinks, project):
    from google.cloud import bigquery
    c = bigquery.Client(project=project or "production-498516")
    sql = """SELECT funded_organization_identifier_permalink AS p,
                    SUM(money_raised_value_usd) AS total, MAX(announced_on) AS last_round
             FROM `production-498516.Crunchbase_Dataset.funding_rounds`
             WHERE funded_organization_identifier_permalink IN UNNEST(@p) GROUP BY p"""
    job = c.query(sql, job_config=bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("p", "STRING", list(permalinks))],
        maximum_bytes_billed=3 * 10**9))
    return {r.p: (r.total, r.last_round) for r in job.result()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="llm-dev-tools")
    ap.add_argument("--n", type=int, default=21)
    ap.add_argument("--project", default=None)
    args = ap.parse_args(argv)

    from marketbrain.labels.gemini_grounding import make_client
    from marketbrain.markets import get_market
    from marketbrain.signals.llm_funding import ground_funding

    mod = get_market(args.market)
    names = list(mod.names())[:args.n]
    perms = list(mod.permalinks())[:args.n]
    cb = _crunchbase(perms, args.project)
    client = make_client()

    print(f"{'company':18s} {'CB total':>10s} {'LLM total':>10s} {'CB last':>11s} {'LLM last':>11s}  newer?")
    fresher = agree = both = 0
    for name, p in zip(names, perms):
        cb_total, cb_last = cb.get(p, (None, None))
        f = ground_funding(name, context=f"an {args.market} company", client=client)
        ll_total, ll_last = f.total_raised_usd, f.latest_round_date
        newer = ""
        if cb_last and ll_last:
            both += 1
            if ll_last > cb_last:
                newer, fresher = f"+{(ll_last - cb_last).days}d", fresher + 1
        if cb_total and ll_total and 0.5 <= ll_total / cb_total <= 2.0:
            agree += 1
        ct = f"${cb_total/1e6:.0f}M" if cb_total else "-"
        lt = f"${ll_total/1e6:.0f}M" if ll_total else "-"
        print(f"{name[:18]:18s} {ct:>10s} {lt:>10s} {str(cb_last):>11s} {str(ll_last):>11s}  {newer}")
    print(f"\ntotals within 2x on {agree}/{len(names)} (LLM as a reliable funding source); "
          f"LLM found a MORE RECENT round on {fresher}/{both} where both dated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
