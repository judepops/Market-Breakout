"""Build the DISJOINT-market dataset: assign each >=$5M company to one market, recompute features + rate.

    python -m experiments.build_partition

One FLAT funding_rounds scan (one row per round - the nested ARRAY_AGG hung on the slow REST endpoint) for
every >=$5M company in the taxonomy -> assign each company to a single market (deterministic for 0/1 tags;
Gemini picks the primary market for ambiguous multi-tag companies) -> per market run the existing
market_structure.build_features. The markets are independent by construction.

Filters are applied AFTER: this computes + saves EVERY market (with its mature flag / cohort_size / rate as
columns); the maturity + cohort-size filters are applied later, at predict/analysis time, so the full picture
is visible first. The only build-time guard is "enough companies to compute a cohort at all".
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import asdict
from pathlib import Path

warnings.filterwarnings("ignore", message="BigQuery Storage module not found")


def _log(msg):
    print(msg, flush=True)


def _fetch_rounds(project):
    """FLAT: one row per (company round) for >=$5M companies in the taxonomy (fast REST download)."""
    from google.cloud import bigquery

    from marketbrain.markets.partition import FINE_TAXONOMY
    tags = {t for _, t in FINE_TAXONOMY}
    where = " OR ".join(f"funded_organization_categories_value LIKE '%{t}%'" for t in tags)
    sql = f"""
    SELECT funded_organization_identifier_permalink AS permalink,
           funded_organization_identifier_value AS name,
           funded_organization_categories_value AS categories,
           announced_on, investment_type,
           money_raised_value_usd AS raised_usd,
           post_money_valuation_value_usd AS post_usd
    FROM `production-498516.Crunchbase_Dataset.funding_rounds`
    WHERE funded_organization_funding_total_value_usd >= 5000000
      AND announced_on IS NOT NULL AND funded_organization_identifier_permalink IS NOT NULL
      AND ({where})
    """
    client = bigquery.Client(project=project or "production-498516")
    job = client.query(sql, job_config=bigquery.QueryJobConfig(maximum_bytes_billed=15 * 10**9))
    _log("[partition] querying funding_rounds (flat) ...")
    df = job.result().to_dataframe()
    _log(f"[partition] {df['permalink'].nunique():,} companies, {len(df):,} rounds; "
         f"billed {job.total_bytes_billed/1e9:.1f} GB")
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/raw/labels")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--w", type=int, default=7)
    ap.add_argument("--q", type=float, default=0.25)
    ap.add_argument("--batch", type=int, default=30)
    ap.add_argument("--no-llm", action="store_true",
                    help="assign ambiguous companies by priority (most-specific tag) instead of Gemini - instant")
    ap.add_argument("--project", default=None)
    args = ap.parse_args(argv)

    import pandas as pd

    from marketbrain.enrichment.classify import classify_batch
    from marketbrain.features.market_structure import build_features
    from marketbrain.labels.gemini_grounding import make_client
    from marketbrain.markets.partition import assign_deterministic, tags_present

    df = _fetch_rounds(args.project)
    df["announced_on"] = pd.to_datetime(df["announced_on"])
    df["year"] = df["announced_on"].dt.year

    # one row per company for assignment
    comp = df.groupby("permalink").agg(cname=("name", "first"), categories=("categories", "first"))
    assignment, ambiguous = {}, []
    for perm, row in comp.iterrows():
        a = assign_deterministic(row["categories"])
        if a is None:
            continue
        if a == "AMBIGUOUS":
            ambiguous.append((perm, row["cname"] or perm, tags_present(row["categories"]), row["categories"] or ""))
        else:
            assignment[perm] = a
    _log(f"[partition] deterministic: {len(assignment):,} single-tag; ambiguous (LLM): {len(ambiguous):,}")

    if args.no_llm:
        for perm, _n, cands, _cat in ambiguous:
            assignment[perm] = cands[0]                 # priority-order: most-specific tag
        _log("[partition] --no-llm: ambiguous assigned by priority (most-specific tag)")
    else:
        client = make_client()
        for i in range(0, len(ambiguous), args.batch):
            chunk = ambiguous[i:i + args.batch]
            try:
                res = classify_batch([(n, c, cat) for _, n, c, cat in chunk], client=client)
            except Exception as e:  # noqa: BLE001
                _log(f"  classify batch {i}: {e}")
                res = {}
            for j, (perm, _n, cands, _cat) in enumerate(chunk):
                assignment[perm] = res.get(j, cands[0])  # LLM choice, else fall back to most-specific tag
            if (i // args.batch) % 10 == 0:
                _log(f"  classified {min(i + args.batch, len(ambiguous)):,}/{len(ambiguous):,} ambiguous")

    df["market"] = df["permalink"].map(assignment)
    rdf = df.dropna(subset=["market"])

    # compute features for EVERY market (filters applied later, not here)
    out_rows = []
    for mk, g in rdf.groupby("market"):
        if g["permalink"].nunique() < 5:           # cannot form a meaningful cohort - skip (not a "filter")
            continue
        f = build_features(g, mk, k=args.k, w=args.w, q=args.q, as_of_year=2026)
        out_rows.append(asdict(f))
    table = pd.DataFrame(out_rows)
    out = Path(args.labels) / "partition_features.parquet"
    table.to_parquet(out, index=False)

    _log(f"\n{len(table)} disjoint markets computed (ALL saved; filter later). "
         f"For reference: {int((table['mature'] & (table['cohort_size'] >= 30)).sum())} are mature + cohort>=30.")
    _log(f"{'market':24s} {'cohort':>7s} {'mature':>6s} {'rate':>6s}")
    for _, r in table.sort_values("target_rate", ascending=False, na_position="last").iterrows():
        rate = f"{r['target_rate']:.1%}" if pd.notna(r["target_rate"]) else "  --"
        _log(f"{r['market']:24s} {int(r['cohort_size']):>7,} {str(bool(r['mature'])):>6s} {rate:>6s}")
    _log(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
