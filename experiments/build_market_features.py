"""Phase 1 — build the market x feature table (event-time formation cohorts) for the crude predictor.

    python -m experiments.build_market_features --out data/raw/labels

For every catalog market, pull its funding rounds (cached locally), compute the vintage-aligned
formation-cohort features + the later outlier-rate target (markets.features.market_structure), and assemble
one row per market. Immature markets (formation horizon t0+k+w not yet passed) keep their features but a null
target and are excluded from the predictor. Writes market_features.parquet.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import asdict
from pathlib import Path

warnings.filterwarnings("ignore", message="BigQuery Storage module not found")


def _rounds(spec, cache_dir, project):
    """Fetch (and cache) a market's funding rounds, so re-tuning k/w/q does not re-bill BigQuery."""
    import pandas as pd

    cache = Path(cache_dir) / f"{spec.name}.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        df["announced_on"] = pd.to_datetime(df["announced_on"])
        df["year"] = df["announced_on"].dt.year
        return df
    from marketbrain.features.market_structure import fetch_rounds
    df, billed = fetch_rounds([list(g) for g in spec.tag_groups], project=project)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    print(f"  [{spec.name}] fetched {len(df):,} rounds ({billed/1e6:.0f} MB billed)")
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/labels")
    ap.add_argument("--cache", default="data/raw/rounds")
    ap.add_argument("--project", default=None)
    ap.add_argument("--k", type=int, default=3, help="feature-window length (years)")
    ap.add_argument("--w", type=int, default=7, help="outcome-window length (years)")
    ap.add_argument("--q", type=float, default=0.25, help="formation-year quantile for t0")
    ap.add_argument("--min-cohort", type=int, default=20, help="min formation-cohort size for a usable market")
    ap.add_argument("--as-of", type=int, default=2026)
    args = ap.parse_args(argv)

    import pandas as pd

    from marketbrain.features.market_structure import build_features
    from marketbrain.markets.catalog import CATALOG

    rows = []
    for spec in CATALOG:
        df = _rounds(spec, args.cache, args.project)
        if df.empty or df["permalink"].nunique() < 10:
            print(f"  [{spec.name}] too little data ({len(df)} rounds) - skipped")
            continue
        f = build_features(df, spec.name, k=args.k, w=args.w, q=args.q, as_of_year=args.as_of)
        row = asdict(f)
        row["kind"] = spec.kind
        row["coverage"] = ",".join(spec.coverage)
        row["footprint_rich"] = spec.footprint_rich
        rows.append(row)

    table = pd.DataFrame(rows)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    out = Path(args.out) / "market_features.parquet"
    table.to_parquet(out, index=False)

    usable = table[table["mature"] & (table["cohort_size"] >= args.min_cohort)]
    print(f"\n{'market':24s} {'kind':9s} {'cohort':>7s} {'mature':>6s} {'rate':>6s} {'rich':>5s}  coverage")
    for _, r in table.sort_values("footprint_rich", ascending=False).iterrows():
        rate = f"{r['target_rate']:.1%}" if pd.notna(r["target_rate"]) else "  --"
        ok = "yes" if (r["mature"] and r["cohort_size"] >= args.min_cohort) else " no"
        print(f"{r['market']:24s} {r['kind']:9s} {int(r['cohort_size']):>7,} {str(bool(r['mature'])):>6s} "
              f"{rate:>6s} {ok:>5s}  {r['coverage']}")
    rich = usable[usable["footprint_rich"]]
    print(f"\n{len(usable)} usable markets (mature + cohort>={args.min_cohort}); "
          f"of those {len(rich)} are footprint-rich (Phase-2 corpus targets). "
          f"k={args.k} w={args.w} q={args.q}. wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
