"""Phase 2 - build the INNOVATION dimension (patents) across all patent-covered mature markets.

ONE combined BigQuery scan: monthly patent counts per market (assignee-matched against each market's top
companies), over the global date span. Then each market's series is windowed to its own formation window
[t0, t0+k] and collapsed to {level, growth} (signal_features), and z-scored across markets. Writing this as a
single scan keeps cost at ~one patents query (~7 GB) instead of one-per-market.

    python -m experiments.build_innovation
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="BigQuery Storage module not found")


def _combined_query(info, lo_year, hi_year, *, project, max_bytes=30 * 10**9):
    """info = [(market, regex)]; return {market -> {'YYYY-MM' -> patent_count}} via one scan."""
    from google.cloud import bigquery

    client = bigquery.Client(project=project or "production-498516")
    counts = ",\n".join(
        f"COUNTIF(EXISTS(SELECT 1 FROM UNNEST(assignee_harmonized) a "
        f"WHERE REGEXP_CONTAINS(REGEXP_REPLACE(UPPER(a.name), r'[.,]', ''), @p{i}))) AS m{i}"
        for i in range(len(info)))
    sql = f"""
    SELECT DIV(public_d, 100) AS yyyymm, {counts}
    FROM (
      SELECT IF(publication_date > 0, publication_date, grant_date) AS public_d, assignee_harmonized
      FROM `patents-public-data.patents.publications`
    )
    WHERE public_d BETWEEN @lo AND @hi
    GROUP BY yyyymm ORDER BY yyyymm
    """
    params = [bigquery.ScalarQueryParameter("lo", "INT64", lo_year * 10000 + 101),
              bigquery.ScalarQueryParameter("hi", "INT64", hi_year * 10000 + 1231)]
    params += [bigquery.ScalarQueryParameter(f"p{i}", "STRING", rgx) for i, (_, rgx) in enumerate(info)]
    dry = client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params, dry_run=True))
    print(f"[innovation] dry-run {dry.total_bytes_processed / 1e9:.1f} GB for {len(info)} markets "
          f"({lo_year}-{hi_year})")
    job = client.query(sql, job_config=bigquery.QueryJobConfig(
        query_parameters=params, maximum_bytes_billed=max_bytes))
    out = {m: {} for m, _ in info}
    for r in job.result():
        ym = int(r.yyyymm)
        mon = f"{ym // 100:04d}-{ym % 100:02d}"
        for i, (m, _) in enumerate(info):
            v = getattr(r, f"m{i}")
            if v:
                out[m][mon] = int(v)
    print(f"[innovation] billed {job.total_bytes_billed / 1e9:.1f} GB")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/raw/labels")
    ap.add_argument("--cache", default="data/raw/rounds")
    ap.add_argument("--figdir", default="artifacts/figures")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--top-n", type=int, default=60)
    ap.add_argument("--project", default=None)
    args = ap.parse_args(argv)

    import numpy as np
    import pandas as pd

    from marketbrain.features.signal_features import reduce_series, zscore
    from marketbrain.markets.catalog import CATALOG
    from marketbrain.signals.patents import _assignee_regex

    ft = pd.read_parquet(Path(args.labels) / "market_features.parquet").set_index("market")
    # run ALL mature markets, not just the patent-covered ones: same scan cost, and it MEASURES the patent
    # footprint everywhere (confirming software markets are really ~0, instead of assuming it from coverage).
    specs = [m for m in CATALOG if m.name in ft.index and bool(ft.loc[m.name, "mature"])]
    print(f"{len(specs)} mature markets (all, not just patent-covered); resolving assignees ...")

    info, t0s = [], {}
    for m in specs:
        names = m.assignees(project=args.project, top_n=args.top_n)
        if names:
            info.append((m.name, _assignee_regex(names)))
            t0s[m.name] = int(ft.loc[m.name, "t0"])

    cache = Path(args.cache) / f"innovation_series_k{args.k}_n{args.top_n}.parquet"
    if cache.exists():
        print(f"using cached series: {cache}")
        sdf = pd.read_parquet(cache)
        series = {mk: dict(zip(g["ym"], g["patents"])) for mk, g in sdf.groupby("market")}
    else:
        lo, hi = min(t0s.values()), max(t0s.values()) + args.k
        series = _combined_query(info, lo, hi, project=args.project)
        rows = [(mk, ym, n) for mk, s in series.items() for ym, n in s.items()]
        pd.DataFrame(rows, columns=["market", "ym", "patents"]).to_parquet(cache, index=False)

    est = {m.name: ("pat" in m.coverage) for m in CATALOG}
    rows = []
    for mk, _ in info:
        f = reduce_series(series.get(mk, {}), t0s[mk], args.k)
        rows.append({"market": mk, "t0": t0s[mk], "innovation_level": f.level,
                     "innovation_growth": round(f.growth, 4), "est_patent_market": est.get(mk, False)})
    df = pd.DataFrame(rows)
    # normalize: log-compress the huge level range, then z-score across markets; growth z-scored directly
    df["innovation_level_z"] = zscore([float(np.log1p(v)) for v in df["innovation_level"]])
    df["innovation_growth_z"] = zscore(list(df["innovation_growth"]))
    df = df.sort_values("innovation_level", ascending=False).reset_index(drop=True)
    out = Path(args.labels) / "innovation_dimension.parquet"
    df.to_parquet(out, index=False)

    print(f"\n{'market':22s} {'window':>9s} {'patents':>10s} {'growth/mo':>9s} {'level-z':>7s}")
    for _, r in df.iterrows():
        print(f"{r['market']:22s} {int(r['t0'])}-{int(r['t0'])+args.k-1:>4} {int(r['innovation_level']):>10,} "
              f"{r['innovation_growth']:>+9.3f} {r['innovation_level_z']:>7.2f}")
    print(f"\n{len(df)} markets with an innovation dimension. wrote {out}")
    _plot(df, args.figdir)
    return 0


def _plot(df, figdir):
    import matplotlib.pyplot as plt

    from marketbrain.figures.style import C_GNN, apply_style, save
    apply_style()
    d = df.sort_values("innovation_level")
    fig, ax = plt.subplots(figsize=(7.0, 0.34 * len(d) + 1.0))
    ax.barh(range(len(d)), [max(v, 1) for v in d["innovation_level"]], color=C_GNN)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d["market"], fontsize=6.5)
    ax.set_xscale("log")
    ax.set_xlabel("patents in the formation window (log scale)")
    ax.set_title("Innovation dimension: patents per market over its formation window")
    png, _ = save(fig, figdir, "innovation_dimension")
    print(f"saved {png}")


if __name__ == "__main__":
    raise SystemExit(main())
