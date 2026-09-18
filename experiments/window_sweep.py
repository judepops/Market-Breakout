"""Sensitivity sweep: does the static funding signal stay null across formation/outcome window choices?

    python -m experiments.window_sweep

Recomputes the cohort, a properly windowed outcome (reached $500M WITHIN the outcome window), and the
median-first-round vs rate correlation for every (k, w) in a grid. If it is ~0 everywhere, the null is not an
artefact of choosing 3 and 7.
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
OUTLIER, FLOOR, ASOF = 500e6, 5e6, 2026


def _fetch():
    from google.cloud import bigquery
    from marketbrain.markets.partition import FINE_TAXONOMY
    tags = {t for _, t in FINE_TAXONOMY}
    where = " OR ".join(f"funded_organization_categories_value LIKE '%{t}%'" for t in tags)
    sql = f"""
    SELECT funded_organization_identifier_permalink AS permalink,
           ANY_VALUE(funded_organization_categories_value) AS categories,
           ARRAY_AGG(STRUCT(announced_on AS d, money_raised_value_usd AS amt) ORDER BY announced_on) AS rounds
    FROM `production-498516.Crunchbase_Dataset.funding_rounds`
    WHERE funded_organization_funding_total_value_usd >= {int(FLOOR)}
      AND announced_on IS NOT NULL AND funded_organization_identifier_permalink IS NOT NULL AND ({where})
    GROUP BY permalink
    """
    c = bigquery.Client(project="production-498516")
    job = c.query(sql, job_config=bigquery.QueryJobConfig(maximum_bytes_billed=25 * 10**9))
    df = job.result().to_dataframe()
    print(f"[fetch] {len(df):,} companies; billed {job.total_bytes_billed/1e9:.1f} GB", flush=True)
    return df


def main():
    import numpy as np
    import pandas as pd
    from marketbrain.markets.partition import assign_deterministic, tags_present

    df = _fetch()

    def assign(c):
        a = assign_deterministic(c)
        return None if a is None else (tags_present(c)[0] if a == "AMBIGUOUS" else a)
    df["market"] = df["categories"].map(assign)
    df = df.dropna(subset=["market"])

    # per company: first year, first round size, and the year cumulative raised first crosses $500M (else None)
    fy, fr, cy = [], [], []
    for r in df["rounds"]:
        ds = [pd.Timestamp(x["d"]).year for x in r]
        am = [x["amt"] or 0 for x in r]
        fy.append(ds[0])
        fr.append(next((x["amt"] for x in r if x["amt"]), np.nan))
        cum = np.cumsum(am)
        idx = np.argmax(cum >= OUTLIER) if (np.array(cum) >= OUTLIER).any() else None
        cy.append(ds[idx] if idx is not None else None)
    df["fy"], df["fr"], df["cross"] = fy, fr, cy

    pf = pd.read_parquet("data/raw/labels/partition_features.parquet")
    t0 = {r["market"]: int(r["t0"]) for _, r in pf.iterrows()}

    def sp(a, b):
        return float(pd.Series(a).rank().corr(pd.Series(b).rank()))

    print("\nmedian first round vs outlier rate  (Spearman r, [#mature markets])")
    print(f"{'':6s}" + "".join(f"  w={w:<8d}" for w in (5, 7, 10)))
    for k in (2, 3, 4):
        cells = []
        for w in (5, 7, 10):
            rows = []
            for mk, g in df.groupby("market"):
                if mk not in t0:
                    continue
                start = t0[mk]
                if start + k + w > ASOF:                        # immature under this window
                    continue
                coh = g[(g["fy"] >= start) & (g["fy"] < start + k)]
                if len(coh) < 8:
                    continue
                out = coh["cross"].apply(lambda c: c is not None and c <= start + k + w)
                rows.append({"rate": out.mean(), "mf": coh["fr"].median()})
            m = pd.DataFrame(rows).dropna()
            cells.append(f"{sp(m['mf'], m['rate']):+.2f} [{len(m):2d}]" if len(m) >= 6 else "  n/a   ")
        print(f"k={k}:  " + "  ".join(cells))
    print("\n(if r is ~0 across the whole grid, the null is not an artefact of choosing k=3, w=7)")


if __name__ == "__main__":
    main()
