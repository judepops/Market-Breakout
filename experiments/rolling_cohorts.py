"""Rolling cohorts: many leakage-safe windows per market, the clean way to get more power than one cohort.

    python -m experiments.rolling_cohorts

Single cohort (~5k) showed r~0 (underpowered); all companies (~27k) showed r~0.45 (biased by vintage).
Rolling cohorts keep the leakage-safe windowing but use EVERY valid 3-year formation block per market, so we
get many more clean data points. Does the powered, clean number land near 0.45?
"""
from __future__ import annotations

import warnings
import pathlib

warnings.filterwarnings("ignore")
OUTLIER, FLOOR, K, W, ASOF = 500e6, 5e6, 3, 7, 2026
FIG = pathlib.Path("artifacts/figures")
RED, SLATE, GREY = "#8b0000", "#2c3e50", "#b9bcc4"


def _fetch():
    from google.cloud import bigquery
    from marketbrain.markets.partition import FINE_TAXONOMY
    tags = {t for _, t in FINE_TAXONOMY}
    where = " OR ".join(f"funded_organization_categories_value LIKE '%{t}%'" for t in tags)
    sql = f"""
    SELECT funded_organization_identifier_permalink AS permalink,
           ANY_VALUE(funded_organization_categories_value) AS categories,
           MAX(funded_organization_funding_total_value_usd) AS total,
           MIN(announced_on) AS first_date,
           ARRAY_AGG(money_raised_value_usd ORDER BY announced_on LIMIT 1)[OFFSET(0)] AS first_round
    FROM `production-498516.Crunchbase_Dataset.funding_rounds`
    WHERE funded_organization_funding_total_value_usd >= {int(FLOOR)}
      AND announced_on IS NOT NULL AND funded_organization_identifier_permalink IS NOT NULL AND ({where})
    GROUP BY permalink
    """
    c = bigquery.Client(project="production-498516")
    job = c.query(sql, job_config=bigquery.QueryJobConfig(maximum_bytes_billed=20 * 10**9))
    df = job.result().to_dataframe()
    print(f"[fetch] {len(df):,} companies; billed {job.total_bytes_billed/1e9:.1f} GB", flush=True)
    return df


def main():
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from marketbrain.markets.partition import assign_deterministic, tags_present

    df = _fetch()
    df["first_year"] = pd.to_datetime(df["first_date"]).dt.year
    df["is_outlier"] = df["total"] >= OUTLIER

    def assign(c):
        a = assign_deterministic(c)
        return None if a is None else (tags_present(c)[0] if a == "AMBIGUOUS" else a)
    df["market"] = df["categories"].map(assign)
    df = df.dropna(subset=["market", "first_year", "first_round"])

    # rolling 3-year formation blocks per market, outcome window must have elapsed by ASOF
    pts = []
    for mk, g in df.groupby("market"):
        for y in range(2000, ASOF - K - W + 1):                 # y+K+W <= ASOF
            coh = g[(g["first_year"] >= y) & (g["first_year"] < y + K)]
            if len(coh) >= 10:
                pts.append({"market": mk, "start": y, "n": len(coh),
                            "rate": coh["is_outlier"].mean(), "medfirst": coh["first_round"].median()})
    p = pd.DataFrame(pts).dropna(subset=["medfirst"])
    print(f"[rolling] {len(p)} clean cohort-windows across {p['market'].nunique()} markets "
          f"(vs 44 single cohorts); {int(p['n'].sum()):,} company-slots")

    def sp(a, b):
        return float(pd.Series(a).rank().corr(pd.Series(b).rank()))

    def spw(a, b, w):
        a, b, w = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy(), np.asarray(w, float)
        am, bm = np.average(a, weights=w), np.average(b, weights=w)
        return float(np.average((a - am) * (b - bm), weights=w)
                     / np.sqrt(np.average((a - am)**2, weights=w) * np.average((b - bm)**2, weights=w)))
    r, rw = sp(p["medfirst"], p["rate"]), spw(p["medfirst"], p["rate"], p["n"])
    # within-market (demeaned) to remove cross-market confounding, using only time variation
    pp = p.copy()
    pp["mf_d"] = pp.groupby("market")["medfirst"].transform(lambda s: s.rank() - s.rank().mean())
    pp["rt_d"] = pp.groupby("market")["rate"].transform(lambda s: s.rank() - s.rank().mean())
    multi = pp.groupby("market").filter(lambda d: len(d) >= 3)
    r_within = float(np.corrcoef(multi["mf_d"], multi["rt_d"])[0, 1]) if len(multi) > 10 else float("nan")
    print(f"[rolling] median first round vs rate:  unweighted r={r:+.2f}   size-weighted r={rw:+.2f}")
    print(f"[rolling] within-market (time variation only) r={r_within:+.2f}  (n={len(multi)} windows in "
          f"{multi['market'].nunique()} markets with >=3 windows)")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, ax = plt.subplots(figsize=(7.0, 4.8), dpi=150)
    sz = p["n"] / p["n"].max() * 220 + 14
    ax.scatter(p["medfirst"] / 1e6, p["rate"] * 100, s=sz, color=RED, alpha=.45, edgecolor="white", linewidth=.5, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("median first round ($M, log)", fontsize=9, color=SLATE)
    ax.set_ylabel("outlier rate (%)", fontsize=9, color=SLATE)
    ax.set_title(f"Rolling cohorts: clean and powered\n{len(p)} leakage safe windows  ·  "
                 f"r = {r:+.2f} unweighted, {rw:+.2f} size weighted  ·  bubble = window size",
                 fontsize=10, color=SLATE, fontweight="bold", loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GREY)
    ax.spines["bottom"].set_color(GREY)
    ax.tick_params(colors=SLATE, length=0)
    ax.grid(color="#eeeef1", zorder=0)
    plt.tight_layout()
    fig.savefig(FIG / "rolling_cohorts.png", bbox_inches="tight", facecolor="white")
    print(f"wrote {FIG/'rolling_cohorts.png'}")


if __name__ == "__main__":
    main()
