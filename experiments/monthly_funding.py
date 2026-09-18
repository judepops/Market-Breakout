"""Step 1 of the lead-lag plan: the monthly funding panel (the TARGET series), plus the disjointness pre-flight.

    python -m experiments.monthly_funding

One flat funding_rounds scan -> assign each company to ONE market (the disjoint partition) -> confirm
disjointness (each company contributes to exactly one market) -> aggregate $ raised to (market, month) ->
macro-normalise by total monthly funding (the secular-growth / vintage guard). Writes the panel + a smoke
figure of a few markets' funding over time.
"""
from __future__ import annotations
import warnings
import pathlib

warnings.filterwarnings("ignore")
FLOOR = 5e6
OUT = pathlib.Path("data/raw/timeseries")
FIG = pathlib.Path("artifacts/figures")
RED, SLATE, GREY = "#8b0000", "#2c3e50", "#b9bcc4"


def _fetch():
    from google.cloud import bigquery
    from marketbrain.markets.partition import FINE_TAXONOMY
    tags = {t for _, t in FINE_TAXONOMY}
    where = " OR ".join(f"funded_organization_categories_value LIKE '%{t}%'" for t in tags)
    sql = f"""
    SELECT funded_organization_identifier_permalink AS permalink,
           funded_organization_categories_value AS categories,
           announced_on, money_raised_value_usd AS raised
    FROM `production-498516.Crunchbase_Dataset.funding_rounds`
    WHERE funded_organization_funding_total_value_usd >= {int(FLOOR)}
      AND announced_on IS NOT NULL AND funded_organization_identifier_permalink IS NOT NULL AND ({where})
    """
    c = bigquery.Client(project="production-498516")
    job = c.query(sql, job_config=bigquery.QueryJobConfig(maximum_bytes_billed=20 * 10**9))
    df = job.result().to_dataframe()
    print(f"[fetch] {len(df):,} rounds, {df['permalink'].nunique():,} companies; "
          f"billed {job.total_bytes_billed/1e9:.1f} GB", flush=True)
    return df


def main():
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from marketbrain.markets.partition import assign_deterministic, tags_present

    OUT.mkdir(parents=True, exist_ok=True)
    df = _fetch()

    # assign each company to ONE market (deterministic; priority-order for ambiguous) -- at company level
    comp = df.groupby("permalink")["categories"].first()

    def assign(c):
        a = assign_deterministic(c)
        return None if a is None else (tags_present(c)[0] if a == "AMBIGUOUS" else a)
    perm_market = {p: assign(c) for p, c in comp.items()}

    # ---- DISJOINTNESS PRE-FLIGHT (the locked guard before any aggregate stat) ----
    assigned = {p: m for p, m in perm_market.items() if m}
    multi = sum(1 for m in perm_market.values() if isinstance(m, list))    # should be 0 by construction
    print(f"[pre-flight] {len(assigned):,} companies assigned to exactly one market; "
          f"{len(perm_market)-len(assigned):,} dropped (no taxonomy tag); companies in >1 market: {multi}")
    print("[pre-flight] DISJOINT by construction (one market per company) -> the shuffle null is valid.")

    df["market"] = df["permalink"].map(assigned)
    df = df.dropna(subset=["market", "raised"])
    df["month"] = pd.to_datetime(df["announced_on"]).values.astype("datetime64[M]")

    # monthly $ raised per market
    panel = df.groupby(["market", "month"], as_index=False)["raised"].sum().rename(columns={"raised": "usd"})
    # macro-normalise by total funding that month (the secular-growth guard)
    tot = panel.groupby("month")["usd"].transform("sum")
    panel["usd_share"] = panel["usd"] / tot
    panel.to_parquet(OUT / "monthly_funding.parquet", index=False)
    print(f"[panel] {len(panel):,} (market, month) rows across {panel['market'].nunique()} markets, "
          f"{panel['month'].min().date()} to {panel['month'].max().date()} -> {OUT/'monthly_funding.parquet'}")

    # ---- smoke figure: a few markets' funding over time (12-month rolling, share) ----
    show = ["fintech", "machine-learning", "developer-tools", "semiconductors", "robotics"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, ax = plt.subplots(figsize=(8.6, 4.4), dpi=150)
    cols = [RED, SLATE, "#c0863a", "#4a7a6a", "#7a6b8b"]
    for mk, col in zip(show, cols):
        s = panel[panel["market"] == mk].set_index("month")["usd_share"].sort_index()
        s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="MS")).fillna(0)
        roll = s.rolling(12, min_periods=3).mean() * 100
        ax.plot(roll.index, roll.values, color=col, lw=2.0, label=mk.replace("-", " "), zorder=3)
    ax.set_xlabel("month", fontsize=9, color=SLATE)
    ax.set_ylabel("share of total monthly VC (%, 12mo rolling)", fontsize=9, color=SLATE)
    ax.set_title("Step 1 smoke test: monthly funding panel (the target series)\n"
                 "macro-normalised share of total funding, so it is not just 'everything grew'",
                 fontsize=10.5, color=SLATE, fontweight="bold", loc="left", pad=10)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.spines["left"].set_color(GREY)
    ax.spines["bottom"].set_color(GREY)
    ax.tick_params(colors=SLATE, length=0)
    ax.grid(color="#eeeef1", zorder=0)
    ax.set_xlim(pd.Timestamp("2008-01-01"), panel["month"].max())
    plt.tight_layout()
    fig.savefig(FIG / "monthly_funding_smoke.png", bbox_inches="tight", facecolor="white")
    print(f"wrote {FIG/'monthly_funding_smoke.png'}")


if __name__ == "__main__":
    main()
