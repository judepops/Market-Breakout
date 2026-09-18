"""Compare the clean cohort sample (~5k) against using every company (~34k): does more data help or bias?

    python -m experiments.cohort_vs_all

Two scatter panels (capital vs outlier rate) on the same markets, one scored on the formation-window cohort,
one scored on every company in the market. Plus a holdout-significance panel with bootstrap error bars.
"""
from __future__ import annotations

import warnings
import pathlib

warnings.filterwarnings("ignore")
OUTLIER, FLOOR = 500e6, 5e6
FIG = pathlib.Path("artifacts/figures")
RED, SLATE, GREY, MUTED = "#8b0000", "#2c3e50", "#b9bcc4", "#6b6b70"


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
    df["first_date"] = pd.to_datetime(df["first_date"])
    df["first_year"] = df["first_date"].dt.year
    df["is_outlier"] = df["total"] >= OUTLIER

    def assign(c):
        a = assign_deterministic(c)
        if a is None:
            return None
        if a == "AMBIGUOUS":
            return tags_present(c)[0]
        return a
    df["market"] = df["categories"].map(assign)
    df = df.dropna(subset=["market"])

    pf = pd.read_parquet("data/raw/labels/partition_features.parquet")
    win = {r["market"]: (int(r["t0"]), int(r["k"]), bool(r["mature"])) for _, r in pf.iterrows()}

    rows = []
    for mk, g in df.groupby("market"):
        if mk not in win:
            continue
        t0, k, mature = win[mk]
        if not mature:
            continue
        coh = g[(g["first_year"] >= t0) & (g["first_year"] < t0 + k)]
        if len(coh) < 5 or len(g) < 5:
            continue
        rows.append({
            "market": mk,
            "n_all": len(g), "rate_all": g["is_outlier"].mean(),
            "medfirst_all": g["first_round"].median(),
            "n_coh": len(coh), "rate_coh": coh["is_outlier"].mean(),
            "medfirst_coh": coh["first_round"].median(),
        })
    m = pd.DataFrame(rows).dropna(subset=["medfirst_all", "medfirst_coh"])
    print(f"[markets] {len(m)} mature markets | companies: cohort {int(m['n_coh'].sum()):,} vs all {int(m['n_all'].sum()):,}")

    def sp(a, b):
        return float(pd.Series(a).rank().corr(pd.Series(b).rank()))

    def spw(a, b, w):
        a = pd.Series(a).rank().to_numpy()
        b = pd.Series(b).rank().to_numpy()
        w = np.asarray(w, float)
        am, bm = np.average(a, weights=w), np.average(b, weights=w)
        return float(np.average((a - am) * (b - bm), weights=w)
                     / np.sqrt(np.average((a - am)**2, weights=w) * np.average((b - bm)**2, weights=w)))
    r_coh = sp(m["medfirst_coh"], m["rate_coh"])
    r_all = sp(m["medfirst_all"], m["rate_all"])
    rw_coh = spw(m["medfirst_coh"], m["rate_coh"], m["n_coh"])
    rw_all = spw(m["medfirst_all"], m["rate_all"], m["n_all"])
    print("[median first round vs rate]")
    print(f"  cohort  ~5k : unweighted r={r_coh:+.2f}   size-weighted r={rw_coh:+.2f}")
    print(f"  all    ~27k : unweighted r={r_all:+.2f}   size-weighted r={rw_all:+.2f}")

    # ---- two-panel scatter ----
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), dpi=150)
    for ax, (xr, yr, nr, r, lab, n) in zip(axes, [
        ("medfirst_coh", "rate_coh", "n_coh", r_coh, "Cohort only (clean, leakage safe)", int(m["n_coh"].sum())),
        ("medfirst_all", "rate_all", "n_all", r_all, "Every company (more data, biased)", int(m["n_all"].sum())),
    ]):
        sz = (m[nr] / m[nr].max() * 260 + 18)
        ax.scatter(m[xr] / 1e6, m[yr] * 100, s=sz, color=RED, alpha=.55, edgecolor="white", linewidth=.6, zorder=3)
        ax.set_xscale("log")
        ax.set_xlabel("median first round ($M, log)", fontsize=9, color=SLATE)
        ax.set_ylabel("outlier rate (%)", fontsize=9, color=SLATE)
        ax.set_title(f"{lab}\nr = {r:+.2f}   ·   {n:,} companies", fontsize=10, color=SLATE, fontweight="bold", loc="left")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_color(GREY)
        ax.spines["bottom"].set_color(GREY)
        ax.tick_params(colors=SLATE, length=0)
        ax.grid(color="#eeeef1", zorder=0)
    fig.suptitle("Same markets, two denominators: bubble size = number of companies",
                 fontsize=11, color=SLATE, fontweight="bold", x=0.012, ha="left")
    plt.tight_layout()
    fig.savefig(FIG / "cohort_vs_all.png", bbox_inches="tight", facecolor="white")
    print(f"wrote {FIG/'cohort_vs_all.png'}")

    # ---- holdout significance (bootstrap) ----
    y = m["rate_coh"].to_numpy(float)
    X = np.c_[np.log(m["medfirst_coh"].to_numpy(float)), np.log(m["n_coh"].to_numpy(float))]
    n = len(y)
    pred = np.zeros(n)
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        A = np.c_[np.ones(len(tr)), X[tr]]
        b = np.linalg.lstsq(A, y[tr], rcond=None)[0]
        pred[i] = np.r_[1, X[i]] @ b
    err_model = np.abs(pred - y)
    err_base = np.abs(np.array([y[[j for j in range(n) if j != i]].mean() for i in range(n)]) - y)
    rng = np.random.RandomState(0)
    bm, bb = [], []
    for _ in range(2000):
        idx = rng.randint(0, n, n)
        bm.append(err_model[idx].mean() * 100)
        bb.append(err_base[idx].mean() * 100)
    bm, bb = np.array(bm), np.array(bb)
    diff = bb - bm
    print(f"[holdout] model MAE {err_model.mean()*100:.2f}pp [{np.percentile(bm,5):.2f},{np.percentile(bm,95):.2f}]  "
          f"baseline {err_base.mean()*100:.2f}pp [{np.percentile(bb,5):.2f},{np.percentile(bb,95):.2f}]  "
          f"improvement {diff.mean():.2f}pp, P(model better)={(diff>0).mean():.0%}")

    fig2, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    vals = [err_base.mean() * 100, err_model.mean() * 100]
    los = [err_base.mean() * 100 - np.percentile(bb, 5), err_model.mean() * 100 - np.percentile(bm, 5)]
    his = [np.percentile(bb, 95) - err_base.mean() * 100, np.percentile(bm, 95) - err_model.mean() * 100]
    ax.bar([0, 1], vals, yerr=[los, his], capsize=7, color=[GREY, RED], width=.6, zorder=3,
           error_kw={"ecolor": SLATE, "elinewidth": 1.3})
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["guess the average\n(baseline)", "use the features\n(our model)"], fontsize=9)
    ax.set_ylabel("holdout error (pp, lower is better)", fontsize=9, color=SLATE)
    ax.set_title(f"Predicting from features beats the baseline\n"
                 f"{diff.mean():.1f} pp better, model wins in {(diff>0).mean():.0%} of bootstrap resamples (90% bars)",
                 fontsize=10, color=SLATE, fontweight="bold", loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GREY)
    ax.spines["bottom"].set_color(GREY)
    ax.tick_params(colors=SLATE, length=0)
    ax.grid(axis="y", color="#eeeef1", zorder=0)
    plt.tight_layout()
    fig2.savefig(FIG / "holdout_significance.png", bbox_inches="tight", facecolor="white")
    print(f"wrote {FIG/'holdout_significance.png'}")


if __name__ == "__main__":
    main()
