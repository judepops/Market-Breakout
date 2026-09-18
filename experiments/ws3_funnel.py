"""WS3 — the sourcing FUNNEL: does the detector's market signal concentrate investable outliers? The funnel
claim survives every prediction-null: even if emergence isn't a company FEATURE, narrowing the universe to
detector-flagged (durably-emerging) markets could raise the per-company outlier hit-rate a picker then sees.
This tests that necessary condition on local data.

    ~/miniconda3/envs/marketbrain/bin/python -m experiments.ws3_funnel

For each bridged company, its "market quality" = its nearest theme's detector durability score (sig_theme).
We rank companies by that and measure the outlier hit-rate in the top decile/quartile vs the base rate
(a precision-at-K funnel), and the hit-rate by theme class (durable / fad / never). Outcome: raised >=$50M and
>=$500M (and is_outlier). Bootstrap CIs. A lift >1 means sourcing from detector-flagged markets is enriched;
lift ~1 means the funnel adds nothing (consistent with the flat 20%~16% theme-level converter base-rate).
"""
from __future__ import annotations
import pathlib
import warnings

warnings.filterwarnings("ignore")
SB = pathlib.Path("artifacts/stageB")


def main():
    import numpy as np, pandas as pd

    c = pd.read_parquet(SB / "company_level_test.parquet")
    keep = [x for x in ["permalink", "theme_id", "sim", "sig_theme", "is_outlier"] if x in c.columns]
    c = c[keep].copy()
    lab = pd.read_parquet(SB / "themes.parquet")[["theme_id", "label", "sig_score"]]
    c = c.merge(lab, on="theme_id", how="left")
    raised = pd.read_parquet("data/raw/labels/company_rounds.parquet").groupby("permalink")["raised_usd"].sum()
    c["raised"] = c["permalink"].map(raised).fillna(0.0)
    c = c[c["sim"] >= 0.6].reset_index(drop=True)
    # market-quality score per company = its theme's detector durability score
    c["mkt_score"] = c["sig_score"].fillna(c.get("sig_theme")).fillna(0.0)
    n = len(c)
    print(f"[ws3] {n} bridged companies (sim>=0.6), {c['theme_id'].nunique()} markets", flush=True)

    def boot_rate(mask, y, B=2000, seed=0):
        yy = y[mask]
        if len(yy) == 0:
            return np.nan, np.nan, np.nan, 0
        rng = np.random.RandomState(seed)
        rates = yy[rng.randint(0, len(yy), size=(B, len(yy)))].mean(1)
        return float(yy.mean()), float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5)), int(yy.sum())

    for thr, tag in [(50e6, ">=$50M"), (500e6, ">=$500M")]:
        y = (c["raised"] >= thr).to_numpy().astype(int)
        base = y.mean()
        print(f"\n[ws3] outcome {tag}: base outlier rate = {base:.1%} ({int(y.sum())}/{n})")
        # precision-at-K by market-quality rank
        order = np.argsort(-c["mkt_score"].to_numpy())
        for frac, name in [(0.10, "top decile"), (0.25, "top quartile"), (0.50, "top half")]:
            k = max(int(frac * n), 1)
            sel = np.zeros(n, bool); sel[order[:k]] = True
            r, lo, hi, npos = boot_rate(sel, y)
            lift = r / base if base > 0 else np.nan
            print(f"   {name:12s} (n={k:4d}): hit-rate {r:.1%} [{lo:.1%},{hi:.1%}]  lift {lift:.2f}x  ({npos} outliers)", flush=True)
        # by theme class
        for cls in ["positive", "hyped_died", "never"]:
            m = (c["label"] == cls).to_numpy()
            if m.sum() == 0:
                continue
            r, lo, hi, npos = boot_rate(m, y)
            print(f"   market class {cls:11s} (n={int(m.sum()):4d}): hit-rate {r:.1%} [{lo:.1%},{hi:.1%}]  lift {r/base if base>0 else float('nan'):.2f}x", flush=True)

    print("\n[ws3] read: top-decile lift >> 1 => the detector's market score concentrates outliers => the funnel")
    print("[ws3]       adds sourcing value even though emergence is not a per-company predictive feature.")
    print("[ws3]       lift ~ 1 => the funnel is flat (consistent with converter base-rate 20%~16%); detector's")
    print("[ws3]       market value would then be timing/coverage, not outlier-density enrichment.")


if __name__ == "__main__":
    main()
