"""M0.3 — the converter BASE RATE: what fraction of durable research themes ever 'convert' into a big-raise
company? Turns the plan's spine ("converters are rare and idiosyncratic -> reasoning, not statistics") from an
argument into a measured number.

    ~/miniconda3/envs/marketbrain/bin/python -m experiments.converter_baserate

Inputs (all cached -- no re-embedding, no corpus load):
  artifacts/stageB/themes.parquet             364 detector themes with label (positive / hyped_died / ...)
  artifacts/stageB/company_level_test.parquet company -> nearest-theme bridge (permalink, theme_id, sim) + founded_on
  data/raw/labels/company_rounds.parquet      raised_usd per round -> total raised per company

Definition: a theme CONVERTS if >= 1 bridged company (sim >= tau) reaches >= $X total raised. Reported over
tau in {0.60, 0.70} x X in {$50M, $500M}, for durable-positive themes vs hyped-died themes (the contrast), with
a founded-after-T0 variant (company founded in/after the theme's emergence). Cluster bootstrap over themes.
HONESTY: this is a LOWER BOUND -- Crunchbase only sees funded companies (survivorship), the bridge is
nearest-theme-only (a company can only convert its single closest theme), and recent themes are right-censored.
"""
from __future__ import annotations
import pathlib
import warnings

warnings.filterwarnings("ignore")
SB = pathlib.Path("artifacts/stageB")


def main():
    import numpy as np
    import pandas as pd

    th = pd.read_parquet(SB / "themes.parquet")[["theme_id", "label", "T0"]]
    comp = pd.read_parquet(SB / "company_level_test.parquet")[["permalink", "theme_id", "sim", "founded_on"]]
    raised = pd.read_parquet("data/raw/labels/company_rounds.parquet").groupby("permalink")["raised_usd"].sum()
    comp["total_raised"] = comp["permalink"].map(raised).fillna(0.0)
    comp["found_ym"] = comp["founded_on"].astype(str).str[:7]
    th["T0_ym"] = th["T0"].astype(str).str[:7]
    comp = comp.merge(th[["theme_id", "T0_ym"]], on="theme_id", how="left")
    comp["post_T0"] = comp["found_ym"] >= comp["T0_ym"]

    print(f"[baserate] {len(th)} themes ({(th['label'] == 'positive').sum()} durable-positive, "
          f"{(th['label'] == 'hyped_died').sum()} hyped-died); {len(comp)} bridged companies", flush=True)

    def theme_converts(theme_ids, tau, x, post_t0_only):
        c = comp[(comp["sim"] >= tau) & (comp["total_raised"] >= x)]
        if post_t0_only:
            c = c[c["post_T0"]]
        conv = set(c["theme_id"])
        return np.array([t in conv for t in theme_ids])

    def boot_ci(hits, B=4000, seed=0):
        rng = np.random.RandomState(seed)
        n = len(hits)
        if n == 0:
            return np.nan, np.nan
        draws = rng.randint(0, n, size=(B, n))
        rates = hits[draws].mean(axis=1)
        return float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))

    rows = []
    for post_t0 in (False, True):
        tag = "founded>=T0" if post_t0 else "any-time    "
        print(f"\n[baserate] company-founding window: {tag.strip()}")
        print(f"{'group':14s}{'bridge tau':>11s}{'threshold':>11s}{'converted':>11s}{'rate':>8s}{'95% CI':>18s}")
        for grp in ("positive", "hyped_died"):
            tids = th.loc[th["label"] == grp, "theme_id"].to_numpy()
            for tau in (0.60, 0.70):
                for x in (50e6, 500e6):
                    hits = theme_converts(tids, tau, x, post_t0)
                    lo, hi = boot_ci(hits)
                    rows.append({"window": tag.strip(), "group": grp, "tau": tau, "x": int(x / 1e6),
                                 "n_themes": len(tids), "n_conv": int(hits.sum()),
                                 "rate": float(hits.mean()), "ci_lo": lo, "ci_hi": hi})
                    print(f"{grp:14s}{tau:>11.2f}{'>=$' + str(int(x/1e6)) + 'M':>11s}"
                          f"{int(hits.sum()):>5d}/{len(tids):<5d}{hits.mean():>8.1%}"
                          f"   [{lo:.1%}, {hi:.1%}]", flush=True)

    out = pd.DataFrame(rows)
    out.to_parquet(SB / "converter_baserate.parquet")
    hp = out[(out.group == "positive") & (out.tau == 0.60) & (out.x == 500) & (out.window == "founded>=T0")].iloc[0]
    print(f"\n[baserate] HEADLINE (lower bound): only {hp['rate']:.0%} of durable research themes ever produce a "
          f"post-emergence >=$500M-raise company")
    print("[baserate] caveats: LOWER BOUND (Crunchbase survivorship; nearest-theme-only bridge; right-censoring of")
    print("[baserate] recent themes). The positive-vs-hyped_died contrast is the informative comparison; #4's powered")
    print("[baserate] null already showed converters share no statistical signature -> reasoning, not statistics.")
    print(f"[baserate] wrote {SB}/converter_baserate.parquet")


if __name__ == "__main__":
    main()
