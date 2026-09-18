"""Step 3 (MILESTONE / decision gate): does a signal LEAD funding? The cheap monthly smoke test.

    python -m experiments.cross_corr_smoke --signal research   (or attention / adoption)

Per market: align the macro-normalised MONTHLY signal share and funding share, smooth both (6-month rolling,
to tame funding lumpiness), take log-growth, and cross-correlate at lags 0-24 months. The lag of peak
correlation is the lead (positive = signal leads funding). Aggregate the per-market leads, then a shuffle null
(re-pair each signal with a random WRONG market's funding). Green if the leads cluster positive and the real
peak correlations beat the shuffle.
"""
from __future__ import annotations
import warnings
import pathlib

warnings.filterwarnings("ignore")
MAXLAG = 24            # months
MIN_OVERLAP = 36       # months of usable overlap required
SMOOTH = 6             # rolling-mean window (months)
FIG = pathlib.Path("artifacts/figures")
TS = pathlib.Path("data/raw/timeseries")
RED, SLATE, GREY = "#8b0000", "#2c3e50", "#b9bcc4"


def _growth(series, idx):
    """Reindex to monthly idx, fill 0, smooth, log-growth -> numpy array (with leading NaNs)."""
    import numpy as np
    s = series.reindex(idx).fillna(0.0)
    sm = s.rolling(SMOOTH, min_periods=3).mean()
    return np.diff(np.log(sm.clip(lower=1e-9).values))


def _best_lag(sg, fg):
    """Peak corr over lags -MAXLAG..+MAXLAG. lag>0 = signal leads funding; lag<0 = funding leads signal."""
    import numpy as np
    best_r, best_l = -2.0, 0
    for lag in range(-MAXLAG, MAXLAG + 1):
        if lag >= 0:
            a, b = (sg[:len(sg) - lag], fg[lag:]) if lag else (sg, fg)
        else:
            a, b = sg[-lag:], fg[:len(fg) + lag]
        n = min(len(a), len(b))
        if n < MIN_OVERLAP or np.std(a[:n]) < 1e-9 or np.std(b[:n]) < 1e-9:
            continue
        r = float(np.corrcoef(a[:n], b[:n])[0, 1])
        if r > best_r:
            best_r, best_l = r, lag
    return best_l, best_r


def main(argv=None):
    import argparse
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--signal", default="research",
                    choices=["research", "attention", "adoption", "wikipedia"])
    args = ap.parse_args(argv)

    fund = pd.read_parquet(TS / "monthly_funding.parquet")
    fund["month"] = pd.to_datetime(fund["month"])
    fseries = {mk: g.set_index("month")["usd_share"].sort_index() for mk, g in fund.groupby("market")}

    sig = pd.read_parquet(TS / f"monthly_{args.signal}.parquet")
    sig["month"] = pd.to_datetime(dict(year=sig["year"], month=sig["month"], day=1))
    sseries = {mk: g.set_index("month")["share"].sort_index() for mk, g in sig.groupby("market")}

    def pair(mk_sig, mk_fund):
        s, f = sseries[mk_sig], fseries.get(mk_fund)
        if f is None:
            return None
        lo, hi = max(s.index.min(), f.index.min()), min(s.index.max(), f.index.max())
        idx = pd.date_range(lo, hi, freq="MS")
        if len(idx) < MIN_OVERLAP + SMOOTH:
            return None
        sg, fg = _growth(s, idx), _growth(f, idx)
        valid = ~(np.isnan(sg) | np.isnan(fg))          # drop the leading NaNs from smoothing, jointly aligned
        sg, fg = sg[valid], fg[valid]
        if len(sg) < MIN_OVERLAP:
            return None
        return _best_lag(sg, fg)

    rows = []
    for mk in sorted(set(sseries) & set(fseries)):
        res = pair(mk, mk)
        if res:
            rows.append({"market": mk, "lead_mo": res[0], "peak_r": res[1]})
    d = pd.DataFrame(rows)
    if len(d) < 6:
        print(f"only {len(d)} markets usable — {args.signal} panel may be incomplete; re-run when it finishes.")
        return
    med_lead = d["lead_mo"].median()
    print(f"[smoke] signal={args.signal} | {len(d)} markets | median lead = {med_lead:.0f} months "
          f"({(d['lead_mo']>0).mean():.0%} show {args.signal} leading) | median peak r = {d['peak_r'].median():.2f}")

    rng = np.random.RandomState(0)
    sh = []
    for _ in range(200):
        mks = list(d["market"])
        for mk, wrong in zip(mks, rng.permutation(mks)):
            if mk != wrong:
                res = pair(mk, wrong)
                if res:
                    sh.append(res[1])
    real_med, shuf_med = d["peak_r"].median(), float(np.median(sh))
    p = float(np.mean(np.array(sh) >= real_med))
    verdict = (f"GREEN: {args.signal} appears to lead funding, beats shuffle"
               if med_lead > 0 and p < 0.05 else "FLAT/NULL: no clear lead beyond chance — early honest read")
    print(f"[shuffle] real median peak r {real_med:.2f} vs shuffle {shuf_med:.2f} (p={p:.3f})")
    print(f"[verdict] {verdict}")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=150)
    ax.hist(d["lead_mo"], bins=range(-MAXLAG, MAXLAG + 4, 3), color=RED, alpha=.8, zorder=3, edgecolor="white")
    ax.axvline(0, color=GREY, lw=1)
    ax.axvline(med_lead, color=SLATE, ls="--", lw=1.5, zorder=4, label=f"median lead {med_lead:.0f} mo")
    ax.set_xlabel(f"lead of {args.signal} over funding (months, peak cross-correlation)", fontsize=9, color=SLATE)
    ax.set_ylabel("number of markets", fontsize=9, color=SLATE)
    ax.set_title(f"Smoke test: does {args.signal} lead funding?\n{len(d)} markets · "
                 f"real median peak r {real_med:.2f} vs shuffle {shuf_med:.2f} (p={p:.3f})",
                 fontsize=10.5, color=SLATE, fontweight="bold", loc="left", pad=10)
    ax.legend(frameon=False, fontsize=8.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GREY)
    ax.spines["bottom"].set_color(GREY)
    ax.tick_params(colors=SLATE, length=0)
    plt.tight_layout()
    out = FIG / f"cross_corr_smoke_{args.signal}.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
