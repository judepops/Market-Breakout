"""Phase 1 (the pre-registered PRIMARY test): S-curve event-lag. Does a signal's TAKEOFF lead funding's?

    python -m experiments.s_curve_lag --signal wikipedia   (research / attention / adoption / wikipedia)

For each market: build the cumulative signal-share and cumulative funding-share, fit a growth curve
(logistic / Gompertz, best R^2), find the INFLECTION (takeoff = fastest-growth date) of each, with boundary
censoring. Each series's takeoff date is computed once; lag = funding-takeoff - signal-takeoff (months), >0
means signal leads. Shuffle null re-pairs each signal with a wrong market's funding. Unlike the
cross-correlation, this compares two dates, not noisy month-to-month wiggles.
"""
from __future__ import annotations
import warnings
import pathlib

warnings.filterwarnings("ignore")
TS = pathlib.Path("data/raw/timeseries")
FIG = pathlib.Path("artifacts/figures")
R2_GATE = 0.90
CENSOR = 0.05
MIN_MONTHS = 48
FLOOR = "2010-01-01"
RED, SLATE, GREY = "#8b0000", "#2c3e50", "#b9bcc4"


def _logistic(t, L, k, t0):
    import numpy as np
    return L / (1 + np.exp(-np.clip(k * (t - t0), -30, 30)))


def _gompertz(t, L, k, t0):
    import numpy as np
    return L * np.exp(-np.exp(-np.clip(k * (t - t0), -30, 30)))


def _infl_pos(y):
    """y = cumulative normalised [0,1]. Returns fractional month index of takeoff, or None."""
    import numpy as np
    from scipy.optimize import curve_fit
    t = np.arange(len(y), dtype=float)
    best = None
    for f, p0 in [(_logistic, [1.0, 0.1, len(y) / 2]), (_gompertz, [1.0, 0.1, len(y) / 2])]:
        try:
            popt, _ = curve_fit(f, t, y, p0=p0, maxfev=20000,
                                bounds=([0.5, 1e-3, -len(y)], [2.0, 5.0, 2 * len(y)]))
            r2 = 1 - np.sum((y - f(t, *popt)) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)
            if best is None or r2 > best[0]:
                best = (r2, f, popt)
        except Exception:  # noqa: BLE001
            continue
    if best is None or best[0] < R2_GATE:
        return None
    _, f, popt = best
    tg = np.linspace(0, len(y) - 1, 1000)
    ti = tg[np.argmax(np.gradient(f(tg, *popt), tg))]
    if ti < CENSOR * (len(y) - 1) or ti > (1 - CENSOR) * (len(y) - 1):
        return None
    return ti


def _takeoff_month(series, floor):
    """Absolute month number (year*12+month, fractional) of this series's takeoff, or None.

    Both the signal and funding series are floored at the SAME date so their cumulative takeoffs are
    comparable (else funding's earlier start gives it a spurious head start)."""
    import pandas as pd
    s = series[series.index >= pd.Timestamp(floor)]
    if len(s) < MIN_MONTHS:
        return None
    idx = pd.date_range(s.index.min(), s.index.max(), freq="MS")
    c = s.reindex(idx).fillna(0.0).cumsum().to_numpy()
    if c[-1] <= 0:
        return None
    ti = _infl_pos(c / c[-1])
    if ti is None:
        return None
    return idx[0].year * 12 + (idx[0].month - 1) + ti


def main(argv=None):
    import argparse
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--signal", default="wikipedia",
                    choices=["research", "attention", "adoption", "wikipedia"])
    args = ap.parse_args(argv)

    fund = pd.read_parquet(TS / "monthly_funding.parquet")
    fund["month"] = pd.to_datetime(fund["month"])
    sig = pd.read_parquet(TS / f"monthly_{args.signal}.parquet")
    sig["month"] = pd.to_datetime(dict(year=sig["year"], month=sig["month"], day=1))

    floor = "2015-07-01" if args.signal == "wikipedia" else "2012-01-01"   # same window for both series
    f_takeoff = {mk: _takeoff_month(g.set_index("month")["usd_share"].sort_index(), floor)
                 for mk, g in fund.groupby("market")}
    s_takeoff = {mk: _takeoff_month(g.set_index("month")["share"].sort_index(), floor)
                 for mk, g in sig.groupby("market")}

    markets = [mk for mk in sorted(set(s_takeoff) & set(f_takeoff))
               if s_takeoff[mk] is not None and f_takeoff[mk] is not None]
    d = pd.DataFrame([{"market": mk, "lead_mo": f_takeoff[mk] - s_takeoff[mk]} for mk in markets])
    if len(d) < 6:
        print(f"only {len(d)} markets had clean takeoffs in BOTH curves (signal={args.signal}) — too few.")
        return
    med = d["lead_mo"].median()
    print(f"[s-curve] signal={args.signal} | {len(d)} markets with clean takeoffs | "
          f"median lead = {med:+.0f} months | {(d['lead_mo']>0).mean():.0%} show {args.signal} leading")

    rng = np.random.RandomState(0)
    sf, ff = [s_takeoff[mk] for mk in markets], [f_takeoff[mk] for mk in markets]
    sh_meds = []
    for _ in range(5000):
        perm = rng.permutation(len(markets))
        sh_meds.append(np.median([ff[j] - sf[perm[i]] for i, j in enumerate(range(len(markets)))]))
    sh_meds = np.array(sh_meds)
    p = float(np.mean(sh_meds >= med))
    print(f"[shuffle] real median lead {med:+.0f} mo vs shuffle median {np.median(sh_meds):+.0f} mo  (p={p:.3f})")
    verdict = (f"GREEN: {args.signal} takeoff leads funding takeoff, beats shuffle"
               if med > 0 and p < 0.05 else "FLAT/NULL: takeoff lead not beyond chance")
    print(f"[verdict] {verdict}")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=150)
    ax.hist(d["lead_mo"], bins=range(-66, 72, 6), color=RED, alpha=.8, zorder=3, edgecolor="white")
    ax.axvline(0, color=GREY, lw=1)
    ax.axvline(med, color=SLATE, ls="--", lw=1.5, label=f"median {med:+.0f} mo")
    ax.set_xlabel(f"takeoff lead of {args.signal} over funding (months)", fontsize=9, color=SLATE)
    ax.set_ylabel("markets", fontsize=9, color=SLATE)
    ax.set_title(f"S-curve event-lag: does {args.signal} take off before funding?\n"
                 f"{len(d)} markets · median {med:+.0f} mo · p={p:.3f} vs shuffle",
                 fontsize=10, color=SLATE, fontweight="bold", loc="left", pad=10)
    ax.legend(frameon=False, fontsize=8.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GREY)
    ax.spines["bottom"].set_color(GREY)
    ax.tick_params(colors=SLATE, length=0)
    plt.tight_layout()
    out = FIG / f"s_curve_lag_{args.signal}.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
