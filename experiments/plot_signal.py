"""Plot a signal trajectory from the corpus, in the Vela house style (generic).

    python -m experiments.plot_signal arxiv      --root data/raw --out artifacts/figures
    python -m experiments.plot_signal wikipedia  --root data/raw --out artifacts/figures
    python -m experiments.plot_signal hackernews --root data/raw --out artifacts/figures

The series is read straight from the bitemporal store (knowledge-dated, so the plot is
leakage-safe). A reference line marks ChatGPT's public launch — the trajectory should
inflect there if the signal is real (a standing sanity-check norm).
"""

from __future__ import annotations

import argparse
from datetime import date

import pandas as pd

from marketbrain.corpus import CorpusStore
from marketbrain.figures.style import C_BASE_RATE, C_GNN, FULL_WIDTH, apply_style, save
from marketbrain.signals import REGISTRY

_LABELS = {
    "arxiv": ("(a) LLM dev-tools — arXiv research momentum", "arXiv submissions / month"),
    "wikipedia": ("(b) LLM dev-tools — Wikipedia attention", "pageviews / month"),
    "hackernews": ("(c) LLM dev-tools — Hacker News volume", "HN stories / month"),
    "npm": ("(d) LLM dev-tools — npm (JS) adoption", "npm downloads / month"),
    "edgar": ("(e) LLM dev-tools — funding (SEC Form D filings)", "Form D filings / month"),
    "pypi": ("(f) LLM dev-tools — PyPI (Python) adoption", "PyPI downloads / month"),
    "crunchbase": ("(g) LLM dev-tools — funding (Crunchbase $ raised)", "$ raised / month"),
    "emergence": ("(h) LLM dev-tools — research emergence (Carley/Garner)", "emergence score (Σ√EScore·n)"),
    "patents": ("(i) LLM dev-tools — patents assigned to roster companies", "patents first public / month"),
    "github": ("(j) LLM dev-tools — GitHub stars (adoption)", "stars gained / month"),
}


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description="Plot a signal trajectory in house style.")
    ap.add_argument("signal", choices=sorted(REGISTRY))
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--market", default="llm-dev-tools")
    ap.add_argument("--out", default="artifacts/figures")
    args = ap.parse_args(argv)

    feature = REGISTRY[args.signal].FEATURE
    df = CorpusStore(args.root).load_observations()
    df = df[(df["feature"] == feature) & (df["market"] == args.market)].copy()
    if df.empty:
        print(f"[plot] no observations for {feature}/{args.market} — collect it first")
        return 1
    df["reference_date"] = pd.to_datetime(df["reference_date"])
    df = df.sort_values("reference_date")

    title, ylabel = _LABELS.get(args.signal, (f"{args.market} — {feature}", feature))
    apply_style()
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 3.0))
    ax.plot(df["reference_date"], df["value"], color=C_GNN, lw=1.9, marker="o", ms=3, zorder=5,
            label=ylabel)
    ax.axvline(pd.Timestamp(date(2022, 11, 30)), color=C_BASE_RATE, ls=":", lw=0.9, zorder=2)
    ax.text(pd.Timestamp(date(2022, 12, 10)), ax.get_ylim()[1] * 0.96, "ChatGPT",
            color=C_BASE_RATE, fontsize=6, va="top")
    ax.set_title(title)
    ax.set_xlabel("month (knowledge-dated, leakage-safe)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper left")

    png, pdf = save(fig, args.out, f"{args.signal}_{args.market}_trajectory")
    print(f"saved {png}")
    print(f"saved {pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
