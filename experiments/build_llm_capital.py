"""LLM funding for every market's cohort companies - the at-scale Crunchbase-parallel capital signal.

    python -m experiments.build_llm_capital --top-n 15

For each mature market, the top-N companies by Crunchbase funding (they dominate the capital feature), then a
grounded-Gemini funding call per company (total raised + latest round), deduped across markets and cached so
the batch is resumable. A later step compares the LLM capital aggregate to the Crunchbase one and re-runs the
predictor. Honest caveats (logged): the LLM returns TOTAL (not formation-window early) funding, and is
unreliable on decade-old rounds - both will show up in the results.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="BigQuery Storage module not found")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/raw/labels")
    ap.add_argument("--out", default="data/raw/labels/llm_capital.parquet")
    ap.add_argument("--top-n", type=int, default=15)
    ap.add_argument("--project", default=None)
    args = ap.parse_args(argv)

    import pandas as pd

    from marketbrain.labels.gemini_grounding import make_client
    from marketbrain.markets.catalog import CATALOG
    from marketbrain.signals.llm_funding import ground_funding

    ft = pd.read_parquet(Path(args.labels) / "market_features.parquet").set_index("market")
    specs = [m for m in CATALOG if m.name in ft.index and bool(ft.loc[m.name, "mature"])]

    # the full >= $5M fundable universe per market (fetch_universe already floors at $5M); dedup by permalink.
    # --top-n caps per market (0 = no cap = every >=$5M company, the comprehensive run).
    targets = {}
    print(f"resolving the >=$5M universe for {len(specs)} markets ...")
    for m in specs:
        comps = sorted(m.universe(project=args.project), key=lambda c: c.funding_total_usd or 0, reverse=True)
        for c in (comps[:args.top_n] if args.top_n else comps):
            targets.setdefault(c.permalink, (c.name, c.funding_total_usd, m.name))
    print(f"{len(targets)} distinct >=$5M companies to ground "
          f"(~{len(targets) * 7 / 3600:.1f} h at ~7s/grounded call)")

    out = Path(args.out)
    done = {}
    if out.exists():
        done = {r.permalink: True for r in pd.read_parquet(out).itertuples()}
        print(f"resuming: {len(done)} already grounded")

    rounds_out = Path(args.labels) / "llm_capital_rounds.parquet"   # the DATED round history (for windowing)
    client = make_client()
    rows = pd.read_parquet(out).to_dict("records") if out.exists() else []
    rounds = pd.read_parquet(rounds_out).to_dict("records") if rounds_out.exists() else []
    n_new = n_unver = 0
    for perm, (name, cb_total, market) in targets.items():
        if perm in done:
            continue
        verified, latest, llm_total, co_rounds = False, None, None, ()
        try:
            f = ground_funding(name or perm, context=f"a {market} company", client=client)
            llm_total, verified, latest, co_rounds = f.total_raised_usd, f.verified, f.latest_round_date, f.rounds
        except Exception as e:  # noqa: BLE001
            print(f"  {name}: {e}")
        n_new += 1
        n_unver += (not verified)
        rows.append({"permalink": perm, "name": name, "market": market,
                     "cb_total_usd": int(cb_total) if cb_total else None,
                     "llm_total_usd": int(llm_total) if (verified and llm_total) else None,  # trust only verified
                     "verified": bool(verified),
                     "llm_latest_date": str(latest) if (verified and latest) else None,
                     "n_rounds": len(co_rounds)})
        if verified:  # save each DATED round so we can window it to a market's formation window
            for r in co_rounds:
                if r.date:
                    rounds.append({"permalink": perm, "date": str(r.date),
                                   "amount_usd": r.amount_usd, "stage": r.stage})
        if n_new % 10 == 0:
            pd.DataFrame(rows).to_parquet(out, index=False)
            pd.DataFrame(rounds).to_parquet(rounds_out, index=False)
            print(f"  {n_new} grounded, {n_unver} unverified (<3 src/null); {len(rounds)} dated rounds captured")
    pd.DataFrame(rows).to_parquet(out, index=False)
    pd.DataFrame(rounds).to_parquet(rounds_out, index=False)
    print(f"\ndone: {len(rows)} companies; {n_unver}/{n_new} new unverified; "
          f"{len(rounds)} dated rounds. wrote {out} + {rounds_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
