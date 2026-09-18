"""Collect a deterministic signal into the corpus (generic dispatcher).

    python -m experiments.collect_signal arxiv      --start 2022-06 --end 2025-12 --root data/raw
    python -m experiments.collect_signal wikipedia  --start 2022-06 --end 2025-12 --root data/raw
    python -m experiments.collect_signal hackernews --start 2022-06 --end 2025-12 --root data/raw

Fetches monthly values, writes leakage-safe Observations into the bitemporal store,
rebuilds + validates each monthly cut (across all features present). The leakage
assertion is the gate.
"""

from __future__ import annotations

import argparse

from marketbrain.corpus import CorpusStore, build_panel
from marketbrain.schema import observations_to_frame
from marketbrain.signals import REGISTRY
from marketbrain.validation import ERROR, run_all_checks


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description="Collect a deterministic signal into the corpus.")
    ap.add_argument("signal", choices=sorted(REGISTRY), help="which signal to collect")
    ap.add_argument("--start", default="2022-06", help="first month 'YYYY-MM'")
    ap.add_argument("--end", default="2025-12", help="last month 'YYYY-MM'")
    ap.add_argument("--market", default="llm-dev-tools")
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--sleep", type=float, default=None, help="seconds between API calls")
    args = ap.parse_args(argv)

    module = REGISTRY[args.signal]
    kwargs = {} if args.sleep is None else {"sleep": args.sleep}
    print(f"[collect] {module.FEATURE} for {args.market}: {args.start}..{args.end}")
    observations = module.collect(args.start, args.end, market=args.market, **kwargs)
    if not observations:
        print("[collect] no observations returned")
        return 1

    store = CorpusStore(args.root)
    store.append_observations(observations_to_frame(observations))
    all_obs = store.load_observations()
    months = sorted({o.reference_date.strftime("%Y-%m") for o in observations})
    errors = 0
    for m in months:
        panel = build_panel(all_obs, m)
        store.write_cut(m, panel)
        failures = run_all_checks(observations=all_obs, panel=panel,
                                  raise_on_error=False, verbose=False)
        errors += sum(1 for f in failures if f.severity == ERROR)

    print(f"[collect] ingested {len(observations)} obs over {len(months)} months; "
          f"integrity errors: {errors}")
    for o in observations:
        print(f"  {o.reference_date.strftime('%Y-%m')}  {int(o.value):9d}")
    print("[collect] OK" if errors == 0 else "[collect] FAILED — integrity errors")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
