"""CLI: load the corpus + a monthly panel cut and run every integrity check.

Exits non-zero on any error-severity failure, so it can gate CI / a PR.

    python -m experiments.validate_cut 2024-03 --root data/raw
"""

from __future__ import annotations

import argparse
import sys

from marketbrain.corpus import CorpusStore, build_panel
from marketbrain.validation import IntegrityError, run_all_checks


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description="Validate a monthly corpus cut.")
    ap.add_argument("month", help="month to validate, 'YYYY-MM'")
    ap.add_argument("--root", default="data/raw", help="corpus root (default: data/raw)")
    args = ap.parse_args(argv)

    store = CorpusStore(args.root)
    observations = store.load_observations()
    panel = build_panel(observations, args.month)
    print(f"[validate_cut] {args.month}: {len(observations)} observations, {len(panel)} panel cells")

    try:
        run_all_checks(observations=observations, panel=panel, raise_on_error=True)
    except IntegrityError:
        print("[validate_cut] FAILED — integrity errors above", file=sys.stderr)
        return 1
    print("[validate_cut] OK — no integrity errors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
