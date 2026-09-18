"""Build the success LABEL for a market (Crunchbase base + Gemini web-grounding) and persist it.

    python -m experiments.build_labels --market llm-dev-tools --root data/raw/labels
    python -m experiments.build_labels --no-ground          # Crunchbase-only (no LLM)

Needs GOOGLE_APPLICATION_CREDENTIALS (Vela service account) for BigQuery and GEMINI_API_KEY in .env for
grounding. Writes labels to ``data/raw/labels/<market>.parquet`` — SEPARATE from the feature corpus.
"""

from __future__ import annotations

import argparse

from marketbrain.labels import build_label, store


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description="Build a market success label (Crunchbase + Gemini).")
    ap.add_argument("--market", default="llm-dev-tools")
    ap.add_argument("--root", default="data/raw/labels")
    ap.add_argument("--threshold", type=int, default=500_000_000)
    ap.add_argument("--no-ground", action="store_true", help="skip Gemini; Crunchbase-only")
    ap.add_argument("--project", default=None)
    args = ap.parse_args(argv)

    print(f"[labels] building {args.market} (ground={not args.no_ground}, threshold=${args.threshold:,})")
    label, outcomes = build_label.run(
        args.market, threshold_usd=args.threshold, ground=not args.no_ground, project=args.project,
    )
    path = store.write(label, outcomes, market=args.market, root=args.root)
    n = sum(1 for o in outcomes if o.is_outlier)
    v = sum(1 for o in outcomes if o.verified)
    print(f"\n[labels] market is_outlier={label.is_outlier}  outliers={n}/{len(outcomes)}  "
          f"grounded-verified={v}  first_crossing={label.label_reference_date}")
    print(f"[labels] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
