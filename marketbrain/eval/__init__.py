"""Pre-registered Phase-0 evaluation for idea-level market-emergence detection.

Pure, leak-free evaluation machinery -- the durable-breakout LABEL, the negative-control set, the train/test
SPLITS (temporal + idea-type holdout), the imbalance-appropriate METRICS (AUPRC, bootstrap CI, shuffle null),
and the BASELINES to beat (momentum + LLM-categorisation). Frozen by Plans/2026-06-30_phase0-preregistration.md;
this package contains NO detector logic. See eval_preregister for the immutable registration artifact.
"""
