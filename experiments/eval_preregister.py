"""Freeze the Phase-0 pre-registration into an immutable artifact -- run ONCE, before the eval/method is built.

    python -m experiments.eval_preregister

Hashes the protocol doc (Plans/2026-06-30_phase0-preregistration.md) and snapshots the frozen thresholds into
artifacts/eval/PREREGISTRATION.json. The point (see the doc's falsification clause) is to make "we found a
signal" falsifiable: the label gates, splits, baseline, null protocol, and decision rule are committed BEFORE
any breakout-detector exists. Any later change is an amendment, logged in the doc's section 11, and results are
reported both pre- and post-amendment. The hash lets a reviewer verify the doc was not edited after the fact.
"""
from __future__ import annotations
import hashlib
import json
import pathlib

DOC = pathlib.Path("../Plans/2026-06-30_phase0-preregistration.md")
OUT = pathlib.Path("artifacts/eval/PREREGISTRATION.json")
REGISTERED = "2026-06-30"

# The frozen protocol constants (mirror the doc; the doc is the source of truth, this is the machine-checkable snapshot).
FROZEN = {
    "label_durable_breakout": {
        "G_REL": 4.0,            # peak share >= 4x baseline (growth gate)
        "G_ABS": 0.003,          # peak share >= 0.3% of monthly cs.* docs (absolute scale gate)
        "P_RET": 0.5,            # plateau >= 50% of peak, 12-24mo after peak (persistence gate)
        "P_FLOOR_MULT": 2.0,     # plateau >= 2x birth baseline (did-not-return-to-obscurity gate)
        "persist_lo_months": 12, "persist_hi_months": 24,
        "birth_min_records": 5,  # first 3-mo window with >= 5 records = birth
        "base_months": 6,        # baseline = median share over [T0, T0+5]
        "smooth_months": 3,      # 3-month centred rolling mean of share
    },
    "detector_firewall": {
        "T_dec_offset_months": 9,        # decision point = T0 + 9 (early window only)
        "no_raw_category_feature": True,  # category identity is time-biased -> banned as a feature
        "cross_domain_spread_allowed": True,
        "venue_as_of_only": True,
    },
    "splits": {
        "temporal_cutoff": "2022-01",        # train T_dec <= 2021-12, test >= 2022-01
        "rolling_origins": ["2020-06", "2021-06", "2022-06", "2023-06"],
        "parent_folds": 5,                    # GroupKFold by idea-type parent; train∩test parents == empty
    },
    "metric": {
        "primary": "AUPRC", "bootstrap_resamples": 2000,
        "precision_at_k": [5, 10, 20],
        "null_permutations": 5000, "null_seed": 0, "alpha": 0.05,
        "decision": "beat max(B1,B2) on Split-1 test AND Split-2 grouped-OOF with separated CIs, AND beat shuffle null p<0.05",
    },
    "baselines": ["B1_momentum_escore", "B2_momentum_plus_llm_categorisation"],
    "sensitivity_grid": {"G_REL": [3, 4, 5], "G_ABS": [0.002, 0.003, 0.005], "P_RET": [0.4, 0.5, 0.6]},
    "falsifiable": True,                       # a NULL result is a committed, reportable outcome
    "normaliser_note": "N_cs(T) = total monthly doc count from OUR corpus (NOT monthly_research.py totalResults, an API artifact)",
}


def main():
    if not DOC.exists():
        raise FileNotFoundError(f"pre-registration doc not found at {DOC} (run from the repo root)")
    doc_text = DOC.read_text()
    doc_hash = hashlib.sha256(doc_text.encode()).hexdigest()
    record = {
        "registered": REGISTERED,
        "doc_path": str(DOC),
        "doc_sha256": doc_hash,
        "doc_chars": len(doc_text),
        "frozen": FROZEN,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=2, sort_keys=True))
    print(f"[prereg] froze {DOC.name} (sha256 {doc_hash[:16]}..., {len(doc_text):,} chars) -> {OUT}")
    print(f"[prereg] label gates G_REL={FROZEN['label_durable_breakout']['G_REL']} "
          f"G_ABS={FROZEN['label_durable_breakout']['G_ABS']} P_RET={FROZEN['label_durable_breakout']['P_RET']}; "
          f"T_dec=T0+{FROZEN['detector_firewall']['T_dec_offset_months']}mo; null B={FROZEN['metric']['null_permutations']}")


if __name__ == "__main__":
    main()
