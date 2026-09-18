"""Diagnostic: why aren't the mega-markets (LLM, diffusion, RL) detector themes? Trace specific terms
through gating + labelling on the broad-7 corpus.

    python -m experiments.stageB_trace_terms
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

PROBES = ["large language models", "large language", "language models", "diffusion models", "diffusion",
          "reinforcement learning", "deep learning", "machine learning", "graph neural networks",
          "federated learning", "transformer", "transformers", "object detection", "knowledge distillation",
          "contrastive learning", "generative adversarial", "self-supervised", "vision transformer",
          "foundation models", "prompt", "in-context learning", "retrieval augmented"]


def main():
    import numpy as np
    import pandas as pd
    from marketbrain.corpus.load_full import load_corpus
    from marketbrain.eval.corpus_inputs import cs_denominator, corpus_term_events
    from marketbrain.eval.label import EmergenceLabelConfig, durable_breakout_label
    from marketbrain.markets.granularity_gates import _pmi_ok, _noun_headed, _is_fragment, GENERIC, DF_CEIL

    cfg = EmergenceLabelConfig()
    corpus = load_corpus(["cs.LG", "cs.CL", "cs.AI", "cs.RO", "cs.SE", "cs.DC", "stat.ML"])
    denom = cs_denominator(corpus)
    grid = list(denom.index)
    corpus_end, n = grid[-1], len(corpus)
    te = corpus_term_events(corpus, min_doc_freq=40, ngram_max=3)
    df_count = te.groupby("term")["record_id"].nunique().to_dict()
    nh = set(_noun_headed([t for t in df_count if t in PROBES]))
    denom_arr = denom.reindex(grid).to_numpy()
    print(f"n={n:,}  df_ceil(0.08)={0.08*n:.0f}  df_ceil(0.35)={0.35*n:.0f}\n", flush=True)
    print(f"{'term':26s}{'df':>7s}{'df%':>6s} {'frag':>4s}{'genc':>5s}{'pmi':>4s}{'noun':>5s}  {'label':>9s}  T0..Tpeak", flush=True)
    for t in PROBES:
        c = df_count.get(t)
        if c is None:
            print(f"{t:26s}   -- not extracted (df<40 or absent)"); continue
        frag = _is_fragment(t)
        ws = t.split()
        genc = (len(ws) == 1 and t in GENERIC) or (len(ws) > 1 and all(w in GENERIC for w in ws))
        pmi = _pmi_ok(t, df_count, n)
        noun = t in nh
        # label (regardless of gates, to see censoring/never)
        pivot = te[te["term"] == t].groupby("month")["record_id"].nunique().reindex(grid, fill_value=0)
        share = pd.Series(np.divide(pivot.to_numpy(), denom_arr, out=np.zeros(len(grid)), where=denom_arr > 0), index=grid)
        res = durable_breakout_label(share, pivot.astype(int), cfg, corpus_end=corpus_end)
        gated = (c <= DF_CEIL * n) and not frag and not genc and pmi and noun
        print(f"{t:26s}{c:>7d}{100*c/n:>5.1f}% {str(frag):>4s}{str(genc):>5s}{str(pmi):>4s}{str(noun):>5s}  "
              f"{res['label']:>9s}  {res.get('T0')}..{res.get('T_peak')}  gate={'PASS' if gated else 'DROP'}", flush=True)


if __name__ == "__main__":
    main()
