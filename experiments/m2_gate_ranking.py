"""M2.0 gate — the FAIR final check before retiring the thesis engine as a predictor. The absolute-conviction
grader failed (AUC 0.433) because it rated everything ~92/100 — no dynamic range. This swaps it for FORCED
PAIRWISE RANKING: show the blinded theses of one durable and one fad theme, ask which capability is more likely
to prove durable/foundational. Pairwise accuracy has the dynamic range an absolute score lacks.

    ~/miniconda3/envs/marketbrain/bin/python -m experiments.m2_gate_ranking [--n 12 --pairs 40]

Same redact + engine as m2_gate_pilot (imported verbatim, blind to labels). Grader = gemini-2.5-pro, each pair
judged in BOTH orders to cancel position bias. Metric: fraction of pairs where the durable theme is chosen
(chance = 0.50), with a binomial 95% CI. PASS if CI excludes 0.50 (the engine's reasoning carries extractable
durability signal). NULL if CI includes 0.50 -> the engine is a writing aid only, no tear-sheets (per the plan).
"""
from __future__ import annotations
import argparse
import json
import pathlib
import warnings
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore")
OUT = pathlib.Path("artifacts/m2_gate")
FLASH, PRO = "gemini-2.5-flash", "gemini-2.5-pro"

RANK = """You are an INDEPENDENT evaluator comparing two emerging technologies, each described by an investment
thesis written by an analyst who did NOT know its outcome. Without trying to recognize the specific technologies,
decide which ONE is more likely to prove a DURABLE, widely-adopted, foundational technology (rather than a hyped
idea that fades). You MUST choose one. Judge only the substance of the reasoning.
Return ONLY JSON: {{"choice": "A" or "B", "reason": "..."}}.

THESIS A:
{a}

THESIS B:
{b}"""


def main(argv=None):
    import numpy as np
    import pandas as pd
    from marketbrain.labels.gemini_grounding import make_client
    from experiments.m2_gate_pilot import REDACT, ENGINE, _call, main as _pilot  # reuse identical engine

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12, help="themes per class")
    ap.add_argument("--pairs", type=int, default=40)
    ap.add_argument("--excerpts", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)

    scores = pd.read_parquet("artifacts/phase1/scores.parquet")
    scores["nw"] = scores["theme_id"].str.split().str.len()
    pool = scores[scores["nw"] >= 2]
    rng = np.random.RandomState(0)
    def pick(lbl):
        return list(rng.permutation(pool[pool["label"] == lbl]["theme_id"].tolist()))[: args.n]
    dur, fad = pick("positive"), pick("hyped_died")

    corpus = pd.read_parquet("data/derived/corpus.parquet")[["month", "title", "abstract"]].dropna().sort_values("month")
    blob = (corpus["title"].str.lower() + ". " + corpus["abstract"].str.lower())
    client = make_client()

    def thesis_for(theme):
        hit = corpus[blob.str.contains(theme, regex=False, na=False)].head(args.excerpts)
        if hit.empty:
            return None
        exc = "\n---\n".join(f"[{r.month}] {r.title.strip()}\n{r.abstract.strip()[:900]}" for r in hit.itertuples())
        dossier = _call(client, FLASH, REDACT.format(excerpts=exc))
        if "_error" in dossier:
            return None
        thesis = _call(client, FLASH, ENGINE.format(dossier=json.dumps(dossier)))
        return None if "_error" in thesis else json.dumps(thesis)

    print(f"[rank] generating blinded theses for {len(dur)} durable + {len(fad)} fad themes...", flush=True)
    allt = dur + fad
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        th = dict(zip(allt, ex.map(thesis_for, allt)))
    dur = [t for t in dur if th.get(t)]; fad = [t for t in fad if th.get(t)]
    print(f"[rank] theses ready: {len(dur)} durable, {len(fad)} fad", flush=True)

    # random durable x fad pairs
    pairs = []
    seen = set()
    while len(pairs) < args.pairs and len(seen) < len(dur) * len(fad):
        d = dur[rng.randint(len(dur))]; f = fad[rng.randint(len(fad))]
        if (d, f) in seen:
            continue
        seen.add((d, f)); pairs.append((d, f))

    def judge(pair):
        d, f = pair
        # order 1: A=durable, B=fad ; order 2: A=fad, B=durable  (cancel position bias)
        r1 = _call(client, PRO, RANK.format(a=th[d], b=th[f]))
        r2 = _call(client, PRO, RANK.format(a=th[f], b=th[d]))
        c1 = r1.get("choice"); c2 = r2.get("choice")
        picks_dur = []
        if c1 in ("A", "B"):
            picks_dur.append(c1 == "A")     # A is durable in order 1
        if c2 in ("A", "B"):
            picks_dur.append(c2 == "B")     # B is durable in order 2
        return {"dur": d, "fad": f, "picks_dur": picks_dur}

    print(f"[rank] judging {len(pairs)} durable-vs-fad pairs (both orders)...", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(judge, pairs))

    votes = [v for r in res for v in r["picks_dur"]]        # each order = one vote (durable chosen? T/F)
    n = len(votes); k = int(sum(votes))
    acc = k / n if n else float("nan")
    # binomial 95% CI (normal approx)
    se = (acc * (1 - acc) / n) ** 0.5 if n else float("nan")
    lo, hi = acc - 1.96 * se, acc + 1.96 * se
    # consistent pairs (both orders agree the durable one is better) — position-bias-free
    consistent = [all(r["picks_dur"]) for r in res if len(r["picks_dur"]) == 2]
    cons_rate = float(np.mean(consistent)) if consistent else float("nan")

    print(f"\n[rank] ===== M2.0 COMPARATIVE (RANKING) GATE =====")
    print(f"[rank] durable chosen in {k}/{n} judgments = {acc:.1%}  95% CI [{lo:.1%}, {hi:.1%}]  (chance 50%)")
    print(f"[rank] position-bias-free: durable wins BOTH orders in {cons_rate:.1%} of pairs (chance 25%)")
    verdict = ("PASS — engine reasoning ranks durable>fad above chance (CI excludes 50%)" if lo > 0.5
               else ("REVERSED (picks fads)" if hi < 0.5 else "NULL — at chance: engine is a writing aid only, no tear-sheets"))
    print(f"[rank] VERDICT: {verdict}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ranking_summary.json").write_text(json.dumps(
        {"n_votes": n, "durable_chosen": k, "accuracy": acc, "ci_lo": lo, "ci_hi": hi,
         "consistent_both_orders": cons_rate, "n_pairs": len(pairs), "verdict": verdict}, indent=2))
    print(f"[rank] wrote {OUT}/ranking_summary.json")
    print("[rank] read: this is the FINAL check. NULL here => Layer-2 engine cannot predict; keep only as a")
    print("[rank]       narrative/sourcing aid. PASS => the absolute-score readout hid real signal; engine earns more.")


if __name__ == "__main__":
    main()
