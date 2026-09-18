"""M2.0 GATE (pilot) — the falsification test for the thesis engine, built BEFORE the engine is trusted.
Pre-registered logic: an LLM can narrate any theme; the only thing that matters is whether it can
DISCRIMINATE durable breakouts from fads on DE-IDENTIFIED dossiers, graded blind by a DIFFERENT model.

    ~/miniconda3/envs/marketbrain/bin/python -m experiments.m2_gate_pilot [--n 15]

Pipeline, per theme (all three stages blind to the label):
  1. REDACT  (gemini-2.5-flash) : real early abstracts -> a de-identified dossier (name stripped, paraphrased).
  2. ENGINE  (gemini-2.5-flash) : dossier -> reasoning chain + one-line thesis (capabilities/apps/market/why-now).
  3. GRADE   (gemini-2.5-pro)   : thesis ONLY -> conviction 0-100 + a concept guess (leakage probe).
Primary readout: ROC-AUC of conviction vs the true durable-breakout label (arXiv phase1). Leakage rate =
fraction where the grader correctly names the concept (de-identification failed). Report AUC on all + on the
non-leaked subset. Ground truth = research durability (a necessary substrate; market conversion is downstream).

KILL CRITERION (from the build plan): if conviction AUC ~ 0.5 (no separation) even on the named/hindsight-
available set, the engine is an eloquent narrative generator with no discriminative power -> downgrade to
writing aid, no tear-sheets. If AUC is meaningfully > 0.5 with low leakage, the engine earns the next stage.
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

REDACT = """You are a redactor preparing a BLINDED technical dossier for a fair evaluation. Below are excerpts
from early research papers about ONE technical concept. Write a de-identified dossier describing the concept's
technical mechanism and the problem it solves, WITHOUT naming it and WITHOUT the specific proper-noun term(s)
that uniquely identify it (paraphrase those away). Do NOT speculate about its popularity, success, or future.
Return ONLY JSON: {{"problem": "...", "mechanism": "...", "approach": "...", "inputs_outputs": "..."}}.

EXCERPTS:
{excerpts}"""

ENGINE = """You are an investment-thesis reasoning engine. Given a BLINDED description of an emerging technical
capability, reason about its trajectory. Do NOT try to name the specific method. Reason concretely about:
(1) capabilities it unlocks, (2) applications/products it makes feasible, (3) markets that could form,
(4) why-now enabling conditions, (5) the kinds of companies that would build on it. Then a one-line thesis.
Return ONLY JSON: {{"capabilities": [...], "applications": [...], "markets": [...], "why_now": "...",
"company_types": [...], "thesis_one_line": "..."}}.

BLINDED DESCRIPTION:
{dossier}"""

GRADE = """You are an INDEPENDENT evaluator. Below is an investment thesis about an emerging technology, written
by an analyst who did NOT know its eventual outcome. Without trying to recognize the specific technology, rate
your CONVICTION (integer 0-100) that the underlying capability will prove a DURABLE, widely-adopted, foundational
technology, as opposed to a hyped idea that fades. Judge only the substance of the reasoning. Also GUESS what the
specific technology might be (a leakage check), and give a one-sentence rationale.
Return ONLY JSON: {{"conviction": <int 0-100>, "concept_guess": "...", "rationale": "..."}}.

THESIS:
{thesis}"""


def _call(client, model, prompt, tries=3):
    from google.genai import types
    cfg = types.GenerateContentConfig(temperature=0, response_mime_type="application/json")
    for i in range(tries):
        try:
            r = client.models.generate_content(model=model, contents=prompt, config=cfg)
            return json.loads(r.text)
        except Exception as e:
            if i == tries - 1:
                return {"_error": str(e)[:120]}
    return {}


def main(argv=None):
    import numpy as np
    import pandas as pd
    from marketbrain.labels.gemini_grounding import make_client
    from marketbrain.eval import metrics

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15, help="themes per class (durable / fad)")
    ap.add_argument("--min-words", type=int, default=2, help="prefer multi-word (real-concept) themes")
    ap.add_argument("--excerpts", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)

    scores = pd.read_parquet("artifacts/phase1/scores.parquet")
    scores["nw"] = scores["theme_id"].str.split().str.len()
    pool = scores[scores["nw"] >= args.min_words]
    rng = np.random.RandomState(0)
    def pick(lbl):
        s = pool[pool["label"] == lbl]["theme_id"].tolist()
        return list(rng.permutation(s))[: args.n]
    themes = [(t, 1) for t in pick("positive")] + [(t, 0) for t in pick("hyped_died")]
    print(f"[m2] pilot themes: {sum(y for _,y in themes)} durable + {sum(1-y for _,y in themes)} fad "
          f"(multi-word, from arXiv phase1)", flush=True)

    print("[m2] loading corpus for excerpts...", flush=True)
    corpus = pd.read_parquet("data/derived/corpus.parquet")[["month", "title", "abstract"]].dropna()
    corpus = corpus.sort_values("month")
    blob = (corpus["title"].str.lower() + ". " + corpus["abstract"].str.lower())

    def excerpts_for(theme):
        hit = corpus[blob.str.contains(theme, regex=False, na=False)].head(args.excerpts)
        if hit.empty:
            return None
        return "\n---\n".join(f"[{r.month}] {r.title.strip()}\n{r.abstract.strip()[:900]}" for r in hit.itertuples())

    client = make_client()

    def run_theme(item):
        theme, y = item
        exc = excerpts_for(theme)
        if not exc:
            return {"theme": theme, "label": y, "skip": "no excerpts"}
        dossier = _call(client, FLASH, REDACT.format(excerpts=exc))
        if "_error" in dossier:
            return {"theme": theme, "label": y, "skip": "redact:" + dossier["_error"]}
        thesis = _call(client, FLASH, ENGINE.format(dossier=json.dumps(dossier)))
        if "_error" in thesis:
            return {"theme": theme, "label": y, "skip": "engine:" + thesis["_error"]}
        # grader sees ONLY the thesis (no dossier, no label, no name)
        grade = _call(client, PRO, GRADE.format(thesis=json.dumps(thesis)))
        if "_error" in grade or "conviction" not in grade:
            return {"theme": theme, "label": y, "skip": "grade:" + str(grade)[:80]}
        guess = str(grade.get("concept_guess", "")).lower()
        tw = set(theme.split())
        leaked = bool(tw & set(guess.split())) and len(tw & set(guess.split())) >= max(1, len(tw) // 2)
        return {"theme": theme, "label": y, "conviction": int(grade["conviction"]),
                "guess": guess, "leaked": leaked, "thesis": thesis.get("thesis_one_line", "")}

    print(f"[m2] running redact->engine->grade on {len(themes)} themes ({args.workers} workers)...", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(run_theme, themes))
    ok = [r for r in rows if "conviction" in r]
    skipped = [r for r in rows if "conviction" not in r]
    print(f"[m2] graded {len(ok)}/{len(themes)}  (skipped {len(skipped)})", flush=True)
    if len(ok) < 8:
        print("[m2] too few graded to judge -- inspect skips:", [r.get('skip') for r in skipped][:5]); return

    df = pd.DataFrame(ok)
    conv = df["conviction"].to_numpy(float)
    y = df["label"].to_numpy(int)
    auc_all = metrics.roc_auc(conv, y)
    clean = df[~df["leaked"]]
    auc_clean = metrics.roc_auc(clean["conviction"].to_numpy(float), clean["label"].to_numpy(int)) if clean["label"].nunique() == 2 else float("nan")
    leak_rate = df["leaked"].mean()
    mp, mf = conv[y == 1].mean(), conv[y == 0].mean()

    print(f"\n[m2] ===== M2.0 GATE PILOT RESULT =====")
    print(f"[m2] conviction AUC vs durability : {auc_all:.3f}   (n={len(df)})")
    print(f"[m2] mean conviction  durable={mp:.1f}  fad={mf:.1f}   (separation {mp-mf:+.1f})")
    print(f"[m2] de-identification leakage    : {leak_rate:.0%} of themes the grader could name")
    print(f"[m2] AUC on non-leaked subset     : {auc_clean:.3f}  (n={len(clean)})")
    verdict = ("PASS — engine discriminates on blinded dossiers" if auc_all >= 0.62 and leak_rate < 0.5
               else ("KILL — no discrimination (auc~0.5): engine is a narrative generator" if auc_all < 0.56
                     else "WEAK/INCONCLUSIVE — modest or leakage-confounded; needs post-cutoff themes"))
    print(f"[m2] VERDICT: {verdict}")
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "pilot_scores.parquet")
    (OUT / "summary.json").write_text(json.dumps(
        {"n": len(df), "auc_all": auc_all, "auc_clean": auc_clean, "leak_rate": float(leak_rate),
         "mean_durable": float(mp), "mean_fad": float(mf), "verdict": verdict,
         "skipped": len(skipped)}, indent=2))
    print(f"[m2] wrote {OUT}/summary.json + pilot_scores.parquet")
    print("[m2] NOTE: ground truth is research-durability (necessary substrate); market conversion is the harder")
    print("[m2]       downstream test. This pilot gates whether the engine can tell real from hype at all.")


if __name__ == "__main__":
    main()
