"""DECISIVE test of the LLM commercial signal: is it PREDICTION or RECOGNITION? The un-blinded LLM predicted
conversion above chance (0.58-0.62), but it knows which ideas became companies. Here we DE-IDENTIFY each idea
(a redactor strips the name + identifying terms, paraphrases) and re-score commercial potential blind. If the
signal survives de-identification -> real predictive commercial judgment (a finance link). If it vanishes ->
it was the LLM recalling known winners, and conversion is unpredictable from de-identified research.

    ~/miniconda3/envs/marketbrain/bin/python -m experiments.converter_predict_deid

Two blind LLM stages per theme (gemini-2.5-flash): redactor (name+terms -> de-identified description) then a
fresh-context commercial scorer (description -> 0-100 + concept guess for a leakage probe). Same converter
outcome, CV-AUC, permutation null as converter_predict. Reports de-id AUC + leakage rate.
"""
from __future__ import annotations
import json
import pathlib
import warnings
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore")
SB = pathlib.Path("artifacts/stageB")
OUT = pathlib.Path("artifacts/converter_predict")

REDACT = """Here is an emerging research concept, given by its defining phrase and associated terms. Rewrite it
as a DE-IDENTIFIED technical description of the method/capability and the problem it addresses, WITHOUT naming
it and WITHOUT any proper-noun or signature term that would identify the specific method. Paraphrase generic
mechanism only; do not hint at its popularity or any company. Return ONLY JSON: {{"description": "..."}}.

PHRASE: {label}
TERMS: {terms}"""

SCORE = """You are a venture analyst. Below is a de-identified description of an emerging research capability.
Without trying to recognise the specific method, estimate the likelihood (0-100) that it leads to at least one
startup raising >=$500M in venture funding within ~7 years, based on commercial applicability, market size, and
defensibility. Also GUESS what the specific technology might be (a leakage check). Return ONLY JSON:
{{"score": <int 0-100>, "concept_guess": "..."}}.

DESCRIPTION: {desc}"""


def _llm(client, prompt, tries=3):
    from google.genai import types
    cfg = types.GenerateContentConfig(temperature=0, response_mime_type="application/json")
    for i in range(tries):
        try:
            return json.loads(client.models.generate_content(model="gemini-2.5-flash", contents=prompt, config=cfg).text)
        except Exception as e:
            if i == tries - 1:
                return {"_err": str(e)[:80]}
    return {}


def main():
    import numpy as np, pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from marketbrain.labels.gemini_grounding import make_client

    th = pd.read_parquet(SB / "themes.parquet")[["theme_id", "label", "T0", "top_terms"]].copy()
    th["T0_ym"] = th["T0"].astype(str).str[:7]
    comp = pd.read_parquet(SB / "company_level_test.parquet")[["permalink", "theme_id", "sim", "founded_on"]]
    raised = pd.read_parquet("data/raw/labels/company_rounds.parquet").groupby("permalink")["raised_usd"].sum()
    comp["tr"] = comp["permalink"].map(raised).fillna(0.0)
    comp["found_ym"] = comp["founded_on"].astype(str).str[:7]
    comp = comp.merge(th[["theme_id", "T0_ym"]], on="theme_id", how="inner")
    comp = comp[(comp["sim"] >= 0.6) & (comp["found_ym"] >= comp["T0_ym"])]
    def converter(x):
        return th["theme_id"].isin(set(comp[comp["tr"] >= x]["theme_id"])).astype(int).to_numpy()

    client = make_client()
    def run(i):
        terms = str(th["top_terms"].iloc[i])[:400]
        d = _llm(client, REDACT.format(label=th["label"].iloc[i], terms=terms))
        if "description" not in d:
            return (None, "", "")
        s = _llm(client, SCORE.format(desc=d["description"]))
        return (s.get("score"), str(s.get("concept_guess", "")).lower(), th["label"].iloc[i].lower())
    print(f"[deid] de-identifying + blind-scoring {len(th)} themes...", flush=True)
    with ThreadPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(run, range(len(th))))
    th["deid"] = pd.to_numeric([r[0] for r in res], errors="coerce")
    th["deid"] = th["deid"].fillna(th["deid"].median())
    # leakage: grader's guess shares the theme's distinctive tokens
    def leaked(guess, name):
        gt, nt = set(guess.split()), set(w for w in name.split() if len(w) > 3)
        return bool(nt) and len(gt & nt) >= max(1, len(nt) // 2)
    th["leaked"] = [leaked(r[1], r[2]) for r in res]
    leak_rate = th["leaked"].mean()

    def cv_auc(x, y, seed=0):
        skf = StratifiedKFold(5, shuffle=True, random_state=seed); oof = np.zeros(len(y))
        for tr_i, te_i in skf.split(x, y):
            if len(np.unique(y[tr_i])) < 2:
                continue
            m = LogisticRegression(max_iter=1000).fit(x[tr_i], y[tr_i])
            oof[te_i] = m.predict_proba(x[te_i])[:, 1]
        return roc_auc_score(y, oof)
    def perm_p(x, y, obs, B=500, seed=0):
        rng = np.random.RandomState(seed)
        return (sum(cv_auc(x, rng.permutation(y)) >= obs for _ in range(B)) + 1) / (B + 1)

    print(f"[deid] de-identification leakage: {leak_rate:.0%} of themes the scorer could name\n", flush=True)
    summary = {"leak_rate": float(leak_rate)}
    for X, tag in [(50e6, ">=$50M"), (500e6, ">=$500M")]:
        y = converter(X); x = th[["deid"]].to_numpy(float)
        auc = cv_auc(x, y); p = perm_p(x, y, auc)
        # also on the non-leaked subset
        mask = ~th["leaked"].to_numpy()
        auc_clean = cv_auc(x[mask], y[mask]) if y[mask].sum() >= 5 and (1 - y[mask]).sum() >= 5 else float("nan")
        verdict = "SURVIVES (real commercial judgment)" if (auc > 0.55 and p < 0.05) else "GONE (was recognition)"
        print(f"[deid] {tag}: base={y.mean():.3f}  de-id LLM CV-AUC={auc:.3f}  perm-p={p:.3f}  (non-leaked {auc_clean:.3f})  -> {verdict}", flush=True)
        summary[tag] = {"base_rate": float(y.mean()), "deid_auc": float(auc), "perm_p": float(p), "auc_nonleaked": float(auc_clean), "verdict": verdict}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "deid_summary.json").write_text(json.dumps(summary, indent=2))
    th[["theme_id", "label", "deid", "leaked"]].to_parquet(OUT / "deid_scores.parquet")
    print(f"\n[deid] wrote {OUT}/deid_summary.json")
    print("[deid] compare to un-blinded (0.577 @$50M, 0.618 @$500M): if de-id holds near those -> PREDICTION;")
    print("[deid]       if it drops to ~0.5 -> the un-blinded signal was RECOGNITION of known winners.")


if __name__ == "__main__":
    main()
