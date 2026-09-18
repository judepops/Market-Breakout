"""Can [structural signal + LLM commercial read] predict which research ideas CONVERT to a big-funded company?
This is the untested combination: the detector score (structure) + an LLM's commercial-potential score (narrative)
-> the real CONVERTER outcome (theme spawns a >=$X company). The converter base-rate showed durability barely
moves conversion (20% vs 16% @ $500M); this asks whether an LLM's market read adds an ORTHOGONAL finance signal.

    ~/miniconda3/envs/marketbrain/bin/python -m experiments.converter_predict

Predictors (per theme): sig_score (detector P(durable)), momentum, llm_commercial (gemini 0-100, UN-BLINDED =
full-hindsight upper bound). Outcome: theme has >=1 bridged company (founded>=T0, sim>=0.6) raising >=$X.
5-fold CV AUC for each predictor + combined (GBM), permutation null (B=500). If even the hindsight LLM+structure
can't beat chance -> the research side genuinely cannot predict conversion (boundary complete). If it can ->
a finance link; then re-run de-identified to check it isn't just the LLM recognising known winners.
"""
from __future__ import annotations
import json
import pathlib
import warnings
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore")
SB = pathlib.Path("artifacts/stageB")
OUT = pathlib.Path("artifacts/converter_predict")

PROMPT = """You are a venture analyst. Below is an emerging research idea (its defining phrase and top associated
terms). Estimate the likelihood that this idea will lead to at least one startup that raises >=$500M in venture
funding within ~7 years. Consider commercial applicability, market size, and defensibility. Return ONLY JSON:
{{"score": <int 0-100>, "reason": "<one sentence>"}}.

IDEA: {label}
TOP TERMS: {terms}"""


def _llm(client, model, prompt, tries=3):
    from google.genai import types
    cfg = types.GenerateContentConfig(temperature=0, response_mime_type="application/json")
    for i in range(tries):
        try:
            return json.loads(client.models.generate_content(model=model, contents=prompt, config=cfg).text)
        except Exception as e:
            if i == tries - 1:
                return {"score": None, "_err": str(e)[:80]}
    return {"score": None}


def main():
    import numpy as np, pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from marketbrain.labels.gemini_grounding import make_client

    th = pd.read_parquet(SB / "themes.parquet")
    cols = [c for c in ["theme_id", "label", "T0", "sig_score", "momentum", "theme_text", "top_terms"] if c in th.columns]
    th = th[cols].copy()
    th["T0_ym"] = th["T0"].astype(str).str[:7]

    # converter outcome via the validated bridge (founded>=T0, sim>=0.6)
    comp = pd.read_parquet(SB / "company_level_test.parquet")[["permalink", "theme_id", "sim", "founded_on"]]
    raised = pd.read_parquet("data/raw/labels/company_rounds.parquet").groupby("permalink")["raised_usd"].sum()
    comp["tr"] = comp["permalink"].map(raised).fillna(0.0)
    comp["found_ym"] = comp["founded_on"].astype(str).str[:7]
    comp = comp.merge(th[["theme_id", "T0_ym"]], on="theme_id", how="inner")
    comp = comp[(comp["sim"] >= 0.6) & (comp["found_ym"] >= comp["T0_ym"])]

    def converter(x):
        conv = set(comp[comp["tr"] >= x]["theme_id"])
        return th["theme_id"].isin(conv).astype(int).to_numpy()

    # LLM commercial score per theme (un-blinded upper bound)
    client = make_client()
    terms = th["top_terms"].astype(str) if "top_terms" in th.columns else th.get("theme_text", th["label"]).astype(str)
    def score_one(i):
        r = _llm(client, "gemini-2.5-flash", PROMPT.format(label=th["label"].iloc[i], terms=terms.iloc[i][:400]))
        return r.get("score")
    print(f"[conv] scoring {len(th)} themes with the LLM commercial predictor...", flush=True)
    with ThreadPoolExecutor(max_workers=8) as ex:
        th["llm"] = list(ex.map(score_one, range(len(th))))
    th["llm"] = pd.to_numeric(th["llm"], errors="coerce").fillna(th["llm"].median() if th["llm"].notna().any() else 50)
    for c in ["sig_score", "momentum"]:
        if c not in th.columns:
            th[c] = 0.0
        th[c] = pd.to_numeric(th[c], errors="coerce").fillna(th[c].median())

    def cv_auc(X, y, seed=0):
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        oof = np.zeros(len(y))
        for tr_i, te_i in skf.split(X, y):
            if len(np.unique(y[tr_i])) < 2:
                continue
            m = HistGradientBoostingClassifier(max_depth=2, max_iter=200, learning_rate=0.05,
                                               min_samples_leaf=15, random_state=0).fit(X[tr_i], y[tr_i])
            oof[te_i] = m.predict_proba(X[te_i])[:, 1]
        return roc_auc_score(y, oof)

    def perm_p(X, y, obs, B=500, seed=0):
        rng = np.random.RandomState(seed)
        cnt = sum(cv_auc(X, rng.permutation(y)) >= obs for _ in range(B))
        return (cnt + 1) / (B + 1)

    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for X_usd, tag in [(50e6, ">=$50M"), (500e6, ">=$500M")]:
        y = converter(X_usd)
        base = y.mean()
        print(f"\n[conv] outcome {tag}: {int(y.sum())}/{len(y)} converters (base rate {base:.3f})", flush=True)
        feats = {
            "sig_score (structure)": th[["sig_score"]].to_numpy(float),
            "momentum": th[["momentum"]].to_numpy(float),
            "LLM commercial (hindsight)": th[["llm"]].to_numpy(float),
            "sig+momentum+LLM (combined)": th[["sig_score", "momentum", "llm"]].to_numpy(float),
        }
        block = {"base_rate": float(base), "n_pos": int(y.sum())}
        for name, X in feats.items():
            auc = cv_auc(X, y)
            p = perm_p(X, y, auc)
            flag = "  <-- BEATS CHANCE" if (auc > 0.5 and p < 0.05) else ""
            print(f"   {name:32s} CV-AUC={auc:.3f}  perm-p={p:.3f}{flag}", flush=True)
            block[name] = {"auc": float(auc), "perm_p": float(p)}
        summary[tag] = block

    th[["theme_id", "label", "sig_score", "momentum", "llm"]].to_parquet(OUT / "scores.parquet")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[conv] wrote {OUT}/summary.json")
    print("[conv] read: any predictor BEATS CHANCE -> a research-side finance link (then de-identify to confirm).")
    print("[conv]       all at chance (esp. the hindsight LLM) -> the boundary is complete; conversion is unpredictable")
    print("[conv]       from the research side, strengthening the paper's negative.")


if __name__ == "__main__":
    main()
