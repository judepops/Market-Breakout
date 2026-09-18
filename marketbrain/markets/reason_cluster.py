"""LLM (Gemini) naming + precision pass over the deterministic phrase clusters -> named emerging markets.

The LLM NAMES and PRUNES clusters of already-granularity-gated phrases (one call per cluster, titles only, not
raw text) and flags non-markets (`is_market`=False kills "percentage-point" / "b-a" / "failure mode"). It can
NOT add phrases or invent granularity -- granularity is fixed upstream by the gates.

Robustness (audit P0): a transient API/parse failure is NOT a verdict. `_call_model` retries with backoff and,
on persistent failure, returns status="error" -- an ABSTAIN that is neither named nor cached (so a real market
is never permanently deleted and mislabeled "junk" by a one-off network blip). Definitive verdicts are cached
by (model, phrases) content-hash so re-runs are free. JSON is pulled with the repo's hardened balanced-brace
extractor (`labels.gemini_grounding._extract_json`), not a greedy regex.
"""
from __future__ import annotations
import hashlib
import json
import pathlib
import time

CACHE = pathlib.Path("data/derived/markets_llm_cache.jsonl")
MODEL = "gemini-2.5-pro"
RETRIES = 3
PROMPT_VERSION = "v2"        # bump on any PROMPT change -> old cache entries are invalidated (audit P0)
PROMPT = (
    "You are curating a taxonomy of EMERGING research micro-markets in AI / dev-tools. Below is a cluster of "
    "technical phrases that co-occur in recent arXiv papers, with example paper titles.\n\n"
    "PHRASES: {phrases}\nEXAMPLE TITLES:\n{titles}\n\n"
    "Decide:\n"
    "1. is_market: TRUE only if the PHRASE(S) THEMSELVES name a real, specific technical idea / method / "
    'capability (e.g. "on-policy distillation", "agentic tool use", "verifiable-reward RL"). Judge the PHRASES, '
    "not just the example papers -- do NOT invent a market from a lone generic or ambiguous word merely because "
    'the example papers share a topic. Set FALSE for: generic words, metrics, statistics, or text artifacts '
    '("percentage-point", "failure mode", "log-probability", "b--"); a single ambiguous common word ("defaults", '
    '"dissociation", "hysteresis"); and BROAD ESTABLISHED fields that are not newly emerging ("semi-supervised '
    'learning", "reinforcement learning").\n'
    "2. market_name: if TRUE, ONE specific name at the altitude of \"RL with verifiable rewards\" (NOT a broad "
    'field like "LLM tooling").\n'
    "3. member_phrases: the subset of the INPUT phrases that genuinely belong. You may NOT add phrases.\n"
    "4. coherence: 0.0-1.0.\n\n"
    'Return ONLY JSON: {{"is_market": bool, "market_name": str, "member_phrases": [str], "coherence": float}}'
)


def _key(phrases, model):
    return hashlib.sha1((PROMPT_VERSION + "|" + model + "|" + "|".join(sorted(phrases))).encode()).hexdigest()


def _load_cache():
    cache = {}
    if CACHE.exists():
        for line in CACHE.read_text().splitlines():
            try:
                d = json.loads(line)
                cache[d["key"]] = d["result"]
            except (json.JSONDecodeError, KeyError):
                pass
    return cache


def _call_model(client, model, prompt):
    """Return (parsed_dict, status). status='ok' on a parsed response; 'error' after RETRIES (transient/abstain)."""
    from ..labels.gemini_grounding import _extract_json
    for attempt in range(RETRIES):
        try:
            resp = client.models.generate_content(model=model, contents=prompt, config={"temperature": 0})
            d = _extract_json(getattr(resp, "text", "") or "")
            if d:
                return d, "ok"
        except Exception:                                  # noqa: BLE001 (network/transport -> retry, then abstain)
            pass
        if attempt < RETRIES - 1:
            time.sleep(2 ** attempt)
    return {}, "error"


def _abstain(phrases):
    return {"status": "error", "is_market": False, "market_name": "",
            "member_phrases": [], "coherence": 0.0, "input_phrases": sorted(phrases)}


def name_clusters(clusters_with_titles, *, client=None, model=MODEL, use_cache=True):
    """clusters_with_titles: iterable of (phrases:list, titles:list). Returns one result dict per cluster.

    Each result carries status: "ok" (a definitive LLM verdict, cached) or "error" (transient failure -> abstain,
    NOT cached; the caller should retry it next run, never count it as junk).
    """
    if client is None:
        from ..labels.gemini_grounding import make_client
        client = make_client()
    cached = _load_cache() if use_cache else {}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for phrases, titles in clusters_with_titles:
        k = _key(phrases, model)
        if k in cached:
            out.append(cached[k])
            continue
        prompt = PROMPT.format(phrases=", ".join(sorted(phrases)), titles="\n".join(f"- {t}" for t in titles[:3]))
        d, status = _call_model(client, model, prompt)
        if status == "error":
            out.append(_abstain(phrases))               # abstain: do NOT cache, do NOT treat as junk
            continue
        pset = set(phrases)
        res = {"status": "ok", "is_market": bool(d.get("is_market", False)),
               "market_name": str(d.get("market_name", "")).strip(),
               "member_phrases": [p for p in d.get("member_phrases", []) if p in pset],
               "coherence": float(d.get("coherence", 0) or 0),
               "input_phrases": sorted(phrases)}
        out.append(res)
        if use_cache:
            with CACHE.open("a") as f:
                f.write(json.dumps({"key": k, "result": res}) + "\n")
    return out
