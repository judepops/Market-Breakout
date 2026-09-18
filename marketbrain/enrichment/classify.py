"""LLM primary-market classifier for AMBIGUOUS (multi-tag) companies - the clean-partition tie-break.

A company tagged with several taxonomy markets is assigned to its PRIMARY one by Gemini (semantic) rather
than by arbitrary tag-priority. Batched (many companies per call) and non-grounded (classification from the
given name + categories), so it is fast and cheap. The model may only choose from each company's own
candidate markets; an out-of-candidates answer is dropped (caller falls back to the first candidate).
"""

from __future__ import annotations

import json
import re

DEFAULT_MODEL = "gemini-2.5-flash"


def _build_prompt(batch) -> str:
    lines = [
        "For each company below, choose the ONE market that best describes its PRIMARY business, chosen "
        "strictly from that company's candidate list. Return ONLY a JSON object mapping each company id "
        '(the number) to its chosen market key, e.g. {"0": "devops", "1": "database"}.', "",
    ]
    for i, (name, cands, cats) in enumerate(batch):
        lines.append(f"{i}. {name} | candidates: {cands} | categories: {cats[:160]}")
    return "\n".join(lines)


def parse_assignments(text: str, batch) -> dict[int, str]:
    """Pure: model JSON {id: key} -> {index: key}, keeping only keys in that company's candidate list."""
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return {}
    try:
        d = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return {}
    out = {}
    for k, v in d.items():
        try:
            i = int(k)
        except (ValueError, TypeError):
            continue
        if 0 <= i < len(batch) and isinstance(v, str) and v in batch[i][1]:
            out[i] = v
    return out


def classify_batch(batch, *, client=None, model: str = DEFAULT_MODEL) -> dict[int, str]:
    """batch = [(name, candidate_keys, categories_str)]; return {index -> chosen market key}. Hits network."""
    if not batch:
        return {}
    if client is None:
        from ..labels.gemini_grounding import make_client
        client = make_client()
    resp = client.models.generate_content(model=model, contents=_build_prompt(batch),
                                          config={"temperature": 0})
    return parse_assignments(resp.text or "", batch)
