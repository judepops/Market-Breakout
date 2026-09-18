"""LLM-scraped funding - a Crunchbase parallel from web-search grounding (Gemini).

A capital signal sourced from funding NEWS rather than Crunchbase's structured (and, in a live system,
lagging) feed. Reuses the label's grounded-Gemini pattern: a natural research prompt so the model actually
searches, structured JSON on the final line, and citations taken from the real grounding_metadata URIs (not
the model's prose). Per company it returns total raised + the most recent round (date + amount) + sources.

This is a FEATURE-side use of an LLM (allowed because it is sourced + citation-backed, and we test it head to
head against Crunchbase rather than trusting it blindly). NOT imported by the deterministic v1 corpus.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date

DEFAULT_MODEL = "gemini-2.5-flash"
MIN_SOURCES = 3  # >= this many distinct grounding sources required to trust a value (anti-hallucination)

_PROMPT = (
    "Using Google Search, research the venture funding HISTORY of the company {name}{ctx}. "
    "List EVERY disclosed funding round you can find - for each round its DATE, amount (USD), and stage "
    "(pre_seed / seed / angel / series_a / series_b / ... / ipo / acquisition). Also the total raised. "
    "Briefly state the facts with figures and dates. Then on the FINAL line output only a JSON object: "
    '{{"total_raised_usd": number or null, "rounds": [{{"date": "YYYY-MM-DD", "amount_usd": number or null, '
    '"stage": string or null}}], "source_url": string or null}}. Dates are REQUIRED for each round.'
)


@dataclass(frozen=True)
class FundingRound:
    date: "date | None"
    amount_usd: "int | None"
    stage: "str | None"


@dataclass(frozen=True)
class FundingResult:
    name: str
    total_raised_usd: "int | None"
    rounds: "tuple[FundingRound, ...]"   # the dated round history (for formation-window analysis)
    latest_round_date: "date | None"     # derived: the most recent dated round
    latest_round_usd: "int | None"
    evidence_url: "str | None"
    grounding_urls: "tuple[str, ...]"
    verified: bool                       # a total was parsed AND >= MIN_SOURCES distinct grounding sources
    raw_text: str


def _num(v) -> "int | None":
    return int(v) if isinstance(v, (int, float)) else None


def _date(s) -> "date | None":
    if not isinstance(s, str):
        return None
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        return None


def _last_json(text: str) -> dict:
    """Extract the last TOP-LEVEL {...} object that parses (not a nested 'rounds' element)."""
    objs, depth, start = [], 0, None
    for i, c in enumerate(text or ""):
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                objs.append(text[start:i + 1])
                start = None
    for s in reversed(objs):
        try:
            d = json.loads(s)
            if isinstance(d, dict):
                return d
        except (json.JSONDecodeError, ValueError):
            continue
    return {}


def parse_funding(name: str, text: str, grounding_urls, *, min_sources: int = MIN_SOURCES) -> FundingResult:
    """Pure: model text + grounding URIs -> a FundingResult with the dated round history (testable offline).

    ``verified`` requires a parsed total AND >= ``min_sources`` distinct grounding sources (multi-source
    corroboration), the same anti-hallucination bar as the grounded label.
    """
    d = _last_json(text)
    rounds = tuple(
        FundingRound(_date(r.get("date")), _num(r.get("amount_usd")),
                     r.get("stage") if isinstance(r.get("stage"), str) else None)
        for r in d.get("rounds", []) if isinstance(r, dict)
    )
    dated = [r for r in rounds if r.date]
    latest = max(dated, key=lambda r: r.date) if dated else None
    total = _num(d.get("total_raised_usd"))
    return FundingResult(
        name=name, total_raised_usd=total, rounds=rounds,
        latest_round_date=(latest.date if latest else None),
        latest_round_usd=(latest.amount_usd if latest else None),
        evidence_url=(grounding_urls[0] if grounding_urls else d.get("source_url") or None),
        grounding_urls=tuple(grounding_urls),
        verified=bool(total is not None and len(grounding_urls) >= min_sources),
        raw_text=text or "")


def ground_funding(name: str, *, context: str = "", client=None, model: str = DEFAULT_MODEL) -> FundingResult:
    """One grounded Gemini call for a company's funding (hits the network)."""
    from google.genai import types

    from ..labels.gemini_grounding import _grounding_uris, make_client
    client = client or make_client()
    prompt = _PROMPT.format(name=name, ctx=(f" ({context})" if context else ""))
    cfg = types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())], temperature=0)
    resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
    return parse_funding(name, resp.text or "", _grounding_uris(resp))
