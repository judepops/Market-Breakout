"""Gemini web-search grounding for the success label — verify + cite + gap-fill Crunchbase.

LABEL-only (allowed to use the future + an LLM); never imported by a feature module. For each company it
asks a grounded Gemini model whether the company reached >=$500M (raise / valuation / M&A / IPO), then:
  * takes the citation from the structured ``grounding_metadata`` URIs (NOT the model's prose, which is
    unreliable — observed it inventing "the source is KuCoin");
  * runs a **deterministic re-check**: ``verified`` only if the parsed value >= threshold AND the answer
    drew on **>= ``MIN_SOURCES`` distinct grounding sources** (multi-source corroboration). One or zero
    sources is treated as un-corroborated and flagged unverified.

Note: ``>= MIN_SOURCES`` grounding chunks means the model triangulated across that many retrieved pages —
a strong anti-hallucination proxy, though not a guarantee that each page independently states the exact
figure. Combined with the Crunchbase structured value (the ``crunchbase+gemini`` path), a verified outlier
has two independent confirmations: the Crunchbase number and the multi-source grounded search.

``google.genai`` is imported lazily so the label-logic tests load without the SDK.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date

from ..env import get_gemini_api_key

DEFAULT_MODEL = "gemini-2.5-flash"
MIN_SOURCES = 3  # distinct grounding sources required to mark a result verified (anti-hallucination)

# A natural research prompt (so the model actually invokes Google Search and returns real citations),
# with a trailing JSON line for structure. A rigid "JSON only" prompt suppresses the search — the model
# then answers from memory and fabricates a source_url, which the re-check (rightly) rejects.
_PROMPT = (
    "Using Google Search, research whether {name} (Crunchbase permalink '{permalink}'), an LLM "
    "developer-tools company, has reached at least ${threshold:,} USD via ANY of: total funding raised, "
    "post-money valuation, or acquisition/M&A price. Briefly state the key facts with figures and dates. "
    "Then, on the FINAL line, output only a JSON object with keys: "
    '{{"is_outlier": boolean, "metric": "raise"|"valuation"|"ma"|"ipo" or null, '
    '"value_usd": number or null, "milestone_date": "YYYY-MM-DD" or null, '
    '"public_date": "YYYY-MM-DD" or null, "source_url": string or null}}.'
)


@dataclass(frozen=True)
class GroundedResult:
    permalink: str
    is_outlier: "bool | None"
    metric: "str | None"
    value_usd: "int | None"
    milestone_date: "date | None"
    public_date: "date | None"
    evidence_url: "str | None"      # taken from grounding_metadata URIs (preferred) or parsed source_url
    grounding_urls: "tuple[str, ...]"
    verified: bool                  # deterministic: value >= threshold AND >= MIN_SOURCES grounding sources
    raw_text: str


def make_client(api_key: "str | None" = None):
    """Construct a google-genai client (lazy import). Reads the key from .env if not given."""
    from google import genai

    return genai.Client(api_key=api_key or get_gemini_api_key())


def _parse_date(s) -> "date | None":
    if not s or not isinstance(s, str):
        return None
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        return None


def _extract_json(text: str) -> dict:
    """Pull the JSON object out of the model text (prose precedes it; the JSON is on the final line).

    Scans flat ``{...}`` blocks and returns the LAST one that parses (the answer), so leading prose that
    happens to contain a brace can't derail it.
    """
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", text, re.DOTALL))):
        try:
            return json.loads(m.group(0))
        except (json.JSONDecodeError, ValueError):
            continue
    return {}


def _grounding_uris(response) -> "tuple[str, ...]":
    """Real cited URLs from grounding_metadata.grounding_chunks (the trustworthy citations)."""
    try:
        gm = response.candidates[0].grounding_metadata
        chunks = gm.grounding_chunks or []
        uris = [c.web.uri for c in chunks if getattr(c, "web", None) and c.web.uri]
        return tuple(dict.fromkeys(uris))  # dedup, keep order
    except (AttributeError, IndexError, TypeError):
        return ()


def parse_response(
    permalink: str, text: str, grounding_urls, *, threshold_usd: int, min_sources: int = MIN_SOURCES
) -> GroundedResult:
    """Pure: turn raw model text + grounding URIs into a re-checked GroundedResult (testable offline)."""
    d = _extract_json(text)
    value = d.get("value_usd")
    value = int(value) if isinstance(value, (int, float)) else None
    url = grounding_urls[0] if grounding_urls else (d.get("source_url") or None)
    verified = bool(value is not None and value >= threshold_usd and len(grounding_urls) >= min_sources)
    return GroundedResult(
        permalink=permalink,
        is_outlier=d.get("is_outlier") if isinstance(d.get("is_outlier"), bool) else None,
        metric=d.get("metric") if d.get("metric") in {"raise", "valuation", "ma", "ipo"} else None,
        value_usd=value,
        milestone_date=_parse_date(d.get("milestone_date")),
        public_date=_parse_date(d.get("public_date")),
        evidence_url=url,
        grounding_urls=tuple(grounding_urls),
        verified=verified,
        raw_text=text,
    )


def ground_company(
    name: str, permalink: str, *, threshold_usd: int = 500_000_000,
    client=None, model: str = DEFAULT_MODEL, min_sources: int = MIN_SOURCES,
) -> GroundedResult:
    """Make one grounded Gemini call for a company and return a re-checked GroundedResult (hits network)."""
    from google.genai import types

    client = client or make_client()
    prompt = _PROMPT.format(name=name, permalink=permalink, threshold=threshold_usd)
    cfg = types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())], temperature=0
    )
    resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
    return parse_response(permalink, resp.text or "", _grounding_uris(resp),
                          threshold_usd=threshold_usd, min_sources=min_sources)
