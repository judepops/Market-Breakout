"""LLM (Gemini) per-market arXiv query generation - clean signal anchors for the research dimension.

Crude name-matching conflated a market with unrelated academic uses of its name ("solar" -> solar PHYSICS,
not solar-energy startups). This asks Gemini for the right arXiv terms + categories per market and builds a
disambiguated query, so the research signal measures the market's actual research. Reuses the label's Gemini
client. The pure parts (parse, build_query) are unit-tested; generation hits the network.
"""

from __future__ import annotations

import json
import re

DEFAULT_MODEL = "gemini-2.5-flash"
# arXiv category codes: cs.LG, physics.app-ph, cond-mat.mtrl-sci, q-fin.GN, math (subcat may be >2 chars / hyphenated)
_VALID_CAT = re.compile(r"^[a-z-]+(\.[A-Za-z-]+)?$")

_PROMPT = (
    'Build an arXiv search query to measure the RESEARCH activity behind the "{market}" technology startup '
    'market - companies building {market} products - while EXCLUDING unrelated academic uses of the same '
    'words (for example "solar" must mean solar-energy technology, NOT solar astrophysics). '
    'Return ONLY a JSON object with keys: '
    '"terms" (3-6 SHORT plain search phrases, no boolean operators) and '
    '"categories" (2-5 arXiv category codes such as cs.LG, physics.app-ph, q-fin.GN). '
    'If the market has essentially no academic research base, return empty lists.'
)


def parse_terms(text: str) -> "tuple[tuple[str, ...], tuple[str, ...]]":
    """Pure: pull (terms, categories) from the model's JSON text; validates category codes, cleans phrases."""
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return (), ()
    try:
        d = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return (), ()
    terms = tuple(t.strip().strip('"').strip() for t in d.get("terms", []) if isinstance(t, str) and t.strip())
    cats = tuple(c.strip() for c in d.get("categories", []) if isinstance(c, str) and _VALID_CAT.match(c.strip()))
    return terms[:6], cats[:5]


def build_arxiv_query(terms: "tuple[str, ...]", categories: "tuple[str, ...]") -> "str | None":
    """Build a disambiguated arXiv query: (cat:.. OR ..) AND (all:"term" OR ..). None if no terms."""
    if not terms:
        return None
    term_q = " OR ".join(f'all:"{t}"' for t in terms)
    if categories:
        cat_q = " OR ".join(f"cat:{c}" for c in categories)
        return f"({cat_q}) AND ({term_q})"
    return f"({term_q})"


def generate_arxiv_query(market: str, *, client=None, model: str = DEFAULT_MODEL) -> "str | None":
    """Ask Gemini for the market's arXiv terms+categories and build the query (hits the network)."""
    if client is None:
        from ..labels.gemini_grounding import make_client
        client = make_client()
    resp = client.models.generate_content(model=model, contents=_PROMPT.format(market=market),
                                          config={"temperature": 0})
    terms, cats = parse_terms(resp.text or "")
    return build_arxiv_query(terms, cats)


_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,38}/[A-Za-z0-9_.-]{1,100}$")
_REPO_PROMPT = (
    'List up to 8 prominent OPEN-SOURCE GitHub repositories, in "owner/repo" form, whose star growth '
    'reflects real DEVELOPER ADOPTION in the "{market}" technology market - the flagship tools/projects '
    'developers in this market actually use. Return ONLY a JSON object: {{"repos": ["owner/repo", ...]}}. '
    'If the market has essentially no open-source adoption (e.g. a hardware or pharma market), return an '
    'empty list. Use real, well-known repositories only.'
)


def parse_repos(text: str) -> "tuple[str, ...]":
    """Pure: pull a validated, deduped 'owner/repo' list from the model's JSON text."""
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return ()
    try:
        repos = json.loads(m.group(0)).get("repos", [])
    except (json.JSONDecodeError, ValueError):
        return ()
    out = []
    for r in repos:
        if isinstance(r, str):
            r = r.strip().removeprefix("https://github.com/").rstrip("/")
            if _REPO_RE.match(r) and r not in out:
                out.append(r)
    return tuple(out[:8])


def generate_github_repos(market: str, *, client=None, model: str = DEFAULT_MODEL) -> "tuple[str, ...]":
    """Ask Gemini for the market's anchor OSS repos (hits the network). Empty tuple if no OSS adoption."""
    if client is None:
        from ..labels.gemini_grounding import make_client
        client = make_client()
    resp = client.models.generate_content(model=model, contents=_REPO_PROMPT.format(market=market),
                                          config={"temperature": 0})
    return parse_repos(resp.text or "")


_HN_PROMPT = (
    'Give the best Hacker News search query to find discussions about the "{market}" technology startup '
    'market - the keywords developers actually use when posting about {market} tools/companies, EXCLUDING '
    'unrelated uses of the words. Return ONLY a JSON object: {{"query": "a short search string, 2-5 words"}}.'
)


def parse_hn_query(text: str) -> "str | None":
    """Pure: pull the cleaned HN search string from the model's JSON text."""
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return None
    try:
        q = json.loads(m.group(0)).get("query")
    except (json.JSONDecodeError, ValueError):
        return None
    q = q.strip() if isinstance(q, str) else None
    return q or None


def generate_hn_query(market: str, *, client=None, model: str = DEFAULT_MODEL) -> "str | None":
    """Ask Gemini for a clean Hacker News search query for the market (hits the network)."""
    if client is None:
        from ..labels.gemini_grounding import make_client
        client = make_client()
    resp = client.models.generate_content(model=model, contents=_HN_PROMPT.format(market=market),
                                          config={"temperature": 0})
    return parse_hn_query(resp.text or "")
