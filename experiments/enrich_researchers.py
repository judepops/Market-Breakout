"""Enrich top researchers with a profile link + startup-founder flag (Gemini + web search).

    python -m experiments.enrich_researchers --snapshot ../market-brain-app/public/mb-data.json

For each unique top researcher on the Durable/Emerging/Fads lists, one web-grounded
Gemini call resolves: affiliation, a direct profile URL (Google Scholar / homepage),
and whether they've co-founded a startup. Disambiguated by the research areas they
appear on. Verify-or-omit: a profile URL is kept only if it's a real http(s) link;
founder flag only with a named company. Resumable cache: data/derived/researchers/.

Requires GEMINI_API_KEY. Grounded calls cost more than plain ones — ~140 researchers.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import time

MODEL = "gemini-3.5-flash"
STORE = pathlib.Path("data/derived/researchers")
CACHE = STORE / "enrichment.json"

PROMPT = """{name} is an AI/ML researcher who has published on: {areas}.

Using web search, find:
- their primary current affiliation (university or company),
- a direct profile URL — prefer Google Scholar, else a personal/university page (null if you cannot find a real one),
- whether they have FOUNDED or CO-FOUNDED a startup company (not just worked at one).

Reply with ONLY a JSON object, no prose:
{{"affiliation": string, "profile_url": string or null, "is_founder": boolean, "founder_company": string or null, "confidence": "high"|"medium"|"low"}}"""


def parse_enrichment(text: str) -> dict | None:
    """Robustly parse the model's JSON; verify-or-omit the link and founder fields."""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    url = d.get("profile_url")
    if not (isinstance(url, str) and url.startswith(("http://", "https://"))):
        url = None
    is_founder = bool(d.get("is_founder")) and bool(d.get("founder_company"))
    return {
        "affiliation": (d.get("affiliation") or "")[:120] or None,
        "profile_url": url,
        "is_founder": is_founder,
        "founder_company": (d.get("founder_company") or None) if is_founder else None,
        "confidence": d.get("confidence") if d.get("confidence") in ("high", "medium", "low") else "low",
    }


def collect_researchers(snapshot: dict, top_n: int = 3) -> dict[str, list[str]]:
    """name -> distinct research areas (theme ids), from the live + fads lists."""
    out: dict[str, set] = {}
    for t in (snapshot.get("live") or []) + (snapshot.get("fads") or []):
        for r in (t.get("researchers") or [])[:top_n]:
            out.setdefault(r["name"], set()).add(t["id"])
    return {name: sorted(areas)[:6] for name, areas in out.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--rps", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0, help="cap researchers (for a test run)")
    args = ap.parse_args(argv)
    STORE.mkdir(parents=True, exist_ok=True)

    from google import genai
    from google.genai import types

    from marketbrain.env import get_gemini_api_key, load_dotenv
    load_dotenv()
    # per-request timeout (ms) so one hung grounded call can't block the whole run
    client = genai.Client(api_key=get_gemini_api_key(),
                          http_options=types.HttpOptions(timeout=30_000))

    snapshot = json.loads(pathlib.Path(args.snapshot).read_text())
    researchers = collect_researchers(snapshot)
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    pending = [n for n in researchers if n not in cache]
    if args.limit:
        pending = pending[: args.limit]
    print(f"[enrich] {len(cache)} cached, {len(pending)} to enrich", flush=True)

    consecutive_fail = 0
    done = 0
    for name in pending:
        areas = ", ".join(researchers[name])
        enr = None
        for attempt in range(3):  # retry transient 5xx with backoff
            try:
                resp = client.models.generate_content(
                    model=MODEL,
                    contents=PROMPT.format(name=name, areas=areas),
                    config=types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())]),
                )
                enr = parse_enrichment(resp.text)
                break
            except Exception as e:  # noqa: BLE001
                if attempt < 2:
                    time.sleep(3 * (attempt + 1))
                else:
                    print(f"[enrich] {name}: {type(e).__name__}, skipping (will retry next run)", flush=True)

        if enr is None:
            # Do NOT cache failures — leave uncached so a later run retries them.
            consecutive_fail += 1
            if consecutive_fail >= 8:
                print("[enrich] 8 consecutive failures — grounding likely degraded; "
                      "stopping so a later run can resume cleanly", flush=True)
                break
            time.sleep(1.0 / args.rps)
            continue

        consecutive_fail = 0
        cache[name] = enr
        done += 1
        if done % 10 == 0:
            CACHE.write_text(json.dumps(cache, indent=2))
            print(f"[enrich] {len(cache)}/{len(researchers)} cached", flush=True)
        time.sleep(1.0 / args.rps)

    CACHE.write_text(json.dumps(cache, indent=2))
    n_found = sum(1 for v in cache.values() if v.get("profile_url"))
    n_founders = sum(1 for v in cache.values() if v.get("is_founder"))
    print(f"[enrich] DONE — {len(cache)} researchers, {n_found} with profile links, "
          f"{n_founders} founders", flush=True)


if __name__ == "__main__":
    main()
