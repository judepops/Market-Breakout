"""LLM concept extraction over the arXiv corpus (v1.1 layer) — Gemini Batch API.

    python -m experiments.llm_concepts --estimate            # token/cost estimate, no API calls
    python -m experiments.llm_concepts --pilot 200           # synchronous sample to eyeball quality
    python -m experiments.llm_concepts --submit              # upload chunks + create batch jobs
    python -m experiments.llm_concepts --collect             # poll jobs, write results to the store
    python -m experiments.llm_concepts --status              # job registry summary

For each paper (title + abstract), extracts 1-4 canonical research concepts via a
schema-constrained Gemini call. Everything is resumable: extracted concepts land in
data/derived/concepts/*.parquet keyed by record_id; --submit only requests records
missing from the store; the job registry (data/derived/concepts/jobs.json) tracks
in-flight Batch API jobs across invocations.

Requires GEMINI_API_KEY in .env (see .env.example). Batch pricing is 50% of standard.
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import time

MODEL = "gemini-3.5-flash"
STORE = pathlib.Path("data/derived/concepts")
JOBS = STORE / "jobs.json"
CHUNK = 20_000  # requests per batch job

# Priced 2026-07-14 (ai.google.dev): standard $1.50/M in, $9/M out; batch = 50%.
PRICE_IN_BATCH = 0.75 / 1e6
PRICE_OUT_BATCH = 4.50 / 1e6
EST_IN_TOKENS = 400   # prompt + title + abstract, average
EST_OUT_TOKENS = 55   # JSON concept list, average

PROMPT = """You are indexing AI research papers into a concept catalog.

From the paper below, extract 1 to 4 specific research concepts: the methods, \
techniques, tasks, or systems the paper is actually about.

Rules:
- Use the canonical, lowercase name the research community uses (e.g. \
"chain-of-thought prompting", "kv-cache", "state space models", "speculative decoding").
- Each concept is at most 4 words.
- Be specific: never output generic fields ("deep learning", "machine learning", \
"neural networks", "artificial intelligence", "large language models").
- Include a model or system name only if the paper introduces or centrally studies it.
- Prefer the idea over the application domain unless the paper is about the domain \
technique itself.

Title: {title}
Abstract: {abstract}"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "concepts": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
    },
    "required": ["concepts"],
}


def build_request(record_id: str, title: str, abstract: str) -> dict:
    """One JSONL line for the Batch API (key = record_id)."""
    return {
        "key": record_id,
        "request": {
            "contents": [{"parts": [{"text": PROMPT.format(title=title, abstract=abstract[:4000])}]}],
            "generation_config": {
                "response_mime_type": "application/json",
                "response_schema": RESPONSE_SCHEMA,
            },
        },
    }


def normalize_concepts(raw) -> list[str]:
    """Clean one response payload into canonical concept strings."""
    if not isinstance(raw, dict):
        return []
    out = []
    for c in raw.get("concepts", [])[:4]:
        if not isinstance(c, str):
            continue
        c = " ".join(c.lower().strip().split())
        if 2 <= len(c) <= 60 and len(c.split()) <= 4:
            out.append(c)
    return list(dict.fromkeys(out))


def parse_result_line(line: str) -> tuple[str, list[str]] | None:
    """(record_id, concepts) from one result JSONL line, or None on error rows."""
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        return None
    key = row.get("key")
    resp = row.get("response")
    if not key or not resp:
        return None
    try:
        text = resp["candidates"][0]["content"]["parts"][0]["text"]
        return key, normalize_concepts(json.loads(text))
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        return key, []


def estimate_cost(n_papers: int) -> dict:
    return {
        "papers": n_papers,
        "est_input_tokens": n_papers * EST_IN_TOKENS,
        "est_output_tokens": n_papers * EST_OUT_TOKENS,
        "est_cost_usd_batch": round(
            n_papers * (EST_IN_TOKENS * PRICE_IN_BATCH + EST_OUT_TOKENS * PRICE_OUT_BATCH), 2),
    }


def _load_store():
    import pandas as pd
    files = sorted(glob.glob(str(STORE / "chunk-*.parquet")))
    if not files:
        return pd.DataFrame(columns=["record_id", "concepts"])
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).drop_duplicates("record_id")


def _pending(corpus):
    done = set(_load_store()["record_id"])
    df = corpus[~corpus["record_id"].isin(done)]
    print(f"[concepts] {len(done):,} extracted, {len(df):,} pending", flush=True)
    return df


def _client():
    from google import genai
    from marketbrain.env import get_gemini_api_key, load_dotenv
    load_dotenv()
    return genai.Client(api_key=get_gemini_api_key())


def _next_chunk_index() -> int:
    files = glob.glob(str(STORE / "chunk-*.parquet"))
    return 1 + max([int(pathlib.Path(f).stem.split("-")[1]) for f in files], default=-1)


def main(argv=None):
    import pandas as pd
    from marketbrain.corpus.load_full import load_corpus

    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", nargs="*", default=["cs.AI"])
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--pilot", type=int, default=0, help="synchronous sample of N papers")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="submit at most N papers (validation runs)")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args(argv)

    STORE.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus(args.cats, cache=False)[["record_id", "month", "title", "abstract"]]

    if args.estimate:
        pend = _pending(corpus)
        print(json.dumps(estimate_cost(len(pend)), indent=2))
        return

    if args.status:
        reg = json.loads(JOBS.read_text()) if JOBS.exists() else {}
        print(json.dumps(reg, indent=2))
        return

    if args.pilot:
        client = _client()
        sample = _pending(corpus).sample(args.pilot, random_state=0)
        rows = []
        for r in sample.itertuples(index=False):
            resp = client.models.generate_content(
                model=MODEL,
                contents=PROMPT.format(title=r.title, abstract=(r.abstract or "")[:4000]),
                config={"response_mime_type": "application/json", "response_schema": RESPONSE_SCHEMA},
            )
            try:
                concepts = normalize_concepts(json.loads(resp.text))
            except (json.JSONDecodeError, TypeError):
                concepts = []
            rows.append({"record_id": r.record_id, "month": r.month, "title": r.title, "concepts": concepts})
            print(f"  {r.title[:70]:70s} -> {concepts}", flush=True)
        out = STORE / "pilot.parquet"
        pd.DataFrame(rows).to_parquet(out)
        print(f"[concepts] pilot written to {out} (NOT merged into the store)", flush=True)
        return

    if args.submit:
        client = _client()
        pend = _pending(corpus)
        if args.limit:
            pend = pend.sample(min(args.limit, len(pend)), random_state=0)
            print(f"[concepts] limited to {len(pend):,} papers for validation", flush=True)
        if pend.empty:
            print("[concepts] nothing to submit")
            return
        reg = json.loads(JOBS.read_text()) if JOBS.exists() else {}
        for start in range(0, len(pend), CHUNK):
            part = pend.iloc[start : start + CHUNK]
            jsonl = STORE / f"requests-{start // CHUNK}.jsonl"
            with jsonl.open("w") as f:
                for r in part.itertuples(index=False):
                    f.write(json.dumps(build_request(r.record_id, r.title or "", r.abstract or "")) + "\n")
            uploaded = client.files.upload(
                file=str(jsonl),
                config={"display_name": jsonl.stem, "mime_type": "jsonl"},
            )
            job = client.batches.create(model=MODEL, src=uploaded.name,
                                        config={"display_name": jsonl.stem})
            reg[job.name] = {"state": "submitted", "n": len(part), "requests_file": str(jsonl)}
            JOBS.write_text(json.dumps(reg, indent=2))
            print(f"[concepts] submitted {job.name} ({len(part):,} papers)", flush=True)
        return

    if args.collect:
        client = _client()
        reg = json.loads(JOBS.read_text()) if JOBS.exists() else {}
        open_jobs = {k: v for k, v in reg.items() if v["state"] not in ("collected", "failed")}
        if not open_jobs:
            print("[concepts] no open jobs")
            return
        month_by_id = dict(zip(corpus["record_id"], corpus["month"]))
        while open_jobs:
            for name, meta in list(open_jobs.items()):
                job = client.batches.get(name=name)
                state = job.state.name
                print(f"[concepts] {name}: {state}", flush=True)
                if state == "JOB_STATE_SUCCEEDED":
                    content = client.files.download(file=job.dest.file_name).decode("utf-8")
                    rows = []
                    for line in content.splitlines():
                        parsed = parse_result_line(line)
                        if parsed:
                            rid, concepts = parsed
                            rows.append({"record_id": rid, "month": month_by_id.get(rid, ""),
                                         "concepts": concepts})
                    if rows:
                        out = STORE / f"chunk-{_next_chunk_index()}.parquet"
                        pd.DataFrame(rows).to_parquet(out)
                        print(f"[concepts] wrote {out}", flush=True)
                    else:
                        print(f"[concepts] WARN {name} succeeded with 0 parseable results", flush=True)
                    reg[name]["state"] = "collected"
                    reg[name]["n_results"] = len(rows)
                    JOBS.write_text(json.dumps(reg, indent=2))
                    print(f"[concepts] collected {len(rows):,} -> {out}", flush=True)
                    del open_jobs[name]
                elif state in ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"):
                    # FAILED can be transient (the service retries internally and later
                    # reports SUCCEEDED) — only give up after 3 consecutive sightings.
                    meta["fail_seen"] = meta.get("fail_seen", 0) + 1
                    if meta["fail_seen"] >= 3:
                        reg[name]["state"] = "failed"
                        JOBS.write_text(json.dumps(reg, indent=2))
                        del open_jobs[name]
            if open_jobs:
                time.sleep(120)
        return

    ap.print_help()


if __name__ == "__main__":
    main()
