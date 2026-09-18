"""Run the LLM seeder (arXiv queries + GitHub repos) for EVERY partition market, so coverage is real.

    python -m experiments.seed_all_markets

Resumable: skips markets already present in each file. A market that genuinely has no open-source adoption
returns an empty repo list (honest masked absence), distinct from never having been seeded.
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


def main():
    import pandas as pd
    from marketbrain.enrichment.llm_terms import generate_arxiv_query, generate_github_repos
    from marketbrain.labels.gemini_grounding import make_client

    pf = pd.read_parquet("data/raw/labels/partition_features.parquet")
    markets = list(pf["market"])                       # all 51 partition markets
    client = make_client()

    arx_p = Path("data/raw/labels/arxiv_queries.json")
    gh_p = Path("data/raw/labels/github_repos.json")
    arx = json.loads(arx_p.read_text()) if arx_p.exists() else {}
    gh = json.loads(gh_p.read_text()) if gh_p.exists() else {}
    print(f"start: {len(markets)} markets | arxiv has {len(arx)}, github has {len(gh)}", flush=True)

    for m in markets:
        name = m.replace("-", " ")
        if m not in arx:
            try:
                arx[m] = generate_arxiv_query(name, client=client)
            except Exception as e:  # noqa: BLE001
                print(f"  arxiv {m}: {e}", flush=True)
                arx[m] = None
            arx_p.write_text(json.dumps(arx))
        if m not in gh:
            try:
                gh[m] = list(generate_github_repos(name, client=client))
            except Exception as e:  # noqa: BLE001
                print(f"  github {m}: {e}", flush=True)
                gh[m] = []
            gh_p.write_text(json.dumps(gh))
        print(f"  {m:24s} arxiv={'Y' if arx.get(m) else '.'}  github={len(gh.get(m) or [])} repos", flush=True)

    mature = set(pf[pf["mature"]]["market"])
    ra = sum(1 for m in mature if arx.get(m))
    ga = sum(1 for m in mature if gh.get(m))
    print(f"\nDONE. mature coverage now: research {ra}/{len(mature)}, adoption {ga}/{len(mature)}", flush=True)


if __name__ == "__main__":
    main()
