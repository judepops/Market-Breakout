"""Generate per-market anchor GitHub repos via Gemini (auto-seeder), cached for the adoption signal.

    python -m experiments.build_market_repos

One Gemini call per market -> its flagship OSS repos (owner/repo). Markets with no open-source adoption
(pharma, aerospace, ...) get an empty list. Cached to a JSON the adoption builder reads.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="BigQuery Storage module not found")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/raw/labels")
    ap.add_argument("--out", default="data/raw/labels/github_repos.json")
    args = ap.parse_args(argv)

    import pandas as pd

    from marketbrain.enrichment.llm_terms import generate_github_repos
    from marketbrain.labels.gemini_grounding import make_client
    from marketbrain.markets.catalog import CATALOG

    ft = pd.read_parquet(Path(args.labels) / "market_features.parquet").set_index("market")
    specs = [m for m in CATALOG if m.name in ft.index and bool(ft.loc[m.name, "mature"])]
    out = Path(args.out)
    have = json.loads(out.read_text()) if out.exists() else {}
    client = make_client()
    print(f"generating GitHub repos for {len(specs)} markets ({len(have)} cached) ...\n")
    for m in specs:
        if m.name in have:
            continue
        try:
            have[m.name] = list(generate_github_repos(m.name, client=client))
        except Exception as e:  # noqa: BLE001
            print(f"  {m.name}: {e}")
            have[m.name] = []
        print(f"{m.name:22s} {have[m.name]}")
        out.write_text(json.dumps(have))
    nrepos = len({r for v in have.values() for r in v})
    print(f"\n{sum(1 for v in have.values() if v)}/{len(have)} markets have repos; {nrepos} distinct. wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
