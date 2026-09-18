"""Canonicalize LLM-extracted concepts into themes (stage 2) + hierarchy (stage 3).

    python -m experiments.concept_themes --build              # embed, merge, write outputs
    python -m experiments.concept_themes --clusters 60        # stage 3: parent clusters + names

Stage 2: embed every unique concept string locally (sentence-transformers, no API),
merge near-duplicates (plural-normalized equality, or embedding cosine >= threshold),
and pick the highest-document-frequency spelling as the canonical name. Outputs:

    data/derived/concepts/concept_map.parquet     concept -> canonical
    data/derived/concepts/concept_events.parquet  (record_id, month, term)  [term = canonical]

`concept_events.parquet` is a drop-in replacement for the n-gram term_events frame —
build_radar_data can pivot it into theme x month series unchanged.

Stage 3: agglomerative clustering of canonical-concept embeddings into parent groups,
named by Gemini (one short call per cluster) -> concept_clusters.parquet.
"""
from __future__ import annotations

import argparse
import json
import pathlib

STORE = pathlib.Path("data/derived/concepts")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SIM_THRESHOLD = 0.92
MIN_DF_THEME = 10  # a canonical concept needs this many distinct papers to be a theme


def norm_plural(term: str) -> str:
    """Plural-insensitive key: 'state space models' == 'state space model'."""
    return " ".join(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in term.split())


def merge_concepts(concepts: list[str], df_counts: dict[str, int], embeddings=None,
                   sim_threshold: float = SIM_THRESHOLD) -> dict[str, str]:
    """Map each concept to its canonical spelling.

    Merges (a) plural-normalized duplicates, and (b) embedding near-duplicates when
    ``embeddings`` (row-aligned with ``concepts``, L2-normalized) is given. Canonical =
    the highest-df member of each merged group (ties: shortest, then alphabetical).
    Union-find keeps merging transitive.
    """
    import numpy as np

    parent = list(range(len(concepts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    by_norm: dict[str, int] = {}
    for i, c in enumerate(concepts):
        key = norm_plural(c)
        if key in by_norm:
            union(by_norm[key], i)
        else:
            by_norm[key] = i

    if embeddings is not None and len(concepts) > 1:
        from sklearn.neighbors import NearestNeighbors

        X = np.asarray(embeddings, dtype=np.float32)
        k = min(6, len(concepts))
        nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(X)
        dist, idx = nn.kneighbors(X)
        for i in range(len(concepts)):
            for d, j in zip(dist[i][1:], idx[i][1:]):
                if 1.0 - d >= sim_threshold:
                    union(i, int(j))

    groups: dict[int, list[int]] = {}
    for i in range(len(concepts)):
        groups.setdefault(find(i), []).append(i)

    mapping: dict[str, str] = {}
    for members in groups.values():
        canon = min(members, key=lambda i: (-df_counts.get(concepts[i], 0), len(concepts[i]), concepts[i]))
        for i in members:
            mapping[concepts[i]] = concepts[canon]
    return mapping


def _load_events():
    import glob

    import pandas as pd
    files = sorted(glob.glob(str(STORE / "chunk-*.parquet")))
    if not files:
        raise FileNotFoundError(f"no concept chunks under {STORE} — run llm_concepts first")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).drop_duplicates("record_id")
    return df.explode("concepts").dropna(subset=["concepts"]).rename(columns={"concepts": "concept"})


def main(argv=None):
    import numpy as np
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--clusters", type=int, default=0, help="stage 3: number of parent clusters")
    ap.add_argument("--min-df", type=int, default=MIN_DF_THEME)
    args = ap.parse_args(argv)

    if args.build:
        ev = _load_events()
        df_counts = ev.groupby("concept")["record_id"].nunique().to_dict()
        concepts = sorted(df_counts)
        print(f"[themes] {len(ev):,} concept events, {len(concepts):,} unique concepts; embedding...",
              flush=True)

        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBED_MODEL)
        emb = model.encode(concepts, normalize_embeddings=True, batch_size=512,
                           show_progress_bar=True)

        mapping = merge_concepts(concepts, df_counts, emb)
        n_merged = sum(1 for k, v in mapping.items() if k != v)
        print(f"[themes] merged {n_merged:,} variants into canonicals", flush=True)

        pd.DataFrame({"concept": list(mapping), "canonical": list(mapping.values())}) \
            .to_parquet(STORE / "concept_map.parquet")
        ev["term"] = ev["concept"].map(mapping)
        events = ev[["record_id", "month", "term"]]
        events.to_parquet(STORE / "concept_events.parquet")

        theme_df = events.groupby("term")["record_id"].nunique().sort_values(ascending=False)
        themes = theme_df[theme_df >= args.min_df]
        print(f"[themes] {len(themes):,} themes with df >= {args.min_df}; top 10:", flush=True)
        for t, n in themes.head(10).items():
            print(f"   {n:5d}  {t}")
        # embeddings for the theme-level canonicals, reused by stage 3.
        # Names as JSON + float array as npz — no pickled object arrays anywhere.
        canon_idx = {c: i for i, c in enumerate(concepts)}
        theme_names = [t for t in themes.index if t in canon_idx]
        (STORE / "theme_names.json").write_text(json.dumps(theme_names))
        np.savez_compressed(STORE / "theme_embeddings.npz",
                            emb=np.asarray(emb[[canon_idx[t] for t in theme_names]], dtype=np.float32))
        return

    if args.clusters:
        from sklearn.cluster import AgglomerativeClustering

        names = json.loads((STORE / "theme_names.json").read_text())
        emb = np.load(STORE / "theme_embeddings.npz")["emb"]  # plain float array, no pickle
        print(f"[themes] clustering {len(names):,} themes into {args.clusters} groups", flush=True)
        labels = AgglomerativeClustering(n_clusters=args.clusters).fit_predict(emb)

        from google import genai
        from marketbrain.env import get_gemini_api_key, load_dotenv
        load_dotenv()
        client = genai.Client(api_key=get_gemini_api_key())

        rows = []
        for cl in range(args.clusters):
            members = [str(n) for n, l in zip(names, labels) if l == cl]
            resp = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=("Name this cluster of AI research concepts with a short, specific, "
                          "lowercase label (2-4 words). Reply with the label only.\n\nConcepts: "
                          + ", ".join(members[:40])),
            )
            cluster_name = " ".join((resp.text or "").lower().strip().split())[:60]
            rows.extend({"term": m, "cluster": cluster_name} for m in members)
            print(f"   [{cl:2d}] {cluster_name}  ({len(members)} themes)", flush=True)
        pd.DataFrame(rows).to_parquet(STORE / "concept_clusters.parquet")
        return

    ap.print_help()


if __name__ == "__main__":
    main()
