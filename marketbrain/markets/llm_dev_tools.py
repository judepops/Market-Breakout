"""The ``llm-dev-tools`` market roster — reusable shared infrastructure.

Each company is **package-anchored** (linked to the PyPI/npm packages that drive its adoption
signal) and **verified against Crunchbase** (a disambiguated ``identifier_permalink``, since
common names like "Cohere"/"Together"/"Chroma" match dozens of unrelated orgs — the real ones
were confirmed by AI category + valuation). Tagged by ``kind`` so the framework subset can be
reported separately from infra/model-lab.

Consumed by: ``signals/crunchbase_funding.py`` (funding $ for these companies), the $500M outlier
label (did any roster company cross $500M), and the package→company enrichment. **Extend by adding
entries** — re-verify the Crunchbase permalink (AI category + plausible valuation) when you do.
A category-tag-based expansion (organizations tagged AI + Developer Tools) can broaden this later.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Company:
    name: str
    cb_permalink: str            # verified Crunchbase organizations.identifier_permalink
    kind: str                    # framework | vector-db | infra-serving | model-lab | local-runtime
    packages: tuple[str, ...]    # anchor packages (PyPI/npm) — the adoption link


# Verified 2026-06 against production-498516.Crunchbase_Dataset.organizations.
ROSTER: tuple[Company, ...] = (
    Company("LangChain", "langchain", "framework", ("langchain", "@langchain/core", "langgraph")),
    Company("LlamaIndex", "llamaindex", "framework", ("llama-index", "llamaindex")),
    Company("CrewAI", "crewai", "framework", ("crewai",)),
    Company("deepset", "deepset", "framework", ("haystack-ai", "farm-haystack")),
    Company("Pinecone", "pinecone", "vector-db", ("pinecone", "pinecone-client")),
    Company("Weaviate", "weaviate", "vector-db", ("weaviate-client",)),
    Company("Qdrant", "qdrant", "vector-db", ("qdrant-client",)),
    Company("LanceDB", "lancedb", "vector-db", ("lancedb",)),
    Company("Chroma", "chroma-4a75", "vector-db", ("chromadb",)),
    Company("Anyscale", "anyscale", "infra-serving", ("ray", "vllm")),
    Company("Baseten", "baseten", "infra-serving", ("truss",)),
    Company("Fireworks AI", "fireworks-ai", "infra-serving", ("fireworks-ai",)),
    Company("Together AI", "together-1a7e", "infra-serving", ("together",)),
    Company("Replicate", "replicate", "infra-serving", ("replicate",)),
    Company("Groq", "groq", "infra-serving", ("groq",)),
    Company("Hugging Face", "hugging-face", "model-lab", ("transformers", "huggingface-hub")),
    Company("Cohere", "cohere-82b8", "model-lab", ("cohere",)),
    Company("Mistral AI", "mistral-ai", "model-lab", ("mistralai",)),
    Company("Perplexity", "perplexity-ai", "model-lab", ()),
    Company("Anysphere (Cursor)", "anysphere", "infra-serving", ()),
    Company("Ollama", "ollama", "local-runtime", ("ollama",)),
)


def permalinks() -> tuple[str, ...]:
    """Crunchbase permalinks for the whole roster (for funding / label queries)."""
    return tuple(c.cb_permalink for c in ROSTER)


def names() -> tuple[str, ...]:
    return tuple(c.name for c in ROSTER)


def by_kind(kind: str) -> tuple[Company, ...]:
    return tuple(c for c in ROSTER if c.kind == kind)


def package_to_company() -> dict[str, str]:
    """Map each anchor package -> company name (for the enrichment step)."""
    return {pkg: c.name for c in ROSTER for pkg in c.packages}


def assignee_names() -> tuple[str, ...]:
    """Patent-assignee match names for the patents signal — the roster's core company names
    (parentheticals stripped). The patents signal counts patents assigned to these companies, so the
    signal is market-specific (it generalises per-market via each roster). For ``llm-dev-tools`` this is
    near-zero (these OSS-heavy companies barely patent) — a real "weak signal here", not missing data.
    """
    return tuple(c.name.split(" (")[0] for c in ROSTER)


# Anchor GitHub repos for the adoption (stars) signal — the OSS-heavy companies. The closed-source ones
# (Pinecone, Together, Fireworks, Perplexity, Anysphere) contribute little/no GitHub signal, which is the
# point: GitHub stars are STRONG exactly where patents were ~0 (the open dev-tools), and vice-versa.
GITHUB_REPOS: tuple[str, ...] = (
    "langchain-ai/langchain", "langchain-ai/langgraph", "run-llama/llama_index", "crewAIInc/crewAI",
    "deepset-ai/haystack", "weaviate/weaviate", "qdrant/qdrant", "lancedb/lancedb", "chroma-core/chroma",
    "ray-project/ray", "vllm-project/vllm", "basetenlabs/truss", "replicate/cog",
    "huggingface/transformers", "huggingface/huggingface_hub", "mistralai/mistral-inference",
    "ollama/ollama",
)


def github_repos() -> tuple[str, ...]:
    """Anchor GitHub repos scoping the stars/adoption signal for this market."""
    return GITHUB_REPOS


# Full-universe spec (the saturation fix) — every fundable (>= $5M) AI company in the infra / generative-AI /
# dev-tools slice, winners AND non-winners, for the outlier-RATE label. Calibrated 2026-06 against the curated
# roster (experiments/calibrate_universe.py): 889 companies, captures 16/21 curated winners, **25.2% outlier
# rate** — i.e. real negatives to learn from (vs the curated roster's saturated 57%). The curated ROSTER above
# stays the *signal* anchor (GitHub repos, packages, patent assignees); the universe is the *outcome/label*
# scope. AI AND (AI-Infrastructure OR Generative-AI OR Developer-Tools) captured the most winners at a clean
# rate; the bare "AI" tag was too broad and any single sub-tag missed the model labs.
UNIVERSE_TAG_GROUPS: tuple[tuple[str, ...], ...] = (
    ("Artificial Intelligence (AI)",),
    ("AI Infrastructure", "Generative AI", "Developer Tools"),
)
UNIVERSE_FOUNDED_FROM: "int | None" = None


def universe(project: "str | None" = None):
    """The full fundable (>= $5M) universe for the outlier-rate label (winners AND negatives)."""
    from .universe import fetch_universe
    comps, _ = fetch_universe([list(g) for g in UNIVERSE_TAG_GROUPS],
                              founded_from=UNIVERSE_FOUNDED_FROM, project=project)
    return comps
