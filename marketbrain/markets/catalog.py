"""Broad-build market catalog — the N≈10 sub-markets for the prediction dataset (the saturation fix, scaled).

Step 2 of the plan: having proved the full-universe rostering + percentile-label machinery on 2 markets,
scale it to ~10 so there's a real cross-market **outlier-rate target** (prediction needs N markets; a
features→rate model on N=2 is meaningless). Each market is a full fundable (>= $5M) universe defined by a
Crunchbase tag spec; `labels/universe_label.py` turns it into the market's outlier RATE + per-company
percentiles.

Mostly single-tag categories (roughly disjoint dev-facing-software sub-markets, each rate-calibrated in
experiments/outlier_rates.py); `llm-dev-tools` is the calibrated AI cut (spec in its module). Deliberate
low-rate **negatives** (IoT, VR) give outcome spread. Substitutions from the suggested set: MLOps -> Machine
Learning, No-Code -> IoT/VR (the No-Code tag returns zero rows; MLOps is not a Crunchbase tag).

This is the TARGET layer only — the per-market signal stacks (layer B: GitHub repos, packages, arXiv terms)
are added as each market graduates to feature extraction. `data-infra` (our other validated market) is the
Database ∪ Analytics composite, so its target is covered by the `database` + `data-analytics` entries here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import llm_dev_tools

_ASSIGNEE_OK = re.compile(r"^[A-Za-z0-9 .&\-]+$")  # the patents collector's allowed assignee-name charset


def clean_assignee(name: "str | None") -> "str | None":
    """Normalise a Crunchbase company name into a patent-assignee match string (or None if unusable).

    Strips characters the patents regex rejects, collapses whitespace, and drops too-short names. The
    patents collector anchors the whole name + legal suffix, so 'Intel' matches 'INTEL CORPORATION' but not
    a longer unrelated name -> precise even across a market's top companies.
    """
    n = re.sub(r"[^A-Za-z0-9 .&\-]", " ", str(name or "")).strip()
    n = re.sub(r"\s+", " ", n)
    return n if len(n) >= 3 and _ASSIGNEE_OK.match(n) else None


@dataclass(frozen=True)
class MarketSpec:
    name: str
    tag_groups: tuple[tuple[str, ...], ...]
    founded_from: "int | None"
    kind: str   # "positive" (dev-facing target) | "negative" (deliberate low-rate)
    note: str = ""
    # expected signal footprint (estimated, confirmed only when the signals are built). Full toolkit:
    #   DEPTH (research/adoption/innovation, need real domain footprint):
    #     gh = GitHub stars, pkg = npm/PyPI, ax = arXiv/emergence (ANY field: cs, q-fin, q-bio, physics, eess),
    #     pat = patents (STRONG for bio/hardware/energy, i.e. exactly where gh/pkg are weak).
    #   ATTENTION (broad/near-universal): wiki = Wikipedia pageviews, hn = Hacker News.
    # footprint_rich = >= 2 DEPTH signals (a Phase-2 corpus target); attention alone is not "rich".
    coverage: tuple[str, ...] = ()
    DEPTH = ("gh", "pkg", "ax", "pat")

    @property
    def n_depth(self) -> int:
        return sum(1 for s in self.coverage if s in self.DEPTH)

    @property
    def footprint_rich(self) -> bool:
        return self.n_depth >= 2

    def universe(self, project: "str | None" = None):
        from .universe import fetch_universe
        comps, _ = fetch_universe([list(g) for g in self.tag_groups],
                                  founded_from=self.founded_from, project=project)
        return comps

    def assignees(self, project: "str | None" = None, top_n: int = 60) -> tuple[str, ...]:
        """Patent-assignee names for the innovation signal: the market's top-`top_n` companies by funding,
        cleaned. (Top-by-funding keeps the dominant patent-holders and bounds the assignee-match noise.)"""
        comps = sorted(self.universe(project), key=lambda c: c.funding_total_usd or 0, reverse=True)[:top_n]
        return tuple(dict.fromkeys(filter(None, (clean_assignee(c.name) for c in comps))))


# Broad dev/technical candidate set. The viability scan (experiments/scan_catalog.py) keeps the ones with a
# mature formation cohort; `coverage` flags which are footprint-rich enough for the Phase-2 signal corpus.
CATALOG: tuple[MarketSpec, ...] = (
    # --- software / dev-technical markets: OSS adoption (gh/pkg) + attention (wiki/hn), AI ones add arXiv ---
    MarketSpec("llm-dev-tools", llm_dev_tools.UNIVERSE_TAG_GROUPS, llm_dev_tools.UNIVERSE_FOUNDED_FROM,
               "positive", "calibrated AI ∩ infra/genai/devtools cut (flagship)", ("gh", "pkg", "ax", "wiki", "hn")),
    MarketSpec("developer-tools", (("Developer Tools",),), None, "positive", "", ("gh", "pkg", "wiki", "hn")),
    MarketSpec("open-source", (("Open Source",),), None, "positive", "", ("gh", "pkg", "wiki", "hn")),
    MarketSpec("devops", (("DevOps",),), None, "positive", "", ("gh", "pkg", "wiki", "hn")),
    MarketSpec("database", (("Database",),), None, "positive", "", ("gh", "pkg", "ax", "wiki", "hn")),
    MarketSpec("data-storage", (("Data Storage",),), None, "positive", "", ("gh", "pkg", "wiki", "hn")),
    MarketSpec("cloud-infrastructure", (("Cloud Infrastructure",),), None, "positive", "", ("gh", "pkg", "wiki", "hn")),
    MarketSpec("cloud-computing", (("Cloud Computing",),), None, "positive", "", ("gh", "pkg", "wiki", "hn")),
    MarketSpec("machine-learning", (("Machine Learning",),), None, "positive", "", ("ax", "gh", "pkg", "wiki", "hn")),
    MarketSpec("artificial-intelligence", (("Artificial Intelligence (AI)",),), None, "positive", "", ("ax", "gh", "pkg", "wiki", "hn")),
    MarketSpec("nlp", (("Natural Language Processing",),), None, "positive", "", ("ax", "gh", "pkg", "wiki", "hn")),
    MarketSpec("computer-vision", (("Computer Vision",),), None, "positive", "", ("ax", "gh", "pkg", "wiki", "hn")),
    MarketSpec("big-data", (("Big Data",),), None, "positive", "", ("gh", "pkg", "wiki", "hn")),
    MarketSpec("data-analytics", (("Analytics",),), None, "positive", "", ("gh", "pkg", "wiki", "hn")),
    MarketSpec("data-management", (("Data Management",),), None, "positive", "", ("gh", "pkg", "wiki", "hn")),
    MarketSpec("mobile", (("Mobile",),), None, "positive", "", ("gh", "pkg", "wiki", "hn")),
    # --- security / crypto: OSS + research (cs.CR) + patents + attention ---
    MarketSpec("cyber-security", (("Cyber Security",),), None, "positive", "", ("gh", "ax", "pat", "wiki", "hn")),
    MarketSpec("network-security", (("Network Security",),), None, "positive", "", ("gh", "ax", "pat", "wiki", "hn")),
    MarketSpec("blockchain", (("Blockchain",),), None, "positive", "", ("gh", "ax", "wiki", "hn")),
    MarketSpec("cryptocurrency", (("Cryptocurrency",),), None, "positive", "", ("gh", "ax", "wiki", "hn")),
    # --- deep-tech: research (arXiv q-bio/q-fin/physics/eess) + PATENTS carry where OSS is thin ---
    MarketSpec("robotics", (("Robotics",),), None, "positive", "", ("ax", "gh", "pat", "wiki", "hn")),
    MarketSpec("computer-hardware", (("Hardware",),), None, "positive", "", ("pat", "gh", "wiki", "hn")),
    MarketSpec("semiconductors", (("Semiconductor",),), None, "positive", "", ("ax", "pat", "wiki", "hn")),
    MarketSpec("internet-of-things", (("Internet of Things",),), None, "positive", "", ("pat", "gh", "wiki", "hn")),
    MarketSpec("biotechnology", (("Biotechnology",),), None, "positive", "patents+q-bio carry it", ("ax", "pat", "wiki", "hn")),
    MarketSpec("medical-devices", (("Medical Device",),), None, "positive", "", ("ax", "pat", "wiki", "hn")),
    MarketSpec("pharmaceutical", (("Pharmaceutical",),), None, "positive", "", ("ax", "pat", "wiki", "hn")),
    MarketSpec("clean-energy", (("Clean Energy",),), None, "positive", "", ("ax", "pat", "wiki", "hn")),
    MarketSpec("automotive", (("Automotive",),), None, "positive", "AV research + patents", ("ax", "pat", "gh", "wiki", "hn")),
    MarketSpec("aerospace", (("Aerospace",),), None, "positive", "", ("ax", "pat", "wiki", "hn")),
    MarketSpec("quantum-computing", (("Quantum Computing",),), None, "positive", "", ("ax", "pat", "gh", "wiki", "hn")),
    MarketSpec("fintech", (("FinTech",),), None, "positive", "q-fin + OSS + patents", ("ax", "gh", "pat", "wiki", "hn")),
    MarketSpec("augmented-reality", (("Augmented Reality",),), None, "negative", "low-rate", ("ax", "pat", "gh", "wiki", "hn")),
    MarketSpec("virtual-reality", (("Virtual Reality",),), None, "negative", "low-rate", ("ax", "pat", "gh", "wiki", "hn")),
    # --- more software / systems (OSS adoption + attention; AI-ish ones add arXiv) ---
    MarketSpec("web-development", (("Web Development",),), None, "positive", "", ("gh", "pkg", "wiki", "hn")),
    MarketSpec("software-engineering", (("Software Engineering",),), None, "positive", "", ("gh", "pkg", "wiki", "hn")),
    MarketSpec("virtualization", (("Virtualization",),), None, "positive", "", ("gh", "pkg", "wiki", "hn")),
    MarketSpec("operating-systems", (("Operating Systems",),), None, "positive", "", ("gh", "ax", "wiki", "hn")),
    MarketSpec("search-engine", (("Search Engine",),), None, "positive", "", ("ax", "gh", "wiki", "hn")),
    MarketSpec("speech-recognition", (("Speech Recognition",),), None, "positive", "", ("ax", "gh", "pkg", "wiki", "hn")),
    MarketSpec("image-recognition", (("Image Recognition",),), None, "positive", "", ("ax", "gh", "wiki", "hn")),
    MarketSpec("bioinformatics", (("Bioinformatics",),), None, "positive", "OSS + q-bio", ("ax", "gh", "pkg", "wiki", "hn")),
    MarketSpec("cloud-security", (("Cloud Security",),), None, "positive", "", ("gh", "ax", "pat", "wiki", "hn")),
    MarketSpec("privacy", (("Privacy",),), None, "positive", "", ("gh", "ax", "wiki", "hn")),
    MarketSpec("network-hardware", (("Network Hardware",),), None, "positive", "", ("pat", "gh", "wiki", "hn")),
    MarketSpec("embedded-systems", (("Embedded Systems",),), None, "positive", "", ("gh", "pat", "wiki", "hn")),
    # --- more deep-tech / physical (patents + arXiv carry it; some add OSS) ---
    MarketSpec("telecommunications", (("Telecommunications",),), None, "positive", "", ("ax", "pat", "wiki", "hn")),
    MarketSpec("electric-vehicle", (("Electric Vehicle",),), None, "positive", "", ("ax", "pat", "wiki", "hn")),
    MarketSpec("battery", (("Battery",),), None, "positive", "energy storage", ("ax", "pat", "wiki", "hn")),
    MarketSpec("solar", (("Solar",),), None, "positive", "", ("ax", "pat", "wiki", "hn")),
    MarketSpec("cleantech", (("CleanTech",),), None, "positive", "", ("ax", "pat", "wiki", "hn")),
    MarketSpec("nanotechnology", (("Nanotechnology",),), None, "positive", "", ("ax", "pat", "wiki", "hn")),
    MarketSpec("advanced-materials", (("Advanced Materials",),), None, "positive", "", ("ax", "pat", "wiki", "hn")),
    MarketSpec("genetics", (("Genetics",),), None, "positive", "q-bio", ("ax", "pat", "wiki", "hn")),
    MarketSpec("agtech", (("AgTech",),), None, "positive", "", ("ax", "pat", "wiki", "hn")),
    MarketSpec("drones", (("Drones",),), None, "positive", "", ("ax", "pat", "gh", "wiki", "hn")),
    MarketSpec("satellite", (("Satellite",),), None, "positive", "", ("ax", "pat", "wiki", "hn")),
    MarketSpec("space-travel", (("Space Travel",),), None, "positive", "", ("ax", "pat", "wiki", "hn")),
    MarketSpec("industrial-automation", (("Industrial Automation",),), None, "positive", "", ("ax", "pat", "wiki", "hn")),
    MarketSpec("health-diagnostics", (("Health Diagnostics",),), None, "positive", "", ("ax", "pat", "wiki", "hn")),
    MarketSpec("3d-printing", (("3D Printing",),), None, "positive", "", ("pat", "gh", "wiki", "hn")),
    MarketSpec("wearables", (("Wearables",),), None, "positive", "", ("pat", "gh", "wiki", "hn")),
    # --- genuinely thin: pure business software, attention + funding only (the moat is distribution, not tech) ---
    MarketSpec("enterprise-software", (("Enterprise Software",),), None, "positive", "thin: no research/OSS trail", ("wiki", "hn")),
    MarketSpec("saas", (("SaaS",),), None, "positive", "thin", ("wiki", "hn")),
    MarketSpec("information-technology", (("Information Technology",),), None, "positive", "thin", ("gh", "wiki", "hn")),
)

_BY_NAME = {m.name: m for m in CATALOG}


def get_spec(name: str) -> MarketSpec:
    try:
        return _BY_NAME[name]
    except KeyError:
        raise KeyError(f"unknown catalog market {name!r}; known: {sorted(_BY_NAME)}") from None


__all__ = ["MarketSpec", "CATALOG", "get_spec"]
