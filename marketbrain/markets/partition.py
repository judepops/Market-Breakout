"""Clean partition of the company universe into DISJOINT markets (each company -> exactly one market).

The catalog markets are overlapping Crunchbase tags (a company with 4 tags is in 4 markets), so the markets
are not statistically independent. Here each >=$5M company is assigned to ONE market:
  * 0 taxonomy tags  -> unassigned (dropped).
  * 1 taxonomy tag   -> that market (deterministic).
  * >=2 taxonomy tags -> AMBIGUOUS: an LLM picks the company's PRIMARY market among its tags (classify.py).
This guarantees independence (one-to-one mapping); the only judgment is which market an ambiguous company
belongs to, which the LLM decides semantically rather than by arbitrary tag-priority.

The taxonomy is fine-grained and ordered specific -> general (so a single-tag fallback still prefers the
specific market); the broad catch-alls (AI / SaaS / Enterprise) sit last. Mobile and Foundational-AI are
dropped (too broad / saturated). market_key -> Crunchbase tag.
"""

from __future__ import annotations

# (market_key, Crunchbase category substring). Specific -> general.
FINE_TAXONOMY: tuple[tuple[str, str], ...] = (
    ("agentic-ai", "Agentic AI"), ("generative-ai", "Generative AI"), ("ai-infra", "AI Infrastructure"),
    ("nlp", "Natural Language Processing"), ("computer-vision", "Computer Vision"),
    ("speech-recognition", "Speech Recognition"), ("image-recognition", "Image Recognition"),
    ("recommendation", "Recommendation Engine"), ("semantic-search", "Semantic Search"),
    ("deep-learning", "Deep Learning"), ("machine-learning", "Machine Learning"),
    ("predictive-analytics", "Predictive Analytics"),
    ("developer-apis", "Developer APIs"), ("developer-platform", "Developer Platform"),
    ("api-management", "API Management"), ("continuous-integration", "Continuous Integration"),
    ("version-control", "Version Control"), ("testing", "Software Testing"),
    ("developer-tools", "Developer Tools"), ("devops", "DevOps"), ("open-source", "Open Source"),
    ("software-engineering", "Software Engineering"), ("web-development", "Web Development"),
    ("operating-systems", "Operating Systems"), ("virtualization", "Virtualization"), ("serverless", "Serverless"),
    ("low-code", "No-Code"), ("database", "Database"), ("data-storage", "Data Storage"),
    ("data-integration", "Data Integration"), ("data-management", "Data Management"),
    ("data-visualization", "Data Visualization"), ("big-data", "Big Data"),
    ("business-intelligence", "Business Intelligence"), ("analytics", "Analytics"),
    ("cloud-security", "Cloud Security"), ("cloud-infrastructure", "Cloud Infrastructure"),
    ("cloud-computing", "Cloud Computing"), ("identity", "Identity Management"),
    ("fraud-detection", "Fraud Detection"), ("network-security", "Network Security"),
    ("cyber-security", "Cyber Security"), ("privacy", "Privacy"), ("blockchain", "Blockchain"),
    ("cryptocurrency", "Cryptocurrency"), ("iot", "Internet of Things"), ("robotics", "Robotics"),
    ("drones", "Drones"), ("semiconductors", "Semiconductor"), ("quantum", "Quantum Computing"),
    ("ar", "Augmented Reality"), ("vr", "Virtual Reality"), ("fintech", "FinTech"),
    ("marketing-automation", "Marketing Automation"), ("productivity", "Productivity Tools"),
    ("collaboration", "Collaboration"),
    ("artificial-intelligence", "Artificial Intelligence (AI)"),
    ("enterprise-software", "Enterprise Software"), ("saas", "SaaS"),
)
_KEY_TO_TAG = dict(FINE_TAXONOMY)
TAXONOMY_KEYS = tuple(k for k, _ in FINE_TAXONOMY)


def tags_present(categories_value: "str | None") -> list[str]:
    """The taxonomy market_keys whose tag appears in the company's categories (priority order preserved)."""
    cv = (categories_value or "")
    return [k for k, t in FINE_TAXONOMY if t in cv]


def assign_deterministic(categories_value: "str | None") -> "str | None":
    """One market for a company, or None if unassigned, or 'AMBIGUOUS' if it needs the LLM tie-break."""
    present = tags_present(categories_value)
    if not present:
        return None
    if len(present) == 1:
        return present[0]
    return "AMBIGUOUS"
