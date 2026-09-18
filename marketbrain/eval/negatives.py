"""Hand-curated negative-control set (Phase-0 pre-registration §3.2).

Pool B of the negative controls: a frozen, human-named list of ideas that were genuinely hyped on
arXiv then faded (web3, metaverse, capsule networks, NAS, ...). It is a *face-validity anchor* — the
detector should score these LOW — and it makes the eval legible to the boss. It is NOT the primary
discriminative negative set (that is Pool A, the data-derived hyped-then-died themes built by
``label.py``); see §3.1/§3.3.

Curation discipline (§3.2 / §3.4): three items — quantum machine learning, federated learning, and
prompt engineering — are *flagged ambiguous* (real but contested: borderline-durable or absorbed-not-
failed). They are reported separately and are NOT counted toward the headline negative set unless their
own arXiv trajectory independently satisfies the Pool-A data criterion. Flagging them here prevents us
from hand-grading the answer key.

Each fizzle is mapped to arXiv term-sets and we report — per §3.2 — whether it reaches the §2.1 *birth*
threshold in the corpus (``>= birth_min_records`` records in a 3-month window). A fizzle that does not is
reported as "not present in corpus" rather than forced in. Everything here is a pure function over the
``term_events`` table from ``signals.arxiv_records.extract_term_events`` (``record_id, month, term,
authors``); no I/O, no network.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# §2.1 birth threshold: a theme is "born" when it first has >= 5 records in a 3-month window.
BIRTH_MIN_RECORDS = 5
_BIRTH_WINDOW_MONTHS = 3


@dataclass(frozen=True)
class FizzleControl:
    """A hand-curated hyped-then-died idea (§3.2).

    ``name`` is the canonical human label; ``terms`` is the arXiv term-set used to count its monthly
    documents (matched case-insensitively against the ``term`` column of ``term_events``); and
    ``flagged_ambiguous`` marks the contested items (§3.2 items 8/9/17) reported separately.
    """

    name: str
    terms: tuple[str, ...]
    flagged_ambiguous: bool = False


# §3.2 frozen list (~20). Committed verbatim; additions/removals are logged amendments.
# Flagged ambiguous: items 8 (quantum ML), 9 (federated learning), 17 (prompt engineering).
KNOWN_FIZZLE: tuple[FizzleControl, ...] = (
    FizzleControl("web3", ("web3", "web 3.0")),
    FizzleControl("metaverse", ("metaverse",)),
    FizzleControl("nft", ("nft", "non-fungible token", "non fungible token")),
    FizzleControl("no-code/low-code", ("no-code", "no code", "low-code", "low code")),
    FizzleControl("capsule networks", ("capsule network", "capsule networks", "capsnet")),
    FizzleControl("neuromorphic computing", ("neuromorphic computing", "neuromorphic")),
    FizzleControl("blockchain for AI", ("blockchain for ai", "decentralized ai training",
                                        "decentralized ai", "blockchain ai")),
    FizzleControl("quantum machine learning",
                  ("quantum machine learning", "quantum neural network", "quantum ml"),
                  flagged_ambiguous=True),
    FizzleControl("federated learning", ("federated learning",), flagged_ambiguous=True),
    FizzleControl("GANs for everything", ("generative adversarial network",
                                          "generative adversarial networks",
                                          "generative adversarial", "gan")),
    FizzleControl("meta-learning", ("meta-learning", "meta learning", "learning to learn")),
    FizzleControl("neural architecture search", ("neural architecture search",
                                                 "architecture search", "nas")),
    FizzleControl("graph neural networks for combinatorial optimization",
                  ("graph neural network combinatorial", "gnn combinatorial optimization",
                   "neural combinatorial optimization")),
    FizzleControl("spiking neural networks", ("spiking neural network", "spiking neural networks",
                                              "spiking network")),
    FizzleControl("hyperdimensional computing", ("hyperdimensional computing",
                                                 "hyperdimensional", "vector symbolic")),
    FizzleControl("liquid neural networks", ("liquid neural network", "liquid neural networks",
                                             "liquid time-constant")),
    FizzleControl("prompt engineering", ("prompt engineering",), flagged_ambiguous=True),
    FizzleControl("AI safety via debate", ("ai safety via debate", "safety via debate",
                                           "debate")),
    FizzleControl("swarm intelligence", ("swarm intelligence",)),
    FizzleControl("extreme learning machines", ("extreme learning machine",
                                                "extreme learning machines")),
)

# The contested subset (§3.2 items 8/9/17), reported separately from the headline negatives.
FLAGGED_AMBIGUOUS: tuple[FizzleControl, ...] = tuple(f for f in KNOWN_FIZZLE if f.flagged_ambiguous)


def _monthly_record_counts(fizzle: FizzleControl, term_events: pd.DataFrame) -> pd.Series:
    """Unique-record count per month for any of a fizzle's terms (case-insensitive term match).

    Mirrors the ``groupby(...)['record_id'].nunique()`` aggregation used across ``emergence.py`` — a
    record that mentions several of the fizzle's terms is counted once per month.
    """
    if term_events.empty:
        return pd.Series(dtype="int64")
    wanted = {t.lower() for t in fizzle.terms}
    hit = term_events[term_events["term"].str.lower().isin(wanted)]
    if hit.empty:
        return pd.Series(dtype="int64")
    return hit.groupby("month")["record_id"].nunique().sort_index()


def present_in_corpus(
    fizzle: FizzleControl, term_events: pd.DataFrame, min_records: int = BIRTH_MIN_RECORDS
) -> bool:
    """Whether ``fizzle`` reaches the §2.1 birth threshold in the corpus.

    True iff some 3-month window has ``>= min_records`` distinct records mentioning the fizzle's terms
    (the §2.1 birth rule: first month a theme has ``n_theme >= 5`` records in a 3-month window). A
    fizzle that never clears this is "not present in corpus" (§3.2) and is not forced into the set.
    """
    counts = _monthly_record_counts(fizzle, term_events)
    if counts.empty:
        return False
    # Dense monthly index so the rolling window spans calendar gaps (months with zero records).
    full = counts.reindex(pd.period_range(counts.index.min(), counts.index.max(), freq="M")
                          .astype(str), fill_value=0)
    window = full.rolling(_BIRTH_WINDOW_MONTHS, min_periods=1).sum()
    return bool((window >= min_records).any())


def map_fizzles_to_themes(
    term_events: pd.DataFrame, min_records: int = BIRTH_MIN_RECORDS
) -> dict[str, bool]:
    """For each hand-curated fizzle, whether its terms reach the §2.1 birth threshold in the corpus.

    Returns ``{name: present}``. ``present`` follows ``present_in_corpus`` — flagged-ambiguous items are
    included in the map (so their corpus presence is visible) but the caller is responsible for the §3.2
    rule that they do not count toward the headline negatives unless they independently satisfy Pool A.
    """
    return {f.name: present_in_corpus(f, term_events, min_records) for f in KNOWN_FIZZLE}
