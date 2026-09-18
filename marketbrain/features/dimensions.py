"""Abstract the raw signals into 4 comparable DIMENSIONS (the fix for structured signal gaps).

No market has all signals: software has GitHub/packages but no patents; deep-tech has patents/arXiv but no
GitHub; every market has Wikipedia/Hacker News. Comparing raw signals across markets is therefore wrong, a
biotech's "GitHub stars = 0" means "no OSS channel exists", not "no adoption". So the model's inputs are 4
dimensions, each fed by whichever signals a market HAS:

    attention   <- wikipedia, hackernews          (universal backbone: every market)
    research    <- arxiv (any field: cs, q-fin, q-bio, physics, eess, ...)
    adoption    <- github_stars, npm/pypi
    innovation  <- patents

Two kinds of "empty" are handled DIFFERENTLY (this is the whole point):
  * a signal present but valued zero      -> a REAL ZERO (informative: tracked, just not adopted).
  * a dimension with NO channel for a market -> UNMEASURED, masked (covered=False), NEVER forced to 0, so we
    never tell the model "biotech has zero adoption". The coverage mask records which dimensions are real.

Attention is universal, so it is the cross-domain-comparable spine; the other dimensions are comparable
within the domains that share their channels (reinforces domain-aware prediction). This module is pure
(no network); Phase 2 collects the raw signal values, this rolls them into the 4 dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass

# signal token -> dimension. Tokens match markets.catalog coverage (gh/pkg/ax/pat/wiki/hn).
SIGNAL_TO_DIM: dict[str, str] = {
    "wiki": "attention", "hn": "attention",
    "ax": "research",
    "gh": "adoption", "pkg": "adoption",
    "pat": "innovation",
}
DIMENSIONS: tuple[str, ...] = ("attention", "research", "adoption", "innovation")


def channels_for(dim: str) -> tuple[str, ...]:
    """The signal tokens that feed a dimension."""
    return tuple(s for s, d in SIGNAL_TO_DIM.items() if d == dim)


@dataclass(frozen=True)
class DimensionValue:
    value: "float | None"    # aggregate of the market's available channels; None when the dimension is unmeasured
    covered: bool            # does the market have ANY channel for this dimension?
    n_channels: int          # how many of the dimension's signals the market actually has


def covered_dimensions(coverage: "tuple[str, ...]") -> set[str]:
    """Which of the 4 dimensions a market can measure, given the signals it has a channel for."""
    return {SIGNAL_TO_DIM[s] for s in coverage if s in SIGNAL_TO_DIM}


def roll_up(coverage: "tuple[str, ...]", signal_values: "dict[str, float] | None" = None) -> "dict[str, DimensionValue]":
    """Roll raw signal values into the 4 dimensions for one market.

    ``coverage`` = the signal tokens this market HAS a channel for (from catalog). ``signal_values`` = the
    measured value per signal token (Phase 2 fills this; defaults to 0 for a present channel). A dimension
    with no channel is **masked** (covered=False, value=None); a present-but-zero channel is a real 0.
    """
    vals = signal_values or {}
    out: dict[str, DimensionValue] = {}
    for dim in DIMENSIONS:
        chans = [s for s in channels_for(dim) if s in coverage]
        if not chans:
            out[dim] = DimensionValue(value=None, covered=False, n_channels=0)
        else:
            # mean over the channels the market has; a tracked-but-zero channel counts as 0 (a real zero)
            v = sum(float(vals.get(s, 0.0)) for s in chans) / len(chans)
            out[dim] = DimensionValue(value=v, covered=True, n_channels=len(chans))
    return out
