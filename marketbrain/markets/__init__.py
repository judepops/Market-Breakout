"""Market rosters — reusable company lists shared across signals, the label, and enrichment.

A roster is the answer to "which companies are in this market", resolved to stable Crunchbase
identifiers. It is shared infrastructure: the Crunchbase funding signal, the $500M outlier label,
and the package→company enrichment all consume the same roster rather than each burying its own
throwaway basket.

Each market module exposes the same interface (``permalinks`` / ``names`` / ``assignee_names`` /
``github_repos``) and is registered in ``MARKETS`` so signals + the label can dispatch by market name
(``get_market("data-infra").permalinks()``). Adding a market = a config module + one registry entry.
"""

from __future__ import annotations

from types import ModuleType

from . import data_infra, llm_dev_tools

MARKETS: dict[str, ModuleType] = {
    "llm-dev-tools": llm_dev_tools,
    "data-infra": data_infra,
}


def get_market(market: str) -> ModuleType:
    """Return the roster module for a market name (raises with the known names if unknown)."""
    try:
        return MARKETS[market]
    except KeyError:
        raise KeyError(f"unknown market {market!r}; known: {sorted(MARKETS)}") from None


__all__ = ["llm_dev_tools", "data_infra", "MARKETS", "get_market"]
