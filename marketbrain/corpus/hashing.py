"""Content hashing for provenance and immutable monthly cuts.

``content_hash`` keys a captured source's bytes (provenance). ``cut_hash`` produces
a deterministic digest of a monthly panel cut so it can be recorded in the manifest
and treated as immutable (lock provenance at creation).
"""

from __future__ import annotations

import hashlib

import pandas as pd

_KEY_COLS = ("feature", "market", "month")


def content_hash(data: "str | bytes") -> str:
    """sha256 hex digest of a captured passage / snapshot's bytes."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def cut_hash(panel: pd.DataFrame) -> str:
    """Deterministic digest of a monthly panel cut.

    Rows are sorted by (feature, market, month) and columns ordered before
    serialization so the hash depends only on content, not on row/column order.
    """
    if panel.empty:
        return hashlib.sha256(b"").hexdigest()
    sort_cols = [c for c in _KEY_COLS if c in panel.columns]
    ordered = panel.sort_values(sort_cols).sort_index(axis=1).reset_index(drop=True)
    payload = ordered.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
