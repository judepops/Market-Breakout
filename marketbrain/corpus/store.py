"""Append-only parquet store for the bitemporal corpus.

Design rules (from the methodology):
  * never delete — supersede by end-dating the prior row and appending the revision;
  * each monthly panel cut is immutable and recorded with its ``cut_hash`` in a
    committed manifest so provenance is locked at creation even though the data
    blobs themselves are git-ignored.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..schema.table import OBSERVATION_COLUMNS
from .hashing import cut_hash

OBS_FILE = "observations.parquet"
CUTS_DIR = "cuts"
MANIFEST = "MANIFEST.tsv"


class CorpusStore:
    """Filesystem-backed store rooted at ``root`` (e.g. ``data/raw``)."""

    def __init__(self, root: "str | Path"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / CUTS_DIR).mkdir(exist_ok=True)

    # --- bitemporal observation log -------------------------------------- #
    def load_observations(self) -> pd.DataFrame:
        path = self.root / OBS_FILE
        if not path.exists():
            return pd.DataFrame(columns=list(OBSERVATION_COLUMNS))
        return pd.read_parquet(path)

    def append_observations(self, observations: pd.DataFrame) -> None:
        """Append rows to the log (append-only; existing rows are never rewritten)."""
        if observations.empty:
            return
        existing = self.load_observations()
        combined = pd.concat([existing, observations], ignore_index=True)
        combined.to_parquet(self.root / OBS_FILE, index=False)

    # --- immutable monthly cuts ------------------------------------------ #
    def write_cut(self, month: str, panel: pd.DataFrame) -> str:
        """Persist a monthly panel cut, record its hash in the manifest, return the hash."""
        h = cut_hash(panel)
        panel.to_parquet(self.root / CUTS_DIR / f"cut_{month}.parquet", index=False)
        self._record(month, h, len(panel))
        return h

    def load_cut(self, month: str) -> pd.DataFrame:
        return pd.read_parquet(self.root / CUTS_DIR / f"cut_{month}.parquet")

    def _record(self, month: str, h: str, n_rows: int) -> None:
        line = f"{month}\t{h}\t{n_rows}\n"
        manifest = self.root / MANIFEST
        prior = manifest.read_text() if manifest.exists() else "month\tcut_hash\tn_rows\n"
        prior = "".join(ln for ln in prior.splitlines(keepends=True)
                        if not ln.startswith(f"{month}\t"))
        manifest.write_text(prior + line)
