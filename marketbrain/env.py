"""Tiny .env loader + secret accessors (no third-party dependency).

Secrets (API keys) are kept out of the code and out of git: real values live in a
gitignored ``.env`` at the repo root; ``.env.example`` is the committed template.
This module reads ``.env`` into ``os.environ`` (without clobbering vars already set
in the real environment) and exposes typed accessors.

These keys belong to the **v1.1** unstructured-text / LLM-extraction component, not
the v1 deterministic corpus ("No LLMs in the v1 corpus").

    from marketbrain.env import get_gemini_api_key
    key = get_gemini_api_key()  # raises if unset
"""

from __future__ import annotations

import os
from pathlib import Path

# Repo root = two levels up from this file (marketbrain/env.py -> marketbrain -> repo).
_REPO_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: str | Path | None = None, *, override: bool = False) -> dict[str, str]:
    """Parse a ``.env`` file and populate ``os.environ``.

    Real environment variables win over ``.env`` unless ``override=True``. Returns the
    parsed key/value pairs (whether or not they were applied). Missing file -> ``{}``.
    Supports ``KEY=value``, ``#`` comments, blank lines, optional surrounding quotes,
    and a leading ``export``.
    """
    dotenv_path = Path(path) if path is not None else _REPO_ROOT / ".env"
    parsed: dict[str, str] = {}
    if not dotenv_path.is_file():
        return parsed

    for raw in dotenv_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        parsed[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return parsed


def get_secret(name: str, *, required: bool = True) -> str | None:
    """Return secret ``name`` from the environment (loading ``.env`` on first miss).

    Raises ``RuntimeError`` with a fix hint if ``required`` and still unset/empty.
    """
    value = os.environ.get(name)
    if not value:
        load_dotenv()
        value = os.environ.get(name)
    if not value:
        if required:
            raise RuntimeError(
                f"{name} is not set. Add it to .env (copy .env.example -> .env and "
                f"fill it in) or export {name} in your shell."
            )
        return None
    return value


def get_gemini_api_key(*, required: bool = True) -> str | None:
    """The Google Gemini API key for the v1.1 unstructured-text component."""
    return get_secret("GEMINI_API_KEY", required=required)
