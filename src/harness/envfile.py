"""Minimal .env loader — no python-dotenv dependency needed for three KEY=VALUE lines
(OPENAI_API_KEY, OPENAI_BASE_URL, EXPERIMENT_MODEL).

Deliberately does not override a variable already exported in the shell (os.environ.setdefault
semantics) -- an operator's explicit `export OPENAI_API_KEY=...` should always win over a
stale .env file.
"""
from __future__ import annotations

import os

#: `harness/retrieval.py` must export a dummy OPENAI_API_KEY before it imports pyserini, which
#: constructs an openai client at module-import time. That import happens when a script imports
#: `harness.runners` -- i.e. BEFORE `main()` calls `load_dotenv` -- so under plain setdefault
#: semantics the dummy permanently shadows the real key from .env, and every experiment call
#: fails with `401 Incorrect API key provided: unused-h***`. A value equal to this placeholder
#: therefore counts as ABSENT here, so the .env value wins. (Cost one dead gate run to find.)
PYSERINI_API_KEY_PLACEHOLDER = "unused-harness-does-not-call-openai-via-pyserini"


def load_dotenv(path: str = ".env") -> dict[str, str]:
    """Returns what it set (for logging/debugging); silently does nothing if path is missing --
    not every environment needs one (tests use backend='echo' and no keys at all)."""
    set_vars: dict[str, str] = {}
    if not os.path.exists(path):
        return set_vars
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            # An EMPTY value means "not set", and must not be exported. `OPENAI_BASE_URL=` with
            # nothing after it is the natural way to write "use the default endpoint", but
            # exporting "" breaks the openai SDK in a way the harness cannot defend against:
            # llm.py passes `os.environ.get("OPENAI_BASE_URL") or None`, and the SDK then falls
            # back to reading the SAME empty variable itself, ending up with "" as the base URL
            # and failing every call with APIConnectionError "Request URL is missing an
            # 'http://' or 'https://' protocol". Diagnosed on the real .env.
            # Treating empty as absent also means a blank line can never shadow a
            # real value exported in the shell.
            existing = os.environ.get(key)
            if key and value and (existing is None
                                  or existing == PYSERINI_API_KEY_PLACEHOLDER):
                os.environ[key] = value
                set_vars[key] = value
    return set_vars
