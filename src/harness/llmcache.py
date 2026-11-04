"""Content-addressed LLM call cache: append-only JSONL, loaded into memory on open.

Why: (1) re-runs are free, so crashes and config tweaks don't burn budget;
(2) provider drift is detectable — same key, different live answer means the
pinned snapshot moved; (3) the cache itself is a releasable reproducibility
artifact (it contains prompts over public topic/PTKB data only — never licensed
collection text; a harness invariant)."""
from __future__ import annotations

import json
import os
import sys
import threading
from typing import Any


class JSONLCache:
    def __init__(self, path: str) -> None:
        self.path = path
        self._mem: dict[str, str] = {}
        # meta is kept in memory too, so a CACHE HIT can still report the token usage the
        # original call cost. Without this, a re-run (all hits) would report zero tokens and
        # RQ1's cost axis would collapse to "cached runs are free", which is true of money
        # but not of what the experiment cost to produce.
        self._meta: dict[str, dict] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        # Malformed lines are SKIPPED, not fatal. Two processes appending to one cache can
        # interleave a partial line (observed Aug 22: a grid run and an intrinsic run sharing
        # runs/llm_cache.jsonl produced one corrupt line out of 4703, which then crashed every
        # subsequent run at startup). A cache is a performance artifact; one bad byte must not
        # brick the harness. The count is surfaced in stats() so this stays visible rather than
        # silently eating data -- a nonzero value means some entry was lost and will be
        # recomputed, and that concurrent writers shared a cache when they should not have.
        self.corrupt_lines = 0
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        key, value = rec["key"], rec["value"]
                    except (ValueError, KeyError, TypeError):
                        self.corrupt_lines += 1
                        continue
                    self._mem[key] = value
                    if rec.get("meta"):
                        self._meta[key] = rec["meta"]
        if self.corrupt_lines:
            print(f"  WARNING {path}: skipped {self.corrupt_lines} unparseable cache line(s) "
                  "-- entries lost and will be recomputed. Concurrent runs must not share a "
                  "cache; pass --cache to give each its own.", file=sys.stderr)
        else:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)

    def get(self, key: str) -> str | None:
        with self._lock:
            if key in self._mem:
                self.hits += 1
                return self._mem[key]
            self.misses += 1
            return None

    def get_meta(self, key: str) -> dict | None:
        """Meta recorded when this key was first written (backend, model, token usage).
        Available on hits as well as misses — see the note in __init__."""
        with self._lock:
            return self._meta.get(key)

    def put(self, key: str, value: str, meta: dict[str, Any] | None = None) -> None:
        with self._lock:
            self._mem[key] = value
            rec = {"key": key, "value": value}
            if meta:
                rec["meta"] = meta
                self._meta[key] = meta
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"entries": len(self._mem), "hits": self.hits, "misses": self.misses,
                    "corrupt_lines": self.corrupt_lines}
