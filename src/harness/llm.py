"""Single LLM entry point. Every model call in the experiment goes through here,
and therefore through the cache. A raw API call anywhere else in the codebase is a bug
(a harness invariant).

Backends:
  echo   — offline, deterministic; for tests and plumbing smoke runs.
  openai — any OpenAI-compatible endpoint (OPENAI_API_KEY / OPENAI_BASE_URL).
           temperature is hard-pinned to 0; the model string comes from RunConfig.

The experiment backend must be a boring, snapshot-pinned workhorse model
: structured-output-capable, identical across every condition.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass

from .llmcache import JSONLCache


class DailyQuotaExhausted(RuntimeError):
    """The key's requests-per-day cap is spent. Not retryable on any useful timescale."""


def is_daily_quota(err: Exception) -> bool:
    """Distinguish a per-DAY cap from a per-minute one. Both arrive as HTTP 429.

    Backing off is right for TPM/RPM limits, which clear in seconds. It is useless for RPD:
    the window is a day, while `MAX_ATTEMPTS` of exponential backoff tops out around two
    minutes, so a run burns six more requests against the very quota that is exhausted and
    dies anyway. Observed Aug 23, mid-grid: `Rate limit reached ... on requests per day (RPD):
    Limit 10000, Used 10000`.

    Matching on the message is unavoidable -- the provider signals both with the same class
    and the same status code, and the distinction exists only in prose."""
    text = str(getattr(err, "message", "") or err).lower()
    return "per day" in text or "rpd" in text


@dataclass(frozen=True)
class LLMCallRecord:
    backend: str
    model: str
    prompt: str
    temperature: float = 0.0
    #: canonical JSON of the provider's response_format, or None for a plain-text call. A
    #: structured-output call is a DIFFERENT call and must not share a cache entry with the
    #: same prompt asked as free text.
    response_format: str | None = None

    def cache_key(self) -> str:
        h = hashlib.sha256()
        for part in (self.backend, self.model, f"{self.temperature:.3f}", self.prompt):
            h.update(part.encode("utf-8"))
            h.update(b"\x1e")
        # Folded in ONLY when set, so every key written before structured output existed stays
        # byte-identical and the accumulated cache keeps its value.
        if self.response_format is not None:
            h.update(self.response_format.encode("utf-8"))
            h.update(b"\x1e")
        return h.hexdigest()


class LLMBackend:
    def __init__(self, backend: str, model: str, cache: JSONLCache) -> None:
        if backend not in ("echo", "openai"):
            raise ValueError(f"unknown backend: {backend}")
        self.backend = backend
        self.model = model
        self.cache = cache
        self._client = None
        # Per-RUN token accounting (RQ1: "at what fraction of B1's token cost"). Counted for
        # cache hits too, from the meta stored with the original call: the cost of producing a
        # condition is a property of the condition, not of whether this particular process
        # happened to re-use a cache. `n_calls` counts every complete() the run made.
        self.n_calls = 0
        self.n_uncached_calls = 0
        self.tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def _record_usage(self, usage: dict | None) -> None:
        if not usage:
            return
        for key in self.tokens:
            value = usage.get(key)
            if isinstance(value, (int, float)):
                self.tokens[key] += int(value)

    def usage_summary(self) -> dict:
        """What this run spent, for the manifest and the cost figure."""
        return {"n_calls": self.n_calls, "n_uncached_calls": self.n_uncached_calls,
                **self.tokens}

    def complete(self, prompt: str, response_format: dict | None = None) -> str:
        """`response_format` is the provider's structured-output spec (e.g.
        `{"type": "json_object"}`), passed straight through to the backend.

        Adapters whose upstream library parses JSON out of the reply need this. Without it the
        model answers in prose, the library's `json.loads` raises, and -- because these
        libraries catch their own exceptions -- the mechanism silently does nothing while the
        run still completes and reports plausible metrics. That is exactly how A-Mem's
        evolution step failed on 429 of 479 turns."""
        rf = json.dumps(response_format, sort_keys=True) if response_format else None
        rec = LLMCallRecord(self.backend, self.model, prompt, response_format=rf)
        key = rec.cache_key()
        self.n_calls += 1
        cached = self.cache.get(key)
        if cached is not None:
            meta = self.cache.get_meta(key) or {}
            self._record_usage(meta.get("usage"))
            return cached
        self.n_uncached_calls += 1
        out, usage = self._complete_uncached(prompt, response_format)
        meta: dict = {"backend": self.backend, "model": self.model}
        if usage:
            # Token counts ride along in the cache record ("API budget -- log it
            # anyway"). Cached re-runs cost nothing, so the cache file is the complete ledger
            # of what was actually bought; whatever cost definition the paper settles on can be
            # computed from it after the fact.
            meta["usage"] = usage
        self._record_usage(usage)
        self.cache.put(key, out, meta=meta)
        return out

    # ------------------------------------------------------------------
    #: transient failures worth retrying, by class NAME so the openai package stays an
    #: optional import. 429 is the one that matters: a long grid shares a per-minute token
    #: budget with everything else on the key, so rate limits are expected, not exceptional.
    RETRYABLE = ("RateLimitError", "APIConnectionError", "APITimeoutError",
                 "InternalServerError")
    MAX_ATTEMPTS = 6

    def _call_with_retry(self, kwargs: dict):
        """Exponential backoff on transient API failures.

        Without this a single 429 kills a run that has already spent hours and real money --
        which is exactly what happened to the LongMemEval ceiling run (TPM limit 200k, one
        request asking for 127k tokens). Auth errors, bad requests and context-length errors
        are NOT retried: they are deterministic, and retrying them just burns the budget more
        slowly. Backoff honours the provider's own hint when it gives one.

        A **daily** cap is the exception that must not be retried (see `is_daily_quota`)."""
        import random
        import time
        delay = 2.0
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            try:
                return self._client.chat.completions.create(**kwargs)
            except Exception as e:                      # noqa: BLE001 -- classified below
                if is_daily_quota(e):
                    raise DailyQuotaExhausted(
                        f"the key's requests-per-day quota is exhausted, so retrying cannot "
                        f"help today: {e}\nEverything already computed is in the LLM cache, so "
                        f"re-running this command after the quota window rolls resumes from "
                        f"there and re-spends nothing."
                    ) from e
                if type(e).__name__ not in self.RETRYABLE or attempt == self.MAX_ATTEMPTS:
                    raise
                wait = getattr(e, "retry_after", None) or delay
                wait = min(float(wait), 60.0) + random.uniform(0, 1.0)
                print(f"  {type(e).__name__} (attempt {attempt}/{self.MAX_ATTEMPTS}) -- "
                      f"retrying in {wait:.1f}s", file=sys.stderr)
                time.sleep(wait)
                delay *= 2
        raise RuntimeError("unreachable")               # pragma: no cover

    def _complete_uncached(self, prompt: str,
                           response_format: dict | None = None) -> tuple[str, dict | None]:
        """Returns (text, usage) where usage is {prompt_tokens, completion_tokens,
        total_tokens} when the backend reports it, else None."""
        if self.backend == "echo":
            # Deterministic offline stand-in: echoes the final non-empty line,
            # tagged so tests can assert the prompt actually reached the backend.
            lines = [ln.strip() for ln in prompt.splitlines() if ln.strip()]
            tail = lines[-1] if lines else ""
            return f"[echo:{self.model}] {tail}", None
        # openai-compatible
        if self._client is None:
            try:
                from openai import OpenAI  # type: ignore
            except ImportError as e:  # pragma: no cover
                raise RuntimeError(
                    "openai package not installed; `pip install openai` or use backend='echo'"
                ) from e
            # An EMPTY OPENAI_BASE_URL is worse than an unset one, and passing `or None` is
            # not a defence: with base_url=None the SDK re-reads the variable itself, gets "",
            # and fails every call with APIConnectionError whose real cause is
            # `UnsupportedProtocol: Request URL is missing an 'http://' or 'https://' protocol`.
            # Our own loader skips empty values, but that is not enough -- `import
            # graphiti_core` pulls in python-dotenv somewhere in its chain and re-exports the
            # same .env line as "", poisoning the process for every later call. Deleting the
            # variable is the only fix that survives whatever else the adapters import.
            if not os.environ.get("OPENAI_BASE_URL", "").strip():
                os.environ.pop("OPENAI_BASE_URL", None)
            self._client = OpenAI(
                api_key=os.environ["OPENAI_API_KEY"],
                base_url=os.environ.get("OPENAI_BASE_URL") or None,
            )
        kwargs = {"model": self.model, "temperature": 0.0,
                  "messages": [{"role": "user", "content": prompt}]}
        if response_format:
            kwargs["response_format"] = response_format
        resp = self._call_with_retry(kwargs)
        usage = None
        u = getattr(resp, "usage", None)
        if u is not None:
            usage = {k: getattr(u, k, None)
                     for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
        return resp.choices[0].message.content or "", usage
