"""A per-day rate limit must fail fast; a per-minute one must back off.

Both arrive as HTTP 429 from the same exception class, so the only signal is the message.
Retrying a daily cap spends six more requests against the quota that is already exhausted and
then dies anyway -- which is what happened mid-grid on Aug 23 (`requests per day (RPD): Limit
10000, Used 10000`) to two jobs at once.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from harness.llm import DailyQuotaExhausted, LLMBackend, is_daily_quota
from harness.llmcache import JSONLCache

RPD = ("Rate limit reached for gpt-4o-mini in organization org-x on requests per day "
       "(RPD): Limit 10000, Used 10000, Requested 1. Please try again in 8.64s.")
TPM = ("Rate limit reached for gpt-4o-mini in organization org-x on tokens per min "
       "(TPM): Limit 200000, Used 72973, Requested 127393. Please try again in 109ms.")


class _RateLimitError(Exception):
    """Stands in for openai.RateLimitError -- classified by NAME, so the name must match."""


_RateLimitError.__name__ = "RateLimitError"


def test_daily_and_per_minute_are_told_apart():
    assert is_daily_quota(_RateLimitError(RPD))
    assert not is_daily_quota(_RateLimitError(TPM))


def test_message_attribute_is_consulted_too():
    e = _RateLimitError("429")
    e.message = RPD
    assert is_daily_quota(e)


def _backend(tmp_path, err, calls):
    b = LLMBackend("openai", "m", JSONLCache(str(tmp_path / "c.jsonl")))

    class _Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            raise err

    class _Client:
        chat = type("C", (), {"completions": _Completions()})()

    b._client = _Client()
    return b


def test_daily_quota_raises_immediately_without_burning_retries(tmp_path):
    calls = []
    b = _backend(tmp_path, _RateLimitError(RPD), calls)
    with pytest.raises(DailyQuotaExhausted) as exc:
        b._call_with_retry({"model": "m"})
    assert len(calls) == 1, f"retried a daily cap {len(calls)} times"
    # the message must tell the operator the work is not lost
    assert "cache" in str(exc.value).lower()


def test_per_minute_limit_still_retries(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    calls = []
    b = _backend(tmp_path, _RateLimitError(TPM), calls)
    with pytest.raises(Exception) as exc:
        b._call_with_retry({"model": "m"})
    assert not isinstance(exc.value, DailyQuotaExhausted)
    assert len(calls) == b.MAX_ATTEMPTS, "per-minute limits must still back off"
