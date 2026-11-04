"""The openai backend records token usage in the cache record ("API budget -- log it
anyway"). Tested with a fake client injected in place of the real one -- no network, no key."""
import json
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from harness.llm import LLMBackend
from harness.llmcache import JSONLCache


class _FakeCompletions:
    def __init__(self):
        self.calls = 0
        self.kwargs = []

    def create(self, model, temperature, messages, **extra):
        self.calls += 1
        self.kwargs.append({"model": model, "temperature": temperature, **extra})
        assert temperature == 0.0
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=f"answer to: {messages[0]['content']}"))],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=4, total_tokens=15),
        )


class _FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_FakeCompletions())


def test_openai_backend_records_usage_in_cache_meta(tmp_path):
    cache_path = str(tmp_path / "cache.jsonl")
    cache = JSONLCache(cache_path)
    backend = LLMBackend("openai", "pinned-snapshot-2026-01-01", cache)
    fake = _FakeClient()
    backend._client = fake  # bypass the real OpenAI() construction

    out1 = backend.complete("hello")
    out2 = backend.complete("hello")          # cache hit -- no second API call
    assert out1 == out2 == "answer to: hello"
    assert fake.chat.completions.calls == 1
    stats = cache.stats()
    # subset, not equality: stats() grows (corrupt_lines was added later) and a test that
    # pins the whole dict fails on every future field for no reason
    assert (stats["entries"], stats["hits"], stats["misses"]) == (1, 1, 1)

    records = [json.loads(ln) for ln in open(cache_path, encoding="utf-8") if ln.strip()]
    assert len(records) == 1
    assert records[0]["meta"]["usage"] == {"prompt_tokens": 11, "completion_tokens": 4,
                                           "total_tokens": 15}
    assert records[0]["meta"]["model"] == "pinned-snapshot-2026-01-01"


def test_usage_is_counted_on_cache_hits_too(tmp_path):
    """RQ1 asks what a condition COST to produce. If hits counted zero, a re-run (all hits)
    would report a free condition and the cost axis would be meaningless."""
    cache_path = str(tmp_path / "cache.jsonl")
    backend = LLMBackend("openai", "snap-1", JSONLCache(cache_path))
    backend._client = _FakeClient()
    backend.complete("hello")
    backend.complete("hello")            # hit
    assert backend.usage_summary() == {"n_calls": 2, "n_uncached_calls": 1,
                                       "prompt_tokens": 22, "completion_tokens": 8,
                                       "total_tokens": 30}

    # a fresh process over the same cache: every call is a hit, cost still reported
    fresh = LLMBackend("openai", "snap-1", JSONLCache(cache_path))
    fresh.complete("hello")
    assert fresh.usage_summary() == {"n_calls": 1, "n_uncached_calls": 0,
                                     "prompt_tokens": 11, "completion_tokens": 4,
                                     "total_tokens": 15}


def test_replay_manifest_carries_usage(tmp_path):
    """The cost figure reads this off the manifest/summary, so the plumbing is pinned here."""
    import shutil

    from harness.config import RunConfig
    from harness.runners import run_replay_config

    fixtures = os.path.join(os.path.dirname(__file__), "fixtures")
    data_dir = tmp_path / "data" / "cast"
    data_dir.mkdir(parents=True)
    shutil.copy(os.path.join(fixtures, "cast_smoke.json"), data_dir / "cast19_topics.json")
    config = RunConfig(condition="B0_none", injection="rewrite_only", collection="cast19",
                       backend="echo", model="smoke-model")
    m = run_replay_config(config, data_dir=str(tmp_path / "data"),
                          runs_dir=str(tmp_path / "runs"),
                          cache_path=str(tmp_path / "runs" / "c.jsonl"))
    assert m["usage"]["n_calls"] == 20            # one rewrite per user turn in the fixture
    assert m["usage"]["total_tokens"] == 0        # echo backend reports no usage


def test_echo_backend_has_no_usage(tmp_path):
    cache_path = str(tmp_path / "cache.jsonl")
    backend = LLMBackend("echo", "smoke-model", JSONLCache(cache_path))
    backend.complete("x")
    rec = json.loads(open(cache_path, encoding="utf-8").readline())
    assert "usage" not in rec["meta"]


def test_response_format_is_forwarded_and_keys_a_separate_cache_entry(tmp_path):
    """A structured-output call is a DIFFERENT call: same prompt asked as JSON must not reuse
    the plain-text answer. And keys for plain calls must be unchanged, so the cache built
    before structured output existed keeps its value."""
    from harness.llm import LLMCallRecord

    plain = LLMCallRecord("openai", "m", "p")
    same_plain = LLMCallRecord("openai", "m", "p", response_format=None)
    structured = LLMCallRecord("openai", "m", "p", response_format='{"type": "json_object"}')
    assert plain.cache_key() == same_plain.cache_key()
    assert plain.cache_key() != structured.cache_key()

    cache_path = str(tmp_path / "c.jsonl")
    backend = LLMBackend("openai", "m", JSONLCache(cache_path))
    fake = _FakeClient()
    backend._client = fake
    backend.complete("hello")
    backend.complete("hello", response_format={"type": "json_object"})
    assert fake.chat.completions.calls == 2, "structured call must not hit the plain-text entry"
    assert fake.chat.completions.kwargs[-1]["response_format"] == {"type": "json_object"}
    assert "response_format" not in fake.chat.completions.kwargs[0]


def test_empty_openai_base_url_is_removed_before_the_client_is_built(tmp_path, monkeypatch):
    """An empty OPENAI_BASE_URL breaks every call: with base_url=None the SDK re-reads the
    variable itself, gets "", and raises APIConnectionError (real cause: UnsupportedProtocol).
    Our .env loader skips empty values, but `import graphiti_core` pulls in python-dotenv and
    re-exports the same line as "", so the defence has to live here. Diagnosed after it broke
    a Graphiti probe and would have broken any replay sharing the process."""
    monkeypatch.setenv("OPENAI_BASE_URL", "")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    backend = LLMBackend("openai", "m", JSONLCache(str(tmp_path / "c.jsonl")))
    captured = {}

    class _Client:
        def __init__(self, **kw):
            captured.update(kw)
            self.chat = SimpleNamespace(completions=_FakeCompletions())

    openai = pytest.importorskip("openai")  # optional: not in requirements.txt
    monkeypatch.setattr(openai, "OpenAI", _Client)
    backend.complete("hello")
    assert captured["base_url"] is None
    assert "OPENAI_BASE_URL" not in os.environ, "the poisoned variable must be removed"


def test_a_real_base_url_is_still_honoured(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    backend = LLMBackend("openai", "m", JSONLCache(str(tmp_path / "c.jsonl")))
    captured = {}

    class _Client:
        def __init__(self, **kw):
            captured.update(kw)
            self.chat = SimpleNamespace(completions=_FakeCompletions())

    openai = pytest.importorskip("openai")  # optional: not in requirements.txt
    monkeypatch.setattr(openai, "OpenAI", _Client)
    backend.complete("hello")
    assert captured["base_url"] == "https://proxy.example/v1"


class _FlakyCompletions:
    """Raises `exc` for the first `n_fail` calls, then succeeds."""

    def __init__(self, exc, n_fail):
        self.exc, self.n_fail, self.calls = exc, n_fail, 0

    def create(self, **kw):
        self.calls += 1
        if self.calls <= self.n_fail:
            raise self.exc
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="recovered"))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )


class _RateLimitError(Exception):
    pass


_RateLimitError.__name__ = "RateLimitError"


class _BadRequestError(Exception):
    pass


_BadRequestError.__name__ = "BadRequestError"


def _backend_with(monkeypatch, tmp_path, completions):
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)      # no real waiting in tests
    b = LLMBackend("openai", "m", JSONLCache(str(tmp_path / "c.jsonl")))
    b._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return b


def test_rate_limit_is_retried_then_succeeds(tmp_path, monkeypatch, capsys):
    """A 429 must not kill a run that has already spent hours and real money -- which is
    exactly what happened to the LongMemEval ceiling run."""
    c = _FlakyCompletions(_RateLimitError("429 tokens per min"), n_fail=3)
    b = _backend_with(monkeypatch, tmp_path, c)
    assert b.complete("hello") == "recovered"
    assert c.calls == 4
    assert "RateLimitError (attempt 1/6)" in capsys.readouterr().err


def test_deterministic_errors_are_not_retried(tmp_path, monkeypatch):
    """Retrying a bad request or a context-length error just burns budget more slowly."""
    c = _FlakyCompletions(_BadRequestError("context_length_exceeded"), n_fail=99)
    b = _backend_with(monkeypatch, tmp_path, c)
    with pytest.raises(Exception) as ei:
        b.complete("hello")
    assert type(ei.value).__name__ == "BadRequestError"
    assert c.calls == 1, "must fail on the first attempt, not after backoff"


def test_retry_gives_up_and_raises(tmp_path, monkeypatch):
    c = _FlakyCompletions(_RateLimitError("429"), n_fail=99)
    b = _backend_with(monkeypatch, tmp_path, c)
    with pytest.raises(Exception) as ei:
        b.complete("hello")
    assert type(ei.value).__name__ == "RateLimitError"
    assert c.calls == LLMBackend.MAX_ATTEMPTS
