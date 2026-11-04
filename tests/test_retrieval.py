"""retrieval.py tests. BM25Stage/CrossEncoderStage are duck-typed on purpose, so these fakes
stand in without touching pyserini, Java, or a real model download -- keeps this suite as fast
and offline as the rest of `make test`."""
import importlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

import harness.retrieval as retrieval
from harness.replay import RewriteResult
from harness.retrieval import rerank_query, run_retrieval


class FakeBM25:
    def __init__(self, corpus):
        self.corpus = corpus  # query -> [(doc_id, text, bm25_score), ...]
        self.calls = []

    def search(self, query, k=1000):
        self.calls.append((query, k))
        return self.corpus.get(query, [])


class FakeCE:
    """Deterministic fake: score = passage length. What matters for these tests is *what
    query text and candidates it was called with*, captured in .calls."""

    def __init__(self):
        self.calls = []

    def rerank(self, query, candidates, depth=100, batch_size=32):
        self.calls.append({"query": query, "doc_ids": [c[0] for c in candidates], "depth": depth})
        ranked = sorted(candidates, key=lambda c: len(c[1]), reverse=True)
        return [(doc_id, float(len(text))) for doc_id, text, _ in ranked[:depth]]


CANDIDATES = [("d1", "short", 0.9), ("d2", "a much longer passage of text", 0.5)]


def test_rewrite_only_leaves_query_unchanged_even_with_memory():
    bm25 = FakeBM25({"garage door opener": CANDIDATES})
    ce = FakeCE()
    rerank_query("garage door opener", "- user owns a Honda", "rewrite_only", bm25, ce)
    assert ce.calls[0]["query"] == "garage door opener"


def test_rewrite_plus_rerank_appends_memory_block():
    bm25 = FakeBM25({"garage door opener": CANDIDATES})
    ce = FakeCE()
    rerank_query("garage door opener", "- user owns a Honda", "rewrite_plus_rerank", bm25, ce)
    assert ce.calls[0]["query"] == "garage door opener\n\nUser context:\n- user owns a Honda"


def test_rewrite_plus_rerank_with_empty_memory_is_unchanged():
    bm25 = FakeBM25({"q": CANDIDATES})
    ce = FakeCE()
    rerank_query("q", "", "rewrite_plus_rerank", bm25, ce)
    assert ce.calls[0]["query"] == "q"


def test_unknown_injection_mode_raises():
    with pytest.raises(ValueError):
        rerank_query("q", "", "sideways", FakeBM25({}), FakeCE())


def test_bm25_called_with_depth_1000_and_ce_gets_its_output():
    bm25 = FakeBM25({"q": CANDIDATES})
    ce = FakeCE()
    rerank_query("q", "", "rewrite_only", bm25, ce)
    assert bm25.calls == [("q", 1000)]
    assert ce.calls[0]["doc_ids"] == ["d1", "d2"]
    assert ce.calls[0]["depth"] == 100


def test_run_retrieval_keys_output_by_turn_id():
    rewrites = [
        RewriteResult("d1", "31_1", "raw q1", "", "garage door opener"),
        RewriteResult("d1", "31_2", "raw q2", "- likes tofu", "vegetarian protein sources"),
    ]
    bm25 = FakeBM25({
        "garage door opener": CANDIDATES,
        "vegetarian protein sources": [("d3", "tofu and lentils", 0.7)],
    })
    ce = FakeCE()
    out = run_retrieval(rewrites, "rewrite_only", bm25, ce)
    assert set(out.keys()) == {"31_1", "31_2"}
    # fake CE scores by passage length, descending -- d2's text is longer than d1's
    expected_31_1 = [("d2", float(len(CANDIDATES[1][1]))), ("d1", float(len(CANDIDATES[0][1])))]
    assert out["31_1"] == expected_31_1
    assert [q for q, _ in bm25.calls] == ["garage door opener", "vegetarian protein sources"]


def test_bm25_stage_requires_pyserini(monkeypatch):
    monkeypatch.setattr(retrieval, "HAVE_PYSERINI", False)
    with pytest.raises(RuntimeError, match="pyserini"):
        retrieval.BM25Stage()


def test_cross_encoder_stage_requires_sentence_transformers(monkeypatch):
    monkeypatch.setattr(retrieval, "HAVE_CROSS_ENCODER", False)
    with pytest.raises(RuntimeError, match="sentence-transformers"):
        retrieval.CrossEncoderStage()


def test_import_never_fails_on_missing_openai_api_key(monkeypatch):
    """Regression guard: pyserini.search.lucene (1.6.0) eagerly constructs an openai.OpenAI()
    client at import time and raises if OPENAI_API_KEY is unset -- confirmed by direct import
    in a clean env. harness.retrieval must pre-empt that regardless of what's exported in the
    shell it's run from."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    importlib.reload(retrieval)  # must not raise
    assert os.environ.get("OPENAI_API_KEY")
