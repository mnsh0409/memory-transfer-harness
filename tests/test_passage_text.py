"""BM25Stage hands passage text to the cross-encoder. On the real `cast2019` prebuilt index
`doc.contents()` is None and the passage lives in `doc.raw()` (verified on the live index,
Aug 22 2026) -- so the text lookup must fall back to raw(), and a search whose every hit
comes back textless must raise rather than let the reranker score empty passages."""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from harness.retrieval import BM25Stage, passage_text


class _Doc:
    def __init__(self, contents=None, raw=None):
        self._c, self._r = contents, raw

    def contents(self):
        return self._c

    def raw(self):
        return self._r


def test_passage_text_prefers_contents():
    assert passage_text(_Doc(contents="stored contents", raw="raw text")) == "stored contents"


def test_passage_text_falls_back_to_plain_raw_like_cast2019():
    assert passage_text(_Doc(contents=None, raw="What is throat cancer? ...")) == \
        "What is throat cancer? ..."


def test_passage_text_unwraps_json_raw():
    assert passage_text(_Doc(contents=None, raw='{"id": "d1", "contents": "json passage"}')) == \
        "json passage"
    assert passage_text(_Doc(contents=None, raw='{"id": "d1"}')) == '{"id": "d1"}'  # no text key
    assert passage_text(_Doc(contents=None, raw="{not json")) == "{not json"


def test_passage_text_never_returns_none():
    assert passage_text(None) == ""
    assert passage_text(_Doc(contents=None, raw=None)) == ""
    assert passage_text(_Doc(contents="", raw="")) == ""


class _FakeSearcher:
    def __init__(self, docs):
        self._docs = docs   # docid -> _Doc

    def search(self, query, k=1000):
        return [SimpleNamespace(docid=d, score=10.0 - i) for i, d in enumerate(self._docs)]

    def doc(self, docid):
        return self._docs[docid]


def _stage(searcher):
    stage = BM25Stage.__new__(BM25Stage)   # skip from_prebuilt_index
    stage._searcher = searcher
    return stage


def test_bm25_stage_search_uses_raw_text_fallback():
    stage = _stage(_FakeSearcher({"MARCO_1": _Doc(None, "passage one"),
                                  "CAR_2": _Doc("contents two", None)}))
    out = stage.search("q", k=2)
    assert out == [("MARCO_1", "passage one", 10.0), ("CAR_2", "contents two", 9.0)]


def test_bm25_stage_raises_when_no_hit_has_text():
    stage = _stage(_FakeSearcher({"d1": _Doc(None, None), "d2": _Doc("", "")}))
    with pytest.raises(RuntimeError, match="no passage text"):
        stage.search("what is throat cancer")


def test_bm25_stage_tolerates_a_few_textless_hits():
    stage = _stage(_FakeSearcher({"d1": _Doc(None, None), "d2": _Doc(None, "ok")}))
    assert stage.search("q") == [("d1", "", 10.0), ("d2", "ok", 9.0)]


def test_bm25_stage_empty_result_is_fine():
    assert _stage(_FakeSearcher({})).search("q") == []
