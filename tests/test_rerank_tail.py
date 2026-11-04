"""rerank_query cross-encodes the BM25 top-RERANK_DEPTH and keeps the BM25 tail underneath, so
the run file has BM25_DEPTH docs per turn (R@1000 = first-stage recall) and the cross-encoder
sees RERANK_DEPTH pairs, not BM25_DEPTH. Pinned here because both halves silently change a
reported number if they drift."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eval.extrinsic import load_run
from harness.retrieval import BM25_DEPTH, RERANK_DEPTH, rerank_query
from harness.trec import validate_run, write_run


class _BM25:
    def __init__(self, n):
        self.n = n

    def search(self, query, k=1000):
        return [(f"d{i:04d}", f"passage {i}", 100.0 - i * 0.01) for i in range(min(self.n, k))]


class _CE:
    def __init__(self):
        self.seen = None

    def rerank(self, query, candidates, depth=100, batch_size=32):
        self.seen = [c[0] for c in candidates]
        # reverse the head so reranking is visibly different from BM25 order
        ranked = list(reversed(candidates))[:depth]
        return [(doc_id, 5.0 - i * 0.001) for i, (doc_id, _, _) in enumerate(ranked)]


def test_ce_sees_only_the_head_and_tail_is_kept_below(tmp_path):
    ce = _CE()
    out = rerank_query("q", "", "rewrite_only", _BM25(BM25_DEPTH), ce)
    assert len(ce.seen) == RERANK_DEPTH                       # 100 pairs, not 1000
    assert len(out) == BM25_DEPTH                             # run carries 1000 docs
    head_ids = [d for d, _ in out[:RERANK_DEPTH]]
    assert head_ids[0] == f"d{RERANK_DEPTH - 1:04d}"          # reranked order wins in the head
    assert [d for d, _ in out[RERANK_DEPTH:]] == [f"d{i:04d}" for i in range(RERANK_DEPTH, BM25_DEPTH)]
    scores = [s for _, s in out]
    assert all(a > b for a, b in zip(scores, scores[1:]))     # strictly decreasing throughout
    assert max(scores[RERANK_DEPTH:]) < min(scores[:RERANK_DEPTH])

    # round-trips through the TREC writer/validator/reader with the ordering intact
    path = str(tmp_path / "x.run")
    write_run(path, {"31_1": out}, tag="t")
    assert validate_run(path, expected_qids={"31_1"}) == []
    assert [d for d, _ in load_run(path)["31_1"]] == [d for d, _ in out]


def test_short_candidate_lists_have_no_tail():
    ce = _CE()
    out = rerank_query("q", "", "rewrite_only", _BM25(7), ce)
    assert len(ce.seen) == 7 and len(out) == 7


def test_empty_candidates_is_empty():
    assert rerank_query("q", "", "rewrite_only", _BM25(0), _CE()) == []
