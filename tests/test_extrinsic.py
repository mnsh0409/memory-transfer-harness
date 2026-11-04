"""Extrinsic metric tests.

judged@k and the I/O layer are tested exhaustively here, since this harness is the
authoritative implementation of both. The relevance measures (nDCG/R/MRR) are tested for
internal consistency, plus a cross-check against trec_eval that runs automatically wherever
pytrec_eval is installed -- see test_native_matches_pytrec_eval_cross_check, which is the
test that settles the open gain-convention question recorded in extrinsic.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from eval.extrinsic import (HAVE_PYTREC_EVAL, _native_ndcg_at_k, _native_recall_at_k,
                            _native_reciprocal_rank, evaluate, judged_at_k, load_qrels,
                            load_run)

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
REAL_QRELS = os.path.join(FIXTURES, "cast_smoke_qrels.txt")


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def test_load_real_qrels_fixture():
    qrels = load_qrels(REAL_QRELS)
    assert set(qrels) == {"31_1", "31_2", "31_3"}
    assert all(isinstance(r, int) for docs in qrels.values() for r in docs.values())
    # grade-0 judgments must survive parsing -- judged@k depends on them
    assert any(r == 0 for docs in qrels.values() for r in docs.values())


def test_load_qrels_accepts_both_year_conventions(tmp_path):
    """CAsT 2019 writes "Q0" in column 2, CAsT 2020 writes "0". Both are real; column 2 is
    ignored. A parser that validates it breaks one year or the other."""
    p2019 = tmp_path / "2019.txt"
    p2019.write_text("31_1 Q0 CAR_abc 2\n31_1 Q0 CAR_def 0\n")
    p2020 = tmp_path / "2020.txt"
    p2020.write_text("81_1 0 CAR_abc 2\n81_1 0 CAR_def 0\n")
    assert load_qrels(str(p2019))["31_1"] == {"CAR_abc": 2, "CAR_def": 0}
    assert load_qrels(str(p2020))["81_1"] == {"CAR_abc": 2, "CAR_def": 0}


def test_load_qrels_rejects_malformed_line(tmp_path):
    p = tmp_path / "bad.txt"
    p.write_text("31_1 Q0 CAR_abc\n")
    with pytest.raises(ValueError, match="want 4"):
        load_qrels(str(p))


def test_load_run_orders_by_rank_column_not_score(tmp_path):
    p = tmp_path / "r.run"
    p.write_text(
        "31_1 Q0 dB 1 0.100000 tag\n"
        "31_1 Q0 dA 2 0.900000 tag\n"   # higher score but rank 2 -- rank wins
    )
    assert [d for d, _ in load_run(str(p))["31_1"]] == ["dB", "dA"]


# ---------------------------------------------------------------------------
# judged@k -- authoritative implementation, so tested hardest
# ---------------------------------------------------------------------------

def test_judged_at_k_counts_grade_zero_as_judged():
    """The whole point of judged@k: a doc judged non-relevant is still judged. On real CAsT
    2019 qrels 72% of judgments are grade 0, so getting this wrong would roughly quarter the
    reported pool coverage."""
    qrels = {"q1": {"d1": 0, "d2": 0, "d3": 3}}
    run = {"q1": [("d1", 9.0), ("d2", 8.0), ("dX", 7.0), ("dY", 6.0)]}
    assert judged_at_k(run, qrels, k=4) == {"q1": 0.5}  # d1,d2 judged; dX,dY not


def test_judged_at_k_denominator_is_short_list_length():
    """A turn returning 4 docs is scored out of 4, not out of 10 -- otherwise a truncated
    result list looks like a pooling problem."""
    qrels = {"q1": {"d1": 1, "d2": 1, "d3": 1, "d4": 1}}
    run = {"q1": [("d1", 4.0), ("d2", 3.0), ("d3", 2.0), ("d4", 1.0)]}
    assert judged_at_k(run, qrels, k=10) == {"q1": 1.0}


def test_judged_at_k_empty_ranking_is_zero():
    assert judged_at_k({"q1": []}, {"q1": {"d1": 1}}, k=10) == {"q1": 0.0}


def test_judged_at_k_skips_qids_with_no_qrels():
    out = judged_at_k({"q1": [("d1", 1.0)], "q2": [("d2", 1.0)]}, {"q1": {"d1": 1}}, k=10)
    assert set(out) == {"q1"}


def test_judged_at_k_on_real_fixture_is_a_fraction():
    qrels = load_qrels(REAL_QRELS)
    judged_docs = list(qrels["31_1"])
    run = {"31_1": [(d, 10.0 - i) for i, d in enumerate(judged_docs[:5])]
                    + [("UNJUDGED_1", 1.0), ("UNJUDGED_2", 0.5)]}
    assert judged_at_k(run, qrels, k=10)["31_1"] == pytest.approx(5 / 7)


# ---------------------------------------------------------------------------
# Native relevance measures -- internal consistency
# ---------------------------------------------------------------------------

def test_native_ndcg_perfect_ranking_is_one():
    rels = {"d1": 3, "d2": 2, "d3": 1}
    ranked = [("d1", 9.0), ("d2", 8.0), ("d3", 7.0)]
    assert _native_ndcg_at_k(ranked, rels, 3) == pytest.approx(1.0)


def test_native_ndcg_reversed_ranking_is_worse():
    rels = {"d1": 3, "d2": 2, "d3": 1}
    good = _native_ndcg_at_k([("d1", 9.0), ("d2", 8.0), ("d3", 7.0)], rels, 3)
    bad = _native_ndcg_at_k([("d3", 9.0), ("d2", 8.0), ("d1", 7.0)], rels, 3)
    assert bad < good


def test_native_ndcg_no_relevant_docs_is_zero_not_nan():
    assert _native_ndcg_at_k([("d1", 1.0)], {"d1": 0}, 3) == 0.0


def test_native_ndcg_gain_conventions_differ_on_graded_qrels():
    """Guards the documented risk: linear and exponential gain are NOT interchangeable on
    CAsT's 0-4 scale. If this ever starts passing as equal, someone has broken the gain
    parameter."""
    rels = {"d1": 4, "d2": 1}
    ranked = [("d2", 9.0), ("d1", 8.0)]  # deliberately suboptimal so iDCG != DCG
    lin = _native_ndcg_at_k(ranked, rels, 2, gain="linear")
    exp = _native_ndcg_at_k(ranked, rels, 2, gain="exponential")
    assert lin != pytest.approx(exp)


def test_native_ndcg_unknown_gain_raises():
    with pytest.raises(ValueError):
        _native_ndcg_at_k([("d1", 1.0)], {"d1": 1}, 3, gain="quadratic")


def test_native_recall_ignores_grade_zero_docs():
    rels = {"d1": 0, "d2": 1, "d3": 2}
    assert _native_recall_at_k([("d1", 3.0), ("d2", 2.0)], rels, 1000) == pytest.approx(0.5)


def test_native_reciprocal_rank_finds_first_relevant():
    rels = {"d3": 1}
    assert _native_reciprocal_rank([("d1", 3.0), ("d2", 2.0), ("d3", 1.0)], rels) == pytest.approx(1 / 3)
    assert _native_reciprocal_rank([("d1", 3.0)], rels) == 0.0


# ---------------------------------------------------------------------------
# evaluate()
# ---------------------------------------------------------------------------

def test_evaluate_refuses_unvalidated_metrics_by_default():
    """Silently substituting the harness's own nDCG for trec_eval's is how a wrong number
    reaches a results table. Without pytrec_eval, evaluate() must raise, not guess."""
    if HAVE_PYTREC_EVAL:
        pytest.skip("pytrec_eval installed -- the refusal path can't trigger here")
    with pytest.raises(RuntimeError, match="pytrec_eval"):
        evaluate({"q1": [("d1", 1.0)]}, {"q1": {"d1": 1}})


def test_evaluate_with_native_metrics_labels_them_unvalidated():
    out = evaluate({"q1": [("d1", 1.0)]}, {"q1": {"d1": 1}}, allow_native_metrics=True)
    if not HAVE_PYTREC_EVAL:
        assert "UNVALIDATED" in out["metrics_source"]
    assert "judged_10" in out["aggregate"]


def test_evaluate_reports_coverage_gaps():
    """n_qrels_only > 0 means judged turns the run never answered -- they're excluded from
    the averages, which is exactly the partial-coverage anomaly the protocol says to flag."""
    run = {"q1": [("d1", 1.0)], "q_extra": [("d9", 1.0)]}
    qrels = {"q1": {"d1": 1}, "q_missing": {"d2": 1}}
    out = evaluate(run, qrels, allow_native_metrics=True)
    assert out["coverage"] == {
        "n_run_qids": 2, "n_qrels_qids": 2, "n_scored": 1,
        "n_qrels_only": 1, "n_run_only": 1,
    }
    assert out["aggregate"]["n_scored_turns"] == 1


def test_evaluate_end_to_end_on_real_qrels_fixture():
    qrels = load_qrels(REAL_QRELS)
    run = {qid: [(d, 10.0 - i) for i, d in enumerate(docs)] for qid, docs in qrels.items()}
    out = evaluate(run, qrels, allow_native_metrics=True)
    assert set(out["per_turn"]) == {"31_1", "31_2", "31_3"}
    assert out["coverage"]["n_qrels_only"] == 0
    for m in out["per_turn"].values():
        assert 0.0 <= m["judged_10"] <= 1.0
        assert 0.0 <= m["ndcg_cut_3"] <= 1.0


@pytest.mark.skipif(not HAVE_PYTREC_EVAL, reason="pytrec_eval not installed")
def test_native_matches_pytrec_eval_cross_check():
    """THE test that settles the open gain-convention question (extrinsic.py).

    It could not run in the sandbox this was written in -- pytrec_eval's setup.py fetches
    trec_eval's C sources from a host that was blocked there. Install pytrec_eval and run it:

        pip install pytrec_eval && pytest tests/test_extrinsic.py -k cross_check

    If it fails on ndcg, flip the default `gain` in _native_ndcg_at_k to the other convention
    and record the answer in extrinsic.py. Passing here means the native path is safe as an
    offline fallback; until then it stays labelled UNVALIDATED."""
    from eval.extrinsic import _native_metrics, _pytrec_eval_metrics
    qrels = load_qrels(REAL_QRELS)
    run = {qid: [(d, 10.0 - i) for i, d in enumerate(docs)] for qid, docs in qrels.items()}

    native = _native_metrics(run, qrels, gain="linear")
    trec = _pytrec_eval_metrics(run, qrels)
    mismatches = {
        (qid, measure): (native[qid][measure], trec[qid][measure])
        for qid in native for measure in ("ndcg_cut_3", "ndcg_cut_5", "recall_1000", "recip_rank")
        if abs(native[qid][measure] - trec[qid][measure]) > 1e-6
    }
    assert not mismatches, (
        f"native implementation diverges from trec_eval: {mismatches}. "
        "Fix _native_ndcg_at_k's gain convention (or the measure that mismatched) and "
        "record it in extrinsic.py."
    )
