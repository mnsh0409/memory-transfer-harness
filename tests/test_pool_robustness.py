"""Tests for scripts/pool_robustness -- the answer to the paper's biggest threat.

judged@10 and nDCG@3 move together across every condition in the memory_only cell, so a
reviewer can propose that the whole B1-over-B0 gap is pool coverage rather than effectiveness.
This diagnostic rescores the same run files with two measures that ignore unjudged documents.
Its defining property, and what these tests pin, is that **padding a run with documents nobody
judged must not move its score** -- if it did, the diagnostic would inherit the very bias it
exists to remove.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from scripts.pool_robustness import (bpref_per_qid, build_rows, condensed_ndcg_at_k, main,
                                     render)


def _qrels(tmp_path, rows):
    p = tmp_path / "qrels.txt"
    p.write_text("".join(f"{q} 0 {d} {r}\n" for q, d, r in rows), encoding="utf-8")
    return str(p)


def _run(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for qid, docs in rows.items():
            for rank, doc in enumerate(docs, start=1):
                f.write(f"{qid} Q0 {doc} {rank} {1000 - rank}.0 tag\n")


# --------------------------------------------------------------------------- condensed nDCG

def test_condensed_ndcg_ignores_unjudged_documents():
    """The whole point: a ranking full of unjudged documents scores as if they were absent."""
    rels = {"d1": 2, "d2": 1, "d3": 0}
    clean = [("d1", 9.0), ("d2", 8.0)]
    padded = [("unjudged_a", 9.5), ("d1", 9.0), ("unjudged_b", 8.5), ("d2", 8.0)]
    assert condensed_ndcg_at_k(padded, rels, 3) == pytest.approx(
        condensed_ndcg_at_k(clean, rels, 3))


def test_condensed_ndcg_still_penalises_bad_ordering_among_judged():
    """Removing unjudged documents must not remove the ability to be wrong."""
    rels = {"d1": 2, "d2": 1}
    good = condensed_ndcg_at_k([("d1", 9.0), ("d2", 8.0)], rels, 3)
    bad = condensed_ndcg_at_k([("d2", 9.0), ("d1", 8.0)], rels, 3)
    assert good > bad


def test_condensed_ndcg_is_zero_when_nothing_relevant_is_judged():
    assert condensed_ndcg_at_k([("d1", 1.0)], {"d1": 0}, 3) == 0.0


def test_condensed_ndcg_ideal_spans_all_relevant_not_just_retrieved():
    """A run that finds one of three relevant documents must not score 1.0 -- the ideal
    ranking is over the query's judged relevant set, so this stays comparable across
    conditions that retrieved different amounts."""
    rels = {"d1": 2, "d2": 2, "d3": 2}
    assert condensed_ndcg_at_k([("d1", 9.0)], rels, 3) < 1.0


# --------------------------------------------------------------------------- bpref

def test_bpref_ignores_unjudged_documents():
    qrels = {"q1": {"rel1": 1, "nonrel1": 0}}
    clean = {"q1": [("rel1", 9.0), ("nonrel1", 8.0)]}
    padded = {"q1": [("unjudged", 9.5), ("rel1", 9.0), ("junk", 8.5), ("nonrel1", 8.0)]}
    assert bpref_per_qid(padded, qrels)["q1"] == pytest.approx(bpref_per_qid(clean, qrels)["q1"])


def test_bpref_punishes_judged_nonrelevant_ranked_above_relevant():
    qrels = {"q1": {"rel1": 1, "nonrel1": 0}}
    good = bpref_per_qid({"q1": [("rel1", 9.0), ("nonrel1", 8.0)]}, qrels)["q1"]
    bad = bpref_per_qid({"q1": [("nonrel1", 9.0), ("rel1", 8.0)]}, qrels)["q1"]
    assert good > bad


# --------------------------------------------------------------------------- end to end

def _fixture(tmp_path):
    """Two conditions over two queries. `padded` retrieves the same judged documents in the
    same order as `clean` but with unjudged documents interleaved, so its judged@10 is far
    worse while its true effectiveness is identical."""
    runs_dir = tmp_path / "runs"
    qrels_rows = [("q1", "rel1", 2), ("q1", "non1", 0), ("q2", "rel2", 1), ("q2", "non2", 0)]
    qrels_path = _qrels(tmp_path, qrels_rows)

    stats = {
        "cast19.B0_none.aaa": {"collection": "cast19", "condition": "B0_none",
                               "injection": "rewrite_only", "context": "memory_only",
                               "ndcg_cut_3": 0.10, "judged_10": 0.90, "memory_k": 5,
                               "ablation": {}},
        "cast19.B1_full_history.bbb": {"collection": "cast19", "condition": "B1_full_history",
                                       "injection": "rewrite_only", "context": "memory_only",
                                       "ndcg_cut_3": 0.90, "judged_10": 0.20, "memory_k": 5,
                                       "ablation": {}},
    }
    os.makedirs(runs_dir, exist_ok=True)
    (runs_dir / "stats_summary.json").write_text(json.dumps(stats), encoding="utf-8")
    # B0: relevant documents ranked last. B1: relevant first, but padded with unjudged.
    _run(str(runs_dir / "trec" / "cast19.B0_none.aaa.run"),
         {"q1": ["non1", "rel1"], "q2": ["non2", "rel2"]})
    _run(str(runs_dir / "trec" / "cast19.B1_full_history.bbb.run"),
         {"q1": ["rel1", "u1", "u2", "non1"], "q2": ["rel2", "u3", "u4", "non2"]})
    return str(runs_dir), qrels_path


def test_the_condition_with_worse_pool_coverage_can_still_win(tmp_path):
    """The fixture is the reviewer's objection inverted: B1 has the *worse* judged@10 and must
    still win on both robust measures, because it ranks the relevant documents first."""
    runs_dir, qrels_path = _fixture(tmp_path)
    res = build_rows(runs_dir, "cast19", qrels_path=qrels_path)
    b0, b1 = res["runs"]["B0_none"], res["runs"]["B1_full_history"]
    assert b1["judged_10"] < b0["judged_10"]          # worse coverage
    assert sum(b1["bpref"].values()) > sum(b0["bpref"].values())
    assert sum(b1["cndcg_3"].values()) > sum(b0["cndcg_3"].values())


def test_render_names_the_baseline_and_reports_both_measures(tmp_path):
    runs_dir, qrels_path = _fixture(tmp_path)
    text = render(build_rows(runs_dir, "cast19", qrels_path=qrels_path))
    assert "Paired vs B0_none" in text
    assert "bpref" in text and "cnDCG@3" in text
    # the interpretation rule must travel with the numbers, not live only in the docstring
    assert "do not report the nDCG gap as an" in text


def test_headroom_shares_are_suppressed_when_the_gap_is_tiny(tmp_path):
    """A near-zero denominator produced a '+215% of B1-B0' label in the figures once. The
    same guard has to hold here or the diagnostic reintroduces the defect it postdates."""
    runs_dir, qrels_path = _fixture(tmp_path)
    res = build_rows(runs_dir, "cast19", qrels_path=qrels_path)
    for key in ("bpref", "cndcg_3"):
        res["runs"]["B1_full_history"][key] = dict(res["runs"]["B0_none"][key])
    res["runs"]["B1_full_history"]["ndcg_cut_3"] = res["runs"]["B0_none"]["ndcg_cut_3"]
    text = render(res)
    assert "too small to express a share of" in text
    assert "%" not in text.split("Headroom delivered")[1].split("If a share")[0].replace(
        "judged@10", "")


def test_cli_refuses_an_empty_cell(tmp_path, capsys):
    runs_dir, _ = _fixture(tmp_path)
    with pytest.raises(SystemExit):
        main(["--runs-dir", runs_dir, "--collection", "cast20"])


def test_cli_json_mode_emits_means_not_per_qid_dicts(tmp_path, capsys):
    runs_dir, qrels_path = _fixture(tmp_path)
    # point the module's qrels table at the fixture for the CLI path
    import scripts.pool_robustness as pr
    pr.QRELS["cast19"] = qrels_path
    assert main(["--runs-dir", runs_dir, "--collection", "cast19", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload["runs"]["B0_none"]["bpref"], float)
    assert isinstance(payload["runs"]["B0_none"]["cndcg_3"], float)
