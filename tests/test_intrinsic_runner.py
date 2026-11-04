"""Tests for the intrinsic driver (eval/intrinsic_runner.py + scripts/run_intrinsic.py).

All offline: B0/B1 baselines are real MemorySystems and the lexical judge is deterministic, so
these exercise the true code path with no API key. Where the real data files are present they
are used (iKAT 2023 is small and tracked-adjacent; LongMemEval is a separate download).
"""
import csv
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from eval.intrinsic import lexical_judge
from eval.intrinsic_runner import (evidence_texts, longmemeval_dialogue, memory_ndcg_at_k,
                                   replay_for_memory, run_longmemeval_recall,
                                   run_ptkb_reconstruction, stratified_sample, summarise,
                                   write_manual_check)
from harness.memory_base import (Dialogue, FullHistoryMemory, NoPersonalizationMemory,
                                 Turn)

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
IKAT23 = os.path.join(REPO_ROOT, "data", "ikat", "ikat23_topics.json")
LME_S = os.path.join(REPO_ROOT, "data", "longmemeval", "data", "longmemeval_s_cleaned.json")


def _dialogue() -> Dialogue:
    return Dialogue(
        dialogue_id="9-1",
        turns=[
            Turn("9-1-1", "user", "I am a vegetarian and I avoid peanuts."),
            Turn("9-1-2", "system", "Noted."),
            Turn("9-1-3", "user", "What should I cook tonight?"),
        ],
        ptkb=["I am a vegetarian.", "I am allergic to peanuts."],
        meta={"ptkb_by_id": {"1": "I am a vegetarian.", "2": "I am allergic to peanuts."},
              "ptkb_provenance": {"9-1-1": [1], "9-1-3": [1, 2]}},
    )


# ---------------------------------------------------------------------------
# Replay protocol
# ---------------------------------------------------------------------------

def test_replay_queries_before_ingesting_the_current_turn():
    """the harness's replay-order invariant, re-pinned here because this module has its own replay
    loop (no rewrite call) and could drift from harness/replay.py."""
    retrievals = replay_for_memory(_dialogue(), FullHistoryMemory())
    assert [r.turn_id for r in retrievals] == ["9-1-1", "9-1-3"]   # user turns only
    assert retrievals[0].entry_texts == []                          # nothing before turn 1
    joined = " ".join(retrievals[1].entry_texts)
    assert "vegetarian" in joined                                   # earlier turns are there
    assert "cook tonight" not in joined                             # the current turn is NOT


def test_replay_resets_between_dialogues():
    memory = FullHistoryMemory()
    replay_for_memory(_dialogue(), memory)
    again = replay_for_memory(_dialogue(), memory)
    assert again[0].entry_texts == []          # reset() ran; no leakage from the first pass


# ---------------------------------------------------------------------------
# Memory-retrieval nDCG
# ---------------------------------------------------------------------------

def test_memory_ndcg_is_none_when_the_turn_has_no_gold():
    """Not applicable, never 0 -- averaging zeros over unlabelled turns would depress every
    system by the same silent amount."""
    assert memory_ndcg_at_k(["anything"], [], lexical_judge, k=3) is None


def test_memory_ndcg_perfect_and_empty():
    gold = ["I am a vegetarian."]
    assert memory_ndcg_at_k(["I am a vegetarian."], gold, lexical_judge, k=3) == pytest.approx(1.0)
    assert memory_ndcg_at_k([], gold, lexical_judge, k=3) == pytest.approx(0.0)


def test_memory_ndcg_rewards_rank_one():
    gold = ["I am a vegetarian."]
    top = memory_ndcg_at_k(["I am a vegetarian.", "unrelated"], gold, lexical_judge, k=3)
    low = memory_ndcg_at_k(["unrelated", "I am a vegetarian."], gold, lexical_judge, k=3)
    assert top > low


# ---------------------------------------------------------------------------
# Axis 1
# ---------------------------------------------------------------------------

def test_ptkb_reconstruction_b0_scores_zero_recall():
    """B0 stores nothing, so nothing can be reconstructed -- the floor of this axis."""
    out = run_ptkb_reconstruction([_dialogue()], NoPersonalizationMemory(), lexical_judge)
    row = out["rows"][0]
    assert row["n_memories"] == 0 and row["recall"] == 0.0 and row["f1"] == 0.0
    assert row["n_gold_ptkb"] == 2 and row["n_user_turns"] == 2


def test_ptkb_reconstruction_b1_recovers_something_and_logs_decisions():
    out = run_ptkb_reconstruction([_dialogue()], FullHistoryMemory(), lexical_judge)
    row = out["rows"][0]
    assert row["n_memories"] == 3           # every turn, verbatim
    assert row["recall"] > 0.0
    assert row["memory_ndcg_3"] is not None and row["n_turns_with_provenance"] == 2
    assert out["decisions"] and all(d["verdict"] in ("ENTAILED", "PARTIAL", "ABSENT")
                                    for d in out["decisions"])
    assert all(d["dialogue_id"] == "9-1" for d in out["decisions"])

    # The two directions must be DISTINGUISHABLE in the log, not both labelled the same.
    # The judge prompt is symmetric, so recall (gold -> memories) and precision
    # (memory -> gold) look alike after the fact; the 100-decision human check is drawn from
    # this log and is the only external validation this axis has, and an annotator who cannot
    # tell which question a row asks cannot answer it. They were both "reconstruction" until
    # Aug 23, which made every exported sample ambiguous.
    directions = {d["direction"] for d in out["decisions"]}
    assert len(directions) == 2, directions
    recall_label = next(d for d in directions if d.startswith("recall"))
    precision_label = next(d for d in directions if d.startswith("precision"))

    # and each label must sit on rows that really are that direction: a recall row's reference
    # is a gold PTKB statement, a precision row's reference is a stored memory (here, a turn).
    gold = set(_dialogue().ptkb)
    recall_rows = [d for d in out["decisions"] if d["direction"] == recall_label]
    precision_rows = [d for d in out["decisions"] if d["direction"] == precision_label]
    assert recall_rows and precision_rows
    assert all(d["reference"] in gold for d in recall_rows)
    assert all(d["reference"] not in gold for d in precision_rows)
    assert all(set(d["candidates"]) <= gold for d in precision_rows)


@pytest.mark.skipif(not os.path.exists(IKAT23), reason="iKAT 2023 topics not fetched")
def test_axis1_runs_on_the_real_ikat23_file():
    from harness.loaders.ikat import load_ikat_topics

    dialogues = load_ikat_topics(IKAT23)[:3]
    out = run_ptkb_reconstruction(dialogues, FullHistoryMemory(), lexical_judge)
    assert len(out["rows"]) == 3
    assert all(r["n_gold_ptkb"] > 0 for r in out["rows"])
    # ptkb_provenance really is usable as the memory-retrieval qrels (STATUS open question)
    assert any(r["n_turns_with_provenance"] > 0 for r in out["rows"])


# ---------------------------------------------------------------------------
# Axis 2 -- the has_answer trap
# ---------------------------------------------------------------------------

def _question():
    return {
        "question_id": "q1", "question_type": "single-session-user",
        "question": "What car problem did I have?",
        "haystack_session_ids": ["s1", "s2"],
        "answer_session_ids": ["s1"],
        "haystack_sessions": [
            [{"role": "user", "content": "My GPS system stopped working.", "has_answer": "True"},
             {"role": "assistant", "content": "That sounds annoying.", "has_answer": "False"},
             {"role": "user", "content": "I also washed the car.", "has_answer": "False"}],
            [{"role": "user", "content": "Unrelated chat about weather.", "has_answer": "False"}],
        ],
    }


def test_evidence_is_only_the_flagged_messages_not_the_whole_answer_session():
    """`has_answer` is the STRING "True"/"False" (896 True vs 10,064 False in the real file),
    so a truthiness check marks everything as evidence and recall looks ~1.0 for every
    system. This test is the guard against that going unnoticed."""
    assert evidence_texts(_question()) == ["My GPS system stopped working."]


def test_evidence_handles_real_booleans_and_missing_keys():
    q = {"haystack_sessions": [[{"role": "user", "content": "a", "has_answer": True},
                                {"role": "user", "content": "b", "has_answer": False},
                                {"role": "user", "content": "c"}]]}
    assert evidence_texts(q) == ["a"]


def test_longmemeval_dialogue_flattens_sessions_in_order():
    d = longmemeval_dialogue(_question())
    assert d.dialogue_id == "q1"
    assert [t.turn_id for t in d.turns][:2] == ["q1:s1:0", "q1:s1:1"]
    assert len(d.turns) == 4
    assert [t.role for t in d.turns][:2] == ["user", "system"]


def test_longmemeval_recall_b1_finds_the_evidence():
    out = run_longmemeval_recall([_question()], FullHistoryMemory(), lexical_judge)
    row = out["rows"][0]
    assert row["n_evidence"] == 1
    assert row["evidence_recall"] == pytest.approx(1.0)   # B1 keeps everything verbatim
    assert row["n_sessions"] == 2


def test_longmemeval_recall_b0_finds_nothing():
    row = run_longmemeval_recall([_question()], NoPersonalizationMemory(),
                                 lexical_judge)["rows"][0]
    assert row["n_retrieved"] == 0 and row["evidence_recall"] == 0.0


def test_longmemeval_recall_is_none_without_labels():
    q = _question()
    for session in q["haystack_sessions"]:
        for m in session:
            m["has_answer"] = "False"
    row = run_longmemeval_recall([q], FullHistoryMemory(), lexical_judge)["rows"][0]
    assert row["n_evidence"] == 0 and row["evidence_recall"] is None   # n/a, not 0


@pytest.mark.skipif(not os.path.exists(LME_S), reason="LongMemEval _S not downloaded")
def test_axis2_runs_on_the_real_longmemeval_file():
    import itertools
    with open(LME_S, encoding="utf-8") as f:
        questions = json.load(f)
    sample = list(itertools.islice(iter(questions), 2))
    out = run_longmemeval_recall(sample, FullHistoryMemory(), lexical_judge)
    assert len(out["rows"]) == 2
    assert all(r["n_sessions"] > 0 for r in out["rows"])
    # the real file must yield SOME flagged evidence, or the has_answer parse has broken again
    assert any(r["n_evidence"] > 0 for r in out["rows"])


# ---------------------------------------------------------------------------
# The manual check -- written, never scored
# ---------------------------------------------------------------------------

def _decisions(n_per=40):
    out = []
    for verdict in ("ENTAILED", "PARTIAL", "ABSENT"):
        for i in range(n_per):
            out.append({"direction": "reconstruction", "dialogue_id": f"d{i}",
                        "reference": f"{verdict} ref {i}", "candidates": [f"mem {i}"],
                        "verdict": verdict})
    return out


def test_sample_is_stratified_and_deterministic():
    a = stratified_sample(_decisions(), n=30)
    b = stratified_sample(_decisions(), n=30)
    assert [d["reference"] for d in a] == [d["reference"] for d in b]   # fixed seed
    assert {d["verdict"] for d in a} == {"ENTAILED", "PARTIAL", "ABSENT"}
    assert len(a) == 30


def test_sample_tops_up_when_a_stratum_is_thin():
    decisions = _decisions(2) + [{"direction": "reconstruction", "dialogue_id": "x",
                                  "reference": f"bulk {i}", "candidates": ["m"],
                                  "verdict": "ABSENT"} for i in range(100)]
    assert len(stratified_sample(decisions, n=50)) == 50


def test_manual_check_file_leaves_the_human_column_empty(tmp_path):
    path = str(tmp_path / "mc.csv")
    n = write_manual_check(path, _decisions(), n=25)
    assert n == 25
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 25
    assert all(r["human_verdict"] == "" for r in rows), \
        "a session must never produce the agreement number this file exists to collect"
    assert all(r["judge_verdict"] in ("ENTAILED", "PARTIAL", "ABSENT") for r in rows)
    assert {"reference", "candidates", "direction", "dialogue_id"} <= set(rows[0])


def test_summarise_ignores_not_applicable():
    rows = [{"recall": 1.0}, {"recall": None}, {"recall": 0.0}]
    out = summarise(rows, ["recall"])
    assert out["mean_recall"] == pytest.approx(0.5) and out["n_recall"] == 2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(IKAT23), reason="iKAT 2023 topics not fetched")
def test_run_intrinsic_cli_axis1(tmp_path):
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps({
        "condition": "B1_full_history", "injection": "rewrite_only", "collection": "ikat23",
        "backend": "echo", "model": "smoke-model",
    }))
    runs_dir = tmp_path / "runs"
    result = subprocess.run(
        [sys.executable, "-m", "scripts.run_intrinsic", "--config", str(config_path),
         "--runs-dir", str(runs_dir), "--manual-check-dir", str(tmp_path / "manual_check"),
         "--cache", str(runs_dir / "cache.jsonl"), "--env", str(tmp_path / "none.env"),
         "--lexical-judge", "--limit", "2", "--manual-check-n", "10"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert "ptkb_reconstruction" in result.stdout
    assert "A HUMAN fills in" in result.stdout

    csvs = list((runs_dir / "intrinsic").glob("*.per_dialogue.csv"))
    assert len(csvs) == 1
    with open(csvs[0], newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 and {"precision", "recall", "f1"} <= set(rows[0])

    summary = json.load(open(runs_dir / "intrinsic_summary.json", encoding="utf-8"))
    row = next(iter(summary.values()))
    assert row["axis"] == "ptkb_reconstruction"
    assert "UNVALIDATED" in row["judge_source"]      # lexical judge is labelled, traceable
    mc = list((tmp_path / "manual_check").glob("*.csv"))
    assert len(mc) == 1
