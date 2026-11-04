import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from harness.config import RunConfig
from harness.llm import LLMBackend
from harness.llmcache import JSONLCache
from harness.memory_base import (Dialogue, FullHistoryMemory, GoldPTKBMemory,
                                 NoPersonalizationMemory, Turn)
from harness.replay import replay_dialogue
from harness.trec import validate_run, write_run
from eval.intrinsic import lexical_judge, score_reconstruction

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "topic_smoke.json")


def load_fixture() -> Dialogue:
    with open(FIXTURE, encoding="utf-8") as f:
        d = json.load(f)
    return Dialogue(
        dialogue_id=d["dialogue_id"],
        turns=[Turn(**t) for t in d["turns"]],
        ptkb=d["ptkb"],
    )


def test_run_id_deterministic_and_sensitive():
    a = RunConfig("S1_mem0", "rewrite_only", "ikat23")
    b = RunConfig("S1_mem0", "rewrite_only", "ikat23")
    c = RunConfig("S1_mem0", "rewrite_only", "ikat23", ablation={"infer": False})
    assert a.run_id == b.run_id
    assert a.run_id != c.run_id
    with pytest.raises(ValueError):
        RunConfig("S9_nope", "rewrite_only", "ikat23")


def test_cache_roundtrip(tmp_path):
    path = str(tmp_path / "cache.jsonl")
    c1 = JSONLCache(path)
    assert c1.get("k") is None
    c1.put("k", "v")
    assert c1.get("k") == "v"
    c2 = JSONLCache(path)  # reload from disk
    assert c2.get("k") == "v"
    assert c2.stats()["entries"] == 1


def test_replay_one_code_path_all_baselines(tmp_path):
    dialogue = load_fixture()
    cache = JSONLCache(str(tmp_path / "cache.jsonl"))
    backend = LLMBackend("echo", "smoke-model", cache)
    n_user = sum(1 for t in dialogue.turns if t.role == "user")

    for mem in (NoPersonalizationMemory(), FullHistoryMemory(), GoldPTKBMemory()):
        results = replay_dialogue(dialogue, mem, backend)
        assert len(results) == n_user, mem.name
        assert all(r.query for r in results), mem.name

    # B0 sees no memory; B2 sees gold PTKB on every turn
    b0 = replay_dialogue(load_fixture(), NoPersonalizationMemory(), backend)
    assert all(r.memory_block == "" for r in b0)
    b2 = replay_dialogue(load_fixture(), GoldPTKBMemory(), backend)
    assert all("vegetarian" in r.memory_block for r in b2)

    # B1 accumulates: first user turn sees nothing, later turns see prior turns only
    b1 = replay_dialogue(load_fixture(), FullHistoryMemory(), backend)
    assert b1[0].memory_block == ""
    assert "meal-prepping" in b1[1].memory_block
    assert "allergy" not in b1[1].memory_block  # current turn never pre-ingested


def test_cache_makes_second_replay_free(tmp_path):
    dialogue = load_fixture()
    cache = JSONLCache(str(tmp_path / "cache.jsonl"))
    backend = LLMBackend("echo", "smoke-model", cache)
    replay_dialogue(dialogue, NoPersonalizationMemory(), backend)
    misses_first = cache.stats()["misses"]
    replay_dialogue(load_fixture(), NoPersonalizationMemory(), backend)
    assert cache.stats()["misses"] == misses_first  # zero new misses


def test_trec_write_validate_append_only(tmp_path):
    path = str(tmp_path / "run.txt")
    write_run(path, {"q1": [("d2", 0.5), ("d1", 0.9)], "q2": [("d3", 1.0)]}, tag="smoke")
    assert validate_run(path, expected_qids={"q1", "q2"}) == []
    assert validate_run(path, expected_qids={"q1", "q2", "q3"}) != []
    with pytest.raises(FileExistsError):
        write_run(path, {"q1": [("d1", 1.0)]}, tag="smoke")


def test_intrinsic_reconstruction_lexical():
    gold = ["I am a vegetarian.", "I am allergic to peanuts."]
    memories = ["User is a vegetarian", "User lives in Taipei"]
    s = score_reconstruction(gold, memories, lexical_judge)
    assert 0.0 < s.recall < 1.0      # one of two gold statements covered
    assert s.n_gold == 2 and s.n_memories == 2
    perfect = score_reconstruction(gold, gold, lexical_judge)
    assert perfect.f1 == 1.0


def test_judge_verdict_parsing_handles_fenced_json():
    """38% of cached judge replies came back markdown-fenced, because the frozen prompt asks
    for JSON without the request setting response_format. Parsing must not depend on the
    keyword fallback, which takes the first verdict word anywhere in the reply."""
    from eval.intrinsic import _parse_verdict

    assert _parse_verdict('{"verdict": "PARTIAL", "note": "x"}') == "PARTIAL"
    assert _parse_verdict('```json\n{"verdict": "ENTAILED"}\n```') == "ENTAILED"
    assert _parse_verdict('```\n{"verdict": "ABSENT"}\n```') == "ABSENT"
    # the failure the keyword scan would get wrong, now handled by the fence-stripped parse
    assert _parse_verdict('```json\n{"verdict": "ABSENT", "note": "not ENTAILED"}\n```') == "ABSENT"
    # unparseable text still falls back rather than raising
    assert _parse_verdict("the answer is PARTIAL") == "PARTIAL"
    assert _parse_verdict("no verdict here") == "ABSENT"


def test_b1_window_returns_only_the_last_w_turns():
    """B1-last-w (follow-up ablation): window=w must keep exactly the w most recent ingested turns,
    and the default must stay unbounded -- the reported B1 must be untouched by the feature."""
    from harness.memory_base import FullHistoryMemory, Turn

    turns = [Turn(turn_id=f"d1_{i}", role="user", text=f"utterance {i}") for i in range(1, 6)]
    full, last2 = FullHistoryMemory(), FullHistoryMemory(window=2)
    for t in turns:
        full.ingest_turn("d1", t)
        last2.ingest_turn("d1", t)
    assert [e.entry_id for e in full.query("d1", "q", k=5)] == [t.turn_id for t in turns]
    assert [e.entry_id for e in last2.query("d1", "q", k=5)] == ["d1_4", "d1_5"]


def test_m1_manual_has_no_memory_system():
    """M1's rewrites are the released manual utterances; asking the registry for a memory
    system must fail loudly, not fall through to some default."""
    import pytest as _pytest

    from harness.registry import make_memory_system

    with _pytest.raises(ValueError, match="make_manual_reference"):
        make_memory_system("M1_manual", {})
