"""The `context` axis is what makes B1_full_history a real information ceiling.

Under the default `history_plus_memory`, `replay.py` puts the verbatim transcript in the
prompt's `{history}` slot for EVERY condition, so B1 -- whose memory *is* that transcript --
adds nothing over B0 and the two tie on 152 of 173 judged turns (paper, Section 3.4). Under
`memory_only` the history slot is empty for every condition, so the memory system is the sole
channel carrying prior turns: B0 sees the bare utterance, B1 sees everything, and S1-S5 sit
between them by exactly how much their abstraction preserved.

These tests pin both halves: that the slot really is empty, and that the two designs can never
be silently mixed in one table or one Holm family.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from harness.config import CONTEXTS, RunConfig
from harness.llm import LLMBackend
from harness.llmcache import JSONLCache
from harness.memory_base import Dialogue, FullHistoryMemory, NoPersonalizationMemory, Turn
from harness.replay import replay_dialogue


def _dialogue():
    return Dialogue(dialogue_id="d1", turns=[
        Turn(turn_id="d1_1", role="user", text="What is throat cancer?"),
        Turn(turn_id="d1_2", role="system", text="A cancer of the pharynx."),
        Turn(turn_id="d1_3", role="user", text="Is it treatable?"),
    ])


class _CapturingBackend(LLMBackend):
    """Records the prompts it was asked to complete, so the slots can be inspected."""

    def __init__(self, tmp_path):
        super().__init__("echo", "smoke-model", JSONLCache(str(tmp_path / "c.jsonl")))
        self.prompts = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return super().complete(prompt)


def test_default_context_puts_the_transcript_in_the_history_slot(tmp_path):
    b = _CapturingBackend(tmp_path)
    replay_dialogue(_dialogue(), NoPersonalizationMemory(), b)
    assert "What is throat cancer?" in b.prompts[-1]      # prior turn reached the prompt


def test_memory_only_empties_the_history_slot_for_every_condition(tmp_path):
    for memory in (NoPersonalizationMemory(), FullHistoryMemory()):
        b = _CapturingBackend(tmp_path / memory.name)
        os.makedirs(tmp_path / memory.name, exist_ok=True)
        replay_dialogue(_dialogue(), memory, b, context="memory_only")
        last = b.prompts[-1]
        conversation = last.split("Conversation:")[1].split("Latest utterance:")[0]
        assert conversation.strip() == "", f"{memory.name} still received history"


def test_memory_only_leaves_b1_as_the_only_condition_with_context(tmp_path):
    """The point of the fix: under memory_only, B1 carries the prior turn and B0 does not."""
    os.makedirs(tmp_path / "b0", exist_ok=True)
    os.makedirs(tmp_path / "b1", exist_ok=True)
    b0 = _CapturingBackend(tmp_path / "b0")
    replay_dialogue(_dialogue(), NoPersonalizationMemory(), b0, context="memory_only")
    b1 = _CapturingBackend(tmp_path / "b1")
    replay_dialogue(_dialogue(), FullHistoryMemory(), b1, context="memory_only")

    assert "throat cancer" not in b0.prompts[-1].split("Latest utterance:")[0]
    assert "throat cancer" in b1.prompts[-1].split("Latest utterance:")[0]
    assert b0.prompts[-1] != b1.prompts[-1]


def test_context_is_part_of_run_identity():
    a = RunConfig("B1_full_history", "rewrite_only", "cast19")
    b = RunConfig("B1_full_history", "rewrite_only", "cast19", context="memory_only")
    assert a.run_id != b.run_id, "the two designs must not share artifacts"
    assert a.context == "history_plus_memory"
    with pytest.raises(ValueError):
        RunConfig("B0_none", "rewrite_only", "cast19", context="sideways")
    assert set(CONTEXTS) == {"history_plus_memory", "memory_only"}


def test_stats_never_mixes_the_two_designs_in_one_cell_or_family(tmp_path):
    """A memory_only run and a history_plus_memory run of the same condition must land in
    different tables and different Holm families -- comparing across designs is meaningless,
    and pooling them would silently average two different experiments."""
    from eval.stats import build_comparisons, cell_name, collect_stats, load_runs
    from tests.test_stats import BASELINE_B0, BASELINE_B1, _write_run

    runs_dir = str(tmp_path / "runs")
    turns = {"31_1": 0.1, "31_2": 0.2, "31_3": 0.3, "32_1": 0.4}
    for ctx in ("history_plus_memory", "memory_only"):
        _write_run(runs_dir, BASELINE_B0, "cast19", turns, context=ctx)
        _write_run(runs_dir, BASELINE_B1, "cast19", {k: v + 0.1 for k, v in turns.items()},
                   context=ctx)

    records = load_runs(runs_dir)
    assert {cell_name(r) for r in records} == {"cast19/rewrite_only",
                                              "cast19/rewrite_only/memory_only"}
    rows = build_comparisons(records)["comparisons"]
    families = {r["family"] for r in rows if r["metric"] == "ndcg_cut_3"}
    assert families == {"main/cast19/rewrite_only/ndcg_cut_3",
                        "main/cast19/rewrite_only/memory_only/ndcg_cut_3"}
    # and no pooled row was invented by treating the two designs as two collections
    assert not [r for r in rows if str(r["collection"]).startswith("pooled(")]

    summary = collect_stats(runs_dir)
    assert set(summary["tables"]) == {"cast19/rewrite_only",
                                      "cast19/rewrite_only/memory_only"}
