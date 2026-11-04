"""Format-neutral rendering is a load-bearing claim, so it gets a test rather than a comment.

Paper §3.4 and §4.2 both rest on one property: only an entry's *text* reaches the rewrite
prompt. If a system's native structure -- graph edge types, note tags, similarity scores,
entry ids -- ever leaked into the memory block, the comparison would reward whichever format
the prompt happens to read best, and every cross-system number in the paper would be measuring
prompt affinity instead of memory quality.

`format_memory_block` currently reads `e.text` and nothing else, so the property holds by
construction. These tests exist so that it keeps holding: adding `e.meta` to the rendering to
"give the model more signal" is a natural-looking change that would silently invalidate the
comparison, and it should fail a test rather than pass review.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from harness.llm import LLMBackend
from harness.llmcache import JSONLCache
from harness.memory_base import Dialogue, MemoryEntry, MemorySystem, Turn
from harness.replay import format_memory_block, replay_dialogue


def _entry(entry_id: str, text: str, **meta) -> MemoryEntry:
    return MemoryEntry(entry_id=entry_id, text=text, meta=meta)


def test_only_entry_text_is_rendered():
    """Ids, scores, and every metadata key stay out of the block."""
    entries = [
        _entry("uuid-8f2c", "The user is vegetarian.",
               score=0.93, edge_type="HAS_PREFERENCE", valid_from="2023-01-01"),
        _entry("uuid-11ab", "The user is allergic to peanuts.",
               tags=["health", "diet"], neighbours=["uuid-8f2c"], evolution_step=3),
    ]
    block = format_memory_block(entries)

    assert block == "- The user is vegetarian.\n- The user is allergic to peanuts."
    for leak in ("uuid-8f2c", "uuid-11ab", "0.93", "HAS_PREFERENCE", "valid_from",
                 "tags", "health", "neighbours", "evolution_step", "score"):
        assert leak not in block, f"{leak!r} leaked into the memory block"


def test_rendering_preserves_the_order_the_system_returned():
    """The block must not re-rank. A system's own ordering is part of what is evaluated."""
    entries = [_entry(f"e{i}", f"fact {i}") for i in range(5)]
    lines = format_memory_block(entries).splitlines()
    assert lines == [f"- fact {i}" for i in range(5)]


def test_empty_retrieval_renders_as_the_empty_string():
    """B0's block, and any turn where a system returns nothing, must be exactly empty --
    not a placeholder the prompt could read as a signal."""
    assert format_memory_block([]) == ""


def test_structure_heavy_system_reaches_the_prompt_as_text_only(tmp_path):
    """End-to-end: a system whose value lives in metadata contributes only its text."""

    class _GraphLikeMemory(MemorySystem):
        name = "graph_like"

        def ingest_turn(self, dialogue_id, turn):
            pass

        def query(self, dialogue_id, utterance, k=5):
            return [_entry("n1", "The user cooks at home.",
                           edge="PREFERS", confidence=0.88, supersedes="n0")]

        def reset(self, dialogue_id=None):
            pass

        def export_store(self, dialogue_id):
            return []

    prompts = []

    class _Capturing(LLMBackend):
        def complete(self, prompt):
            prompts.append(prompt)
            return super().complete(prompt)

    backend = _Capturing("echo", "smoke-model", JSONLCache(str(tmp_path / "c.jsonl")))
    dialogue = Dialogue(dialogue_id="d1", turns=[
        Turn(turn_id="d1_1", role="user", text="Any dinner ideas?"),
    ])
    replay_dialogue(dialogue, _GraphLikeMemory(), backend)

    knowledge = prompts[-1].split("User knowledge")[1].split("Conversation:")[0]
    assert "The user cooks at home." in knowledge
    for leak in ("PREFERS", "0.88", "supersedes", "n1", "confidence"):
        assert leak not in knowledge, f"{leak!r} reached the rewriter"
