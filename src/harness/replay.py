"""The replay protocol, one code path for every condition.

For each dialogue:
  for each turn in order:
    if user turn: memory.query() FIRST (memory must not see the current turn),
                  format memory block, rewrite the utterance via the frozen prompt;
    then memory.ingest_turn() for user and system turns alike.

Output: one RewriteResult per user turn — the standalone query that feeds
first-stage retrieval. Retrieval/reranking are separate stages so the same
rewrites serve both injection modes.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

from .llm import LLMBackend
from .memory_base import Dialogue, GoldPTKBMemory, MemoryEntry, MemorySystem

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "prompts")


def load_prompt(name: str) -> str:
    path = os.path.abspath(os.path.join(PROMPTS_DIR, f"{name}.txt"))
    with open(path, encoding="utf-8") as f:
        return f.read()


def format_memory_block(entries: list[MemoryEntry]) -> str:
    """Format-neutral rendering (plan step 5): verbatim entry text, one per line.
    No system-specific structure may leak through here — that would bias the
    comparison toward whichever system's native format the prompt reads best."""
    return "\n".join(f"- {e.text}" for e in entries)


@dataclass(frozen=True)
class RewriteResult:
    dialogue_id: str
    turn_id: str
    utterance: str
    memory_block: str
    query: str


def replay_dialogue(
    dialogue: Dialogue,
    memory: MemorySystem,
    backend: LLMBackend,
    rewrite_prompt_name: str = "rewrite_v1",
    memory_k: int = 5,
    context: str = "history_plus_memory",
) -> list[RewriteResult]:
    """`context` (config.CONTEXTS) decides what fills the prompt's `{history}` slot.
    Under "memory_only" it stays empty for every condition, so the memory system is the only
    channel carrying prior turns -- which is what makes B1 an information ceiling over B0
    instead of a duplicate of it. The frozen prompt is NOT edited either way; only what is
    substituted into its slots changes."""
    template = load_prompt(rewrite_prompt_name)
    if isinstance(memory, GoldPTKBMemory):
        memory.register_dialogue(dialogue)

    results: list[RewriteResult] = []
    history_lines: list[str] = []

    for turn in dialogue.turns:
        if turn.role == "user":
            entries = memory.query(dialogue.dialogue_id, turn.text, k=memory_k)
            mem_block = format_memory_block(entries)
            history_text = "" if context == "memory_only" else "\n".join(history_lines)
            prompt = (
                template
                .replace("{memory_block}", mem_block)
                .replace("{history}", history_text)
                .replace("{utterance}", turn.text)
            )
            query = backend.complete(prompt).strip()
            results.append(
                RewriteResult(
                    dialogue_id=dialogue.dialogue_id,
                    turn_id=turn.turn_id,
                    utterance=turn.text,
                    memory_block=mem_block,
                    query=query,
                )
            )
        memory.ingest_turn(dialogue.dialogue_id, turn)
        history_lines.append(f"{turn.role}: {turn.text}")

    return results


def write_rewrites(path: str, results: list[RewriteResult]) -> None:
    """Cache run_replay's output for run_retrieval to consume (protocol's Fixed CLI Contract:
    'dialogues -> rewrites (cached)'). Append-only, same reasoning as trec.write_run: a script
    re-run against the same config should fail loudly, not silently clobber what a later stage
    may already be reading."""
    if os.path.exists(path):
        raise FileExistsError(
            f"{path} exists — rewrite caches are append-only; new config -> new run_id"
        )
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def read_rewrites(path: str) -> list[RewriteResult]:
    with open(path, encoding="utf-8") as f:
        return [RewriteResult(**json.loads(line)) for line in f if line.strip()]
