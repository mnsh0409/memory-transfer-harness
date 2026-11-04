"""S3 — A-Mem adapter (agentic Zettelkasten: notes + LLM-driven evolution of neighbours).

Upstream: https://github.com/agiresearch/A-mem, package `agentic_memory`, pinned commit in
requirements.lock. Import-guarded like the other adapters: the harness and its tests run
without A-Mem installed.

Four things about upstream that this adapter has to work around, each verified by reading the
installed source rather than the README:

1. **Per-dialogue isolation is not a feature upstream has.** `AgenticMemorySystem` keeps its
   notes in a single process-wide ChromaDB collection literally named "memories", and its
   constructor calls `client.reset()` on the shared in-process client. So the only safe
   pattern is exactly one live instance at a time, rebuilt whenever the dialogue changes —
   which is what `reset()`/`_system_for()` below do. Two instances alive at once would share
   (and wipe) each other's collection.
2. **The LLM is used only by the evolution step.** `analyze_content()` exists but nothing
   calls it (grep the package), so notes are stored with their raw text and default
   context/tags. `add_note` → `process_memory` → LLM, and `process_memory` returns early
   without an LLM call while the store is empty. Consequence: with `evolution=False` this
   adapter makes **zero** LLM calls, which is what makes the frozen scope ablation-2 arm cheap
   and offline-testable.
3. **Upstream's OpenAI controller calls the API directly at temperature 0.7** and builds its
   own client from `OPENAI_API_KEY`. That breaks two harness invariants at once (every call
   through `harness.llm` + the cache; temp 0 everywhere). This adapter therefore injects a
   thin controller that delegates to the harness `LLMBackend` — **transport only. The prompt
   A-Mem sends is passed through byte-for-byte**; we evaluate the library as shipped, we just
   refuse to let it open its own untracked, non-deterministic connection.
4. **There is no evolution on/off switch.** the frozen scope ablation 2 needs one, so
   `evolution=False` overrides `process_memory` to the identity decision (`False, note`) —
   the documented no-evolution behaviour of the same code path, not a reimplementation and
   not a prompt edit.
"""
from __future__ import annotations

import json

from harness.memory_base import MemoryEntry, MemorySystem, Turn

try:
    from agentic_memory.memory_system import AgenticMemorySystem
    HAVE_AMEM = True
except Exception:  # broad: upstream imports chromadb/nltk/litellm at module import
    HAVE_AMEM = False

EMBED_MODEL = "all-MiniLM-L6-v2"   # upstream default; local sentence-transformers, no API
EVO_THRESHOLD = 100                # upstream default (consolidation cadence)


class HarnessLLMController:
    """Adapts `harness.llm.LLMBackend` to the `get_completion` interface A-Mem calls.

    A-Mem invokes `self.llm_controller.llm.get_completion(prompt, response_format=..., ...)`
    and expects a JSON string back. Everything here goes through the harness cache at
    temperature 0.

    `response_format` is FORWARDED. It used to be accepted and ignored, on the reasoning that
    A-Mem "falls back to no evolution if it cannot parse", judged conservative at the time. It
    was not: without the structured-output spec the model replies in prose, A-Mem's
    `json.loads` raises on every note, upstream swallows it, and the run completes normally
    with evolution having never happened -- 429 failures in 479 turns, which made the
    `evolution` flag a no-op and the frozen scope ablation 2 an off-vs-off non-experiment. A silent
    fallback that disables the mechanism under test is not conservative."""

    def __init__(self, backend) -> None:
        self._backend = backend
        self.calls = 0

    def get_completion(self, prompt: str, response_format: dict | None = None,
                       temperature: float | None = None) -> str:   # noqa: ARG002
        self.calls += 1
        return self._backend.complete(prompt, response_format=response_format)


class AMemMemory(MemorySystem):
    name = "S3_amem"
    capabilities = frozenset({"evolution"})

    def __init__(self, evolution: bool = True, backend=None, llm_model: str = "harness",
                 embed_model: str = EMBED_MODEL) -> None:
        if not HAVE_AMEM:
            raise RuntimeError(
                "A-Mem not installed. pip install -e <clone of agiresearch/A-mem> at the "
                "commit in requirements.lock (needs chromadb + sentence-transformers)."
            )
        if evolution and backend is None:
            raise ValueError(
                "A-Mem's evolution step calls an LLM; pass the harness LLMBackend so the call "
                "is cached and temp-0 (ARCHITECTURE.md), or construct with evolution=False."
            )
        self._evolution = evolution
        self._backend = backend
        self._llm_model = llm_model
        self._embed_model = embed_model
        self._dialogue_id: str | None = None
        self._system = None

    # ------------------------------------------------------------------
    def _system_for(self, dialogue_id: str):
        """Exactly one live AgenticMemorySystem, rebuilt when the dialogue changes.
        Rebuilding is what isolates dialogues (see module docstring, point 1)."""
        if self._system is not None and self._dialogue_id == dialogue_id:
            return self._system
        self._system = self._build()
        self._dialogue_id = dialogue_id
        return self._system

    def _build(self):
        system = AgenticMemorySystem(
            model_name=self._embed_model,
            llm_backend="openai",
            llm_model=self._llm_model,
            evo_threshold=EVO_THRESHOLD,
            # placeholder: upstream's controller demands a key at construction, but we replace
            # the controller below, so no OpenAI client of theirs is ever used.
            api_key="unused-harness-routes-llm-calls-through-harness.llm",
        )
        if self._backend is not None:
            system.llm_controller.llm = HarnessLLMController(self._backend)
        if not self._evolution:
            # the frozen scope ablation 2: no neighbour rewriting, no evolution LLM call.
            system.process_memory = lambda note: (False, note)
        return system

    # ------------------------------------------------------------------
    def ingest_turn(self, dialogue_id: str, turn: Turn) -> None:
        system = self._system_for(dialogue_id)
        # Speaker is kept in the note text: the note is the unit A-Mem reasons over, and a
        # note that drops "who said this" is a different (lossier) memory than the one the
        # library is designed to hold. Same `role: text` convention B1 renders with, so no
        # condition gets a format advantage in the rewrite prompt (ARCHITECTURE.md point 5).
        system.add_note(content=f"{turn.role}: {turn.text}")

    def query(self, dialogue_id: str, utterance: str, k: int) -> list[MemoryEntry]:
        system = self._system_for(dialogue_id)
        hits = system.search_agentic(utterance, k=k) or []
        return [
            MemoryEntry(
                entry_id=str(h.get("id", i)),
                text=h.get("content", ""),
                meta={"score": h.get("score"), "is_neighbor": h.get("is_neighbor", False),
                      "context": h.get("context")},
            )
            for i, h in enumerate(hits[:k])
        ]

    def reset(self, dialogue_id: str) -> None:
        """Drop everything for this dialogue. Rebuilds the underlying system, which is the
        only reliable way to clear upstream's process-wide collection."""
        self._system = None
        self._dialogue_id = None

    def export_store(self, dialogue_id: str) -> list[MemoryEntry]:
        """TRUE DUMP: every note held for this dialogue, read
        straight off `AgenticMemorySystem.memories`, never through the retriever — a search
        would silently cap the intrinsic axis at whatever k it was given."""
        system = self._system_for(dialogue_id)
        out = []
        for note_id, note in system.memories.items():
            meta = {"context": getattr(note, "context", None),
                    "tags": getattr(note, "tags", None),
                    "keywords": getattr(note, "keywords", None),
                    "links": getattr(note, "links", None),
                    "evolved": bool(getattr(note, "evolution_history", None))}
            out.append(MemoryEntry(entry_id=str(note_id), text=note.content,
                                   meta=json.loads(json.dumps(meta, default=str))))
        return out
