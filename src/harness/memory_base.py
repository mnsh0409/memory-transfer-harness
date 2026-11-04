"""Core abstraction: every memory system — and every baseline — is a MemorySystem.

Design rule (ARCHITECTURE.md): baselines B0/B1/B2 are *degenerate* MemorySystems so
the replay/rewrite/retrieval pipeline has exactly one code path. If downstream code
ever special-cases a baseline, the design is broken.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Turn:
    turn_id: str          # e.g. "9-1-2" (iKAT style) or "cast-77-3"
    role: str             # "user" | "system"
    text: str


@dataclass
class Dialogue:
    dialogue_id: str
    turns: list[Turn]
    ptkb: list[str] = field(default_factory=list)   # gold persona statements (may be empty, e.g. CAsT)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryEntry:
    entry_id: str
    text: str
    meta: dict[str, Any] = field(default_factory=dict)


class MemorySystem(abc.ABC):
    """Contract every adapter implements. Per-dialogue isolation is mandatory:
    state keyed by dialogue_id, never shared across dialogues."""

    name: str = "abstract"
    #: capability flags an ablation config may flip, e.g. {"infer", "evolution", "edge_invalidation"}
    capabilities: frozenset[str] = frozenset()

    @abc.abstractmethod
    def ingest_turn(self, dialogue_id: str, turn: Turn) -> None:
        """Feed one turn (user or system) into memory. Called in strict dialogue order."""

    @abc.abstractmethod
    def query(self, dialogue_id: str, utterance: str, k: int) -> list[MemoryEntry]:
        """Retrieve up to k memory entries relevant to the current utterance.
        Must NOT see the current turn via ingest first — replay calls query before ingest."""

    @abc.abstractmethod
    def reset(self, dialogue_id: str) -> None:
        """Drop all state for this dialogue."""

    def export_store(self, dialogue_id: str) -> list[MemoryEntry]:
        """Full dump of stored memories for intrinsic evaluation. Default: query with empty
        utterance and large k; adapters should override with a true dump."""
        return self.query(dialogue_id, "", k=10_000)


# --------------------------------------------------------------------------------------
# Degenerate baselines
# --------------------------------------------------------------------------------------

class NoPersonalizationMemory(MemorySystem):
    """B0 — floor. Memory always empty; rewrite sees conversation only."""

    name = "B0_none"

    def ingest_turn(self, dialogue_id: str, turn: Turn) -> None:  # noqa: ARG002
        return None

    def query(self, dialogue_id: str, utterance: str, k: int) -> list[MemoryEntry]:  # noqa: ARG002
        return []

    def reset(self, dialogue_id: str) -> None:  # noqa: ARG002
        return None

    def export_store(self, dialogue_id: str) -> list[MemoryEntry]:  # noqa: ARG002
        return []


class FullHistoryMemory(MemorySystem):
    """B1 — the cheap alternative: verbatim prior transcript, no abstraction.

    `window` keeps only the last N ingested turns (None = unbounded). It exists for the
    B1-last-w follow-up ablation: if a short window matches full history on this collection, the
    task is recency-dominated context abstraction and the paper must say so. It is an
    ablation of the reference, never the reference itself."""

    name = "B1_full_history"
    capabilities = frozenset({"window"})

    def __init__(self, window: int | None = None) -> None:
        if window is not None and window < 1:
            raise ValueError("window must be >= 1 or None")
        self._window = window
        self._store: dict[str, list[Turn]] = {}

    def ingest_turn(self, dialogue_id: str, turn: Turn) -> None:
        self._store.setdefault(dialogue_id, []).append(turn)

    def query(self, dialogue_id: str, utterance: str, k: int) -> list[MemoryEntry]:  # noqa: ARG002
        turns = self._store.get(dialogue_id, [])
        if self._window is not None:
            turns = turns[-self._window:]
        return [
            MemoryEntry(entry_id=t.turn_id, text=f"{t.role}: {t.text}", meta={"kind": "raw_turn"})
            for t in turns
        ]  # deliberately ignores k: B1 is "everything, verbatim" (or the last `window` turns)

    def reset(self, dialogue_id: str) -> None:
        self._store.pop(dialogue_id, None)


class GoldPTKBMemory(MemorySystem):
    """B2 — oracle. Gold persona statements, bypassing construction entirely.
    Only meaningful on collections with personas (iKAT); on CAsT it degrades to B0."""

    name = "B2_gold_ptkb"

    def __init__(self) -> None:
        self._ptkb: dict[str, list[str]] = {}

    def register_dialogue(self, dialogue: Dialogue) -> None:
        self._ptkb[dialogue.dialogue_id] = list(dialogue.ptkb)

    def ingest_turn(self, dialogue_id: str, turn: Turn) -> None:  # noqa: ARG002
        return None  # oracle knowledge is fixed; conversation adds nothing

    def query(self, dialogue_id: str, utterance: str, k: int) -> list[MemoryEntry]:  # noqa: ARG002
        stmts = self._ptkb.get(dialogue_id, [])
        return [
            MemoryEntry(entry_id=f"ptkb-{i}", text=s, meta={"kind": "gold_ptkb"})
            for i, s in enumerate(stmts)
        ]

    def reset(self, dialogue_id: str) -> None:
        self._ptkb.pop(dialogue_id, None)
