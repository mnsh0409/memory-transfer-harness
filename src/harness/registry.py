"""Collection -> loader and condition -> MemorySystem dispatch, shared by every scripts/run_*.py
entry point so that mapping lives in exactly one place (ARCHITECTURE.md: one code path).

Data file layout (what scripts/fetch_free_data.sh is expected to produce under data/ — not
written yet, see the task list): one topic JSON per collection, at the paths in
COLLECTION_PATHS below. If fetch_free_data.sh ends up naming things differently, change the
paths here, not the loaders themselves.
"""
from __future__ import annotations

import os

from adapters.amem_adapter import AMemMemory
from adapters.gam_adapter import GAMMemory
from adapters.graphiti_adapter import GraphitiMemory
from adapters.mem0_adapter import Mem0Memory

from .loaders.cast import load_cast_topics
from .loaders.ikat import load_ikat_topics
from .memory_base import (Dialogue, FullHistoryMemory, GoldPTKBMemory,
                           MemorySystem, NoPersonalizationMemory)

# collection name (config.py's COLLECTIONS) -> (loader function, relative path under data_dir)
COLLECTION_PATHS: dict[str, tuple[callable, str]] = {
    "cast19": (load_cast_topics, os.path.join("cast", "cast19_topics.json")),
    "cast20": (load_cast_topics, os.path.join("cast", "cast20_topics.json")),
    "ikat23": (load_ikat_topics, os.path.join("ikat", "ikat23_topics.json")),
    "ikat24": (load_ikat_topics, os.path.join("ikat", "ikat24_topics.json")),
    "ikat25": (load_ikat_topics, os.path.join("ikat", "ikat25_topics.json")),
    # longmemeval is session-shaped, not Dialogue-shaped, and is intrinsic-only --
    # scripts/run_intrinsic.py owns it directly rather than going through this loader registry.
}

# Adapters not yet built. Kept as an explicit map, not a bare
# NotImplementedError, so "what's left" is readable straight out of this file.
_NOT_YET_BUILT = {
    "S4_letta": "Letta adapter not built yet (dropped under a pre-agreed timebox; paper, Section 4)",
}


def load_dialogues(collection: str, data_dir: str = "data") -> list[Dialogue]:
    if collection not in COLLECTION_PATHS:
        known = ", ".join(sorted(COLLECTION_PATHS))
        raise ValueError(
            f"no loader registered for collection {collection!r}. Known: {known}. "
            "(longmemeval is intrinsic-only -- see scripts/run_intrinsic.py, not this registry.)"
        )
    loader, rel_path = COLLECTION_PATHS[collection]
    path = os.path.join(data_dir, rel_path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- run scripts/fetch_free_data.sh first: the qrels and "
            "topics files must be present under data/"
        )
    return loader(path)


def make_memory_system(condition: str, ablation: dict[str, bool] | None = None,
                       backend=None) -> MemorySystem:
    """condition -> MemorySystem.

    `backend` is the harness LLMBackend (harness/llm.py). Adapters whose upstream library
    would otherwise open its own uncached, non-temp-0 API connection take it and route
    through it instead (ARCHITECTURE.md: "a raw API call anywhere else in experiment code is a
    bug"). Adapters that don't need it ignore it."""
    ablation = ablation or {}
    if condition == "B0_none":
        return NoPersonalizationMemory()
    if condition == "B1_full_history":
        # `window` (B1-last-w) is a follow-up ablation, not part of the frozen grid.
        return FullHistoryMemory(window=ablation.get("window"))
    if condition == "B2_gold_ptkb":
        return GoldPTKBMemory()
    if condition == "M1_manual":
        raise ValueError(
            "M1_manual has no memory system and no replay -- its rewrites come verbatim from "
            "the track-released manual utterances via scripts/make_manual_reference.py; only "
            "run_retrieval and run_eval apply to it."
        )
    if condition == "S1_mem0":
        return Mem0Memory(infer=ablation.get("infer", True),
                          extraction_prompt=ablation.get("extraction_prompt"))
    if condition == "S3_amem":
        # the frozen scope ablation 2 flips `evolution`. With evolution off A-Mem makes no LLM
        # calls at all, so the backend is genuinely optional there (see amem_adapter).
        return AMemMemory(evolution=ablation.get("evolution", True), backend=backend)
    if condition == "S2_graphiti":
        # the frozen scope ablation 3 flips `edge_invalidation`. Embedded Kuzu by default (no Docker
        # on the evaluation machine); pass uri/user/password for a real Neo4j.
        return GraphitiMemory(backend=backend,
                              edge_invalidation=ablation.get("edge_invalidation", True))
    if condition == "S5_gam":
        # GAM is LLM-driven on BOTH ingest and query and has no offline mode, so the backend
        # is mandatory. the frozen scope lists no GAM ablation.
        return GAMMemory(backend=backend)
    if condition in _NOT_YET_BUILT:
        raise NotImplementedError(f"{condition}: {_NOT_YET_BUILT[condition]}")
    raise ValueError(f"unknown condition {condition!r}")
