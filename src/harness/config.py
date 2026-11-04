"""Run configuration + deterministic run IDs.

A run is uniquely determined by its config; the run ID is a content hash of the
canonical config JSON. New config → new run ID → new artifacts directory. Run
artifacts are append-only (a harness invariant): never overwrite a TREC run file.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

CONDITIONS = (
    "B0_none", "B1_full_history", "B2_gold_ptkb",
    "S1_mem0", "S2_graphiti", "S3_amem", "S4_letta", "S5_gam",
    # M1: the track-released MANUAL rewrites through the same frozen retrieval stack -- the
    # human-rewrite reference every CAsT paper reports and internal review required. It has
    # no memory system and no replay; scripts/make_manual_reference.py writes its rewrites
    # file directly from the released data, and retrieval/eval treat it like any other run.
    "M1_manual",
)
INJECTIONS = ("rewrite_only", "rewrite_plus_rerank")

# Which channels fill the rewrite prompt's context slots. This is the axis that decides
# whether B1_full_history is a real information ceiling or a no-op:
#
#   history_plus_memory  {history} = the verbatim transcript, {memory_block} = what the system
#                        returned. Memory is a SUPPLEMENT to the conversation. B1 is degenerate
#                        here -- its memory IS the transcript, so B1 == B0 plus a duplicate
#                        (measured: 73.5% byte-identical rewrites; paper, Section 3.4).
#   memory_only          {history} = "" for every condition; the memory system is the ONLY
#                        channel carrying prior turns. B0 sees the bare utterance (true floor),
#                        B1 sees everything (true ceiling), and S1-S5 sit between them by
#                        exactly how much their abstraction preserved. This is the design RQ1's
#                        "fraction of the B1 ceiling recovered" actually requires.
#
# Default stays history_plus_memory so existing run_ids and artifacts keep their meaning.
CONTEXTS = ("history_plus_memory", "memory_only")
COLLECTIONS = ("ikat23", "ikat24", "ikat25", "cast19", "cast20", "longmemeval")


@dataclass(frozen=True)
class RunConfig:
    condition: str                      # one of CONDITIONS
    injection: str                      # one of INJECTIONS
    collection: str                     # one of COLLECTIONS
    backend: str = "openai"             # "echo" for smoke
    model: str = "PINNED_IN_DOTENV"     # experiment backend snapshot id
    memory_k: int = 5                   # memory entries retrieved per turn
    ablation: dict[str, bool] = field(default_factory=dict)   # e.g. {"infer": False}
    rewrite_prompt: str = "rewrite_v1"  # prompts/<name>.txt — frozen after Aug 5
    context: str = "history_plus_memory"   # one of CONTEXTS — see below

    def __post_init__(self) -> None:
        if self.condition not in CONDITIONS:
            raise ValueError(f"unknown condition {self.condition!r}")
        if self.context not in CONTEXTS:
            raise ValueError(f"unknown context mode {self.context!r}")
        if self.injection not in INJECTIONS:
            raise ValueError(f"unknown injection {self.injection!r}")
        if self.collection not in COLLECTIONS:
            raise ValueError(f"unknown collection {self.collection!r}")

    def canonical_json(self) -> str:
        d = asdict(self)
        d["ablation"] = dict(sorted(d["ablation"].items()))
        return json.dumps(d, sort_keys=True, separators=(",", ":"))

    @property
    def run_id(self) -> str:
        h = hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()[:12]
        return f"{self.collection}.{self.condition}.{self.injection}.{h}"
