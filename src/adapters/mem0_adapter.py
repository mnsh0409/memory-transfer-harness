"""S1 — Mem0 adapter. PIN: mem0ai 0.1.x (pre-v3).

Maps the pre-v3 pipeline (extract -> ADD/UPDATE/DELETE) onto MemorySystem.
`infer` capability flag: ablation sets infer=False (raw storage, no extraction).
Import-guarded: the harness and tests run without mem0 installed.
"""
from __future__ import annotations

import os

from harness.memory_base import MemoryEntry, MemorySystem, Turn

try:
    from mem0 import Memory  # type: ignore
    HAVE_MEM0 = True
except ImportError:
    HAVE_MEM0 = False


#: Mem0's default Qdrant location. Hardcoded upstream, hence this whole function.
DEFAULT_QDRANT_PATH = "/tmp/qdrant"


def _default_config() -> dict:
    """Make a Mem0 process isolatable, so two of them can run at once.

    **Two Mem0 processes cannot share a machine by default.** Zero-arg `Memory()` puts Qdrant
    at a fixed `/tmp/qdrant` and takes an exclusive lock, so the second process dies with
    `PermissionError: [WinError 32] ... /tmp/qdrant/.lock` while resetting a store the first is
    using. Relocating only that path is not enough -- Mem0 keeps a second store under
    `~/.mem0/migrations_qdrant`, which collides the same way.

    A crash is the *benign* outcome. The dangerous interleaving is a starting process wiping
    the store a running process is mid-dialogue on: that run completes and reports
    ordinary-looking metrics for a mechanism that silently lost its memory, which is precisely
    the failure class this paper is about.

    **The single knob is `MEM0_DIR`**, which upstream already honours for the migrations store.
    When it is set we put the vector store and the history DB underneath it too, so one
    variable isolates a whole process. It must be exported **before** `import mem0`: upstream
    reads it into a module-level constant at import time, so setting it from Python after the
    import silently does nothing. Set it in the job script, not in the adapter.

    Unset, every path is byte-identical to Mem0's own defaults, so runs already scored
    reproduce exactly. Storage location has no effect on what is stored or retrieved, and
    per-dialogue isolation is enforced by `reset()` either way.

    Audited when this was found (Aug 23): the four Mem0 replays scored to that point started
    Aug 22 15:00 and Aug 23 00:22 / 02:37 / 07:42 -- none overlapped, so no reported number is
    affected."""
    mem0_dir = os.environ.get("MEM0_DIR", "").strip()
    if not mem0_dir:
        return {"vector_store": {"provider": "qdrant",
                                 "config": {"path": DEFAULT_QDRANT_PATH}}}
    return {
        "vector_store": {"provider": "qdrant",
                         "config": {"path": os.path.join(mem0_dir, "qdrant")}},
        "history_db_path": os.path.join(mem0_dir, "history.db"),
    }


class Mem0Memory(MemorySystem):
    name = "S1_mem0"
    capabilities = frozenset({"infer"})

    def __init__(self, config: dict | None = None, infer: bool = True,
                 extraction_prompt: str | None = None) -> None:
        """`extraction_prompt` names a versioned file in prompts/ whose text replaces Mem0's
        shipped fact-extraction prompt via the library's own `custom_fact_extraction_prompt`
        config hook — a configuration the library exposes, not a library edit. It exists for
        the follow-up ablation internal review proposed: shipped Mem0 extracts *personal* facts
        and therefore stores nothing from information-seeking dialogue (§6.2 of the paper);
        this arm asks whether a topical extraction target closes that gap within the same
        pipeline. It is NOT part of the frozen grid and appears in no submitted number."""
        if not HAVE_MEM0:
            raise RuntimeError("pip install 'mem0ai<0.2' (pre-v3 pin — see ARCHITECTURE.md)")
        cfg = dict(config) if config else _default_config()
        if extraction_prompt:
            prompt_path = os.path.join(os.path.dirname(__file__), "..", "..", "prompts",
                                       f"{extraction_prompt}.txt")
            with open(os.path.abspath(prompt_path), encoding="utf-8") as f:
                cfg["custom_fact_extraction_prompt"] = f.read()
        self._m = Memory.from_config(cfg)
        self._infer = infer

    def ingest_turn(self, dialogue_id: str, turn: Turn) -> None:
        self._m.add(
            [{"role": "user" if turn.role == "user" else "assistant", "content": turn.text}],
            user_id=dialogue_id,          # per-dialogue isolation (ARCHITECTURE.md)
            infer=self._infer,
        )

    def query(self, dialogue_id: str, utterance: str, k: int) -> list[MemoryEntry]:
        res = self._m.search(utterance, user_id=dialogue_id, limit=k)
        hits = res.get("results", res) if isinstance(res, dict) else res
        return [
            MemoryEntry(entry_id=str(h.get("id", i)), text=h.get("memory", ""),
                        meta={"score": h.get("score")})
            for i, h in enumerate(hits)
        ]

    def reset(self, dialogue_id: str) -> None:
        self._m.delete_all(user_id=dialogue_id)

    def export_store(self, dialogue_id: str) -> list[MemoryEntry]:
        res = self._m.get_all(user_id=dialogue_id)
        items = res.get("results", res) if isinstance(res, dict) else res
        return [
            MemoryEntry(entry_id=str(h.get("id", i)), text=h.get("memory", ""), meta={})
            for i, h in enumerate(items)
        ]
