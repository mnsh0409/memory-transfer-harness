"""S5 — GAM adapter (General Agentic Memory: compile-at-query deep research over a memory tree).

Upstream: https://github.com/VectorSpaceLab/general-agentic-memory, package `gam`, commit
pinned in requirements.lock. Import-guarded like the other adapters.

GAM is the odd one out in this grid and the mapping deserves reading before you trust a GAM
number:

* **It is not a store of retrievable entries.** GAM keeps raw content in an agent file system
  (chunks + LLM-written memory/TLDR headers, organised into a taxonomy tree) and does its
  work at *query* time: `request()` runs a multi-round ReAct exploration and returns a single
  synthesised answer, not a ranked list of memories. So `query(..., k)` returns **one**
  MemoryEntry and **k is inapplicable** — noted here because the frozen scope ablation 4 sweeps
  k ∈ {3,5,10} on the best system, and that sweep is a no-op if the best system is GAM.
* **What we ask it is an adapter decision, and it is measurement-affecting.** Handing GAM the
  raw user utterance would have it *answer the search question* ("which proteins are safe
  given my allergy?" → "tofu and lentils"), which is retrieval work no other condition does —
  its memory block would carry answer content rather than user knowledge, and any nDCG gain
  would be unattributable. `USER_KNOWLEDGE_FRAMING` therefore asks GAM for what it knows
  about *the user* that bears on the utterance, which is the comparable question — the same
  thing Mem0/A-Mem/Graphiti return from their stores, and what "memory as an automatically
  built PTKB" means in the plan. The alternative is one constant away; flip it before a
  scored run, never after.
* **Cost.** Ingest is LLM chunking + memory generation per turn; each query is up to
  `max_iterations` (upstream default 10) exploration rounds. GAM is by a wide margin the most
  expensive condition in the grid. Everything still goes through the harness cache, so
  re-runs are free, but the first pass is not.
* **Temperature.** Upstream defaults to 0.3 and builds its own OpenAI client. The harness
  requires temp 0 through `harness/llm.py`, so this adapter injects `HarnessGenerator` for
  both the memory-building and chat agents. Prompts are passed through unchanged.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from harness.memory_base import MemoryEntry, MemorySystem, Turn

try:
    from gam import Workflow
    from gam.generators.base import BaseGenerator
    HAVE_GAM = True
except Exception:  # upstream pulls tiktoken/pydantic/docker at import time
    HAVE_GAM = False
    BaseGenerator = object  # type: ignore


# ---------------------------------------------------------------------------
# GAM is POSIX-only. This is why the first GAM run stored nothing.
# ---------------------------------------------------------------------------

def _posix_shell_patch_applied() -> bool:
    """GAM edits its knowledge base by shelling out **Unix** commands, so on Windows it
    silently stores nothing. This routes those commands through bash.

    `LocalWorkspace.run` executes with `subprocess.run(code, shell=True)`, which on Windows
    resolves to `cmd.exe`. GAM's `TextGAMAgent` builds directories with `mkdir -p "..."` and
    writes every file with a **bash heredoc**, `cat > "..." << 'EOF_FILE_CONTENT'`. Measured
    on this box:

        cmd.exe   mkdir_rc=0  write_rc=1  tree=['-p', 'sub']     <- stray dir, NO file
        bash -c   mkdir_rc=0  write_rc=0  tree=['sub','sub/f.md']

    `mkdir -p` half-works — cmd's own `mkdir` treats `-p` as another directory to create and
    still exits 0, so GAM's exit-code check passes and a junk `-p` directory appears in every
    store. The heredoc then fails, no file is ever written, and `add()` returns
    `BatchProcessingResult(success=True, errors=[], created_files=[])`. **GAM reports success
    having stored nothing** — the same silent-failure class as A-Mem's evolution and
    Graphiti's dropped edges, and the reason the first GAM run produced a run file identical
    to B0 after 5,379 calls and 3.35M tokens.

    This changes only the *shell GAM's own commands run in* — not its logic, its prompts, or
    what it decides to store. It is the same kind of substitution as injecting the harness LLM
    backend: transport, not behaviour. No-op on POSIX, where `shell=True` is already `/bin/sh`.
    """
    if os.name != "nt" or not HAVE_GAM:
        return False
    try:
        from gam.workspaces.local_workspace import LocalWorkspace
    except Exception:  # pragma: no cover - upstream layout change
        return False
    if getattr(LocalWorkspace, "_harness_posix_patch", False):
        return True

    bash = shutil.which("bash")
    if not bash:  # pragma: no cover - machine-dependent
        print("WARNING: GAM needs a POSIX shell and no bash is on PATH. It will store "
              "NOTHING and report success. Do not report a GAM number from this box.",
              file=sys.stderr)
        return False

    def run(self, code, timeout=120, workdir=None):
        exec_workdir = self.root_path if workdir is None else Path(workdir).resolve()
        if not Path(exec_workdir).exists():
            return f"Error: Working directory {exec_workdir} does not exist", "-1"
        try:
            result = subprocess.run([bash, "-c", code], cwd=str(exec_workdir),
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    timeout=timeout)
            return result.stdout.decode("utf-8", "replace"), str(result.returncode)
        except subprocess.TimeoutExpired:
            return f"Error: command timed out after {timeout}s", "-1"
        except Exception as e:  # noqa: BLE001
            return f"Error: {e}", "-1"

    LocalWorkspace.run = run
    LocalWorkspace._harness_posix_patch = True
    return True


POSIX_SHELL_PATCHED = _posix_shell_patch_applied()

#: What GAM is asked at query time. See the module docstring: asking for user knowledge rather
#: than for the answer is what keeps GAM comparable to the other conditions.
def _strip_reasoning(text: str) -> str:
    """Remove chain-of-thought scaffolding from GAM's compiled answer.

    GAM's researcher agent is a ReAct-style loop and its replies routinely carry `<think>`
    blocks -- 37% of its cached replies do. That is internal and expected, but the *final*
    compiled answer sometimes arrives still wrapped in it, and whatever this function returns
    becomes the memory block the frozen rewrite prompt reads. Observed directly: a probe on a
    built store returned `<think> I have not yet gathered specific information ...` as the
    entry text, which would have entered the prompt as if it were user knowledge.

    Strips complete `<think>...</think>` spans, and an unterminated trailing `<think>` (a
    truncated reasoning tail). If nothing survives, the caller treats it as an empty compile,
    which is the honest reading: the system produced reasoning but no answer."""
    import re

    text = re.sub(r"(?is)<think>.*?</think>", " ", text)
    text = re.sub(r"(?is)<think>.*$", " ", text)
    text = re.sub(r"(?is)</?think>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


USER_KNOWLEDGE_FRAMING = (
    "What do we know about this user that is relevant to their request: {utterance}\n"
    "Answer with the user's own facts, preferences and constraints only. "
    "Do not answer the request itself."
)


class HarnessGenerator(BaseGenerator):  # type: ignore[misc]
    """GAM `BaseGenerator` backed by `harness.llm.LLMBackend` (temp 0, cached).

    GAM calls `generate_single(prompt=..., schema=...)` / `generate_batch(...)` and expects
    `{"text": str, "json": dict|None, "response": dict}`. We hand back the harness's text and
    parse JSON best-effort (json_repair when available, since GAM itself depends on it and its
    prompts ask for JSON); a parse failure yields `json=None`, which upstream already handles
    as "no structured result"."""

    def __init__(self, backend) -> None:  # deliberately does not call super().__init__
        self._backend = backend
        self.calls = 0

    # -- helpers -------------------------------------------------------
    @staticmethod
    def _as_prompt(prompt, messages) -> str:
        if prompt is not None:
            return prompt
        if not messages:
            return ""
        return "\n\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)

    @staticmethod
    def _parse_json(text: str):
        try:
            return json.loads(text)
        except Exception:
            pass
        try:
            from json_repair import repair_json  # GAM's own dependency
            out = json.loads(repair_json(text))
            return out if isinstance(out, (dict, list)) else None
        except Exception:
            return None

    def _one(self, prompt: str) -> dict:
        self.calls += 1
        text = self._backend.complete(prompt)
        return {"text": text, "json": self._parse_json(text), "response": {}}

    # -- BaseGenerator interface --------------------------------------
    def generate_single(self, prompt=None, messages=None, schema=None,
                        extra_params=None) -> dict:   # noqa: ARG002
        return self._one(self._as_prompt(prompt, messages))

    def generate_batch(self, prompts=None, messages_list=None, schema=None,
                       extra_params=None) -> list[dict]:   # noqa: ARG002
        if prompts:
            return [self._one(p) for p in prompts]
        return [self._one(self._as_prompt(None, m)) for m in (messages_list or [])]


class GAMMemory(MemorySystem):
    name = "S5_gam"
    capabilities = frozenset()          # the frozen scope lists no GAM ablation

    def __init__(self, backend=None, root_dir: str = None, verbose: bool = False,
                 framing: str = USER_KNOWLEDGE_FRAMING, max_iterations: int = 10) -> None:
        if not HAVE_GAM:
            raise RuntimeError(
                "GAM not installed. pip install --no-build-isolation -e <clone of "
                "VectorSpaceLab/general-agentic-memory> at the commit in requirements.lock."
            )
        if backend is None:
            raise ValueError(
                "GAM calls an LLM for both ingest and query; pass the harness LLMBackend so "
                "calls are cached and temp-0 (ARCHITECTURE.md). GAM has no offline mode."
            )
        self._backend = backend
        self._generator = HarnessGenerator(backend)
        self._root = Path(root_dir or Path("runs") / "gam_stores").resolve()
        self._verbose = verbose
        self._framing = framing
        self._max_iterations = max_iterations
        self._dialogue_id: str | None = None
        self._wf = None

    # ------------------------------------------------------------------
    def _dir_for(self, dialogue_id: str) -> Path:
        """One GAM directory per dialogue — this *is* the per-dialogue isolation (ARCHITECTURE.md).
        Dialogue ids contain '_' or '-' only (see loaders), so they are safe path segments."""
        return self._root / dialogue_id.replace("/", "_")

    def _wf_for(self, dialogue_id: str):
        if self._wf is not None and self._dialogue_id == dialogue_id:
            return self._wf
        self._wf = Workflow(
            "text",
            gam_dir=str(self._dir_for(dialogue_id)),
            generator=self._generator,          # memory-building agent
            chat_generator=self._generator,     # Q&A agent -- same backend, same cache
            verbose=self._verbose,
            max_iterations=self._max_iterations,
        )
        self._dialogue_id = dialogue_id
        return self._wf

    # ------------------------------------------------------------------
    def ingest_turn(self, dialogue_id: str, turn: Turn) -> None:
        wf = self._wf_for(dialogue_id)
        # Same `role: text` convention as B1 and the other adapters, so no condition gets a
        # format advantage in the frozen rewrite prompt (ARCHITECTURE.md point 5).
        wf.add(content=f"{turn.role}: {turn.text}")

    def query(self, dialogue_id: str, utterance: str, k: int) -> list[MemoryEntry]:  # noqa: ARG002
        """Compile-at-query: one synthesised entry. `k` is inapplicable (module docstring)."""
        wf = self._wf_for(dialogue_id)
        if self._is_empty(wf):
            return []                      # nothing ingested yet -> no memory, like every system
        result = wf.request(self._framing.format(utterance=utterance))
        answer = _strip_reasoning((getattr(result, "answer", "") or "").strip())
        if not answer:
            return []
        return [MemoryEntry(
            entry_id=f"{dialogue_id}:compiled",
            text=answer,
            meta={"kind": "compiled_at_query",
                  "confidence": getattr(result, "confidence", None),
                  "sources": list(getattr(result, "sources", []) or []),
                  "k_inapplicable": True},
        )]

    @staticmethod
    def _is_empty(wf) -> bool:
        try:
            return bool(wf.tree.is_empty)
        except Exception:
            return False

    def reset(self, dialogue_id: str) -> None:
        """Delete this dialogue's GAM directory. GAM persists to disk, so a soft reset would
        leak the previous dialogue's memory into the next one."""
        self._wf = None
        self._dialogue_id = None
        shutil.rmtree(self._dir_for(dialogue_id), ignore_errors=True)

    def export_store(self, dialogue_id: str) -> list[MemoryEntry]:
        """TRUE DUMP: every file node in the GAM tree, with its LLM-written summary. Reads the
        tree directly rather than issuing a `request()` — a request would return one
        synthesised answer, which is emphatically not a dump of what was stored."""
        wf = self._wf_for(dialogue_id)
        try:
            tree = wf.reload_tree()
        except Exception:
            tree = wf.tree
        root = getattr(tree, "root", None)
        if root is None:
            return []
        out = []
        for node in root.get_all_files():
            text = (getattr(node, "content", "") or "").strip()
            summary = (getattr(node, "summary", "") or "").strip()
            if not text and not summary:
                continue
            out.append(MemoryEntry(
                entry_id=str(getattr(node, "name", len(out))),
                text=text or summary,
                meta={"summary": summary, "kind": "gam_node"},
            ))
        return out
