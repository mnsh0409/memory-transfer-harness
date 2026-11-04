"""Two Mem0 processes must be able to run at once, and the default must not have moved.

Mem0 pins its Qdrant store to `/tmp/qdrant` and its migrations store to `~/.mem0`, both with
exclusive locks, so a second Mem0 process dies on `PermissionError: [WinError 32] ... .lock`
-- or, on an unluckier interleaving, wipes the store a running process is mid-dialogue on and
lets that run report ordinary-looking metrics for a mechanism that lost its memory. `MEM0_DIR`
is the single knob that relocates all of it.

The load-bearing test here is the *unset* one: several Mem0 runs are already scored and in the
paper, and they must reproduce byte-identically. If the no-MEM0_DIR config ever stops matching
Mem0's own defaults, those runs are no longer reproducible and this test fails.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from adapters.mem0_adapter import DEFAULT_QDRANT_PATH, _default_config

mem0_configs = pytest.importorskip("mem0.configs.base",
                                   reason="mem0 not installed; adapter is import-guarded")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("MEM0_DIR", raising=False)


def test_unset_reproduces_mem0s_own_defaults():
    """The already-scored runs ran on Mem0's defaults. This is what keeps them reproducible."""
    upstream = mem0_configs.MemoryConfig()
    ours = _default_config()
    assert ours["vector_store"]["provider"] == upstream.vector_store.provider
    assert ours["vector_store"]["config"]["path"] == upstream.vector_store.config.path
    assert DEFAULT_QDRANT_PATH == upstream.vector_store.config.path
    # history_db_path is NOT overridden when unset -- mem0's own default must stand
    assert "history_db_path" not in ours


def test_mem0_dir_relocates_both_stores(monkeypatch, tmp_path):
    home = str(tmp_path / "mem0home")
    monkeypatch.setenv("MEM0_DIR", home)
    cfg = _default_config()
    assert cfg["vector_store"]["config"]["path"].startswith(home)
    assert cfg["history_db_path"].startswith(home)
    # and it is no longer the shared location that caused the collision
    assert cfg["vector_store"]["config"]["path"] != DEFAULT_QDRANT_PATH


def test_two_processes_get_disjoint_paths(monkeypatch, tmp_path):
    """The isolation property itself: different MEM0_DIR -> no shared file anywhere."""
    paths = []
    for name in ("proc_a", "proc_b"):
        monkeypatch.setenv("MEM0_DIR", str(tmp_path / name))
        cfg = _default_config()
        paths.append({cfg["vector_store"]["config"]["path"], cfg["history_db_path"]})
    assert not (paths[0] & paths[1]), "two Mem0 processes would still share a file"


def test_blank_mem0_dir_is_treated_as_unset(monkeypatch):
    """An exported-but-empty var is how this would silently half-apply in a shell script --
    the same empty-string trap that OPENAI_BASE_URL sprang three separate times."""
    monkeypatch.setenv("MEM0_DIR", "   ")
    assert _default_config()["vector_store"]["config"]["path"] == DEFAULT_QDRANT_PATH
    assert "history_db_path" not in _default_config()
