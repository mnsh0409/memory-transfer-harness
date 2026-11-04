"""Tests for the small pieces scripts/run_replay.py and scripts/run_retrieval.py are built on:
.env loading, manifest read/write, rewrite-cache read/write, and the collection/condition
registry. All offline -- no real API keys, mem0, or pyserini needed to exercise the plumbing."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from adapters.mem0_adapter import HAVE_MEM0
from harness.envfile import load_dotenv
from harness.manifest import read_manifest, write_manifest
from harness.memory_base import (FullHistoryMemory, GoldPTKBMemory,
                                  NoPersonalizationMemory)
from harness.registry import load_dialogues, make_memory_system
from harness.replay import RewriteResult, read_rewrites, write_rewrites

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


# ---------------------------------------------------------------------------
# envfile
# ---------------------------------------------------------------------------

def test_load_dotenv_parses_and_skips_comments_and_blanks(tmp_path, monkeypatch):
    monkeypatch.delenv("SOME_TEST_KEY", raising=False)
    envfile = tmp_path / ".env"
    envfile.write_text('# comment\n\nSOME_TEST_KEY=abc123\nQUOTED="with spaces"\n')
    set_vars = load_dotenv(str(envfile))
    assert set_vars == {"SOME_TEST_KEY": "abc123", "QUOTED": "with spaces"}
    assert os.environ["SOME_TEST_KEY"] == "abc123"


def test_load_dotenv_treats_an_empty_value_as_unset(tmp_path, monkeypatch):
    """`OPENAI_BASE_URL=` (the natural way to write "use the default endpoint") must not be
    exported as "". llm.py maps "" to None, but the openai SDK then re-reads the same variable
    itself and uses "" as the base URL, failing every call with APIConnectionError. Diagnosed
    on the real .env before the gate run."""
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("SOME_TEST_KEY", raising=False)
    envfile = tmp_path / ".env"
    envfile.write_text("OPENAI_BASE_URL=\nSOME_TEST_KEY=real\n")
    set_vars = load_dotenv(str(envfile))
    assert set_vars == {"SOME_TEST_KEY": "real"}
    assert "OPENAI_BASE_URL" not in os.environ


def test_load_dotenv_overrides_the_pyserini_placeholder_key(tmp_path, monkeypatch):
    """harness.retrieval exports a dummy OPENAI_API_KEY at import time so pyserini can import;
    scripts import harness.runners (and therefore retrieval) BEFORE main() loads .env, so under
    plain setdefault semantics the dummy shadows the real key and every experiment call 401s
    with 'Incorrect API key provided: unused-h***'. This killed a gate run."""
    from harness.envfile import PYSERINI_API_KEY_PLACEHOLDER
    monkeypatch.setenv("OPENAI_API_KEY", PYSERINI_API_KEY_PLACEHOLDER)
    envfile = tmp_path / ".env"
    envfile.write_text("OPENAI_API_KEY=sk-the-real-key\n")
    load_dotenv(str(envfile))
    assert os.environ["OPENAI_API_KEY"] == "sk-the-real-key"


def test_load_dotenv_import_order_does_not_decide_the_key(tmp_path, monkeypatch):
    """The end-to-end shape of the bug: import retrieval first (as runners does), then load
    .env, and the real key must still win."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import importlib

    import harness.retrieval as retrieval
    importlib.reload(retrieval)                      # sets the placeholder, as at script import
    envfile = tmp_path / ".env"
    envfile.write_text("OPENAI_API_KEY=sk-real\nEXPERIMENT_MODEL=gpt-4o-mini-2024-07-18\n")
    load_dotenv(str(envfile))
    assert os.environ["OPENAI_API_KEY"] == "sk-real"


def test_load_dotenv_empty_value_cannot_shadow_the_shell(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.example/v1")
    envfile = tmp_path / ".env"
    envfile.write_text("OPENAI_BASE_URL=\n")
    load_dotenv(str(envfile))
    assert os.environ["OPENAI_BASE_URL"] == "https://proxy.example/v1"


def test_load_dotenv_never_overrides_existing_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SOME_TEST_KEY", "already-set")
    envfile = tmp_path / ".env"
    envfile.write_text("SOME_TEST_KEY=from-file\n")
    load_dotenv(str(envfile))
    assert os.environ["SOME_TEST_KEY"] == "already-set"


def test_load_dotenv_missing_file_is_a_noop():
    assert load_dotenv("/nonexistent/path/.env") == {}


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------

def test_manifest_roundtrip_and_append_only(tmp_path):
    path = str(tmp_path / "m.json")
    write_manifest(path, {"run_id": "x", "n": 3})
    assert read_manifest(path) == {"run_id": "x", "n": 3}
    with pytest.raises(FileExistsError):
        write_manifest(path, {"run_id": "x", "n": 4})


# ---------------------------------------------------------------------------
# rewrite cache
# ---------------------------------------------------------------------------

def test_rewrite_cache_roundtrip_and_append_only(tmp_path):
    path = str(tmp_path / "rewrites.jsonl")
    results = [
        RewriteResult("31", "31_1", "raw one", "", "query one"),
        RewriteResult("31", "31_2", "raw two", "- likes tofu", "query two"),
    ]
    write_rewrites(path, results)
    assert read_rewrites(path) == results
    with pytest.raises(FileExistsError):
        write_rewrites(path, results)


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

def test_load_dialogues_dispatches_cast(tmp_path):
    cast_dir = tmp_path / "cast"
    cast_dir.mkdir()
    (cast_dir / "cast19_topics.json").write_text(
        open(os.path.join(FIXTURES, "cast_smoke.json"), encoding="utf-8").read()
    )
    dialogues = load_dialogues("cast19", data_dir=str(tmp_path))
    assert {d.dialogue_id for d in dialogues} == {"31", "32"}


def test_load_dialogues_dispatches_ikat(tmp_path):
    ikat_dir = tmp_path / "ikat"
    ikat_dir.mkdir()
    (ikat_dir / "ikat23_topics.json").write_text(
        open(os.path.join(FIXTURES, "ikat_smoke.json"), encoding="utf-8").read()
    )
    dialogues = load_dialogues("ikat23", data_dir=str(tmp_path))
    assert {d.dialogue_id for d in dialogues} == {"9-1", "9-2"}


def test_load_dialogues_unknown_collection_raises():
    with pytest.raises(ValueError, match="longmemeval"):
        load_dialogues("longmemeval", data_dir="data")


def test_load_dialogues_missing_file_raises_with_hint(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_free_data"):
        load_dialogues("cast19", data_dir=str(tmp_path))


def test_make_memory_system_baselines():
    assert isinstance(make_memory_system("B0_none"), NoPersonalizationMemory)
    assert isinstance(make_memory_system("B1_full_history"), FullHistoryMemory)
    assert isinstance(make_memory_system("B2_gold_ptkb"), GoldPTKBMemory)


def test_make_memory_system_mem0():
    if not HAVE_MEM0:
        with pytest.raises(RuntimeError, match="mem0ai"):
            make_memory_system("S1_mem0")
        return
    try:
        mem = make_memory_system("S1_mem0", ablation={"infer": False})
    except Exception as e:
        # mem0 builds its own OpenAI client eagerly in Memory(), so with the library
        # installed but no key in the environment this is a machine state, not a code fault.
        # (Contrast S3: A-Mem with evolution=False constructs and runs with no key at all.)
        if "api_key" in str(e).lower() or "credential" in str(e).lower():
            pytest.skip("mem0ai installed but no OPENAI_API_KEY -- .env missing")
        raise
    assert mem.name == "S1_mem0"


def test_make_memory_system_amem():
    """S3 landed Aug 22. evolution=False needs no LLM backend (upstream makes no LLM call on
    that path), so the registry must build it without one."""
    from adapters.amem_adapter import HAVE_AMEM
    if HAVE_AMEM:
        mem = make_memory_system("S3_amem", ablation={"evolution": False})
        assert mem.name == "S3_amem" and mem.capabilities == frozenset({"evolution"})
    else:
        with pytest.raises(RuntimeError, match="A-Mem not installed"):
            make_memory_system("S3_amem", ablation={"evolution": False})


def test_make_memory_system_gam():
    """S5 landed Aug 22. Unlike S3 it has no offline mode, so the registry must refuse to
    build it without a backend rather than let it open its own temp-0.3 client."""
    from adapters.gam_adapter import HAVE_GAM
    if HAVE_GAM:
        with pytest.raises(ValueError, match="LLMBackend"):
            make_memory_system("S5_gam")
    else:
        with pytest.raises(RuntimeError, match="GAM not installed"):
            make_memory_system("S5_gam")


def test_make_memory_system_graphiti():
    """S2 landed Aug 22. Like S5 it has no offline mode, so the registry must refuse to build
    it without a backend rather than let it open its own client."""
    from adapters.graphiti_adapter import HAVE_GRAPHITI
    if HAVE_GRAPHITI:
        with pytest.raises(ValueError, match="LLMBackend"):
            make_memory_system("S2_graphiti")
    else:
        with pytest.raises(RuntimeError, match="graphiti-core not installed"):
            make_memory_system("S2_graphiti")


def test_make_memory_system_not_yet_built_adapters_say_so():
    for condition in ("S4_letta",):
        with pytest.raises(NotImplementedError, match="not built yet"):
            make_memory_system(condition)


def test_make_memory_system_unknown_condition_raises():
    with pytest.raises(ValueError):
        make_memory_system("B9_nope")


# ---------------------------------------------------------------------------
# cache robustness
# ---------------------------------------------------------------------------

def test_cache_skips_corrupt_lines_instead_of_crashing(tmp_path, capsys):
    """Two processes appending to one cache can interleave a partial line. That happened
    (grid + intrinsic sharing runs/llm_cache.jsonl: 1 bad line in 4703) and crashed every
    later run at startup. A cache must degrade, not brick the harness."""
    from harness.llmcache import JSONLCache
    path = tmp_path / "c.jsonl"
    good = JSONLCache(str(path))
    good.put("k1", "v1")
    good.put("k2", "v2", meta={"usage": {"total_tokens": 5}})
    with open(path, "a", encoding="utf-8") as f:
        f.write("organized into a main category for cultural expressions, wit\n")  # real corruption
        f.write('{"key": "k3"}\n')                                                 # no "value"
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"key": "k4", "value": "v4"}\n')

    reloaded = JSONLCache(str(path))
    assert reloaded.get("k1") == "v1"
    assert reloaded.get("k4") == "v4", "entries after the corruption must still load"
    assert reloaded.get("k3") is None
    assert reloaded.stats()["corrupt_lines"] == 2
    assert reloaded.get_meta("k2") == {"usage": {"total_tokens": 5}}
    assert "skipped 2 unparseable cache line" in capsys.readouterr().err


def test_clean_cache_reports_no_corruption(tmp_path):
    from harness.llmcache import JSONLCache
    c = JSONLCache(str(tmp_path / "c.jsonl"))
    c.put("k", "v")
    assert JSONLCache(str(tmp_path / "c.jsonl")).stats()["corrupt_lines"] == 0
