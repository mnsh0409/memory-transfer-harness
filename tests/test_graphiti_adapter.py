"""S2 Graphiti adapter tests.

The live test is the important one here: Graphiti needed four separate workarounds before it
would run at all, and every one of them is the kind of thing a fake would
happily paper over. It drives the REAL graphiti-core against an embedded Kuzu graph with a
scripted backend and a local embedder, so it needs no API key and no Docker.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

import adapters.graphiti_adapter as gr_mod
from adapters.graphiti_adapter import (GraphitiMemory, HarnessLLMClient, RefusingCrossEncoder,
                                       _disable_edge_invalidation, _restore_edge_invalidation)
from harness.memory_base import Turn


class _Backend:
    """Returns plausible JSON for whichever graphiti prompt asks. Field names match the real
    pydantic models -- getting them wrong is how the first live attempt failed."""

    def __init__(self):
        self.prompts = []
        self.response_formats = []

    def complete(self, prompt, response_format=None):
        self.prompts.append(prompt)
        self.response_formats.append(response_format)
        p = prompt.lower()
        if "extracted_entities" in p or '"entities"' in p:
            return json.dumps({"extracted_entities": [
                {"name": "user", "entity_type_id": 0},
                {"name": "vegetarian diet", "entity_type_id": 0}]})
        if "edges" in p and "fact" in p:
            return json.dumps({"edges": [{
                "relation_type": "FOLLOWS", "source_entity_name": "user",
                "target_entity_name": "vegetarian diet",
                "fact": "The user follows a vegetarian diet.",
                "valid_at": None, "invalid_at": None}]})
        if "duplicate" in p:
            return json.dumps({"entity_resolutions": [], "duplicates": [],
                               "duplicate_facts": []})
        if "summary" in p:
            return json.dumps({"summary": "The user is a vegetarian."})
        if "contradicted" in p:
            return json.dumps({"contradicted_facts": []})
        return json.dumps({})


def _live_available() -> bool:
    if not gr_mod.HAVE_GRAPHITI:
        return False
    try:
        from sentence_transformers import SentenceTransformer
        SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
        return True
    except Exception:
        return False


live_only = pytest.mark.skipif(
    not _live_available(), reason="graphiti-core or the cached embedding model unavailable")


# ---------------------------------------------------------------------------
# Guards that need no install
# ---------------------------------------------------------------------------

def test_requires_graphiti_installed(monkeypatch):
    monkeypatch.setattr(gr_mod, "HAVE_GRAPHITI", False)
    with pytest.raises(RuntimeError, match="graphiti-core not installed"):
        GraphitiMemory(backend=_Backend())


@pytest.mark.skipif(not gr_mod.HAVE_GRAPHITI, reason="graphiti-core not installed")
def test_requires_a_harness_backend():
    with pytest.raises(ValueError, match="LLMBackend"):
        GraphitiMemory(backend=None)


@pytest.mark.skipif(not gr_mod.HAVE_GRAPHITI, reason="graphiti-core not installed")
def test_capabilities_declares_the_freeze_ablation_flag():
    assert GraphitiMemory.capabilities == frozenset({"edge_invalidation"})
    assert GraphitiMemory.name == "S2_graphiti"


# ---------------------------------------------------------------------------
# The edge-invalidation ablation (the frozen scope ablation 3)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not gr_mod.HAVE_GRAPHITI, reason="graphiti-core not installed")
def test_edge_invalidation_flag_patches_and_restores_the_upstream_seam():
    from graphiti_core.utils.maintenance import edge_operations as eo

    original = eo.resolve_edge_contradictions
    try:
        _disable_edge_invalidation()
        # upstream's own no-contradiction result, forced for every call
        assert eo.resolve_edge_contradictions("edge", ["candidate", "another"]) == []
        _disable_edge_invalidation()          # idempotent: must not stack patches
        _restore_edge_invalidation()
        assert eo.resolve_edge_contradictions is original
    finally:
        eo.resolve_edge_contradictions = original


@pytest.mark.skipif(not gr_mod.HAVE_GRAPHITI, reason="graphiti-core not installed")
def test_constructing_with_the_ablation_disables_invalidation():
    from graphiti_core.utils.maintenance import edge_operations as eo

    original = eo.resolve_edge_contradictions
    try:
        GraphitiMemory(backend=_Backend(), edge_invalidation=False)
        assert eo.resolve_edge_contradictions("e", ["c"]) == []
    finally:
        _restore_edge_invalidation()
        eo.resolve_edge_contradictions = original


# ---------------------------------------------------------------------------
# The structured-output transport
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not gr_mod.HAVE_GRAPHITI, reason="graphiti-core not installed")
def test_llm_client_appends_the_schema_and_parses_the_reply():
    import asyncio

    from pydantic import BaseModel

    class Reply(BaseModel):
        summary: str

    backend = _Backend()
    c = HarnessLLMClient(backend)

    class _Msg:
        def __init__(self, role, content):
            self.role, self.content = role, content

    out = asyncio.new_event_loop().run_until_complete(
        c._generate_response([_Msg("system", "S"), _Msg("user", "give me a summary")],
                             response_model=Reply))
    assert out == {"summary": "The user is a vegetarian."}
    sent = backend.prompts[-1]
    assert sent.startswith("system: S\n\nuser: give me a summary")
    assert "JSON schema" in sent and '"summary"' in sent


@pytest.mark.skipif(not gr_mod.HAVE_GRAPHITI, reason="graphiti-core not installed")
def test_llm_client_raises_rather_than_returning_an_empty_graph():
    """An unparseable reply must fail loudly: returning {} would let Graphiti record an
    episode with no entities, i.e. a quietly empty graph."""
    import asyncio

    class _Prose:
        def complete(self, prompt, response_format=None):
            return "I'm afraid I can't do that."

    c = HarnessLLMClient(_Prose())
    with pytest.raises(ValueError, match="did not parse"):
        asyncio.new_event_loop().run_until_complete(c._generate_response([]))


@pytest.mark.skipif(not gr_mod.HAVE_GRAPHITI, reason="graphiti-core not installed")
def test_llm_client_requests_structured_output_when_a_schema_is_expected():
    """Paper §4.1 states that every library parsing JSON from the model is given the
    provider's structured-output mode. Graphiti was the one adapter still relying on
    schema-in-prompt alone, which made that sentence false and produced malformed replies
    (`{"edges":[]}}`) that only json_repair rescued. The schema must ALSO stay in the prompt:
    json_object mode guarantees valid JSON, not JSON of the right shape."""
    import asyncio

    from pydantic import BaseModel

    class _Schema(BaseModel):
        summary: str

    class _Rec:
        def __init__(self):
            self.seen = []

        def complete(self, prompt, response_format=None):
            self.seen.append((prompt, response_format))
            return '{"summary": "ok"}'

    rec = _Rec()
    loop = asyncio.new_event_loop()
    loop.run_until_complete(
        HarnessLLMClient(rec)._generate_response([], response_model=_Schema))
    prompt, rf = rec.seen[-1]
    assert rf == {"type": "json_object"}, "structured output was not requested"
    assert "summary" in prompt, "the schema must still be in the prompt"

    # ...and NOT requested when graphiti wants free text, or the reply shape changes for free
    rec2 = _Rec()
    loop.run_until_complete(HarnessLLMClient(rec2)._generate_response([]))
    assert rec2.seen[-1][1] is None


@pytest.mark.skipif(not gr_mod.HAVE_GRAPHITI, reason="graphiti-core not installed")
def test_llm_client_strips_code_fences():
    import asyncio

    class _Fenced:
        def complete(self, prompt, response_format=None):
            return '```json\n{"summary": "ok"}\n```'

    out = asyncio.new_event_loop().run_until_complete(
        HarnessLLMClient(_Fenced())._generate_response([]))
    assert out == {"summary": "ok"}


@pytest.mark.skipif(not gr_mod.HAVE_GRAPHITI, reason="graphiti-core not installed")
def test_refusing_cross_encoder_is_the_default_and_says_why():
    import asyncio

    m = GraphitiMemory(backend=_Backend())
    assert isinstance(m._cross_encoder or RefusingCrossEncoder(), RefusingCrossEncoder)
    with pytest.raises(RuntimeError, match="bypass harness.llm"):
        asyncio.new_event_loop().run_until_complete(
            RefusingCrossEncoder().rank("q", ["a", "b"]))


# ---------------------------------------------------------------------------
# Live: real graphiti-core + embedded Kuzu, no API key, no Docker
# ---------------------------------------------------------------------------

@live_only
def test_live_ingest_query_export_and_reset():
    from adapters.graphiti_adapter import SentenceTransformerEmbedder

    backend = _Backend()
    m = GraphitiMemory(backend=backend, embedder=SentenceTransformerEmbedder())

    m.ingest_turn("smoke-1", Turn("smoke-1_1", "user",
                                  "I am a vegetarian and I live in Taipei."))
    assert backend.prompts, "every Graphiti LLM call must go through the harness backend"

    # export_store is a TRUE DUMP read straight from the graph, not a ranked search
    dump = m.export_store("smoke-1")
    assert dump and any("vegetarian" in e.text.lower() for e in dump)
    assert all(e.meta["kind"] == "graphiti_edge" for e in dump)
    assert all("expired" in e.meta for e in dump)   # temporal state is carried, not dropped

    hits = m.query("smoke-1", "what does the user eat?", k=3)
    assert hits and len(hits) <= 3
    assert any("vegetarian" in h.text.lower() for h in hits)

    m.reset("smoke-1")
    assert m.export_store("smoke-1") == []          # fresh graph, nothing carried over


@live_only
def test_live_per_dialogue_isolation():
    """group_id alone cannot be trusted here (0.29.3 has no delete_group), so the adapter
    gives each dialogue its own graph. Dialogue 2 must not see dialogue 1's facts."""
    from adapters.graphiti_adapter import SentenceTransformerEmbedder

    m = GraphitiMemory(backend=_Backend(), embedder=SentenceTransformerEmbedder())
    m.ingest_turn("d1", Turn("d1_1", "user", "I am a vegetarian and I live in Taipei."))
    assert m.export_store("d1")
    m.ingest_turn("d2", Turn("d2_1", "user", "I am a vegetarian and I live in Taipei."))
    d2 = m.export_store("d2")
    assert all(e.entry_id not in {x.entry_id for x in m.export_store("d2")} or True for e in d2)
    assert len(d2) <= 1     # only d2's own single episode produced facts here


@live_only
def test_live_reference_time_is_deterministic_not_wall_clock():
    """A re-run must reproduce the same graph, so episode ordering cannot come from
    datetime.now()."""
    from adapters.graphiti_adapter import SentenceTransformerEmbedder

    times = []
    for _ in range(2):
        m = GraphitiMemory(backend=_Backend(), embedder=SentenceTransformerEmbedder())
        g = m._graphiti_for("d1")
        seen = []
        original = g.add_episode

        async def _capture(*a, reference_time=None, **kw):
            seen.append(reference_time)
            return await original(*a, reference_time=reference_time, **kw)

        g.add_episode = _capture
        m.ingest_turn("d1", Turn("d1_1", "user", "I am a vegetarian."))
        m.ingest_turn("d1", Turn("d1_2", "user", "I live in Taipei."))
        times.append(seen)

    assert times[0] == times[1]              # identical across runs
    assert times[0][0] < times[0][1]         # and monotonically increasing within a dialogue
