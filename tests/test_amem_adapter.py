"""S3 A-Mem adapter tests.

Two layers, both offline:

* **Contract tests with a fake `AgenticMemorySystem`** — always run, no A-Mem install needed.
  They pin the things that would silently corrupt a run: per-dialogue isolation, export_store
  being a true dump rather than a search, the evolution flag actually suppressing the
  evolution step, and every LLM call going through the harness backend (never upstream's own
  temperature-0.7 OpenAI client).
* **A live smoke test** against the real library, skipped unless A-Mem is importable and its
  embedding model is cached locally. It runs the smoke fixture with `evolution=False`, which
  upstream implements with zero LLM calls -- so it needs no API key.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

import adapters.amem_adapter as amem_mod
from adapters.amem_adapter import AMemMemory, HarnessLLMController
from harness.memory_base import Turn


# ---------------------------------------------------------------------------
# Fake upstream, shaped like the real AgenticMemorySystem
# ---------------------------------------------------------------------------

class _FakeNote:
    def __init__(self, content, note_id):
        self.content = content
        self.id = note_id
        self.context = "General"
        self.tags = []
        self.keywords = []
        self.links = []
        self.evolution_history = []


class _FakeLLMController:
    """Stands in for upstream's `llm_controller`, whose `.llm` the adapter replaces."""
    def __init__(self):
        self.llm = object()


class _FakeAgenticMemorySystem:
    instances: list["_FakeAgenticMemorySystem"] = []

    def __init__(self, model_name=None, llm_backend=None, llm_model=None,
                 evo_threshold=None, api_key=None):
        self.memories = {}
        self.llm_controller = _FakeLLMController()
        self.init_kwargs = dict(model_name=model_name, llm_backend=llm_backend,
                                llm_model=llm_model, evo_threshold=evo_threshold,
                                api_key=api_key)
        self.evolution_calls = 0
        _FakeAgenticMemorySystem.instances.append(self)

    # upstream calls process_memory inside add_note; the adapter may override it
    def process_memory(self, note):
        self.evolution_calls += 1
        return True, note

    def add_note(self, content, time=None, **kwargs):
        note_id = f"n{len(self.memories)}"
        note = _FakeNote(content, note_id)
        _evolved, note = self.process_memory(note)
        self.memories[note_id] = note
        return note_id

    def search_agentic(self, query, k=5):
        return [{"id": nid, "content": n.content, "context": n.context, "score": 0.5,
                 "is_neighbor": False}
                for nid, n in list(self.memories.items())[:k]]


class _FakeBackend:
    def __init__(self):
        self.prompts = []
        self.response_formats = []

    def complete(self, prompt, response_format=None):
        self.prompts.append(prompt)
        self.response_formats.append(response_format)
        return '{"should_evolve": false}'


@pytest.fixture
def fake_amem(monkeypatch):
    _FakeAgenticMemorySystem.instances = []
    # raising=False: when A-Mem isn't installed the import guard leaves the name undefined,
    # and these contract tests must still run on a bare checkout.
    monkeypatch.setattr(amem_mod, "AgenticMemorySystem", _FakeAgenticMemorySystem,
                        raising=False)
    monkeypatch.setattr(amem_mod, "HAVE_AMEM", True)
    return _FakeAgenticMemorySystem


TURNS = [
    Turn("d1_1", "user", "I want to start meal-prepping."),
    Turn("d1_2", "system", "Batch-cook grains on Sunday."),
    Turn("d1_3", "user", "What proteins work for me?"),
]


def test_requires_amem_installed(monkeypatch):
    monkeypatch.setattr(amem_mod, "HAVE_AMEM", False)
    with pytest.raises(RuntimeError, match="A-Mem not installed"):
        AMemMemory(evolution=False)


def test_evolution_on_requires_a_harness_backend(fake_amem):
    """Upstream would otherwise build its own OpenAI client at temperature 0.7 -- two
    harness invariants broken silently. Refuse instead."""
    with pytest.raises(ValueError, match="LLMBackend"):
        AMemMemory(evolution=True)
    assert AMemMemory(evolution=True, backend=_FakeBackend()) is not None


def test_ingest_and_query_roundtrip(fake_amem):
    m = AMemMemory(evolution=False)
    for t in TURNS:
        m.ingest_turn("d1", t)
    entries = m.query("d1", "proteins", k=5)
    assert [e.text for e in entries] == [f"{t.role}: {t.text}" for t in TURNS]
    assert all(e.entry_id for e in entries)
    assert entries[0].meta["score"] == 0.5


def test_query_respects_k(fake_amem):
    m = AMemMemory(evolution=False)
    for t in TURNS:
        m.ingest_turn("d1", t)
    assert len(m.query("d1", "q", k=2)) == 2


def test_speaker_role_is_kept_in_the_note_text(fake_amem):
    m = AMemMemory(evolution=False)
    m.ingest_turn("d1", TURNS[0])
    assert list(fake_amem.instances[-1].memories.values())[0].content.startswith("user: ")


def test_per_dialogue_isolation_rebuilds_the_system(fake_amem):
    """Upstream keeps one process-wide ChromaDB collection, so a second dialogue must get a
    fresh instance -- otherwise dialogue 2 retrieves dialogue 1's notes."""
    m = AMemMemory(evolution=False)
    m.ingest_turn("d1", TURNS[0])
    m.ingest_turn("d2", TURNS[2])          # switching dialogue rebuilds
    assert len(fake_amem.instances) == 2
    texts = [e.text for e in m.query("d2", "q", k=10)]
    assert texts == ["user: What proteins work for me?"]
    assert "meal-prepping" not in " ".join(texts)


def test_reset_drops_state(fake_amem):
    m = AMemMemory(evolution=False)
    m.ingest_turn("d1", TURNS[0])
    m.reset("d1")
    assert m.query("d1", "q", k=10) == []
    assert len(fake_amem.instances) == 2   # rebuilt after reset


def test_export_store_is_a_true_dump_not_a_search(fake_amem):
    """The intrinsic axis reads export_store directly: it must return every note, including
    more than any search k would give back."""
    m = AMemMemory(evolution=False)
    for i in range(12):
        m.ingest_turn("d1", Turn(f"d1_{i}", "user", f"fact number {i}"))
    dump = m.export_store("d1")
    assert len(dump) == 12                      # not capped at the default k=5
    assert len(m.query("d1", "", k=5)) == 5     # ...which a search would have been
    assert dump[0].text == "user: fact number 0"
    assert set(dump[0].meta) == {"context", "tags", "keywords", "links", "evolved"}


def test_evolution_flag_off_suppresses_the_evolution_step(fake_amem):
    m = AMemMemory(evolution=False)
    for t in TURNS:
        m.ingest_turn("d1", t)
    assert fake_amem.instances[-1].evolution_calls == 0


def test_evolution_flag_on_runs_it_through_the_harness_backend(fake_amem):
    backend = _FakeBackend()
    m = AMemMemory(evolution=True, backend=backend)
    for t in TURNS:
        m.ingest_turn("d1", t)
    system = fake_amem.instances[-1]
    assert system.evolution_calls == 3
    # upstream's own client is replaced by the harness-backed controller
    assert isinstance(system.llm_controller.llm, HarnessLLMController)
    assert system.init_kwargs["api_key"].startswith("unused-harness")


def test_harness_controller_forwards_response_format():
    """It used to DROP response_format, on the reasoning that A-Mem falls back to "no
    evolution" when it cannot parse. That silently disabled the mechanism under test on 429 of
    479 turns and made the frozen scope ablation 2 an off-vs-off non-experiment, so
    forwarding it is now part of the contract."""
    backend = _FakeBackend()
    c = HarnessLLMController(backend)
    out = c.get_completion("PROMPT", response_format={"type": "json_schema"}, temperature=0.7)
    assert out == '{"should_evolve": false}'
    assert backend.prompts == ["PROMPT"]        # prompt passed through byte-for-byte
    assert backend.response_formats == [{"type": "json_schema"}]
    assert c.calls == 1


def test_harness_controller_passes_none_when_amem_asks_for_plain_text():
    backend = _FakeBackend()
    HarnessLLMController(backend).get_completion("PROMPT")
    assert backend.response_formats == [None]


def test_capabilities_declares_the_freeze_ablation_flag():
    assert AMemMemory.capabilities == frozenset({"evolution"})
    assert AMemMemory.name == "S3_amem"


# ---------------------------------------------------------------------------
# Live smoke test -- real library, no API key (evolution=False makes no LLM calls)
# ---------------------------------------------------------------------------

def _amem_live_available() -> bool:
    if not amem_mod.HAVE_AMEM:
        return False
    try:                                    # embedding model must already be cached
        from sentence_transformers import SentenceTransformer
        SentenceTransformer(amem_mod.EMBED_MODEL, local_files_only=True)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _amem_live_available(),
                    reason="A-Mem or its cached embedding model unavailable")
def test_live_smoke_on_the_fixture_dialogue():
    """The smoke-test pattern required before an adapter touches real data:
    replay the fixture, retrieve, and dump the store -- against the real A-Mem."""
    import json
    from harness.memory_base import Dialogue

    fixture = os.path.join(os.path.dirname(__file__), "fixtures", "topic_smoke.json")
    with open(fixture, encoding="utf-8") as f:
        d = json.load(f)
    dialogue = Dialogue(dialogue_id=d["dialogue_id"], turns=[Turn(**t) for t in d["turns"]],
                        ptkb=d["ptkb"])

    m = AMemMemory(evolution=False)
    m.reset(dialogue.dialogue_id)
    for turn in dialogue.turns:
        m.ingest_turn(dialogue.dialogue_id, turn)

    dump = m.export_store(dialogue.dialogue_id)
    assert len(dump) == len(dialogue.turns)
    hits = m.query(dialogue.dialogue_id, "Which proteins are safe for me?", k=3)
    assert 0 < len(hits) <= 3
    assert all(h.text for h in hits)
    # retrieval is semantic: the protein turn should surface for a protein question
    assert any("protein" in h.text.lower() or "tofu" in h.text.lower() for h in hits)

    m.reset(dialogue.dialogue_id)
    assert m.export_store(dialogue.dialogue_id) == []
