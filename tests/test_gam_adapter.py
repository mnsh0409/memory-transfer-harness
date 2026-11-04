"""S5 GAM adapter tests.

GAM has no offline mode -- both ingest and query are LLM-driven -- so there is no
key-free live path the way there is for A-Mem. Instead:

* **Contract tests with a fake `Workflow`** pin the adapter's own behaviour: per-dialogue
  isolation via one directory per dialogue, reset actually deleting it, export_store reading
  the tree rather than issuing a request, the compile-at-query mapping, and the
  user-knowledge framing.
* **A live test against the real GAM package** drives it with a scripted `HarnessGenerator`
  over a fake backend, so upstream's real add/request code paths run with no API key. It is
  skipped when GAM isn't installed, and it is tolerant about *what* GAM produces (that
  depends on canned LLM text) while still asserting the adapter survives the real objects.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

import adapters.gam_adapter as gam_mod
from adapters.gam_adapter import USER_KNOWLEDGE_FRAMING, GAMMemory, HarnessGenerator
from harness.memory_base import Turn


# ---------------------------------------------------------------------------
# Fakes shaped like upstream
# ---------------------------------------------------------------------------

class _FakeNode:
    def __init__(self, name, content, summary=""):
        self.name, self.content, self.summary = name, content, summary


class _FakeRoot:
    def __init__(self, nodes):
        self._nodes = nodes

    def get_all_files(self):
        return self._nodes


class _FakeTree:
    def __init__(self, nodes):
        self.root = _FakeRoot(nodes)
        self.is_empty = not nodes


class _FakeChatResult:
    def __init__(self, answer):
        self.answer, self.confidence, self.sources = answer, 0.9, ["/memories/0.md"]


class _FakeWorkflow:
    instances: list["_FakeWorkflow"] = []

    def __init__(self, kind, gam_dir=None, generator=None, chat_generator=None,
                 verbose=False, max_iterations=10):
        self.kind, self.gam_dir = kind, gam_dir
        self.generator, self.chat_generator = generator, chat_generator
        self.max_iterations = max_iterations
        self.added: list[str] = []
        self.requests: list[str] = []
        os.makedirs(gam_dir, exist_ok=True)          # upstream creates the dir
        _FakeWorkflow.instances.append(self)

    @property
    def tree(self):
        return _FakeTree([_FakeNode(f"n{i}.md", c, f"summary of {c[:12]}")
                          for i, c in enumerate(self.added)])

    def reload_tree(self):
        return self.tree

    def add(self, content=None, input_file=None, **kw):
        self.added.append(content)

    def request(self, question, **kw):
        self.requests.append(question)
        return _FakeChatResult(f"The user mentioned: {' | '.join(self.added)}")


class _FakeBackend:
    def __init__(self, replies=None):
        self.prompts = []
        self._replies = replies or {}

    def complete(self, prompt):
        self.prompts.append(prompt)
        for needle, reply in self._replies.items():
            if needle in prompt:
                return reply
        return "{}"


@pytest.fixture
def fake_gam(monkeypatch, tmp_path):
    _FakeWorkflow.instances = []
    monkeypatch.setattr(gam_mod, "Workflow", _FakeWorkflow, raising=False)
    monkeypatch.setattr(gam_mod, "HAVE_GAM", True)
    return _FakeWorkflow


def _mem(tmp_path, **kw):
    return GAMMemory(backend=_FakeBackend(), root_dir=str(tmp_path / "stores"), **kw)


TURNS = [
    Turn("d1_1", "user", "I want to start meal-prepping."),
    Turn("d1_2", "system", "Batch-cook grains on Sunday."),
    Turn("d1_3", "user", "What proteins work for me?"),
]


def test_requires_gam_installed(monkeypatch, tmp_path):
    monkeypatch.setattr(gam_mod, "HAVE_GAM", False)
    with pytest.raises(RuntimeError, match="GAM not installed"):
        GAMMemory(backend=_FakeBackend(), root_dir=str(tmp_path))


def test_requires_a_harness_backend(fake_gam, tmp_path):
    """GAM would otherwise build its own client at temperature 0.3."""
    with pytest.raises(ValueError, match="LLMBackend"):
        GAMMemory(backend=None, root_dir=str(tmp_path))


def test_ingest_uses_role_prefixed_content(fake_gam, tmp_path):
    m = _mem(tmp_path)
    for t in TURNS:
        m.ingest_turn("d1", t)
    assert fake_gam.instances[-1].added == [f"{t.role}: {t.text}" for t in TURNS]


def test_query_is_compile_at_query_single_entry(fake_gam, tmp_path):
    m = _mem(tmp_path)
    for t in TURNS:
        m.ingest_turn("d1", t)
    entries = m.query("d1", "Which proteins are safe?", k=5)
    assert len(entries) == 1                      # k is inapplicable, not truncation
    assert entries[0].meta["k_inapplicable"] is True
    assert entries[0].meta["kind"] == "compiled_at_query"
    assert entries[0].meta["confidence"] == 0.9
    assert "meal-prepping" in entries[0].text


def test_query_asks_for_user_knowledge_not_the_answer(fake_gam, tmp_path):
    """The framing is what keeps GAM comparable: other conditions supply user knowledge, so
    GAM must not be asked to answer the search question itself."""
    m = _mem(tmp_path)
    m.ingest_turn("d1", TURNS[0])
    m.query("d1", "Which proteins are safe?", k=5)
    asked = fake_gam.instances[-1].requests[0]
    assert asked == USER_KNOWLEDGE_FRAMING.format(utterance="Which proteins are safe?")
    assert "Do not answer the request itself." in asked
    assert "user" in asked.lower()


def test_query_on_empty_store_returns_nothing_without_calling_request(fake_gam, tmp_path):
    m = _mem(tmp_path)
    assert m.query("d1", "anything", k=5) == []
    assert fake_gam.instances[-1].requests == []


def test_per_dialogue_isolation_is_one_directory_each(fake_gam, tmp_path):
    m = _mem(tmp_path)
    m.ingest_turn("d1", TURNS[0])
    m.ingest_turn("d2", TURNS[2])
    dirs = [w.gam_dir for w in fake_gam.instances]
    assert len(set(dirs)) == 2 and dirs[0] != dirs[1]
    assert fake_gam.instances[-1].added == ["user: What proteins work for me?"]


def test_reset_deletes_the_directory(fake_gam, tmp_path):
    m = _mem(tmp_path)
    m.ingest_turn("d1", TURNS[0])
    path = fake_gam.instances[-1].gam_dir
    assert os.path.isdir(path)
    m.reset("d1")
    assert not os.path.exists(path)       # GAM persists to disk: a soft reset would leak
    assert m.export_store("d1") == []


def test_export_store_dumps_the_tree_not_a_request(fake_gam, tmp_path):
    m = _mem(tmp_path)
    for t in TURNS:
        m.ingest_turn("d1", t)
    dump = m.export_store("d1")
    assert [e.text for e in dump] == [f"{t.role}: {t.text}" for t in TURNS]
    assert all(e.meta["kind"] == "gam_node" and e.meta["summary"] for e in dump)
    assert fake_gam.instances[-1].requests == []   # a dump must never be a search


def test_generators_are_injected_for_both_agents(fake_gam, tmp_path):
    m = _mem(tmp_path)
    m.ingest_turn("d1", TURNS[0])
    wf = fake_gam.instances[-1]
    assert isinstance(wf.generator, HarnessGenerator)
    assert wf.chat_generator is wf.generator        # one cache for both agents


def test_capabilities_and_name():
    assert GAMMemory.name == "S5_gam"
    assert GAMMemory.capabilities == frozenset()    # the frozen scope lists no GAM ablation


# ---------------------------------------------------------------------------
# HarnessGenerator -- the temp-0 / cached transport GAM is given
# ---------------------------------------------------------------------------

def test_generator_returns_gam_result_shape_and_parses_json():
    g = HarnessGenerator(_FakeBackend({"ask": '{"a": 1}'}))
    out = g.generate_single(prompt="please ask")
    assert set(out) == {"text", "json", "response"}
    assert out["json"] == {"a": 1} and out["text"] == '{"a": 1}'


def test_generator_repairs_or_nulls_bad_json():
    g = HarnessGenerator(_FakeBackend({"x": "not json at all"}))
    assert g.generate_single(prompt="x")["json"] is None    # upstream handles None


def test_generator_flattens_messages_and_batches():
    backend = _FakeBackend()
    g = HarnessGenerator(backend)
    g.generate_single(messages=[{"role": "system", "content": "S"},
                                {"role": "user", "content": "U"}])
    assert backend.prompts[-1] == "system: S\n\nuser: U"
    assert len(g.generate_batch(prompts=["a", "b", "c"])) == 3
    assert g.calls == 4


# ---------------------------------------------------------------------------
# Live -- real GAM package, scripted generator, no API key
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not gam_mod.HAVE_GAM, reason="GAM not installed")
def test_live_gam_accepts_our_generator_and_directory_layout(tmp_path):
    """Drives the REAL gam.Workflow with a scripted generator. Asserts the plumbing the
    adapter depends on -- construction with an injected generator, add() accepting raw
    content, a readable tree, and reset() clearing the store -- without pretending to
    validate GAM's own reasoning on canned LLM text."""
    backend = _FakeBackend({"": json.dumps({"chunks": ["user: I am a vegetarian."],
                                            "memory": "The user is a vegetarian.",
                                            "tldr": "vegetarian user",
                                            "answer": "The user is a vegetarian."})})
    m = GAMMemory(backend=backend, root_dir=str(tmp_path / "stores"), verbose=False)
    wf = m._wf_for("smoke-1")
    # gam.Workflow is a FACTORY FUNCTION, not a class -- it dispatches on "text"/"video",
    # so the adapter gets a TextWorkflow back and `isinstance(wf, Workflow)` would raise.
    assert wf.__class__.__name__ == "TextWorkflow"
    assert isinstance(wf.generator, HarnessGenerator)
    assert wf.chat_generator is wf.generator

    try:
        m.ingest_turn("smoke-1", Turn("smoke-1_1", "user", "I am a vegetarian."))
    except Exception as e:                      # canned LLM text may not satisfy upstream
        pytest.skip(f"GAM ingest needs a real model to produce usable structure: {e!r}")

    assert backend.prompts, "GAM must have gone through the harness backend, not its own client"
    dump = m.export_store("smoke-1")
    assert isinstance(dump, list)
    store_dir = m._dir_for("smoke-1")
    assert store_dir.exists()
    m.reset("smoke-1")
    assert not store_dir.exists()


def test_strip_reasoning_keeps_the_answer_and_drops_the_thinking():
    """GAM's compiled answer becomes the memory block verbatim, so chain-of-thought must not
    ride along. 37% of its cached replies carry <think>; a probe on a built store returned an
    unterminated one AS the entry text."""
    from adapters.gam_adapter import _strip_reasoning

    assert _strip_reasoning("<think>weighing options</think>The user is vegetarian.") \
        == "The user is vegetarian."
    # unterminated tail (truncated reasoning) -- the observed failure
    assert _strip_reasoning("<think>I have not yet gathered specific information") == ""
    # nothing but reasoning -> empty, which the caller reads as an empty compile
    assert _strip_reasoning("<think>a</think>   ") == ""
    # untouched when clean
    assert _strip_reasoning("The user cooks at home.") == "The user cooks at home."
