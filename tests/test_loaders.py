"""Loader tests. Fixtures are real 2-topic slices cut directly from the official files
(treccastweb 2019 evaluation topics; irlabamsterdam iKAT 2023 test topics) — not hand-written
stand-ins — so a format change upstream is exactly what would break these."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from harness.loaders.cast import load_cast_topics, parse_cast_topics
from harness.loaders.ikat import load_ikat_topics, parse_ikat_topics

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def load_raw(name):
    with open(os.path.join(FIXTURES, name), encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CAsT
# ---------------------------------------------------------------------------

def test_cast_loader_shape():
    dialogues = load_cast_topics(os.path.join(FIXTURES, "cast_smoke.json"))
    assert {d.dialogue_id for d in dialogues} == {"31", "32"}
    for d in dialogues:
        assert d.ptkb == []
        assert len(d.turns) > 0
        assert all(t.role == "user" for t in d.turns)


def test_cast_turn_id_matches_qrels_convention():
    """<topic>_<turn> with an underscore — this is what the 2019/2020 qrels key on."""
    dialogues = parse_cast_topics(load_raw("cast_smoke.json"))
    topic31 = next(d for d in dialogues if d.dialogue_id == "31")
    assert [t.turn_id for t in topic31.turns[:3]] == ["31_1", "31_2", "31_3"]
    assert topic31.turns[0].text == "What is throat cancer?"


def test_cast_loader_never_uses_rewritten_utterance():
    """2020 topics carry automatic_rewritten_utterance alongside raw_utterance; the loader
    must always take the raw form, since the harness does its own rewriting."""
    raw = load_raw("cast_smoke.json")
    raw[0]["turn"][0] = dict(raw[0]["turn"][0])
    raw[0]["turn"][0]["automatic_rewritten_utterance"] = "SHOULD NOT APPEAR"
    dialogues = parse_cast_topics(raw)
    first_turn_text = dialogues[0].turns[0].text
    assert "SHOULD NOT APPEAR" not in first_turn_text
    assert first_turn_text == raw[0]["turn"][0]["raw_utterance"].strip()


def test_cast_loader_cleans_whitespace():
    dialogues = parse_cast_topics(load_raw("cast_smoke.json"))
    assert all(not t.text.startswith(" ") and not t.text.endswith(" ") for d in dialogues for t in d.turns)


# ---------------------------------------------------------------------------
# iKAT
# ---------------------------------------------------------------------------

def test_ikat_loader_shape():
    dialogues = load_ikat_topics(os.path.join(FIXTURES, "ikat_smoke.json"))
    assert {d.dialogue_id for d in dialogues} == {"9-1", "9-2"}
    for d in dialogues:
        assert len(d.ptkb) > 0
        assert len(d.turns) > 0
        assert all(t.role == "user" for t in d.turns)


def test_ikat_turn_id_matches_qrels_convention():
    """<topic>-<turn> with a hyphen (topic itself already contains one, e.g. "9-1" -> "9-1-2");
    do not collapse this to the CAsT underscore convention."""
    dialogues = parse_ikat_topics(load_raw("ikat_smoke.json"))
    topic = next(d for d in dialogues if d.dialogue_id == "9-1")
    assert [t.turn_id for t in topic.turns[:2]] == ["9-1-1", "9-1-2"]


def test_ikat_ptkb_populated_and_ordered():
    dialogues = parse_ikat_topics(load_raw("ikat_smoke.json"))
    topic = next(d for d in dialogues if d.dialogue_id == "9-1")
    assert "I'm vegetarian." in topic.ptkb
    assert len(topic.ptkb) == len(topic.meta["ptkb_by_id"])


def test_ikat_loader_never_uses_resolved_utterance():
    raw = load_raw("ikat_smoke.json")
    raw[0]["turns"][0] = dict(raw[0]["turns"][0])
    raw[0]["turns"][0]["resolved_utterance"] = "SHOULD NOT APPEAR"
    dialogues = parse_ikat_topics(raw)
    assert "SHOULD NOT APPEAR" not in dialogues[0].turns[0].text


def test_ikat_loader_rejects_positional_topic_numbers():
    """The real 2024 topics file numbers topics 0,1,2,... as plain ints instead of using
    '<persona>-<dialogue>' ids. Turn ids built from those align with no qrels, and would fail
    silently rather than loudly -- so the loader refuses."""
    raw = [{"number": 0, "title": "Traveling to Egypt", "ptkb": {"1": "x"},
            "turns": [{"turn_id": 1, "utterance": "hi"}]}]
    with pytest.raises(ValueError, match="positional index"):
        parse_ikat_topics(raw)


def test_ikat_loader_positional_numbers_allowed_when_opted_in():
    raw = [{"number": 0, "title": "Traveling to Egypt", "ptkb": {"1": "x"},
            "turns": [{"turn_id": 1, "utterance": "hi"}]}]
    dialogues = parse_ikat_topics(raw, strict_ids=False)
    assert dialogues[0].turns[0].turn_id == "0-1"


def test_ikat_loader_accepts_real_2023_ids():
    """Guard against the 2024 check being too aggressive and rejecting valid 2023 data."""
    dialogues = parse_ikat_topics(load_raw("ikat_smoke.json"))
    assert {d.dialogue_id for d in dialogues} == {"9-1", "9-2"}


def test_ikat_loader_never_surfaces_clueweb_provenance():
    """Free-data-only by scope decision: nothing licensed-corpus-shaped
    should be reachable off a loaded Dialogue, even by accident."""
    raw = load_raw("ikat_smoke.json")
    dialogues = parse_ikat_topics(raw)
    dumped = json.dumps([d.meta for d in dialogues])
    assert "clueweb22" not in dumped
    for d in dialogues:
        assert "response_provenance" not in d.meta
        assert "response" not in d.meta
