"""CAsT 2019/2020 loader (needs NO corpus license, topics+qrels only).

Parses the treccastweb-format topic JSON (github.com/daltonj/treccastweb, as staged into
data/cast/ by scripts/fetch_free_data.sh) into Dialogue objects.

Only `raw_utterance` is ever read. CAsT topic files (2020 especially) also carry
`automatic_rewritten_utterance` / `manual_rewritten_utterance` — those must NOT be used: the
harness performs its own contextual rewriting via prompts/rewrite_v1.txt, and feeding it an
already-rewritten utterance would contaminate the B0/B1/S1-S5 comparison (every condition has
to start from the same raw, context-dependent question).

CAsT topic files carry no system-turn text and no persona data, so every Dialogue here has
ptkb=[] and turns of role="user" only — matching the iKAT loader's shape only in the
Dialogue/Turn contract, not in what's populated (see loaders/ikat.py for the contrast).

Turn id convention: "<topic_number>_<turn_number>" (e.g. "31_1"), matching the qid format the
official 2019/2020 qrels use. Do not switch to a hyphenated id here — that's the iKAT
convention (loaders/ikat.py), not this one; getting it wrong silently breaks qrels alignment
in run_eval rather than raising an error, so it's covered by a unit test below.
"""
from __future__ import annotations

import json

from ..memory_base import Dialogue, Turn


def _clean(text: str) -> str:
    """Collapse whitespace (a few real topic files have trailing/doubled spaces)."""
    return " ".join(text.split())


def parse_cast_topics(raw: list[dict]) -> list[Dialogue]:
    """raw: the parsed JSON list straight from a treccastweb evaluation-topics file
    (2019: evaluation_topics_v1.0.json; 2020: 2020_automatic_evaluation_topics_v1.0.json).
    One Dialogue per topic; dialogue_id is the topic number as a string (used for
    per-dialogue memory isolation — a harness invariant)."""
    dialogues = []
    for topic in raw:
        number = topic["number"]
        turns = [
            Turn(
                turn_id=f"{number}_{turn['number']}",
                role="user",
                text=_clean(turn["raw_utterance"]),
            )
            for turn in topic["turn"]
        ]
        dialogues.append(
            Dialogue(
                dialogue_id=str(number),
                turns=turns,
                ptkb=[],
                meta={
                    "title": topic.get("title", ""),
                    "description": topic.get("description", ""),
                },
            )
        )
    return dialogues


def load_cast_topics(path: str) -> list[Dialogue]:
    """path: a treccastweb topic JSON file, e.g. data/cast/cast19_topics.json."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return parse_cast_topics(raw)
