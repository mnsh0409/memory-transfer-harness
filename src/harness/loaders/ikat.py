"""iKAT 2023/2024 loader (public topics + PTKB qrels, NO passage
corpus needed; this feeds the intrinsic PTKB-reconstruction workstream).

Parses the official topic JSON (github.com/irlabamsterdam/iKAT, as staged into data/ikat/ by
scripts/fetch_free_data.sh) into Dialogue objects with populated ptkb.

Only `utterance` is read for turn text, never `resolved_utterance` — same reason as the CAsT
loader: the harness's own rewrite_v1.txt does the contextual rewriting, and consuming iKAT's
own pre-resolved form would contaminate the B0/B1/S1-S5 comparison.

Deliberately NOT loaded: `response` (the assessor-generated system turn) and
`response_provenance` (a list of clueweb22-en... document ids). Two reasons: (1) neither the
protocol nor the plan asks for system turns on this axis — PTKB reconstruction only needs
utterance + ptkb; (2) response_provenance references the licensed ClueWeb22 corpus by doc id,
and this paper is free-data-only by scope decision — leaving it
out entirely means data/ikat/*.json can never become a vector for licensed-corpus references
leaking into the repo or the LLM cache (a harness invariant). If a later (licensed) workstream
needs it, add it as an explicit opt-in, not a default.

Turn id convention: "<topic_number>-<turn_id>" (e.g. "9-1-2") — note the topic number itself
already contains a hyphen (e.g. "9-1"), so the full id is topic + "-" + turn, NOT topic + "_"
+ turn like CAsT. This is the "(iKAT style)" example already named in memory_base.py's Turn
docstring. Getting the separator wrong silently breaks qrels alignment rather than raising, so
it's covered by a unit test below.
"""
from __future__ import annotations

import json

from ..memory_base import Dialogue, Turn


def _clean(text: str) -> str:
    return " ".join(text.split())


def _ptkb_list(ptkb: dict[str, str]) -> list[str]:
    """iKAT's ptkb is a dict keyed by string statement-number ("1".."n"); sort numerically so
    ordering is deterministic and matches the numbering ptkb_provenance refers to."""
    return [ptkb[k] for k in sorted(ptkb, key=int)]


def _check_topic_number(number: object, index: int) -> None:
    """Guard against a file whose `number` is a positional index rather than a real dialogue
    identifier.

    The 2023 file uses `"9-1"` (persona 9, dialogue 1) — hyphenated, globally meaningful, and
    what the PTKB qrels key on. The 2024 file at
    `irlabamsterdam/iKAT/main/2024/data/2024_test_topics.json` instead numbers its topics
    0,1,2,...,16 as plain ints, and its `title` field is not unique either ("Buying perfume"
    appears at both index 1 and index 3). Turn ids built from that would come out as "0-1",
    "1-1", ... — syntactically fine, silently meaningless, and aligned with no qrels file.

    This module's whole stated hazard is that a wrong id convention breaks qrels alignment
    *without raising*, so it raises here instead of guessing what 2024's real ids should be.
    Resolve by finding the authoritative 2024 topics/qrels pair, then either fix the fetch URL
    or add the correct mapping — do not paper over it downstream. Pass `strict_ids=False` for
    exploratory (non-scoring) work only."""
    if isinstance(number, bool) or not isinstance(number, str) or "-" not in number:
        raise ValueError(
            f"topic at index {index} has number={number!r}, which is not a dialogue "
            f"identifier of the expected '<persona>-<dialogue>' form (e.g. '9-1'). This is "
            f"the known shape of the 2024 topics file, whose `number` is a positional index "
            f"— turn ids derived from it would align with no qrels. Find the authoritative "
            f"2024 topics/qrels pair, or pass strict_ids=False for exploratory work that is "
            f"not scored."
        )


def parse_ikat_topics(raw: list[dict], strict_ids: bool = True) -> list[Dialogue]:
    """raw: the parsed JSON list straight from an official iKAT test-topics file
    (e.g. 2023/data/2023_test_topics.json). One Dialogue per topic.

    strict_ids=True (default) refuses files whose topic numbers are positional indices rather
    than real dialogue ids — see _check_topic_number."""
    dialogues = []
    for index, topic in enumerate(raw):
        number = topic["number"]  # e.g. "9-1" — already hyphenated
        if strict_ids:
            _check_topic_number(number, index)
        ptkb = topic.get("ptkb", {}) or {}
        turns = [
            Turn(
                turn_id=f"{number}-{turn['turn_id']}",
                role="user",
                text=_clean(turn["utterance"]),
            )
            for turn in topic["turns"]
        ]
        dialogues.append(
            Dialogue(
                dialogue_id=str(number),
                turns=turns,
                ptkb=_ptkb_list(ptkb),
                meta={
                    "title": topic.get("title", ""),
                    "ptkb_by_id": dict(ptkb),
                    # per-turn gold PTKB statement numbers actually used — not the text,
                    # and never the clueweb-referencing response_provenance field.
                    "ptkb_provenance": {
                        f"{number}-{turn['turn_id']}": turn.get("ptkb_provenance", [])
                        for turn in topic["turns"]
                    },
                },
            )
        )
    return dialogues


def load_ikat_topics(path: str, strict_ids: bool = True) -> list[Dialogue]:
    """path: an iKAT topic JSON file, e.g. data/ikat/ikat23_topics.json."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return parse_ikat_topics(raw, strict_ids=strict_ids)
