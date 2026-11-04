"""Intrinsic evaluation: PTKB reconstruction (needs NO corpus license).

Recall direction: each gold PTKB statement → is it contained in the exported
memories?  Precision direction: each memory → is it supported by the gold set?
The judge is pluggable: `lexical_judge` (deterministic, for tests/smoke) or an
LLM judge built on the frozen prompts/judge_v1.txt instrument.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass

from harness.llm import LLMBackend
from harness.replay import load_prompt

# judge_fn(reference_statement, candidate_texts) -> verdict in {"ENTAILED","PARTIAL","ABSENT"}
JudgeFn = Callable[[str, list[str]], str]

_STOP = {
    "a", "an", "the", "i", "my", "me", "is", "am", "are", "to", "of", "and", "or",
    "in", "on", "for", "with", "have", "has", "do", "does", "not", "it", "that",
}


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in _STOP}


def lexical_judge(reference: str, candidates: list[str]) -> str:
    """Deterministic fallback: ENTAILED if any candidate covers all content words,
    PARTIAL if any covers half, else ABSENT. For plumbing only — the paper uses
    the LLM judge with a 100-sample human agreement check."""
    ref = _content_words(reference)
    if not ref:
        return "ABSENT"
    best = 0.0
    for c in candidates:
        cw = _content_words(c)
        best = max(best, len(ref & cw) / len(ref))
    if best >= 0.999:
        return "ENTAILED"
    if best >= 0.5:
        return "PARTIAL"
    return "ABSENT"


def _strip_code_fence(raw: str) -> str:
    """Return the JSON body of a ```-fenced reply, or the input unchanged.

    Measured on the cached judge calls: **582 of 1526 (38%) came back fenced**, because the
    model wraps JSON in markdown when not asked for structured output. Stripping the fence
    here rather than passing `response_format` is deliberate -- the request is unchanged, so
    the cache key is unchanged, so no existing judged result is invalidated and no budget is
    re-spent. The frozen prompt is untouched either way."""
    t = raw.strip()
    if not t.startswith("```"):
        return raw
    t = t[3:]
    if t[:4].lower().startswith("json"):
        t = t[4:]
    end = t.rfind("```")
    return (t[:end] if end != -1 else t).strip()


def _parse_verdict(raw: str) -> str:
    """JSON first, then fence-stripped JSON, then a last-resort keyword scan.

    The keyword scan takes the FIRST verdict word anywhere in the reply, so it can in principle
    disagree with the JSON it failed to parse (e.g. "not ENTAILED; verdict ABSENT"). Checked
    against all 1526 cached judge replies: it never disagreed. It stays as a floor, but the two
    parses above it should now handle every real reply."""
    for candidate in (raw, _strip_code_fence(raw)):
        try:
            verdict = json.loads(candidate).get("verdict", "")
        except (json.JSONDecodeError, AttributeError, TypeError):
            continue
        if verdict in ("ENTAILED", "PARTIAL", "ABSENT"):
            return verdict
    m = re.search(r"ENTAILED|PARTIAL|ABSENT", raw)
    return m.group(0) if m else "ABSENT"


def make_llm_judge(backend: LLMBackend, prompt_name: str = "judge_v1") -> JudgeFn:
    template = load_prompt(prompt_name)

    def judge(reference: str, candidates: list[str]) -> str:
        mems = "\n".join(f"[{i}] {c}" for i, c in enumerate(candidates))
        prompt = template.replace("{ptkb_statement}", reference).replace("{memories}", mems)
        raw = backend.complete(prompt)
        verdict = _parse_verdict(raw)
        return verdict if verdict in ("ENTAILED", "PARTIAL", "ABSENT") else "ABSENT"

    return judge


@dataclass(frozen=True)
class ReconstructionScore:
    precision: float
    recall: float
    f1: float
    n_gold: int
    n_memories: int


def score_reconstruction(
    gold: list[str],
    memories: list[str],
    judge: JudgeFn,
    partial_credit: float = 0.5,
    precision_judge: JudgeFn | None = None,
) -> ReconstructionScore:
    """`precision_judge` defaults to `judge` and exists only so a caller can tell the two
    directions apart when recording decisions.

    The two calls ask genuinely different questions — recall asks whether a *gold statement*
    is present in the memories, precision asks whether a *memory* is supported by the gold
    set — but the judge prompt is symmetric, so a decision log that labels both the same way
    is indistinguishable after the fact. That matters because the 100-decision human check is
    drawn from that log and is the only external validation this axis has: an annotator who
    cannot tell which question a row is asking cannot answer it reliably."""
    if not gold and not memories:
        return ReconstructionScore(1.0, 1.0, 1.0, 0, 0)
    credit = {"ENTAILED": 1.0, "PARTIAL": partial_credit, "ABSENT": 0.0}
    pjudge = precision_judge or judge
    recall = (
        sum(credit[judge(g, memories)] for g in gold) / len(gold) if gold else 0.0
    )
    precision = (
        sum(credit[pjudge(m, gold)] for m in memories) / len(memories) if memories else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    )
    return ReconstructionScore(precision, recall, f1, len(gold), len(memories))
