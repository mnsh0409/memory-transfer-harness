"""Driver for the intrinsic axes — the logic behind `scripts/run_intrinsic.py`.

Kept out of scripts/ so it is unit-testable, same split as harness/runners.py.

Two axes, both license-free:

* **Axis 1 — PTKB reconstruction (iKAT 2023).** Replay each persona's dialogue through a
  memory system, dump the store, and judge it against the gold PTKB in both directions →
  per-dialogue precision / recall / F1. `eval/intrinsic.py` already owns the scoring and the
  frozen judge (`prompts/judge_v1.txt`); this module is the driver, the CSV emission, and the
  stratified sample for the human check.
* **Axis 2 — LongMemEval recall.** Replay each question's `_S` haystack sessions, query with
  the question, and measure whether the retrieved memories carry the labelled evidence →
  per-question CSV.

Three decisions worth knowing before reading a number out of these CSVs:

1. **No rewrite call.** This replay does `query()` strictly before `ingest()` — the protocol
   the pins in requirements.lock — but does **not** invoke the rewrite prompt: what a system *stored* does
   not depend on how the query was later rewritten, and skipping it saves the entire rewrite
   budget on an axis that never uses the result. The turn sequence a memory system sees is
   identical to the extrinsic run.
2. **`ptkb_provenance` is the memory-retrieval qrels.** iKAT 2023's per-turn
   `ptkb_provenance` lists the gold PTKB statement numbers relevant to that turn, which is
   exactly the per-turn relevance signal memory-retrieval nDCG@3 needs. No
   separate qrels file is needed, and none exists.
3. **The 100-decision sample is written, never scored.** `manual_check/` gets a stratified
   sample of judge decisions with an empty `human_verdict` column. Agreement is a human
   number; a session that filled it in would be fabricating the paper's validity evidence.
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
from dataclasses import dataclass

from harness.memory_base import Dialogue, MemorySystem

from .intrinsic import JudgeFn, score_reconstruction

VERDICTS = ("ENTAILED", "PARTIAL", "ABSENT")
MANUAL_CHECK_N = 100


# ---------------------------------------------------------------------------
# Replay (query strictly before ingest; no rewrite call -- see module docstring)
# ---------------------------------------------------------------------------

@dataclass
class TurnRetrieval:
    turn_id: str
    utterance: str
    entry_texts: list[str]
    entry_ids: list[str]


def replay_for_memory(dialogue: Dialogue, memory: MemorySystem,
                      memory_k: int = 5) -> list[TurnRetrieval]:
    """Feed a dialogue through a memory system, capturing what each user turn retrieved.

    Mirrors `harness.replay.replay_dialogue`'s protocol exactly — query() before ingest() of
    the current turn, per-dialogue isolation via reset() — minus the rewrite LLM call."""
    memory.reset(dialogue.dialogue_id)
    out: list[TurnRetrieval] = []
    for turn in dialogue.turns:
        if turn.role == "user":
            entries = memory.query(dialogue.dialogue_id, turn.text, k=memory_k)
            out.append(TurnRetrieval(
                turn_id=turn.turn_id,
                utterance=turn.text,
                entry_texts=[e.text for e in entries],
                entry_ids=[e.entry_id for e in entries],
            ))
        memory.ingest_turn(dialogue.dialogue_id, turn)
    return out


# ---------------------------------------------------------------------------
# Memory-retrieval nDCG@k against ptkb_provenance
# ---------------------------------------------------------------------------

def memory_ndcg_at_k(entry_texts: list[str], relevant_statements: list[str],
                     judge: JudgeFn, k: int = 3) -> float | None:
    """nDCG@k over retrieved memory entries, where an entry is relevant if the judge says it
    carries one of the turn's gold PTKB statements.

    Graded like the reconstruction scorer: ENTAILED = 1, PARTIAL = 0.5, ABSENT = 0. Linear
    gain and 1/log2(rank+1) discount, matching the extrinsic side (trec_eval
    uses linear gain — the two axes must not disagree on this or the RQ4 correlation compares
    incompatible scales).

    Returns None when the turn has no gold statements: that is "not applicable", not zero, and
    averaging zeros over unlabelled turns would silently depress every system."""
    if not relevant_statements:
        return None
    credit = {"ENTAILED": 1.0, "PARTIAL": 0.5, "ABSENT": 0.0}
    gains = [credit[judge(text, relevant_statements)] for text in entry_texts[:k]]
    dcg = sum(g / math.log2(rank + 1) for rank, g in enumerate(gains, start=1))
    ideal = sorted([1.0] * len(relevant_statements), reverse=True)[:k]
    idcg = sum(g / math.log2(rank + 1) for rank, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else None


# ---------------------------------------------------------------------------
# Axis 1 — PTKB reconstruction
# ---------------------------------------------------------------------------

def _judge_recording(judge: JudgeFn, log: list[dict], direction: str) -> JudgeFn:
    """Wrap a judge so every decision is captured for the manual-check sample."""
    def wrapped(reference: str, candidates: list[str]) -> str:
        verdict = judge(reference, candidates)
        log.append({"direction": direction, "reference": reference,
                    "candidates": list(candidates), "verdict": verdict})
        return verdict
    return wrapped


def run_ptkb_reconstruction(dialogues: list[Dialogue], memory: MemorySystem, judge: JudgeFn,
                            memory_k: int = 5, ndcg_k: int = 3) -> dict:
    """Per-dialogue P/R/F1 plus memory-retrieval nDCG@k. Returns rows + the judge decision log."""
    rows, decisions = [], []
    for dialogue in dialogues:
        retrievals = replay_for_memory(dialogue, memory, memory_k=memory_k)
        memories = [e.text for e in memory.export_store(dialogue.dialogue_id)]

        recon_log: list[dict] = []
        # The two directions are labelled separately: the judge prompt is symmetric, so
        # without this a human annotating the sample cannot tell which question a row asks
        # (see score_reconstruction). Recall rows read gold -> memories; precision rows read
        # memory -> gold, which is why a precision row's "reference" is a stored memory.
        score = score_reconstruction(
            dialogue.ptkb, memories,
            _judge_recording(judge, recon_log, "recall (is this gold PTKB in the memories?)"),
            precision_judge=_judge_recording(
                judge, recon_log, "precision (is this memory supported by the gold PTKB?)"),
        )
        for d in recon_log:
            d["dialogue_id"] = dialogue.dialogue_id
        decisions.extend(recon_log)

        # memory-retrieval nDCG@k, averaged over turns that carry provenance labels
        ptkb_by_id = dialogue.meta.get("ptkb_by_id", {}) or {}
        provenance = dialogue.meta.get("ptkb_provenance", {}) or {}
        per_turn = []
        for r in retrievals:
            gold_ids = provenance.get(r.turn_id) or []
            gold = [ptkb_by_id[str(i)] for i in gold_ids if str(i) in ptkb_by_id]
            value = memory_ndcg_at_k(r.entry_texts, gold, judge, k=ndcg_k)
            if value is not None:
                per_turn.append(value)
        ndcg = sum(per_turn) / len(per_turn) if per_turn else None

        rows.append({
            "dialogue_id": dialogue.dialogue_id,
            "n_gold_ptkb": score.n_gold,
            "n_memories": score.n_memories,
            "precision": score.precision,
            "recall": score.recall,
            "f1": score.f1,
            f"memory_ndcg_{ndcg_k}": ndcg,
            "n_turns_with_provenance": len(per_turn),
            "n_user_turns": len(retrievals),
        })
    return {"rows": rows, "decisions": decisions}


# ---------------------------------------------------------------------------
# Axis 2 — LongMemEval recall
# ---------------------------------------------------------------------------

def load_longmemeval(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def stratified_question_sample(questions: list[dict], n: int,
                               key: str = "question_type") -> list[dict]:
    """A proportional, deterministic sample across `key` -- for LongMemEval, question type.

    **Taking the first N would be badly biased**: the released `_S` file is *sorted by question
    type*, so `questions[:30]` is 30 `single-session-user` items and none of the other five
    types. A recall number from that prefix would describe the easiest slice of the benchmark
    while looking like it described the benchmark. The full 500 questions carry 246,750 haystack
    turns, and a per-turn-ingesting system makes one model call per turn, so sampling is a
    budget necessity rather than a preference -- which is exactly when a biased sample is most
    tempting.

    Proportional: each type contributes round(n * share), at least one, capped at what exists.
    Deterministic: types in first-appearance order, items in file order, no RNG -- so the
    sample is reproducible from the file alone and needs no seed recorded.
    """
    if n <= 0 or n >= len(questions):
        return list(questions)
    groups: dict[str, list[dict]] = {}
    for q in questions:
        groups.setdefault(str(q.get(key, "")), []).append(q)

    total = len(questions)
    quota = {k: max(1, round(n * len(v) / total)) for k, v in groups.items()}
    # round() can overshoot or undershoot n; settle the difference on the largest strata first
    order = sorted(groups, key=lambda k: (-len(groups[k]), k))
    while sum(quota.values()) > n:
        for k in order:
            if sum(quota.values()) <= n:
                break
            if quota[k] > 1:
                quota[k] -= 1
    while sum(quota.values()) < n:
        for k in order:
            if sum(quota.values()) >= n:
                break
            if quota[k] < len(groups[k]):
                quota[k] += 1

    picked = [q for k in groups for q in groups[k][: quota[k]]]
    # restore original file order so downstream ids stay comparable to an unsampled run
    index = {id(q): i for i, q in enumerate(questions)}
    return sorted(picked, key=lambda q: index[id(q)])


def longmemeval_dialogue(question: dict) -> Dialogue:
    """One LongMemEval question's haystack sessions → one Dialogue.

    Sessions are concatenated in the order the benchmark gives them; each message becomes a
    Turn whose id encodes session and position, so evidence labels stay traceable."""
    from harness.memory_base import Turn

    qid = str(question.get("question_id", "q"))
    session_ids = question.get("haystack_session_ids") or []
    sessions = question.get("haystack_sessions") or []
    turns = []
    for s_index, session in enumerate(sessions):
        s_id = str(session_ids[s_index]) if s_index < len(session_ids) else str(s_index)
        for m_index, message in enumerate(session or []):
            role = "user" if message.get("role") == "user" else "system"
            text = (message.get("content") or "").strip()
            if not text:
                continue
            turns.append(Turn(turn_id=f"{qid}:{s_id}:{m_index}", role=role, text=text))
    return Dialogue(dialogue_id=qid, turns=turns, ptkb=[], meta={"question": question})


def _is_evidence(message: dict) -> bool:
    """`has_answer` is the STRING "True"/"False" in the released files, not a bool.

    Verified on longmemeval_oracle.json: 896 messages carry "True" and **10,064 carry
    "False"** — and `bool("False")` is True in Python. A truthiness check therefore marks
    every message as evidence and makes recall ~1.0 for every system, which would look like a
    result rather than a bug. Parse it explicitly, and treat a missing key as not-evidence."""
    value = message.get("has_answer")
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def evidence_texts(question: dict) -> list[str]:
    """The labelled evidence for a question: exactly the messages flagged `has_answer`.

    There is deliberately no "everything in an answer session" fallback — answer sessions
    contain mostly non-evidence turns (a typical session is 12 messages with 2 flagged), so
    such a fallback would inflate the denominator ~6x. 21 of the 500 oracle questions carry no
    flagged message at all; those score `evidence_recall = None` (not applicable), never 0."""
    out = []
    for session in question.get("haystack_sessions") or []:
        for message in session or []:
            if not isinstance(message, dict):
                continue
            text = (message.get("content") or "").strip()
            if text and _is_evidence(message):
                out.append(text)
    return out


def run_longmemeval_recall(questions: list[dict], memory: MemorySystem, judge: JudgeFn,
                           memory_k: int = 5) -> dict:
    """Per-question retrieved-memory hit rate against the labelled evidence."""
    rows, decisions = [], []
    for question in questions:
        dialogue = longmemeval_dialogue(question)
        replay_for_memory(dialogue, memory, memory_k=memory_k)
        query_text = question.get("question") or ""
        entries = memory.query(dialogue.dialogue_id, query_text, k=memory_k)
        retrieved = [e.text for e in entries]
        gold = evidence_texts(question)

        log: list[dict] = []
        recording = _judge_recording(judge, log, "longmemeval")
        hits = [recording(g, retrieved) for g in gold] if gold else []
        for d in log:
            d["dialogue_id"] = dialogue.dialogue_id
        decisions.extend(log)

        n_hit = sum(1 for v in hits if v in ("ENTAILED", "PARTIAL"))
        rows.append({
            "question_id": dialogue.dialogue_id,
            "question_type": question.get("question_type", ""),
            "n_evidence": len(gold),
            "n_retrieved": len(retrieved),
            "n_evidence_hit": n_hit,
            "evidence_recall": (n_hit / len(gold)) if gold else None,
            "n_sessions": len(question.get("haystack_sessions") or []),
        })
    return {"rows": rows, "decisions": decisions}


# ---------------------------------------------------------------------------
# The human check — written, never scored
# ---------------------------------------------------------------------------

def stratified_sample(decisions: list[dict], n: int = MANUAL_CHECK_N,
                      seed: int = 20260822) -> list[dict]:
    """Stratified by verdict so rare classes are actually inspected: an unstratified sample of
    a store dominated by ABSENT would give a human almost nothing to disagree with.
    Deterministic (fixed seed) so the file can be regenerated identically."""
    rng = random.Random(seed)
    by_verdict: dict[str, list[dict]] = {v: [] for v in VERDICTS}
    for d in decisions:
        by_verdict.setdefault(d["verdict"], []).append(d)
    present = [v for v in by_verdict if by_verdict[v]]
    if not present:
        return []
    per = max(1, n // len(present))
    sample: list[dict] = []
    for verdict in present:
        pool = by_verdict[verdict]
        sample.extend(rng.sample(pool, min(per, len(pool))))
    # top up from everything still unused, so a thin stratum doesn't shrink the sample
    if len(sample) < n:
        chosen = {id(d) for d in sample}
        rest = [d for d in decisions if id(d) not in chosen]
        rng.shuffle(rest)
        sample.extend(rest[: n - len(sample)])
    return sample[:n]


def write_manual_check(path: str, decisions: list[dict], n: int = MANUAL_CHECK_N) -> int:
    """Write the stratified sample with an EMPTY `human_verdict` column.

    The agreement number this file exists to produce is a human's; nothing in this repo may
    fill that column in."""
    sample = stratified_sample(decisions, n=n)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["idx", "dialogue_id", "direction", "reference",
                                          "candidates", "judge_verdict", "human_verdict",
                                          "note"])
        w.writeheader()
        for i, d in enumerate(sample):
            w.writerow({
                "idx": i,
                "dialogue_id": d.get("dialogue_id", ""),
                "direction": d.get("direction", ""),
                "reference": d["reference"],
                "candidates": " | ".join(d["candidates"]),
                "judge_verdict": d["verdict"],
                "human_verdict": "",        # HUMAN FILLS THIS IN. Never a session.
                "note": "",
            })
    return len(sample)


def write_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def summarise(rows: list[dict], keys: list[str]) -> dict:
    """Mean of each key over rows where it is not None (None = not applicable, never 0)."""
    out = {}
    for key in keys:
        vals = [r[key] for r in rows if r.get(key) is not None]
        out[f"mean_{key}"] = (sum(vals) / len(vals)) if vals else None
        out[f"n_{key}"] = len(vals)
    return out
