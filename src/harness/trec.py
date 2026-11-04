"""TREC run file writing + validation (fail fast: a malformed run discovered at
scoring time costs a day; discovered at write time it costs a second)."""
from __future__ import annotations

import os


def write_run(
    path: str,
    results: dict[str, list[tuple[str, float]]],
    tag: str,
    depth: int = 1000,
) -> None:
    """results: qid -> [(doc_id, score), ...] any order; written sorted by score desc."""
    if os.path.exists(path):
        raise FileExistsError(
            f"{path} exists — run artifacts are append-only; new config → new run_id"
        )
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for qid, ranked in results.items():
            top = sorted(ranked, key=lambda p: p[1], reverse=True)[:depth]
            for rank, (doc_id, score) in enumerate(top, start=1):
                f.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} {tag}\n")


def validate_run(path: str, expected_qids: set[str] | None = None) -> list[str]:
    """Returns a list of problems (empty = valid)."""
    problems: list[str] = []
    seen: dict[str, list[tuple[int, float, str]]] = {}
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            parts = line.split()
            if len(parts) != 6:
                problems.append(f"line {i}: {len(parts)} fields (want 6)")
                continue
            qid, q0, doc_id, rank_s, score_s, _tag = parts
            if q0 != "Q0":
                problems.append(f"line {i}: field 2 is {q0!r}, want 'Q0'")
            try:
                rank, score = int(rank_s), float(score_s)
            except ValueError:
                problems.append(f"line {i}: non-numeric rank/score")
                continue
            seen.setdefault(qid, []).append((rank, score, doc_id))
    for qid, rows in seen.items():
        ranks = [r for r, _, _ in rows]
        if ranks != list(range(1, len(rows) + 1)):
            problems.append(f"{qid}: ranks not contiguous from 1")
        scores = [s for _, s, _ in rows]
        if any(a < b for a, b in zip(scores, scores[1:])):
            problems.append(f"{qid}: scores not non-increasing")
        docs = [d for _, _, d in rows]
        if len(docs) != len(set(docs)):
            problems.append(f"{qid}: duplicate doc ids")
    if expected_qids is not None:
        missing = expected_qids - set(seen)
        if missing:
            problems.append(f"missing qids: {sorted(missing)[:5]}{'…' if len(missing) > 5 else ''}")
    return problems
