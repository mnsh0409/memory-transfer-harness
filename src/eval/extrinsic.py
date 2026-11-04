"""Extrinsic evaluation: TREC ranking metrics + judged@10.

Two classes of metric live here, and the distinction matters for what may go in the paper:

  * **nDCG@3/5, R@1000, MRR** — standard TREC measures. The authoritative implementation is
    trec_eval (via pytrec_eval, which wraps the actual trec_eval C code). A native
    reimplementation is also provided, but it is NOT the default and NOT paper-grade: see
    the long note on `_native_ndcg_at_k` about gain conventions. `evaluate()` refuses to use
    it unless explicitly asked.
  * **judged@k** — pool-coverage: the mean fraction of a turn's top-k documents that appear
    in qrels *at any grade, including grade 0*. trec_eval does not compute this, so the
    implementation here is authoritative by definition. Grade 0 counting as "judged" is the
    whole point: on CAsT 2019, 72% of judgments are grade 0 (21,230 of 29,350), so
    judged@10 and "relevant@10" are very different numbers. This is the metric the plan
    (§3 "Pooling honesty") wants in every table to pre-empt the TREC-veteran reviewer.

Qrels format note: the standard TREC 4-column layout is `qid iter docid rel`, but the two
collections this paper uses disagree on column 2 — CAsT 2019 writes `Q0` there, CAsT 2020
writes `0`. Both were checked against the real files. `load_qrels` therefore ignores column 2
entirely; do not add validation on it, it will break one year or the other.
"""
from __future__ import annotations

import math
from collections import defaultdict

# Install note: plain `pip install pytrec_eval` builds from source and its setup.py DOWNLOADS
# trec_eval's C sources at build time, which fails on any host that can't reach that URL.
# `pip install pytrec_eval_terrier` ships a prebuilt wheel with trec_eval already bundled and
# provides this same `pytrec_eval` module -- prefer it. (requirements.txt says the same.)
try:
    import pytrec_eval  # type: ignore
    HAVE_PYTREC_EVAL = True
except ImportError:
    HAVE_PYTREC_EVAL = False


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_qrels(path: str) -> dict[str, dict[str, int]]:
    """qid -> {doc_id: relevance}. Includes grade-0 (judged-but-not-relevant) entries —
    they are essential for judged@k and harmless for the relevance measures, which treat
    gain 0 as no gain anyway."""
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 4:
                raise ValueError(f"{path} line {i}: {len(parts)} fields (want 4: qid iter docid rel)")
            qid, _iter, doc_id, rel = parts  # _iter is "Q0" in 2019, "0" in 2020 — ignored
            qrels[qid][doc_id] = int(rel)
    return dict(qrels)


def load_run(path: str) -> dict[str, list[tuple[str, float]]]:
    """qid -> [(doc_id, score), ...] ordered by the run file's own rank column.
    Use harness.trec.validate_run first if the file's provenance is uncertain; this reader
    trusts the rank ordering rather than re-sorting by score."""
    run: dict[str, list[tuple[int, str, float]]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 6:
                raise ValueError(f"{path} line {i}: {len(parts)} fields (want 6)")
            qid, _q0, doc_id, rank, score, _tag = parts
            run[qid].append((int(rank), doc_id, float(score)))
    return {
        qid: [(doc_id, score) for _rank, doc_id, score in sorted(rows)]
        for qid, rows in run.items()
    }


# ---------------------------------------------------------------------------
# judged@k — authoritative here; trec_eval has no equivalent
# ---------------------------------------------------------------------------

def judged_at_k(
    run: dict[str, list[tuple[str, float]]],
    qrels: dict[str, dict[str, int]],
    k: int = 10,
) -> dict[str, float]:
    """Per-qid fraction of the top-k retrieved docs that appear in qrels at ANY grade.

    Denominator is min(k, len(retrieved)) — a turn that returned only 4 documents is scored
    out of 4, not out of 10, so a short result list can't masquerade as poor pool coverage.
    Only qids present in qrels are scored (see `evaluate` for how coverage is reported)."""
    out = {}
    for qid, ranked in run.items():
        if qid not in qrels:
            continue
        judged_docs = qrels[qid]
        top = ranked[:k]
        if not top:
            out[qid] = 0.0
            continue
        out[qid] = sum(1 for doc_id, _ in top if doc_id in judged_docs) / len(top)
    return out


# ---------------------------------------------------------------------------
# Native metric implementations — cross-check / offline path only, NOT paper-grade
# ---------------------------------------------------------------------------

def _native_ndcg_at_k(ranked: list[tuple[str, float]], rels: dict[str, int], k: int,
                      gain: str = "linear") -> float:
    """nDCG@k with discount 1/log2(rank+1).

    GAIN CONVENTION IS SETTLED: **trec_eval's `ndcg_cut` uses LINEAR gain** (gain =
    relevance level), not the exponential 2^rel - 1 form common in learning-to-rank. Verified
    empirically against trec_eval on real CAsT 2019 qrels: linear reproduces trec_eval's
    per-qid nDCG@3/@5 exactly (0 mismatches at 1e-6 tolerance), while exponential diverges on
    every measured pair (qid 31_1 nDCG@3: 0.2050 exponential vs 0.3100 trec_eval). Do not
    "correct" this to the exponential form -- it would silently make every reported nDCG wrong.

    `tests/test_extrinsic.py -k cross_check` re-verifies this against trec_eval on any machine
    where pytrec_eval is installed. Keep it passing."""
    if gain == "linear":
        def g(r: int) -> float:
            return float(r)
    elif gain == "exponential":
        def g(r: int) -> float:
            return float(2 ** r - 1)
    else:
        raise ValueError(f"unknown gain convention {gain!r}")

    dcg = sum(
        g(rels.get(doc_id, 0)) / math.log2(rank + 1)
        for rank, (doc_id, _score) in enumerate(ranked[:k], start=1)
    )
    ideal_rels = sorted(rels.values(), reverse=True)[:k]
    idcg = sum(g(r) / math.log2(rank + 1) for rank, r in enumerate(ideal_rels, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def _native_recall_at_k(ranked: list[tuple[str, float]], rels: dict[str, int], k: int) -> float:
    """Recall over *relevant* docs (grade >= 1); grade-0 judgments are not relevant."""
    relevant = {doc_id for doc_id, r in rels.items() if r >= 1}
    if not relevant:
        return 0.0
    retrieved = {doc_id for doc_id, _ in ranked[:k]}
    return len(relevant & retrieved) / len(relevant)


def _native_reciprocal_rank(ranked: list[tuple[str, float]], rels: dict[str, int]) -> float:
    for rank, (doc_id, _score) in enumerate(ranked, start=1):
        if rels.get(doc_id, 0) >= 1:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Evaluation entry point
# ---------------------------------------------------------------------------

RELEVANCE_MEASURES = ("ndcg_cut_3", "ndcg_cut_5", "recall_1000", "recip_rank")


def _pytrec_eval_metrics(
    run: dict[str, list[tuple[str, float]]],
    qrels: dict[str, dict[str, int]],
) -> dict[str, dict[str, float]]:
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {"ndcg_cut.3,5", "recall.1000", "recip_rank"}
    )
    pt_run = {qid: {doc_id: score for doc_id, score in ranked} for qid, ranked in run.items()}
    return evaluator.evaluate(pt_run)


def _native_metrics(
    run: dict[str, list[tuple[str, float]]],
    qrels: dict[str, dict[str, int]],
    gain: str = "linear",
) -> dict[str, dict[str, float]]:
    out = {}
    for qid, ranked in run.items():
        if qid not in qrels:
            continue
        rels = qrels[qid]
        out[qid] = {
            "ndcg_cut_3": _native_ndcg_at_k(ranked, rels, 3, gain),
            "ndcg_cut_5": _native_ndcg_at_k(ranked, rels, 5, gain),
            "recall_1000": _native_recall_at_k(ranked, rels, 1000),
            "recip_rank": _native_reciprocal_rank(ranked, rels),
        }
    return out


def evaluate(
    run: dict[str, list[tuple[str, float]]],
    qrels: dict[str, dict[str, int]],
    judged_k: int = 10,
    allow_native_metrics: bool = False,
    gain: str = "linear",
) -> dict:
    """Returns {"per_turn": {qid: {...}}, "aggregate": {...}, "coverage": {...}}.

    Relevance measures come from pytrec_eval (== trec_eval) when it is installed. If it is
    not, this raises unless `allow_native_metrics=True`, because silently substituting an
    unvalidated reimplementation is exactly how a wrong number reaches a results table.
    judged@k is always computed here regardless — trec_eval has no equivalent.

    Following trec_eval's default behaviour, relevance measures are computed over qids
    present in BOTH run and qrels. That makes coverage a correctness concern rather than a
    cosmetic one, so the counts are returned in "coverage" and the caller is expected to
    look at them — `n_qrels_only > 0` means judged turns your run never answered, which
    inflates the averages and is precisely the "partial coverage" anomaly the protocol says
    to flag rather than quietly accept."""
    run_qids, qrels_qids = set(run), set(qrels)
    scored_qids = run_qids & qrels_qids

    if HAVE_PYTREC_EVAL:
        per_turn = _pytrec_eval_metrics(run, qrels)
        metrics_source = "pytrec_eval"
    elif allow_native_metrics:
        per_turn = _native_metrics(run, qrels, gain=gain)
        metrics_source = f"native(gain={gain}, UNVALIDATED — not for reported numbers)"
    else:
        raise RuntimeError(
            "pytrec_eval is not installed, so nDCG/R/MRR cannot be computed against the "
            "authoritative trec_eval implementation. Install it (`pip install pytrec_eval`) "
            "for any number that goes in the paper. To compute plumbing-only numbers with "
            "the harness's own unvalidated implementation, pass allow_native_metrics=True "
            "(or --allow-native-metrics) and do not report the result."
        )

    judged = judged_at_k(run, qrels, k=judged_k)
    judged_key = f"judged_{judged_k}"
    for qid in per_turn:
        per_turn[qid][judged_key] = judged.get(qid, 0.0)

    def mean(key: str) -> float:
        vals = [m[key] for m in per_turn.values() if key in m]
        return sum(vals) / len(vals) if vals else 0.0

    aggregate = {k: mean(k) for k in (*RELEVANCE_MEASURES, judged_key)}
    aggregate["n_scored_turns"] = len(per_turn)

    return {
        "per_turn": per_turn,
        "aggregate": aggregate,
        "coverage": {
            "n_run_qids": len(run_qids),
            "n_qrels_qids": len(qrels_qids),
            "n_scored": len(scored_qids),
            # judged turns the run never answered -> silently excluded from the averages
            "n_qrels_only": len(qrels_qids - run_qids),
            # turns the run answered that have no judgments at all -> unscorable
            "n_run_only": len(run_qids - qrels_qids),
        },
        "metrics_source": metrics_source,
    }
