"""Retrieval stage: BM25 top-1000 -> cross-encoder rerank top-100.

BM25: Pyserini's `cast2019` prebuilt Lucene index — this is the combined MS MARCO v1 passage +
TREC CAR v2 collection built for CAsT, and TREC CAsT 2020 reuses the *same* document collection
(only the topics differ), so this one index serves both cast19 and cast20 configs; there is no
separate "cast2020" prebuilt index and there does not need to be.

IMPORTANT — there is no fallback index. The obvious one to reach for is
"msmarco-v1-passage + trec-car merged by score" if `cast2019` is ever unavailable.
Checked against the installed pyserini's prebuilt-index manifest directly
(pyserini.prebuilt_index_info.TF_INDEX_INFO and every other *_INDEX_INFO dict in that module):
there is no "trec-car" (or any TREC CAR-only) prebuilt index registered anywhere. `cast2019`
itself IS present and loads by name, so the common case is fine — but if it ever genuinely
fails, silently retrieving from msmarco-v1-passage alone would mean scoring every condition
against a corpus quietly missing the entire CAR-derived half of the judged pool, which is a
correctness bug, not a resilience feature (same "anomalies are flagged, never silently
retried" rule as everywhere else in this harness). So BM25Stage does NOT auto-degrade to a
smaller corpus; it raises with an explicit message and leaves the fallback as a decision for
whoever's on call, e.g. building a real merged index from raw MS MARCO + TREC CAR v2 downloads.
That's real, non-trivial index-building work — worth knowing about now rather than mid-grid.

Also worth knowing before running this anywhere: `cast2019` is ~19.8 GiB compressed
("size compressed (bytes)": 21266884884 in the manifest) — bigger than the "a few GB" estimate
floated earlier. Whatever machine runs run_retrieval.py needs real headroom for it, not just
disk for the harness itself.

Reranking: naver/trecdl22-crossencoder-debertav3 (the model name pinned in the protocol) over
the BM25 top-100, via sentence_transformers.CrossEncoder; BM25 ranks 101-1000 are kept below
the reranked head so every run file carries 1000 documents per turn and the frozen R@1000 is
the first-stage recall the plan means by it (see rerank_query). Runs on GPU if available, CPU
otherwise (slower, not blocked -- but on CPU deberta-v3-large does ~10 pairs/s, so keep the
cross-encoder on the GPU for anything beyond a smoke test). Not verified against the live HF Hub from this
sandbox — huggingface.co isn't reachable through its network allowlist (curl gets a 403 from
the local proxy, not from HF) — so confirm the model downloads cleanly on whatever machine
actually runs this.

Injection modes (RunConfig.injection, config.py):
  rewrite_only        -- cross-encoder scores (rewritten query, passage) only. Memory has
                          already shaped the query via the frozen rewrite prompt; nothing
                          further changes at this stage.
  rewrite_plus_rerank -- cross-encoder ALSO sees the raw memory_block appended to the query
                          side, so memory statements that didn't survive into the rewritten
                          query text can still influence the rerank directly.
This is this module's interpretation of the injection axis: ARCHITECTURE.md point 7 says only
that "both injection modes reuse the same replay pass," not what distinguishes them
downstream, and no other doc spells it out further. Revisit it before the
first non-rewrite_only run — this is an assumption, not a spec quote, and it's cheap to change
(it's isolated to the four lines in rerank_query below).

Import-guarded like adapters/mem0_adapter.py: pyserini/torch/sentence-transformers are heavy,
optional-until-needed deps, so the harness and its tests import and run without them.
"""
from __future__ import annotations

import json
import os

from .envfile import PYSERINI_API_KEY_PLACEHOLDER
from .replay import RewriteResult

# pyserini.search.lucene, as of 1.6.0, transitively imports pyserini.encode._openai, which
# constructs an `openai.OpenAI()` client at MODULE IMPORT TIME -- so importing pyserini at all
# raises openai.OpenAIError("Missing credentials") in any process that hasn't already exported
# OPENAI_API_KEY, even though this module never calls OpenAI through pyserini. Confirmed by
# direct import in a clean environment. Set a harmless placeholder first so the import can't
# fail on that account; the harness's own OpenAI usage (harness/llm.py) reads the real key from
# .env independently and is unaffected by this.
# The placeholder is shared with envfile.load_dotenv, which treats it as "no key set" so
# that a real key from .env still wins -- this module is imported before .env is loaded.
os.environ.setdefault("OPENAI_API_KEY", PYSERINI_API_KEY_PLACEHOLDER)

try:
    from pyserini.search.lucene import LuceneSearcher
    HAVE_PYSERINI = True
except Exception:  # broad on purpose -- see the OPENAI_API_KEY note above; not just ImportError
    HAVE_PYSERINI = False

try:
    from sentence_transformers import CrossEncoder
    HAVE_CROSS_ENCODER = True
except Exception:
    HAVE_CROSS_ENCODER = False

CE_MODEL = "naver/trecdl22-crossencoder-debertav3"
BM25_INDEX = "cast2019"  # serves cast19 AND cast20 -- same underlying collection, see module docstring
BM25_DEPTH = 1000
RERANK_DEPTH = 100


class BM25Stage:
    """First-stage retrieval over the CAsT collection. One instance per process — loading the
    prebuilt index is the expensive part, so scripts/run_retrieval.py should build one BM25Stage
    and reuse it across every turn/config in a run, not one per query."""

    def __init__(self, index_name: str = BM25_INDEX) -> None:
        if not HAVE_PYSERINI:
            raise RuntimeError("pip install pyserini (needs Java 21) -- see requirements.txt")
        try:
            self._searcher = LuceneSearcher.from_prebuilt_index(index_name)
        except Exception as e:
            raise RuntimeError(
                f"could not load pyserini prebuilt index {index_name!r}. The "
                "msmarco-v1-passage + trec-car fallback does not "
                "exist in the installed pyserini (no 'trec-car' prebuilt index is registered) "
                "-- do not improvise a smaller-corpus substitute, it silently changes what's "
                "being scored. Fix the index before running the grid."
            ) from e

    def search(self, query: str, k: int = BM25_DEPTH) -> list[tuple[str, str, float]]:
        """Returns (doc_id, passage_text, bm25_score) triples, best first.

        Raises if every retrieved document came back with empty text: the cross-encoder
        would then score (query, "") pairs and the rerank stage would be silently
        meaningless -- the "flag, never silently accept" rule applies to the retrieval stage
        too. (Verified Aug 22 2026 on the real `cast2019` index: `doc.contents()` is None
        there and the passage text is in `doc.raw()`; see `passage_text`.)"""
        hits = self._searcher.search(query, k=k)
        out = []
        for h in hits:
            doc = self._searcher.doc(h.docid)
            out.append((h.docid, passage_text(doc), h.score))
        if out and all(not text for _, text, _ in out):
            raise RuntimeError(
                f"BM25 returned {len(out)} hits for {query[:60]!r} but no passage text for any "
                "of them (contents() and raw() both empty) -- the reranker would score empty "
                "passages. Check the index; do not proceed with this run."
            )
        return out


def passage_text(doc) -> str:
    """Passage text for a pyserini Document, whichever field the prebuilt index stored it in.

    The real `cast2019` prebuilt index returns None from `doc.contents()` and the plain passage
    string from `doc.raw()` (checked on the live index, Aug 22 2026 -- `contents()` was None
    for every hit probed, `raw()` was the passage text). Other prebuilt indexes store a JSON
    blob in raw ({"id": ..., "contents": ...}); that case is unwrapped too. Never returns
    None, so callers can test truthiness."""
    if doc is None:
        return ""
    text = doc.contents()
    if text:
        return text
    raw = doc.raw() or ""
    stripped = raw.lstrip()
    if stripped.startswith("{"):
        try:
            obj = json.loads(stripped)
        except ValueError:
            return raw
        if isinstance(obj, dict):
            for key in ("contents", "text", "passage", "body"):
                if obj.get(key):
                    return str(obj[key])
    return raw


class CrossEncoderStage:
    """Wraps the frozen reranker model. One instance per process for the same reason as
    BM25Stage -- model load is the expensive part, .rerank() calls are cheap by comparison."""

    def __init__(self, model_name: str = CE_MODEL, device: str | None = None) -> None:
        if not HAVE_CROSS_ENCODER:
            raise RuntimeError("pip install torch sentence-transformers -- see requirements.txt")
        self._ce = CrossEncoder(model_name, device=device)

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str, float]],
        depth: int = RERANK_DEPTH,
        batch_size: int = 32,
    ) -> list[tuple[str, float]]:
        """candidates: (doc_id, passage_text, bm25_score) from BM25Stage.search.
        Returns (doc_id, ce_score) for the top `depth`, ce_score highest first."""
        if not candidates:
            return []
        doc_ids = [c[0] for c in candidates]
        pairs = [(query, c[1]) for c in candidates]
        scores = self._ce.predict(pairs, batch_size=batch_size)
        ranked = sorted(zip(doc_ids, scores), key=lambda t: t[1], reverse=True)
        return [(doc_id, float(score)) for doc_id, score in ranked[:depth]]


def rerank_query(
    query: str,
    memory_block: str,
    injection: str,
    bm25: BM25Stage,
    ce: CrossEncoderStage,
) -> list[tuple[str, float]]:
    """One rewritten query -> ranked (doc_id, score) list, honoring the injection mode.
    bm25/ce are duck-typed on purpose (need only .search()/.rerank()) so tests can pass fakes
    without touching pyserini or a real model."""
    if injection not in ("rewrite_only", "rewrite_plus_rerank"):
        raise ValueError(f"unknown injection mode {injection!r}")
    candidates = bm25.search(query, k=BM25_DEPTH)
    ce_query = query
    if injection == "rewrite_plus_rerank" and memory_block:
        ce_query = f"{query}\n\nUser context:\n{memory_block}"
    # Cross-encode the BM25 top-RERANK_DEPTH only (the "rerank top-100" of the protocol), then
    # keep BM25 ranks RERANK_DEPTH+1..BM25_DEPTH underneath the reranked head, scored strictly
    # below it, so the run file carries BM25_DEPTH documents per turn. That is what makes the
    # frozen R@1000 the *first-stage* recall the plan defines ("R@1000 (first-stage
    # sensitivity)") -- a 100-document run would silently report R@100 under that name -- and
    # it costs 100 rather than 1000 cross-encoder pairs per turn.
    head, tail = candidates[:RERANK_DEPTH], candidates[RERANK_DEPTH:]
    reranked = ce.rerank(ce_query, head, depth=RERANK_DEPTH)
    floor = min((s for _, s in reranked), default=0.0)
    seen = {doc_id for doc_id, _ in reranked}
    out = list(reranked)
    for i, (doc_id, _text, _bm25_score) in enumerate(tail):
        if doc_id in seen:
            continue
        out.append((doc_id, floor - 1.0 - i * 1e-5))   # strictly decreasing, below the head
    return out


def run_retrieval(
    rewrites: list[RewriteResult],
    injection: str,
    bm25: BM25Stage,
    ce: CrossEncoderStage,
) -> dict[str, list[tuple[str, float]]]:
    """rewrites: everything run_replay produced for one (condition, collection). Returns the
    qid -> ranked-list mapping harness.trec.write_run expects, keyed by RewriteResult.turn_id
    (which IS the qrels qid for a real loader -- see loaders/cast.py, loaders/ikat.py)."""
    return {
        r.turn_id: rerank_query(r.query, r.memory_block, injection, bm25, ce)
        for r in rewrites
    }
