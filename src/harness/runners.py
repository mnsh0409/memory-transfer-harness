"""Orchestration for the Fixed CLI Contract entry points. Kept out of
scripts/ so it's directly unit-testable (scripts/run_replay.py and scripts/run_retrieval.py
are thin argparse shims that call straight into here) -- same "real logic lives in src/harness,
scripts/ stays thin" split as the rest of this repo (README.md describes scripts/ as just
"commands you run").
"""
from __future__ import annotations

import json
import os
from dataclasses import replace

from eval.extrinsic import evaluate, load_qrels, load_run

from .config import RunConfig
from .llm import LLMBackend
from .llmcache import JSONLCache
from .manifest import write_manifest
from .registry import load_dialogues, make_memory_system
from .replay import read_rewrites, replay_dialogue, write_rewrites
from .retrieval import BM25Stage, CrossEncoderStage, run_retrieval
from .trec import write_run

DEFAULT_CACHE_PATH = os.path.join("runs", "llm_cache.jsonl")

# collection -> qrels file, relative to data_dir. Paths match what registry.COLLECTION_PATHS
# expects for topics, so scripts/fetch_free_data.sh populates one directory per collection.
QRELS_PATHS = {
    "cast19": os.path.join("cast", "cast19_qrels.txt"),
    "cast20": os.path.join("cast", "cast20_qrels.txt"),
    "ikat23": os.path.join("ikat", "ikat23_qrels.txt"),
    "ikat24": os.path.join("ikat", "ikat24_qrels.txt"),
}


def resolve_model(config: RunConfig) -> RunConfig:
    """RunConfig.model defaults to the literal sentinel "PINNED_IN_DOTENV" -- swap in
    EXPERIMENT_MODEL from the environment (.env, loaded by the caller) so every config file
    doesn't have to hardcode the snapshot id and risk drifting from what's actually pinned."""
    if config.model != "PINNED_IN_DOTENV":
        return config
    resolved = os.environ.get("EXPERIMENT_MODEL")
    if not resolved:
        raise RuntimeError(
            "config.model is the PINNED_IN_DOTENV placeholder but EXPERIMENT_MODEL isn't set "
            "in the environment -- check .env (see README, Re-running the experiments)"
        )
    return replace(config, model=resolved)


def run_replay_config(
    config: RunConfig,
    data_dir: str = "data",
    runs_dir: str = "runs",
    cache_path: str = DEFAULT_CACHE_PATH,
) -> dict:
    """dialogues -> rewrites (cached). Returns a small summary dict; writes
    {runs_dir}/rewrites/{run_id}.jsonl and {runs_dir}/manifests/{run_id}.json."""
    config = resolve_model(config)
    dialogues = load_dialogues(config.collection, data_dir=data_dir)
    cache = JSONLCache(cache_path)
    backend = LLMBackend(config.backend, config.model, cache)
    # The same backend the rewrite step uses, handed to adapters whose upstream library would
    # otherwise open its own uncached, non-temp-0 connection (S3 A-Mem's evolution step, S5
    # GAM's ingest and query). One backend => one cache => the cache is the whole spend
    # ledger for the run, and re-runs are free for adapter calls too.
    memory = make_memory_system(config.condition, config.ablation, backend=backend)

    all_results = []
    for dialogue in dialogues:
        memory.reset(dialogue.dialogue_id)  # clean slate even if the adapter persists to disk
        all_results.extend(
            replay_dialogue(
                dialogue, memory, backend,
                rewrite_prompt_name=config.rewrite_prompt,
                memory_k=config.memory_k,
                context=config.context,
            )
        )

    rewrites_path = os.path.join(runs_dir, "rewrites", f"{config.run_id}.jsonl")
    write_rewrites(rewrites_path, all_results)

    manifest = {
        "run_id": config.run_id,
        "stage": "run_replay",
        "config": config.canonical_json(),
        "n_dialogues": len(dialogues),
        "n_rewrites": len(all_results),
        "cache_stats": cache.stats(),
        # RQ1's cost axis. Covers the rewrite calls AND every adapter call routed through this
        # same backend (A-Mem evolution, GAM, Graphiti) -- i.e. everything except Mem0's own
        # internal client, which cannot be routed through the harness.
        "usage": backend.usage_summary(),
        "rewrites_path": rewrites_path,
    }
    write_manifest(os.path.join(runs_dir, "manifests", f"{config.run_id}.replay.json"), manifest)
    return manifest


def run_retrieval_config(
    config: RunConfig,
    runs_dir: str = "runs",
    bm25: object | None = None,
    ce: object | None = None,
) -> dict:
    """rewrites -> TREC run file (BM25 -> cross-encoder). Requires run_replay_config to have
    already produced {runs_dir}/rewrites/{run_id}.jsonl for this exact config.

    bm25/ce accept pre-built BM25Stage/CrossEncoderStage-shaped objects (duck-typed -- see
    retrieval.py) so tests can inject fakes instead of loading the real ~20GiB index and
    downloading a model; scripts/run_retrieval.py leaves both as None and gets the real thing.
    """
    # The resolved model is part of run identity, so EVERY stage must resolve before computing
    # run_id -- otherwise this stage looks for `<hash of PINNED_IN_DOTENV>.jsonl` while replay
    # wrote `<hash of the real snapshot>.jsonl` and the run dies between stages.
    config = resolve_model(config)
    rewrites_path = os.path.join(runs_dir, "rewrites", f"{config.run_id}.jsonl")
    if not os.path.exists(rewrites_path):
        raise FileNotFoundError(
            f"{rewrites_path} not found -- run scripts/run_replay.py with the same config first"
        )
    rewrites = read_rewrites(rewrites_path)

    if bm25 is None:
        bm25 = BM25Stage()
    if ce is None:
        ce = CrossEncoderStage()

    results = run_retrieval(rewrites, config.injection, bm25, ce)

    run_path = os.path.join(runs_dir, "trec", f"{config.run_id}.run")
    write_run(run_path, results, tag=config.run_id)

    manifest = {
        "run_id": config.run_id,
        "stage": "run_retrieval",
        "config": config.canonical_json(),
        "n_queries": len(rewrites),
        "run_path": run_path,
    }
    write_manifest(
        os.path.join(runs_dir, "manifests", f"{config.run_id}.retrieval.json"), manifest
    )
    return manifest


def _replay_usage(config: RunConfig, runs_dir: str) -> dict | None:
    """Token usage recorded by the replay stage for this exact run_id, or None if that
    manifest predates token accounting. Deliberately not recomputed here: scoring must not
    depend on the LLM backend."""
    path = os.path.join(runs_dir, "manifests", f"{config.run_id}.replay.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("usage")
    except (OSError, ValueError):
        return None


def run_eval_config(
    config: RunConfig,
    data_dir: str = "data",
    runs_dir: str = "runs",
    judged_k: int = 10,
    allow_native_metrics: bool = False,
) -> dict:
    """TREC run file + qrels -> metrics incl. judged@k. Writes a per-turn JSONL alongside the
    aggregate row, and appends that row to {runs_dir}/stats_summary.json.

    stats_summary.json is the one place in this harness that is deliberately NOT append-only:
    it's an accumulating index keyed by run_id, and collect_stats/figures read it as a whole.
    Re-scoring an existing run_id overwrites just that key (scoring is a pure function of the
    run file + qrels, so this can't lose information the way overwriting a run file would).
    The per-turn JSONL is append-only like every other run artifact."""
    config = resolve_model(config)   # same run-identity reason as run_retrieval_config
    if config.collection not in QRELS_PATHS:
        raise ValueError(
            f"no qrels registered for collection {config.collection!r} "
            f"(known: {', '.join(sorted(QRELS_PATHS))})"
        )
    qrels_path = os.path.join(data_dir, QRELS_PATHS[config.collection])
    if not os.path.exists(qrels_path):
        raise FileNotFoundError(
            f"{qrels_path} not found -- run scripts/fetch_free_data.sh first"
        )
    run_path = os.path.join(runs_dir, "trec", f"{config.run_id}.run")
    if not os.path.exists(run_path):
        raise FileNotFoundError(
            f"{run_path} not found -- run scripts/run_retrieval.py with the same config first"
        )

    results = evaluate(
        load_run(run_path), load_qrels(qrels_path),
        judged_k=judged_k, allow_native_metrics=allow_native_metrics,
    )

    per_turn_path = os.path.join(runs_dir, "metrics", f"{config.run_id}.per_turn.jsonl")
    if os.path.exists(per_turn_path):
        raise FileExistsError(
            f"{per_turn_path} exists -- run artifacts are append-only; new config -> new run_id"
        )
    os.makedirs(os.path.dirname(per_turn_path), exist_ok=True)
    with open(per_turn_path, "w", encoding="utf-8") as f:
        for qid in sorted(results["per_turn"]):
            f.write(json.dumps({"turn_id": qid, **results["per_turn"][qid]},
                               ensure_ascii=False) + "\n")

    row = {
        "run_id": config.run_id,
        "condition": config.condition,
        "collection": config.collection,
        "injection": config.injection,
        "ablation": dict(sorted(config.ablation.items())),
        "memory_k": config.memory_k,   # collect_stats needs it to tell the k-sweep apart
        # which channels filled the prompt (config.CONTEXTS). Without it, runs from two
        # different designs would be indistinguishable in the tables.
        "context": config.context,
        # what this condition cost to build, carried from the replay manifest so the cost
        # figure reads one file (RQ1). None for runs replayed before token accounting existed.
        "usage": _replay_usage(config, runs_dir),
        **results["aggregate"],
        "coverage": results["coverage"],
        "metrics_source": results["metrics_source"],
    }
    stats_path = os.path.join(runs_dir, "stats_summary.json")
    stats = {}
    if os.path.exists(stats_path):
        with open(stats_path, encoding="utf-8") as f:
            stats = json.load(f)
    stats[config.run_id] = row
    os.makedirs(os.path.dirname(stats_path) or ".", exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")

    manifest = {
        "run_id": config.run_id,
        "stage": "run_eval",
        "config": config.canonical_json(),
        "per_turn_path": per_turn_path,
        "stats_path": stats_path,
        **row,
    }
    write_manifest(os.path.join(runs_dir, "manifests", f"{config.run_id}.eval.json"), manifest)
    return manifest
