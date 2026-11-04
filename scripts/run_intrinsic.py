#!/usr/bin/env python3
"""Intrinsic axes: PTKB reconstruction (iKAT 2023) and LongMemEval recall.
Fixed CLI contract:

    python -m scripts.run_intrinsic --config CFG.json

Thin shim: real logic is eval.intrinsic_runner. The axis is chosen by the config's
collection — `ikat23` → PTKB reconstruction, `longmemeval` → evidence recall.

Writes runs/intrinsic/{run_id}.*.csv, a manifest, a row in runs/intrinsic_summary.json, and
manual_check/{run_id}.csv — the 100-decision stratified sample whose `human_verdict` column a
PERSON fills in. Nothing here ever writes that column: it validates the LLM judge, and so it
validates this whole axis.

Uses the frozen judge (prompts/judge_v1.txt) on EXPERIMENT_MODEL by default. --lexical-judge
swaps in the deterministic offline judge for plumbing runs; its numbers are labelled
UNVALIDATED in the summary and must not be reported.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eval.intrinsic import lexical_judge, make_llm_judge  # noqa: E402
from eval.intrinsic_runner import (MANUAL_CHECK_N, load_longmemeval,  # noqa: E402
                                   run_longmemeval_recall, run_ptkb_reconstruction,
                                   stratified_question_sample, summarise, write_csv,
                                   write_manual_check)
from harness.config import RunConfig  # noqa: E402
from harness.envfile import load_dotenv  # noqa: E402
from harness.llm import LLMBackend  # noqa: E402
from harness.llmcache import JSONLCache  # noqa: E402
from harness.manifest import write_manifest  # noqa: E402
from harness.registry import load_dialogues, make_memory_system  # noqa: E402
from harness.runners import DEFAULT_CACHE_PATH, resolve_model  # noqa: E402

LONGMEMEVAL_S = os.path.join("longmemeval", "data", "longmemeval_s_cleaned.json")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="path to a RunConfig JSON file")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--env", default=".env")
    p.add_argument("--cache", default=DEFAULT_CACHE_PATH)
    p.add_argument("--manual-check-dir", default="manual_check")
    p.add_argument("--manual-check-n", type=int, default=MANUAL_CHECK_N)
    p.add_argument("--ndcg-k", type=int, default=3)
    p.add_argument("--limit", type=int, default=None,
                   help="score only the first N dialogues/questions (smoke runs)")
    p.add_argument("--regenerate-outputs", action="store_true",
                   help="re-render the per-unit CSV and manual-check sample for an already "
                        "scored config from its cached decisions. Refuses if any metric "
                        "moved, and never rewrites the manifest.")
    p.add_argument("--lexical-judge", action="store_true",
                   help="deterministic offline judge instead of the frozen LLM instrument. "
                        "Plumbing only -- output is labelled UNVALIDATED.")
    p.add_argument("--longmemeval-file", default=LONGMEMEVAL_S,
                   help="relative to --data-dir")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    load_dotenv(args.env)
    with open(args.config, encoding="utf-8") as f:
        config = RunConfig(**json.load(f))

    # Check the append-only guard BEFORE doing any work. It used to be enforced only by
    # write_manifest at the very end, by which point the per-unit CSV, the manual-check sample
    # and the summary row had already been overwritten -- so a re-run left outputs and manifest
    # describing different executions, and exited non-zero as if nothing had happened.
    manifest_path = os.path.join(args.runs_dir, "manifests", f"{config.run_id}.intrinsic.json")
    if os.path.exists(manifest_path) and not args.regenerate_outputs:
        raise SystemExit(
            f"{manifest_path} exists -- this config has been scored and artifacts are "
            f"append-only. A changed config produces a new run_id. If you only need to "
            f"re-render the derived outputs (per-unit CSV, manual-check sample) from the same "
            f"cached decisions, pass --regenerate-outputs: it refuses if any metric moved."
        )

    cache = JSONLCache(args.cache)
    if args.lexical_judge:
        judge, judge_source = lexical_judge, "lexical(UNVALIDATED -- not for reported numbers)"
        backend = LLMBackend(config.backend, config.model if config.model != "PINNED_IN_DOTENV"
                             else "unresolved", cache)
    else:
        config = resolve_model(config)
        backend = LLMBackend(config.backend, config.model, cache)
        judge, judge_source = make_llm_judge(backend), f"judge_v1@{config.model}"

    memory = make_memory_system(config.condition, config.ablation, backend=backend)

    if config.collection == "longmemeval":
        path = os.path.join(args.data_dir, args.longmemeval_file)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found -- LongMemEval's _S file is downloaded separately (see its "
                "README: huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)"
            )
        questions = load_longmemeval(path)
        if args.limit:
            # NOT questions[:N]: the released _S file is sorted by question_type, so a prefix
            # is one type only. See eval.intrinsic_runner.stratified_question_sample.
            before = len(questions)
            questions = stratified_question_sample(questions, args.limit)
            kinds = sorted({str(q.get("question_type", "")) for q in questions})
            print(f"SAMPLED {len(questions)} of {before} questions, stratified by "
                  f"question_type over {len(kinds)} types: {', '.join(kinds)}")
        result = run_longmemeval_recall(questions, memory, judge, memory_k=config.memory_k)
        axis, metric_keys = "longmemeval_recall", ["evidence_recall"]
        csv_path = os.path.join(args.runs_dir, "intrinsic", f"{config.run_id}.per_question.csv")
        n_units = len(questions)
    else:
        dialogues = load_dialogues(config.collection, data_dir=args.data_dir)
        if args.limit:
            dialogues = dialogues[: args.limit]
        result = run_ptkb_reconstruction(dialogues, memory, judge, memory_k=config.memory_k,
                                         ndcg_k=args.ndcg_k)
        axis = "ptkb_reconstruction"
        metric_keys = ["precision", "recall", "f1", f"memory_ndcg_{args.ndcg_k}"]
        csv_path = os.path.join(args.runs_dir, "intrinsic", f"{config.run_id}.per_dialogue.csv")
        n_units = len(dialogues)

    rows = result["rows"]
    write_csv(csv_path, rows)

    manual_path = os.path.join(args.manual_check_dir, f"{config.run_id}.csv")
    n_sampled = write_manual_check(manual_path, result["decisions"], n=args.manual_check_n)

    row = {
        "run_id": config.run_id, "axis": axis, "condition": config.condition,
        "collection": config.collection, "ablation": dict(sorted(config.ablation.items())),
        "memory_k": config.memory_k, "n_units": n_units,
        "n_judge_decisions": len(result["decisions"]),
        "judge_source": judge_source, **summarise(rows, metric_keys),
    }
    summary_path = os.path.join(args.runs_dir, "intrinsic_summary.json")
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)

    if args.regenerate_outputs:
        # Re-rendering an export is not re-scoring a run, but that is only true if the numbers
        # are identical. Verify it rather than assert it: any drift means the config is not
        # really the one that was scored, and the append-only rule should bite after all.
        old = summary.get(config.run_id, {})
        moved = {k: (old.get(k), row.get(k)) for k in row
                 if k.startswith(("mean_", "n_")) and old.get(k) != row.get(k)}
        if old and moved:
            raise SystemExit(
                f"--regenerate-outputs refuses: this is not a re-render, the numbers moved: "
                f"{moved}. That means the scored configuration and this one differ; give it a "
                f"new run_id instead of overwriting."
            )
        print(f"REGENERATED derived outputs for {config.run_id} from cached decisions; "
              f"metrics verified unchanged, manifest left as written.")

    summary[config.run_id] = row
    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")

    if not args.regenerate_outputs:
        write_manifest(
            manifest_path,
            {"run_id": config.run_id, "stage": "run_intrinsic",
             "config": config.canonical_json(), "csv_path": csv_path,
             "manual_check_path": manual_path, "cache_stats": cache.stats(), **row},
        )

    means = " ".join(f"{k.replace('mean_', '')}={row[k]:.4f}" if row.get(k) is not None
                     else f"{k.replace('mean_', '')}=n/a"
                     for k in row if k.startswith("mean_"))
    print(f"run_intrinsic {config.run_id} [{axis}]: {means} over {n_units} units "
          f"[{judge_source}]")
    print(f"  per-unit CSV -> {csv_path}")
    print(f"  MANUAL CHECK -> {manual_path}: {n_sampled} decisions sampled. "
          f"A HUMAN fills in `human_verdict`; this script never does.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
