#!/usr/bin/env python3
"""How much does each memory system actually inject? Read-only diagnostic over run artifacts.

    python -m scripts.injection_report --collection cast19

Answers the question that decides whether an extrinsic comparison can say anything at all:
for each condition, on how many turns did the system have any memory to offer, and on how many
did that memory actually change the rewritten query relative to the B0 (no-memory) run?

Why this exists: on CAsT 2019, S1_mem0 stored something on only 13.9% of judged turns and
changed the rewrite on **5 of 173** -- so 168 turns are tied with B0 by construction and no
paired test over them can detect anything. That is a property of the collection (CAsT is
topical information-seeking with no personas, so a fact-extraction memory has almost nothing
to store), not a bug, but it has to be reported next to any "no significant difference"
claim or that claim reads as "memory does not help" when it actually means "we could not
measure". The paper (Section 3.4) describes the same failure mode in B1, where it is a harness artifact.

Reads only what run_replay and run_eval already wrote; computes nothing new and writes
nothing. Safe to run while a grid is in flight.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

BASELINE = "B0_none"


def _load_rewrites(runs_dir: str, run_id: str) -> dict:
    path = os.path.join(runs_dir, "rewrites", f"{run_id}.jsonl")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return {r["turn_id"]: r for r in (json.loads(ln) for ln in f if ln.strip())}


def _judged_turns(runs_dir: str, run_id: str) -> set:
    path = os.path.join(runs_dir, "metrics", f"{run_id}.per_turn.jsonl")
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {json.loads(ln)["turn_id"] for ln in f if ln.strip()}


def build_rows(runs_dir: str, collection: str, injection: str = "rewrite_only",
               context: str = "history_plus_memory") -> list[dict]:
    stats_path = os.path.join(runs_dir, "stats_summary.json")
    with open(stats_path, encoding="utf-8") as f:
        stats = json.load(f)
    # `context` MUST be part of the cell. Without it the B0 of one design becomes the baseline
    # for another design's runs and every number is wrong -- observed the moment the
    # memory_only ceiling runs landed: the table showed two B0_none rows and reported GAM as
    # changing 82% of turns when it had in fact changed none. Rows written before the context
    # axis existed default to the design they ran under.
    cell = {rid: row for rid, row in stats.items()
            if row.get("collection") == collection and row.get("injection") == injection
            and row.get("context", "history_plus_memory") == context}
    base_id = next((rid for rid, row in cell.items()
                    if row["condition"] == BASELINE and not row.get("ablation")), None)
    if base_id is None:
        raise SystemExit(f"no {BASELINE} run scored for {collection}/{injection}/{context} "
                         "yet -- the diagnostic is relative to it")
    base = _load_rewrites(runs_dir, base_id)
    judged = _judged_turns(runs_dir, base_id)

    rows = []
    for rid, row in sorted(cell.items(), key=lambda kv: kv[1]["condition"]):
        rw = _load_rewrites(runs_dir, rid)
        if not rw:
            continue
        label = row["condition"] + ("".join(f"[{k}={v}]" for k, v in
                                            sorted((row.get("ablation") or {}).items())))
        turns = sorted(rw)
        jt = [t for t in turns if t in judged]
        has_mem = [t for t in turns if rw[t]["memory_block"].strip()]
        differs = [t for t in turns
                   if t in base and rw[t]["query"].strip() != base[t]["query"].strip()]
        rows.append({
            "run": label, "run_id": rid, "n_turns": len(turns), "n_judged": len(jt),
            "mem_all": len(has_mem),
            "diff_all": len(differs),
            "mem_judged": sum(1 for t in jt if rw[t]["memory_block"].strip()),
            "diff_judged": sum(1 for t in jt if t in base
                               and rw[t]["query"].strip() != base[t]["query"].strip()),
            "ndcg_cut_3": row.get("ndcg_cut_3"),
        })
    return rows


def render(rows: list[dict], collection: str,
           context: str = "history_plus_memory") -> str:
    # plain ASCII: these tables get pasted into logs, and the console here is cp950
    head = (f"Injection report - {collection} / {context} (relative to {BASELINE})\n\n"
            f"{'run':<26}{'nDCG@3':>9}{'mem/turns':>16}{'changed':>16}"
            f"{'mem/judged':>15}{'changed(j)':>15}\n")
    lines = [head + "-" * 97]
    for r in rows:
        pct = lambda a, b: f"{a}/{b} ({100*a/b:.0f}%)" if b else "-"      # noqa: E731
        nd = "-" if r["ndcg_cut_3"] is None else f"{r['ndcg_cut_3']:.4f}"
        lines.append(f"{r['run']:<26}{nd:>9}{pct(r['mem_all'], r['n_turns']):>16}"
                     f"{pct(r['diff_all'], r['n_turns']):>16}"
                     f"{pct(r['mem_judged'], r['n_judged']):>15}"
                     f"{pct(r['diff_judged'], r['n_judged']):>15}")
    lines.append("")
    lines.append("changed(j) is the ceiling on how many judged turns can differ from B0 at all:")
    lines.append("every other judged turn is TIED by construction, whatever the metric says.")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--collection", default="cast19")
    p.add_argument("--injection", default="rewrite_only")
    p.add_argument("--context", default="history_plus_memory",
                   help="history_plus_memory (default) or memory_only -- different "
                        "experiments; never compare one to the other's baseline")
    p.add_argument("--json", action="store_true", help="emit rows as JSON instead of a table")
    args = p.parse_args(argv)
    rows = build_rows(args.runs_dir, args.collection, args.injection, args.context)
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        print(render(rows, args.collection, args.context))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
