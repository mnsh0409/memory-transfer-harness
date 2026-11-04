#!/usr/bin/env bash
# the frozen scope ablations 1 and 2, under the reportable (memory_only) design.
#
#   1. Mem0 infer=False  -- cast19 is already scored; this adds cast20, so the paper's
#      strongest causal claim ("extraction is what loses the headroom") is replicated on a
#      second, independent year instead of resting on one.
#   2. A-Mem evolution=False -- neither year has been run. Sections 3.4 and 4.2 *predict* this
#      is a null: evolution rewrites a note's context, tags and neighbour links but not its
#      text, and only text reaches the rewriter. An unrun prediction is a caveat; a measured
#      one is a result. Note this replay makes no A-Mem LLM calls at all (evolution is the
#      only thing A-Mem uses the backend for), so the ablation arm is nearly free.
#
# Ablation 3 (Graphiti edge_invalidation=False) is deliberately NOT here: S2 injects on 1% of
# turns and is quarantined, so its ablation would be an uninterpretable delta on an
# uninterpretable baseline.
#
# Phase discipline, learned the hard way: replays are API-bound and retrievals
# are HDD-bound on a 6TB spinning disk, so replays run first as a group and retrievals run
# STRICTLY SERIAL afterwards. Two concurrent retrievals thrash the index and take longer than
# running them in sequence. This job is designed to overlap with an API-only job (LongMemEval)
# -- its long pole is disk, which that job never touches.
set -u
cd "$(dirname "$0")/.."

CACHE=runs/llm_cache.abl2.jsonl   # own cache file: one writer per file, always
CFGS=(runs/configs/25-memory_only-ablation-01-*.json
      runs/configs/25-memory_only-ablation-02-*.json
      runs/configs/25-memory_only-ablation-03-*.json)

echo "=== $(date +%H:%M:%S) PHASE 1 replays (API-bound)"
for cfg in "${CFGS[@]}"; do
  echo "=== $(date +%H:%M:%S) REPLAY $(basename "$cfg")"
  python -m scripts.run_replay --config "$cfg" --cache "$CACHE" || echo "FAILED replay $cfg"
done

echo "=== $(date +%H:%M:%S) PHASE 2 retrieval+eval, strictly serial (HDD)"
for cfg in "${CFGS[@]}"; do
  echo "=== $(date +%H:%M:%S) RETRIEVE $(basename "$cfg")"
  python -m scripts.run_retrieval --config "$cfg" || { echo "FAILED retrieval $cfg"; continue; }
  python -m scripts.run_eval --config "$cfg" || echo "FAILED eval $cfg"
done

echo "=== $(date +%H:%M:%S) ABLATIONS2 DONE"
