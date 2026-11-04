#!/usr/bin/env bash
# Intrinsic axis 2 -- LongMemEval evidence recall. API-only: no retrieval, no Pyserini index,
# no disk contention, so this is safe to run alongside the HDD-bound ablation job.
#
# Why the systems run in full and the ceiling does not. The first attempt (runs/lme.log,
# Aug 23 00:52) asked for B1 over all 500 questions and died on the first 429: B1's memory IS
# the whole session, so each judge prompt ran ~127k tokens against a 200k TPM limit -- roughly
# one call a minute, for a number that is ~1.0 by construction (does the full transcript
# contain the evidence? yes). That is the most expensive possible way to measure a foregone
# conclusion. S1 and S3 return k=5 entries, so their prompts are small and they carry the
# actual finding; B1 is spot-checked on a subsample purely to confirm the ceiling behaves.
#
# The retry/backoff in harness/llm.py postdates that failure (it was written because of it),
# so a 429 now backs off instead of killing the run. The --limit on B1 is a budget decision,
# not a workaround for the crash.
set -u
cd "$(dirname "$0")/.."

CACHE=runs/llm_cache.lme2.jsonl   # own cache file: one writer per file, always

# SAMPLE SIZE IS FORCED, NOT CHOSEN. The 500 questions carry 246,750 haystack turns between
# them (median 491 per question), and a per-turn-ingesting system makes one model call per
# turn -- ~47 hours of wall clock at this account's throughput, for one condition. 30 questions
# is ~2.5h per system.
#
# The SAME N is used for all three conditions on purpose: the sample is deterministic and
# stratified by question_type, so every condition sees the identical question set and B1
# remains a valid ceiling for the systems. Giving the ceiling a different subsample would
# quietly compare two different benchmarks.
#
# `--limit` on longmemeval means the stratified sample, NOT the first N -- the released _S file
# is sorted by question_type, so a prefix would be single-session-user only.
N=30

# Mem0 pins its stores to /tmp/qdrant and ~/.mem0 and locks both, so this job's Mem0 would
# fight the ablation job's Mem0 for them -- the first attempt died exactly there. MEM0_DIR
# relocates all of it, and MUST be exported before python starts: upstream reads it into a
# module-level constant at import time, so setting it from inside Python is a silent no-op.
export MEM0_DIR="${MEM0_DIR:-$PWD/runs/mem0_home_lme}"
mkdir -p "$MEM0_DIR"
echo "MEM0_DIR=$MEM0_DIR (isolated from any concurrent Mem0 run)"

for c in S1_mem0 S3_amem B1_full_history; do
  echo "=== $(date +%H:%M:%S) LME $c (stratified sample of $N)"
  python -m scripts.run_intrinsic --config "runs/configs_intrinsic/lme.$c.json" \
    --cache "$CACHE" --limit "$N" || echo "FAILED lme $c"
done

echo "=== $(date +%H:%M:%S) LME2 DONE"
