#!/usr/bin/env bash
# S5 GAM, supervised so it lands unattended.
#
# GAM is the most expensive condition in the grid: ~10 model calls per turn on ingest plus up
# to max_iterations exploration rounds per query, over 479 turns. At the observed ~750 calls/h
# that is roughly 14 hours and ~10.5k calls -- inside the wall clock, but close enough to the
# account's rolling requests-per-day ceiling that a mid-run stall is likely at least once.
#
# A stall must not cost the run. Every completed model call is in the content-hashed cache, so
# re-running run_replay replays the same dialogues, hits the cache for everything already done,
# and only spends new requests where it left off. This loop therefore just restarts on failure
# and waits for the quota window to roll.
#
# Two things make the restart safe rather than wasteful:
#   * run_replay writes its rewrites file only on success, and that write is append-only, so a
#     killed run leaves no partial artifact to collide with.
#   * GAM's per-dialogue stores are rebuilt from cached calls -- local disk work, no API.
set -u
cd "$(dirname "$0")/.."

CACHE=runs/llm_cache.gam2.jsonl
CFG=runs/configs/15-memory_only-06-cast19.S5_gam.rewrite_only.59663b361840.json
MAX_ATTEMPTS=12
WAIT=1800          # 30 min between attempts; the quota window rolls in well under an hour

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  echo "=== $(date +%H:%M:%S) GAM REPLAY attempt $attempt/$MAX_ATTEMPTS (cached calls replay free)"
  if python -m scripts.run_replay --config "$CFG" --cache "$CACHE"; then
    echo "=== $(date +%H:%M:%S) GAM REPLAY COMPLETE on attempt $attempt"
    break
  fi
  if [ "$attempt" -eq "$MAX_ATTEMPTS" ]; then
    echo "=== $(date +%H:%M:%S) GAM REPLAY GAVE UP after $MAX_ATTEMPTS attempts"
    exit 1
  fi
  echo "=== $(date +%H:%M:%S) attempt $attempt failed ($(wc -l < "$CACHE") calls cached so far); sleeping ${WAIT}s"
  sleep "$WAIT"
done

# Refuse to score a run whose memory block is empty on every turn -- the whole point of the
# POSIX fix is that this condition stores something, and a B0-identical run file is what the
# first GAM attempt produced.
python - <<'PY'
import glob, json, os
rw = sorted(glob.glob("runs/rewrites/cast19.S5_gam.*.jsonl"), key=os.path.getmtime)[-1]
rows = [json.loads(l) for l in open(rw, encoding="utf-8") if l.strip()]
n = sum(1 for r in rows if r["memory_block"].strip())
print(f"GAM injected memory on {n}/{len(rows)} turns  ({rw})")
if n == 0:
    raise SystemExit("GAM STORED NOTHING -- do not score this; the POSIX fix did not take")
PY

echo "=== $(date +%H:%M:%S) GAM RETRIEVE (HDD; nothing else should be on the index)"
python -m scripts.run_retrieval --config "$CFG" || { echo "FAILED retrieval"; exit 1; }
python -m scripts.run_eval --config "$CFG" || echo "FAILED eval"

echo "=== $(date +%H:%M:%S) GAM SUPERVISED DONE"
