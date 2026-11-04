#!/usr/bin/env bash
# S5 GAM, recovered. The first GAM run stored nothing on 479/479 turns and produced a TREC
# file byte-identical to B0's after 5,379 model calls -- because GAM edits its knowledge base
# by shelling out Unix commands and `shell=True` on Windows is cmd.exe, which cannot run its
# `cat > "..." << 'EOF'` file writes. Fixed in the adapter (transport only, not behaviour) and
# pinned by tests/test_gam_posix_shell.py.
#
# This is the memory_only design -- the reportable one -- and no GAM run exists in that cell,
# so nothing collides with the append-only guard.
#
# GAM is by a wide margin the most expensive condition: LLM chunking plus memory generation on
# every ingest, and up to max_iterations exploration rounds on every query. Its replay is
# API-bound and touches no index, so it overlaps safely with an HDD-bound retrieval job; the
# retrieval at the end is the part that must not run beside another retrieval.
set -u
cd "$(dirname "$0")/.."

CACHE=runs/llm_cache.gam2.jsonl    # own cache file: one writer per file, always
CFG=runs/configs/15-memory_only-06-cast19.S5_gam.rewrite_only.59663b361840.json

echo "=== $(date +%H:%M:%S) GAM REPLAY (API-bound) $(basename "$CFG")"
python -m scripts.run_replay --config "$CFG" --cache "$CACHE" || { echo "FAILED replay"; exit 1; }

# Guard: the whole point is that this condition stores something. If the store is empty again,
# say so loudly rather than letting a B0-identical run file into the stats.
python - <<'PY'
import glob, json, os
rw = sorted(glob.glob("runs/rewrites/cast19.S5_gam.*.jsonl"), key=os.path.getmtime)[-1]
rows = [json.loads(l) for l in open(rw, encoding="utf-8") if l.strip()]
n = sum(1 for r in rows if r["memory_block"].strip())
print(f"GAM injected memory on {n}/{len(rows)} turns  ({rw})")
if n == 0:
    raise SystemExit("GAM STORED NOTHING AGAIN -- do not score this; the POSIX fix did not take")
PY

# Two retrievals on this box thrash the 6TB spinning index and finish slower than if run in
# sequence, so wait for the ablation job's serial retrieval queue to drain before starting.
# Waiting on a marker in its log, not on a PID, so this survives that job being restarted.
if [ -f runs/ablations2.log ]; then
  waited=0
  while ! grep -q "ABLATIONS2 DONE" runs/ablations2.log; do
    if [ "$waited" -ge 14400 ]; then
      echo "=== $(date +%H:%M:%S) waited 4h for ablations2; proceeding anyway"
      break
    fi
    [ "$((waited % 600))" -eq 0 ] && \
      echo "=== $(date +%H:%M:%S) holding GAM retrieval; ablations2 still on the index"
    sleep 60
    waited=$((waited + 60))
  done
fi

echo "=== $(date +%H:%M:%S) GAM RETRIEVE (HDD -- must not overlap another retrieval)"
python -m scripts.run_retrieval --config "$CFG" || { echo "FAILED retrieval"; exit 1; }
python -m scripts.run_eval --config "$CFG" || echo "FAILED eval"

echo "=== $(date +%H:%M:%S) GAM2 DONE"
