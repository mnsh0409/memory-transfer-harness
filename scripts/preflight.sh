#!/usr/bin/env bash
# One command that tells you whether this machine can actually run the experiment.
#
#   bash scripts/preflight.sh            # checks only, no big downloads
#   bash scripts/preflight.sh --index    # ALSO pulls the ~19.8 GiB Pyserini cast2019 index
#
# Every check prints PASS / FAIL / WARN and the script exits non-zero if anything a run
# depends on is broken. The point is to fail here, in two minutes, rather than three hours
# into the grid.
#
# Deliberately does NOT pull the Pyserini index by default -- it is ~19.8 GiB compressed and
# you want to start that download knowingly. But DO run it with --index before the grid: an
# index that fails to materialise at hour zero of a scored run is the expensive failure this
# script exists to prevent.

set -uo pipefail

# Interpreter: override with PYTHON=... when `python3` on PATH is not the one you
# installed the deps into (e.g. the Windows Store stub). Everything below uses $PY.
PY="${PYTHON:-python3}"

FAILED=0
pass() { printf '  \033[32mPASS\033[0m  %s\n' "$1"; }
fail() { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; FAILED=1; }
warn() { printf '  \033[33mWARN\033[0m  %s\n' "$1"; }

echo "== Environment =="
$PY --version >/dev/null 2>&1 && pass "$PY $($PY --version 2>&1 | cut -d' ' -f2)" \
  || fail "$PY not found (set PYTHON=... to point at the interpreter with the deps)"

if java -version >/dev/null 2>&1; then
  # grep for the version line specifically: JAVA_TOOL_OPTIONS noise can occupy line 1
  JV=$(java -version 2>&1 | grep -oE '(openjdk|java) version "[^"]+"' | head -1)
  [[ -z "$JV" ]] && JV="(version string unreadable)"
  if java -version 2>&1 | grep -qE 'version "(2[1-9]|[3-9][0-9])'; then
    pass "Java 21+ -- $JV"
  else
    fail "Java present but too old for Pyserini (needs 21): $JV"
  fi
else
  fail "java not found -- Pyserini needs Java 21"
fi

echo
echo "== Test suite =="
# Same command as `make test`, run directly so a box without make (Windows) still gets
# the real verdict.
if $PY -m pytest tests/ -q >/tmp/preflight_tests.log 2>&1; then
  pass "$(tail -1 /tmp/preflight_tests.log)"
else
  fail "test suite failed -- see /tmp/preflight_tests.log"
fi

echo
echo "== Scoring dependency (required before any reported number) =="
if $PY -c "import pytrec_eval" >/dev/null 2>&1; then
  pass "pytrec_eval importable"
  if $PY -m pytest tests/test_extrinsic.py -k cross_check -q >/tmp/preflight_xcheck.log 2>&1; then
    pass "native metrics match trec_eval (nDCG gain convention verified)"
  else
    fail "native/trec_eval cross-check FAILED -- see /tmp/preflight_xcheck.log"
  fi
else
  fail "pytrec_eval missing. Install: pip install pytrec_eval_terrier
         (NOT plain 'pytrec_eval' -- it builds from source and downloads trec_eval's C
          sources at build time, which fails behind a restrictive proxy.)
         run_eval refuses to compute nDCG/R/MRR without it."
fi

echo
echo "== Experiment backend =="
if [[ -f .env ]]; then
  pass ".env present"
  # shellcheck disable=SC1091
  set -a; . ./.env; set +a
  [[ -n "${OPENAI_API_KEY:-}" ]] && pass "OPENAI_API_KEY set" || fail "OPENAI_API_KEY empty in .env"
  if [[ -n "${EXPERIMENT_MODEL:-}" ]]; then
    if [[ "$EXPERIMENT_MODEL" == *"-"*"-"* ]]; then
      pass "EXPERIMENT_MODEL=$EXPERIMENT_MODEL"
    else
      warn "EXPERIMENT_MODEL=$EXPERIMENT_MODEL does not look like a pinned SNAPSHOT id"
      warn "  (the harness requires an exact snapshot, e.g. gpt-4o-mini-2024-07-18 -- an"
      warn "   unpinned alias silently changes under you mid-grid)"
    fi
  else
    fail "EXPERIMENT_MODEL not set in .env"
  fi
else
  fail ".env not found -- see README, Re-running the experiments"
fi

echo
echo "== Data =="
for f in data/cast/cast19_topics.json data/cast/cast19_qrels.txt \
         data/cast/cast20_topics.json data/cast/cast20_qrels.txt; do
  [[ -s "$f" ]] && pass "$f ($(wc -c <"$f") bytes)" || fail "$f missing -- run scripts/fetch_free_data.sh"
done
[[ -s data/ikat/ikat23_topics.json ]] && pass "data/ikat/ikat23_topics.json" \
  || warn "data/ikat/ikat23_topics.json missing (intrinsic axis only, not needed for CAsT runs)"

if [[ -s data/cast/cast19_qrels.txt ]]; then
  $PY - <<'PY'
import sys
sys.path.insert(0, "src")
from eval.extrinsic import load_qrels
q = load_qrels("data/cast/cast19_qrels.txt")
n_judged = len(q)
grades = {}
for docs in q.values():
    for r in docs.values():
        grades[r] = grades.get(r, 0) + 1
print(f"  \033[32mPASS\033[0m  cast19 qrels parse: {n_judged} judged turns, "
      f"grade histogram {dict(sorted(grades.items()))}")
print(f"  \033[33mNOTE\033[0m  {n_judged} is the per-condition n for CAsT 2019 "
      f"(the topics file has 479 turns; only these are judged)")
PY
fi

echo
echo "== Retrieval index =="
if [[ "${1:-}" == "--index" ]]; then
  echo "  pulling cast2019 prebuilt index (~19.8 GiB compressed) -- this takes a while..."
  $PY - <<'PY'
import os, sys
sys.path.insert(0, "src")
os.environ.setdefault("OPENAI_API_KEY", "unused")
try:
    from pyserini.search.lucene import LuceneSearcher
    from harness.retrieval import passage_text
    s = LuceneSearcher.from_prebuilt_index("cast2019")
    hits = s.search("what is throat cancer", k=5)
    texts = [passage_text(s.doc(h.docid)) for h in hits]
    print(f"  \033[32mPASS\033[0m  cast2019 index loaded; {len(hits)} hits for a test query")
    print(f"        top hit: {hits[0].docid} (score {hits[0].score:.3f})")
    if not all(texts):
        # the reranker scores (query, passage) pairs -- an index that returns no passage
        # text would make the whole rerank stage meaningless. On the real cast2019 index
        # the text is in doc.raw(), not doc.contents(); passage_text handles that.
        print("  \033[31mFAIL\033[0m  passage text missing for some hits -- see "
              "harness.retrieval.passage_text; do not run the grid like this")
        sys.exit(1)
    print(f"        passage text retrievable for all {len(texts)} hits "
          f"(top hit {len(texts[0])} chars)")
except Exception as e:
    print(f"  \033[31mFAIL\033[0m  cast2019 prebuilt index unavailable: {e}")
    print("        STOP. There is no msmarco+trec-car fallback: pyserini registers no")
    print("        trec-car prebuilt index -- do not")
    print("        substitute a smaller corpus, it silently changes what is being scored.")
    sys.exit(1)
PY
  [[ $? -ne 0 ]] && FAILED=1
else
  if $PY -c "
import os; os.environ.setdefault('OPENAI_API_KEY','unused')
import pyserini.prebuilt_index_info as p
assert 'cast2019' in p.TF_INDEX_INFO
" >/dev/null 2>&1; then
    pass "pyserini installed and 'cast2019' is in its prebuilt-index manifest"
    warn "index NOT downloaded. Re-run with --index before the grid (~19.8 GiB)."
  else
    fail "pyserini not installed or 'cast2019' missing from its manifest"
  fi
fi

echo
if [[ $FAILED -eq 0 ]]; then
  printf '\033[32mPreflight OK.\033[0m Next: python -m scripts.make_grid --gate-only, then the B1>B0 gate run.\n'
else
  printf '\033[31mPreflight FAILED.\033[0m Fix the items above before running anything scored.\n'
fi
exit $FAILED
