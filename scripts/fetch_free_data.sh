#!/usr/bin/env bash
# Fetch every free (unlicensed) data file the harness needs, into the data/ layout that
# src/harness/registry.py and src/harness/runners.py already expect.
#
# Nothing here needs credentials or a license. The passage corpus is NOT fetched here --
# Pyserini downloads its prebuilt `cast2019` index on first use (~19.8 GiB compressed; see
# src/harness/retrieval.py). This script only pulls topics, qrels, and the LongMemEval repo,
# which together are a few MB.
#
# URL provenance (checked Aug 21 2026):
#   * CAsT topics + qrels come from the ORGANIZERS' GITHUB MIRROR (daltonj/treccastweb), not
#     from trec.nist.gov. Both were verified to return real data. trec.nist.gov/data/cast/*
#     was NOT reachable from the environment this was written in -- if you have a working
#     nist.gov path, either is fine, the files are the same, but the GitHub paths are the
#     ones actually confirmed to work.
#   * iKAT topics come from irlabamsterdam/iKAT, verified.
#   * iKAT PTKB qrels: NOT verified -- see the note in that section before trusting it.
#
# Fails loudly on any bad fetch rather than leaving a truncated or HTML-error file in data/,
# because a 200-byte "404: Not Found" page parsed as JSON produces a confusing failure three
# stages later instead of here.

set -euo pipefail

DATA_DIR="${DATA_DIR:-data}"
mkdir -p "$DATA_DIR/cast" "$DATA_DIR/ikat" "$DATA_DIR/longmemeval"

fetch() {
  local url="$1" dest="$2" min_bytes="${3:-1000}"
  if [[ -s "$dest" ]]; then
    echo "  [skip] $dest already present ($(wc -c <"$dest") bytes)"
    return 0
  fi
  echo "  [get ] $url"
  if ! curl -fsSL --retry 3 --retry-delay 2 "$url" -o "$dest"; then
    echo "  [FAIL] could not fetch $url" >&2
    rm -f "$dest"
    return 1
  fi
  local size
  size=$(wc -c <"$dest")
  if (( size < min_bytes )); then
    echo "  [FAIL] $dest is only ${size} bytes -- almost certainly an error page, not data" >&2
    rm -f "$dest"
    return 1
  fi
  echo "  [ ok ] $dest (${size} bytes)"
}

check_json() {
  python3 -c "import json,sys; json.load(open(sys.argv[1], encoding='utf-8')); print('  [ ok ] valid JSON:', sys.argv[1])" "$1"
}

check_qrels() {
  # 4 columns: qid iter docid rel. Column 2 is "Q0" in CAsT 2019 and "0" in CAsT 2020 --
  # both are real, so it is deliberately not validated (see eval/extrinsic.load_qrels).
  python3 - "$1" <<'PY'
import sys
path = sys.argv[1]
n = 0
with open(path, encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 4:
            sys.exit(f"  [FAIL] {path} line {i}: {len(parts)} fields, want 4")
        int(parts[3])
        n += 1
print(f"  [ ok ] {path}: {n} judgments")
PY
}

echo "== CAsT 2019 =="
fetch "https://raw.githubusercontent.com/daltonj/treccastweb/master/2019/data/evaluation/evaluation_topics_v1.0.json" \
      "$DATA_DIR/cast/cast19_topics.json"
fetch "https://raw.githubusercontent.com/daltonj/treccastweb/master/2019/data/2019qrels.txt" \
      "$DATA_DIR/cast/cast19_qrels.txt"
check_json  "$DATA_DIR/cast/cast19_topics.json"
check_qrels "$DATA_DIR/cast/cast19_qrels.txt"

echo "== CAsT 2020 =="
fetch "https://raw.githubusercontent.com/daltonj/treccastweb/master/2020/2020_automatic_evaluation_topics_v1.0.json" \
      "$DATA_DIR/cast/cast20_topics.json"
fetch "https://raw.githubusercontent.com/daltonj/treccastweb/master/2020/2020qrels.txt" \
      "$DATA_DIR/cast/cast20_qrels.txt"
check_json  "$DATA_DIR/cast/cast20_topics.json"
check_qrels "$DATA_DIR/cast/cast20_qrels.txt"

echo "== iKAT 2023 =="
fetch "https://raw.githubusercontent.com/irlabamsterdam/iKAT/main/2023/data/2023_test_topics.json" \
      "$DATA_DIR/ikat/ikat23_topics.json"
check_json "$DATA_DIR/ikat/ikat23_topics.json"

echo "== iKAT 2024 =="
# This URL RESOLVES and returns valid JSON -- but the file it returns is NOT usable for
# scoring as-is. Its topics are numbered 0,1,2,...,16 (plain positional ints) rather than the
# '<persona>-<dialogue>' ids the 2023 file uses ("9-1"), and its `title` is not unique either
# ("Buying perfume" appears at both index 1 and index 3). Turn ids derived from that align
# with no qrels file. src/harness/loaders/ikat.py REFUSES this file by default for exactly
# that reason -- see _check_topic_number there.
#
# To resolve: find the authoritative 2024 topics/qrels pair (check the repo tree and the
# TREC iKAT 2024 track page) and fix the URL below. Do NOT work around it downstream.
# iKAT 2023 alone is enough to start the intrinsic PTKB axis; 2024 can follow.
if ! fetch "https://raw.githubusercontent.com/irlabamsterdam/iKAT/main/2024/data/2024_test_topics.json" \
           "$DATA_DIR/ikat/ikat24_topics.json"; then
  echo "  [warn] iKAT 2024 topics not fetched -- fix the URL above."
fi
if [[ -s "$DATA_DIR/ikat/ikat24_topics.json" ]]; then
  check_json "$DATA_DIR/ikat/ikat24_topics.json"
  python3 - "$DATA_DIR/ikat/ikat24_topics.json" <<'PY'
import json, sys
raw = json.load(open(sys.argv[1], encoding="utf-8"))
bad = [t.get("number") for t in raw if not isinstance(t.get("number"), str) or "-" not in str(t.get("number"))]
if bad:
    print(f"  [WARN] {sys.argv[1]}: {len(bad)} topics have positional numbers "
          f"(e.g. {bad[:3]}) instead of '<persona>-<dialogue>' ids.")
    print("         The iKAT loader will REFUSE this file for scoring. See the note above.")
PY
fi

# iKAT PTKB qrels: not fetched. The 2023 topics file already carries per-turn
# `ptkb_provenance` (gold PTKB statement numbers per turn), which loaders/ikat.py exposes as
# meta["ptkb_provenance"] -- that is very likely sufficient for the PTKB-reconstruction axis
# without a separate qrels file. Confirm what run_intrinsic.py actually
# needs before hunting for one.

echo "== LongMemEval =="
if [[ -d "$DATA_DIR/longmemeval/.git" ]]; then
  echo "  [skip] repo already cloned"
else
  git clone --depth 1 https://github.com/xiaowu0162/LongMemEval.git "$DATA_DIR/longmemeval" \
    || echo "  [warn] clone failed -- check the repo URL; data files follow its README (HF link)"
fi
echo "  [note] LongMemEval's _S session data is downloaded per its own README (HF link),"
echo "         not by this script."

echo
echo "Done. Present under $DATA_DIR:"
find "$DATA_DIR" -maxdepth 2 -type f \( -name '*.json' -o -name '*.txt' \) -exec ls -la {} \;
