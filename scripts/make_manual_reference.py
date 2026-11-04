#!/usr/bin/env python3
"""Build the M1_manual rewrites files from the track-released manual utterances.

    python -m scripts.make_manual_reference

The human-rewrite reference every CAsT paper reports, and the item internal review listed
first under nonstandard usage ("an automatic condition is called a ceiling and cells exceed
100%"). M1 has no memory system and no replay: the query IS the organizers' released manual
rewrite, verbatim, and it flows through the identical frozen retrieval stack and eval as every
other condition. Zero model calls.

Sources (fetched into data/cast/ from daltonj/treccastweb, the track's own repository):
  cast19_manual_utterances.tsv   turn_id <TAB> resolved utterance   (479 turns)
  cast20_manual_topics.json      per-turn `manual_rewritten_utterance` (216 turns)

Emits, per year: a config JSON (condition M1_manual, memory_only context so it lands in the
same stats cell as the reported design), the rewrites file at the path run_retrieval expects,
and a replay-stage manifest recording provenance. Append-only throughout: refuses to overwrite.
Then run, strictly serially on the index:

    python -m scripts.run_retrieval --config runs/configs_manual/<year>.json
    python -m scripts.run_eval      --config runs/configs_manual/<year>.json
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from harness.config import RunConfig  # noqa: E402
from harness.manifest import write_manifest  # noqa: E402
from harness.registry import load_dialogues  # noqa: E402
from harness.replay import RewriteResult, write_rewrites  # noqa: E402
from harness.runners import resolve_model  # noqa: E402
from harness.envfile import load_dotenv  # noqa: E402


def manual_map_cast19(path: str) -> dict[str, str]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            turn_id, utt = line.split("\t", 1)
            out[turn_id.strip()] = utt.strip()
    return out


def manual_map_cast20(path: str) -> dict[str, str]:
    out = {}
    for topic in json.load(open(path, encoding="utf-8")):
        for turn in topic.get("turn", []):
            out[f"{topic['number']}_{turn['number']}"] = turn["manual_rewritten_utterance"].strip()
    return out


def main(argv=None) -> int:
    load_dotenv(".env")
    sources = {
        "cast19": ("data/cast/cast19_manual_utterances.tsv", manual_map_cast19),
        "cast20": ("data/cast/cast20_manual_topics.json", manual_map_cast20),
    }
    os.makedirs("runs/configs_manual", exist_ok=True)
    for coll, (path, loader) in sources.items():
        manual = loader(path)
        config = resolve_model(RunConfig(condition="M1_manual", injection="rewrite_only",
                                         collection=coll, context="memory_only"))
        dialogues = load_dialogues(coll, data_dir="data")
        results, missing = [], []
        for d in dialogues:
            for t in d.turns:
                if t.role != "user":
                    continue
                if t.turn_id not in manual:
                    missing.append(t.turn_id)
                    continue
                results.append(RewriteResult(dialogue_id=d.dialogue_id, turn_id=t.turn_id,
                                             utterance=t.text, memory_block="",
                                             query=manual[t.turn_id]))
        if missing:
            raise SystemExit(f"{coll}: {len(missing)} user turns have no manual utterance "
                             f"(first: {missing[:5]}) -- refusing a partial reference run")
        rewrites_path = os.path.join("runs", "rewrites", f"{config.run_id}.jsonl")
        write_rewrites(rewrites_path, results)
        write_manifest(
            os.path.join("runs", "manifests", f"{config.run_id}.replay.json"),
            {"run_id": config.run_id, "stage": "make_manual_reference",
             "config": config.canonical_json(), "n_rewrites": len(results),
             "source_file": path,
             "note": "queries are the track-released MANUAL utterances verbatim; no model "
                     "calls were made and no memory system exists for this condition"},
        )
        cfg_path = os.path.join("runs", "configs_manual", f"{coll}.M1_manual.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            f.write(config.canonical_json())
        print(f"{coll}: {len(results)} manual rewrites -> {rewrites_path}")
        print(f"        config -> {cfg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
