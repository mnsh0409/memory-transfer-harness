#!/usr/bin/env python3
"""dialogues -> rewrites (cached). Fixed CLI contract:

    python -m scripts.run_replay --config CFG.json

Thin shim: all real logic is harness.runners.run_replay_config, so it's unit-tested there
without subprocessing this file. Run from the repo root.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from harness.config import RunConfig  # noqa: E402
from harness.envfile import load_dotenv  # noqa: E402
from harness.runners import DEFAULT_CACHE_PATH, run_replay_config  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="path to a RunConfig JSON file")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--cache", default=DEFAULT_CACHE_PATH)
    p.add_argument("--env", default=".env")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    load_dotenv(args.env)
    with open(args.config, encoding="utf-8") as f:
        config = RunConfig(**json.load(f))
    manifest = run_replay_config(
        config, data_dir=args.data_dir, runs_dir=args.runs_dir, cache_path=args.cache
    )
    print(
        f"run_replay {config.run_id}: {manifest['n_dialogues']} dialogues, "
        f"{manifest['n_rewrites']} rewrites, cache={manifest['cache_stats']} "
        f"-> {manifest['rewrites_path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
