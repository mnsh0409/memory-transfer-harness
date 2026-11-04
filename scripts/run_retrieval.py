#!/usr/bin/env python3
"""rewrites -> TREC run file (BM25 -> cross-encoder). Fixed CLI contract:

    python -m scripts.run_retrieval --config CFG.json

Thin shim: all real logic is harness.runners.run_retrieval_config, so it's unit-tested there
(against fakes -- see tests/test_runners.py) without needing pyserini/Java/a GPU/a model
download just to exercise the plumbing. This script itself does need all of those, same as the
protocol always said. Run from the repo root, after scripts/run_replay.py for the same config.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from harness.config import RunConfig  # noqa: E402
from harness.envfile import load_dotenv  # noqa: E402
from harness.runners import run_retrieval_config  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="path to a RunConfig JSON file")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--env", default=".env")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    load_dotenv(args.env)
    with open(args.config, encoding="utf-8") as f:
        config = RunConfig(**json.load(f))
    manifest = run_retrieval_config(config, runs_dir=args.runs_dir)
    print(
        f"run_retrieval {config.run_id}: {manifest['n_queries']} queries "
        f"-> {manifest['run_path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
