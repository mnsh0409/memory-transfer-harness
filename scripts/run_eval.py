#!/usr/bin/env python3
"""run file + qrels -> metrics incl. judged@10. Fixed CLI contract:

    python -m scripts.run_eval --config CFG.json

Thin shim: real logic is harness.runners.run_eval_config. Run from the repo root, after
scripts/run_retrieval.py for the same config.

nDCG/R/MRR come from pytrec_eval (== trec_eval). If it isn't installed this exits with an
error rather than silently falling back to the harness's own unvalidated implementation --
pass --allow-native-metrics to override, and do not put those numbers in the paper.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from harness.config import RunConfig  # noqa: E402
from harness.envfile import load_dotenv  # noqa: E402
from harness.runners import run_eval_config  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="path to a RunConfig JSON file")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--judged-k", type=int, default=10)
    p.add_argument("--env", default=".env")
    p.add_argument(
        "--allow-native-metrics", action="store_true",
        help="compute nDCG/R/MRR with the harness's own UNVALIDATED implementation when "
             "pytrec_eval is unavailable. Plumbing/smoke use only -- never for reported numbers.",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    load_dotenv(args.env)
    with open(args.config, encoding="utf-8") as f:
        config = RunConfig(**json.load(f))
    m = run_eval_config(
        config, data_dir=args.data_dir, runs_dir=args.runs_dir,
        judged_k=args.judged_k, allow_native_metrics=args.allow_native_metrics,
    )
    judged_key = f"judged_{args.judged_k}"
    print(
        f"run_eval {config.run_id}: nDCG@3={m['ndcg_cut_3']:.4f} nDCG@5={m['ndcg_cut_5']:.4f} "
        f"R@1000={m['recall_1000']:.4f} MRR={m['recip_rank']:.4f} "
        f"{judged_key}={m[judged_key]:.4f} over {m['n_scored_turns']} turns "
        f"[{m['metrics_source']}]"
    )
    cov = m["coverage"]
    if cov["n_qrels_only"] or cov["n_run_only"]:
        print(
            f"  COVERAGE ANOMALY -- flag it, do not silently accept: "
            f"{cov['n_qrels_only']} judged turns missing from the run, "
            f"{cov['n_run_only']} run turns with no judgments. "
            f"Averages above are over the {cov['n_scored']} scored turns only."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
