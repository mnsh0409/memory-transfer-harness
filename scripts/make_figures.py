#!/usr/bin/env python3
"""collect_stats output -> paper figures (PDF + PNG). `make figures`:

    python -m scripts.collect_stats && python -m scripts.make_figures

Thin shim: real logic is eval.figures.make_figures. Reads runs/stats/collect_stats.json only,
writes runs/figures/ (regenerated in place) plus figures_manifest.json naming the inputs.
Needs matplotlib (optional dependency; nothing else in the harness does).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eval.figures import make_figures  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stats-dir", default=os.path.join("runs", "stats"))
    p.add_argument("--out-dir", default=os.path.join("runs", "figures"))
    p.add_argument("--metric", default="ndcg_cut_3")
    p.add_argument("--formats", default="pdf,png")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    result = make_figures(args.stats_dir, args.out_dir, metric=args.metric,
                          formats=tuple(f for f in args.formats.split(",") if f))
    for name, paths in result["figures"].items():
        if paths:
            print(f"  {name}: " + ", ".join(paths))
        else:
            print(f"  {name}: skipped (no applicable rows yet)")
    print(f"make_figures: manifest -> {result['manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
