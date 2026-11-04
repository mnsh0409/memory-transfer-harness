#!/usr/bin/env python3
"""Emit every frozen config into runs/configs/. Fixed CLI contract:

    python -m scripts.make_grid

The grid is defined by the frozen scope and implemented in harness.grid. To change what runs, change
the frozen scope first, then harness.grid to match.

    --gate-only     emit just the two B1>B0 gate configs (the frozen scope: no S-run starts until
                    that passes) -- useful for a first pass that can't accidentally kick off
                    the whole grid.
    --best-system   include the k-sweep ablation, scoped to this system. Omit until the main
                    grid is scored and you know which system won on nDCG@3 / CAsT 2019.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from harness.grid import _config_dict, emit_grid, gate_grid  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default=os.path.join("runs", "configs"))
    p.add_argument("--gate-only", action="store_true")
    p.add_argument("--context", default="history_plus_memory",
                   help="history_plus_memory (default) or memory_only -- the latter empties "
                        "the prompt's {history} slot so memory is the only context channel, "
                        "which is what makes B1 a real ceiling (paper, Section 3.4)")
    p.add_argument("--best-system", default=None,
                   help="e.g. S1_mem0 -- adds the k in {3,5,10} sweep on that system")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.gate_only:
        os.makedirs(args.out_dir, exist_ok=True)
        paths = []
        for i, config in enumerate(gate_grid()):
            path = os.path.join(args.out_dir, f"00-gate-{i:02d}-{config.run_id}.json")
            if os.path.exists(path):
                raise FileExistsError(f"{path} exists -- delete {args.out_dir} to regenerate")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(_config_dict(config), f, indent=2, sort_keys=True)
                f.write("\n")
            paths.append(path)
    else:
        paths = emit_grid(args.out_dir, best_system=args.best_system, context=args.context)

    for path in paths:
        print(f"  {path}")
    print(f"{len(paths)} configs -> {args.out_dir}")
    if not args.gate_only and args.best_system is None:
        print("note: k-sweep omitted (no --best-system). Re-emit it once the main grid is "
              "scored; the frozen scope scopes that sweep to the single best system.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
