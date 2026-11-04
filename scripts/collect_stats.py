#!/usr/bin/env python3
"""Scored runs -> paired comparisons (Holm-corrected) + effect sizes + metrics tables.
Fixed CLI contract:

    python -m scripts.collect_stats

Thin shim: real logic is eval.stats.collect_stats. Reads runs/stats_summary.json and the
per-turn JSONL files run_eval wrote; writes runs/stats/ (comparisons.csv, collect_stats.json,
tables/*.md|tex, per_topic_*.csv). Outputs are pure functions of the inputs and are
regenerated in place -- they are NOT run artifacts.

Conventions (see eval/stats.py docstring): pairing unit = turn; n = turns scored in both
runs (173 on CAsT 2019, not 479); Holm within each (group, collection, injection, metric)
family; a per-dialogue test is reported next to every per-turn one as a clustering check.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eval.stats import DEFAULT_ALPHA, PRIMARY_METRIC, collect_stats  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--out-dir", default=None, help="default: <runs-dir>/stats")
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    p.add_argument("--primary-metric", default=PRIMARY_METRIC,
                   help="metric used for the per-topic CSVs and the best-system pick")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    summary = collect_stats(args.runs_dir, out_dir=args.out_dir, alpha=args.alpha,
                            primary_metric=args.primary_metric)
    print(summary["_markdown"])

    primary = [c for c in summary["comparisons"] if c["metric"] == args.primary_metric]
    if primary:
        print(f"Paired comparisons on {args.primary_metric} (alpha={args.alpha}, Holm within "
              f"family; p_dlg = dialogue-level check):")
        for c in primary:
            p = "n/a" if c["p"] is None else f"{c['p']:.4f}"
            ph = "n/a" if c["p_holm"] is None else f"{c['p_holm']:.4f}"
            pd = "n/a" if c["p_dialogue_level"] is None else f"{c['p_dialogue_level']:.4f}"
            dz = "n/a" if c["cohen_dz"] is None else f"{c['cohen_dz']:+.3f}"
            flag = " SIG" if c["significant"] else ""
            unp = f" UNPAIRED={c['n_unpaired']}" if c["n_unpaired"] else ""
            print(f"  [{c['group']}/{c['collection']}] {c['baseline']} -> {c['treatment']}: "
                  f"delta={c['mean_delta']:+.4f} n={c['n']} W/L/T={c['wins']}/{c['losses']}/"
                  f"{c['ties']} dz={dz} p={p} p_holm={ph} p_dlg={pd}{flag}{unp}")
            # treated-subset view: most turns are tied because the condition never injected,
            # so the ITT line above averages over turns that had no treatment at all
            nch = c.get("n_changed") or 0
            if nch:
                pw = c.get("p_wilcoxon_changed")
                pw = "n/a" if pw is None else f"{pw:.4f}"
                print(f"      on the {nch} turn(s) it actually changed: "
                      f"delta={c['mean_delta_changed']:+.4f} wilcoxon p={pw}"
                      f"{'  (too few for a reliable p)' if nch < 6 else ''}")
    if summary["unpaired_runs"]:
        print(f"  WARNING unpaired runs (no un-ablated counterpart scored): "
              f"{', '.join(summary['unpaired_runs'])}")
    pool = {k: v for k, v in (summary.get("pooling_exposure") or {}).items() if v["warn"]}
    if pool:
        print()
        print("POOLING EXPOSURE -- judged@k differs across compared conditions:")
        for k, v in sorted(pool.items()):
            print(f"  {k}: spread {v['spread']:.4f}  "
                  f"({v['min']['run']} {v['min']['value']:.4f} .. "
                  f"{v['max']['run']} {v['max']['value']:.4f})")
        print("  A condition retrieving more UNJUDGED documents is penalised by nDCG whatever")
        print("  its true quality. Where coverage and nDCG move together, the causal story is")
        print("  ambiguous and the paper must say so.")
    if summary["best_system_cast19"]:
        b = summary["best_system_cast19"]
        margin = "" if b.get("margin") is None else f", margin {b['margin']:+.4f} over {b['runner_up']}"
        print()
        print(f"best system on cast19/{args.primary_metric}: {b['condition']} "
              f"({b[args.primary_metric]:.4f}){margin}")
        if b.get("near_tie"):
            print("  WARNING near-tie: the top two are within noise. Do not report this as a "
                  "winner without saying how it was chosen.")
        print(f"  {b['caveat']}")
    print(f"collect_stats: {summary['n_runs']} runs, {len(summary['comparisons'])} comparisons "
          f"-> {os.path.dirname(summary['outputs']['comparisons_csv'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
