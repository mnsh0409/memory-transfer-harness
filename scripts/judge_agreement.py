#!/usr/bin/env python3
"""Turn a filled-in manual_check CSV into the judge-agreement number the paper reports.

    python -m scripts.judge_agreement --blind manual_check/RUN.csv     # step 1: make a copy to annotate
    python -m scripts.judge_agreement manual_check/RUN.filled.csv      # step 2: score it

`run_intrinsic` writes `manual_check/{run_id}.csv` with an empty `human_verdict` column. That
column validates the LLM judge and therefore the whole intrinsic axis, so a human fills it and
nothing in this repo ever may. This script only *reads* it.

Two methodological points it enforces, because both would otherwise quietly produce a wrong
number:

**Blind first.** The file ships with `judge_verdict` visible. Annotating next to the judge's
answer anchors the annotator, and an agreement statistic computed that way is worthless. Use
`--blind` to write an annotation copy with that column removed; score the original afterwards
(rows are matched on `idx`).

**The sample is STRATIFIED by verdict** (`intrinsic_runner.stratified_sample`, so rare classes
are actually inspected), which means it is deliberately *not* representative. Overall agreement
on such a sample does not estimate population agreement — the rare classes are over-weighted.
This script therefore reports agreement PER STRATUM as the headline, and labels the pooled
figure as sample-level only. Report per-stratum numbers, or reweight by the population verdict
counts (printed by `run_intrinsic`) before claiming a single agreement figure.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

VERDICTS = ("ENTAILED", "PARTIAL", "ABSENT")
# what each verdict is worth in the P/R/F1 (eval.intrinsic.score_reconstruction's partial_credit)
CREDIT = {"ENTAILED": 1.0, "PARTIAL": 0.5, "ABSENT": 0.0}


def blind(path: str) -> str:
    """Write an annotation copy without the judge's answer."""
    out = os.path.splitext(path)[0] + ".to_annotate.csv"
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    fields = [c for c in rows[0] if c != "judge_verdict"] if rows else []
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: v for k, v in r.items() if k != "judge_verdict"})
    return out


def _kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Cohen's kappa over the 3 verdict classes."""
    n = len(pairs)
    if n == 0:
        return None
    observed = sum(1 for a, b in pairs if a == b) / n
    expected = 0.0
    for v in VERDICTS:
        pa = sum(1 for a, _ in pairs if a == v) / n
        pb = sum(1 for _, b in pairs if b == v) / n
        expected += pa * pb
    if expected >= 1.0:
        return None                      # degenerate: everything in one class
    return (observed - expected) / (1 - expected)


def score(path: str, judged_path: str | None = None) -> dict:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if rows and "judge_verdict" not in rows[0]:
        # a blinded annotation file: rejoin the judge's answers from the original by idx
        src = judged_path or path.replace(".to_annotate.csv", ".csv")
        with open(src, encoding="utf-8", newline="") as f:
            judge = {r["idx"]: r["judge_verdict"] for r in csv.DictReader(f)}
        for r in rows:
            r["judge_verdict"] = judge.get(r["idx"], "")

    filled, unfilled, invalid = [], 0, []
    for r in rows:
        h = (r.get("human_verdict") or "").strip().upper()
        if not h:
            unfilled += 1
            continue
        if h not in VERDICTS:
            invalid.append((r["idx"], h))
            continue
        filled.append((r["judge_verdict"].strip().upper(), h))

    per_stratum = {}
    for v in VERDICTS:
        pairs = [(j, h) for j, h in filled if j == v]
        if pairs:
            per_stratum[v] = {"n": len(pairs),
                              "agree": sum(1 for j, h in pairs if j == h),
                              "rate": sum(1 for j, h in pairs if j == h) / len(pairs)}
    confusion = {j: {h: 0 for h in VERDICTS} for j in VERDICTS}
    for j, h in filled:
        confusion[j][h] += 1
    credit_delta = [CREDIT[h] - CREDIT[j] for j, h in filled]
    return {
        "path": path, "n_rows": len(rows), "n_filled": len(filled),
        "n_unfilled": unfilled, "invalid": invalid,
        "per_stratum": per_stratum,
        "sample_agreement": (sum(1 for j, h in filled if j == h) / len(filled)) if filled else None,
        "cohen_kappa": _kappa(filled),
        "mean_credit_delta": (sum(credit_delta) / len(credit_delta)) if credit_delta else None,
        "confusion": confusion,
    }


def render(r: dict) -> str:
    L = [f"Judge agreement - {os.path.basename(r['path'])}", ""]
    if r["n_unfilled"]:
        L.append(f"  {r['n_unfilled']} of {r['n_rows']} rows still have an empty human_verdict.")
    if r["invalid"]:
        L.append(f"  INVALID verdicts (must be one of {', '.join(VERDICTS)}): {r['invalid'][:5]}")
    if not r["n_filled"]:
        L.append("  Nothing scored yet.")
        return "\n".join(L)
    L.append(f"  scored: {r['n_filled']} decisions")
    L.append("")
    L.append("  PER STRATUM (the headline -- the sample is stratified, so these are what you report):")
    for v, d in r["per_stratum"].items():
        L.append(f"    judge said {v:<9} n={d['n']:<4} human agreed {d['agree']:<4} ({100*d['rate']:.0f}%)")
    L.append("")
    L.append(f"  sample-level agreement : {100*r['sample_agreement']:.1f}%  "
             "(NOT a population estimate - strata are over-weighted by design)")
    k = r["cohen_kappa"]
    L.append(f"  Cohen's kappa          : {'n/a' if k is None else f'{k:+.3f}'}")
    L.append(f"  mean credit delta      : {r['mean_credit_delta']:+.3f}  "
             "(human minus judge, in P/R/F1 credit; + means the judge under-credited)")
    L.append("")
    L.append("  confusion (rows = judge, cols = human):")
    L.append("             " + "".join(f"{h:>10}" for h in VERDICTS))
    for j in VERDICTS:
        L.append(f"    {j:<9}" + "".join(f"{r['confusion'][j][h]:>10}" for h in VERDICTS))
    return "\n".join(L)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", help="the manual_check CSV (blinded copy or the original)")
    p.add_argument("--blind", action="store_true",
                   help="write a .to_annotate.csv copy without judge_verdict, then exit")
    p.add_argument("--judged", default=None,
                   help="original file to rejoin judge verdicts from (default: infer)")
    args = p.parse_args(argv)
    if args.blind:
        out = blind(args.csv)
        print(f"annotation copy -> {out}")
        print("Fill the human_verdict column with ENTAILED / PARTIAL / ABSENT, then run this "
              "script on it without --blind.")
        return 0
    print(render(score(args.csv, args.judged)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
