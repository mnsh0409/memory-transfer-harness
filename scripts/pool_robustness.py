#!/usr/bin/env python3
"""Does the headline survive when pool coverage is taken out of it? Read-only diagnostic.

    python -m scripts.pool_robustness --collection cast19 --context memory_only

**The problem this exists to answer.** Under `memory_only`, judged@10 and nDCG@3 move together
almost perfectly: on CAsT 2019 the conditions run 0.2930..0.5870 in nDCG@3 and 0.4682..0.7855
in judged@10; on CAsT 2020, 0.1777..0.4790 and 0.4394..0.8514. Unjudged documents are scored
as non-relevant, so a condition whose queries fall outside the 2019/2020 pools is penalised
twice -- once by the metric and once by the pool's construction. Read pessimistically, the
entire B1-over-B0 gap could be pool coverage rather than effectiveness, and *no* comparison in
the design escapes that by having a smaller coverage spread: the within-system Mem0
`infer` ablation moves judged@10 by 0.2931, which is the same order as the B0->B1 gap of
0.3133. Picking the comparison with the smallest spread is not a defence.

**The standard answer, and the one used here.** Two measures that do not reward pool coverage:

  * **bpref** (Buckley & Voorhees) counts, for each relevant document, how many *judged*
    non-relevant documents outrank it. Unjudged documents are ignored entirely rather than
    treated as non-relevant, which is exactly the assumption under suspicion. Computed by
    pytrec_eval (== trec_eval) so it is the authoritative implementation.
  * **Condensed-list nDCG@3** (Sakai) removes unjudged documents from the ranking and scores
    what remains. A condition that retrieves many unjudged documents is neither punished nor
    rewarded for them; it is scored on the judged documents it did retrieve, in their order.

Neither is a substitute for nDCG on a complete pool -- both have known biases of their own, and
bpref in particular can *flatter* a system that retrieves few judged documents. They are used
here in one direction only: if a gap that is large under nDCG survives under both, pool coverage
is not a sufficient explanation for it. If it collapses under both, the nDCG gap should not be
reported as an effectiveness claim. Say which happened; do not report only the favourable one.

Reads run files, qrels and `stats_summary.json`. Writes nothing -- run artifacts stay
append-only and the reported metrics files are untouched.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eval.extrinsic import load_qrels, load_run  # noqa: E402

BASELINE = "B0_none"
QRELS = {"cast19": os.path.join("data", "cast", "cast19_qrels.txt"),
         "cast20": os.path.join("data", "cast", "cast20_qrels.txt")}


def condensed_ndcg_at_k(ranked, rels, k: int) -> float:
    """nDCG@k over the ranking with unjudged documents deleted (Sakai's condensed list).

    Linear gain and 1/log2(rank+1) discount, matching trec_eval's `ndcg_cut` convention as
    settled in eval/extrinsic.py. The ideal ranking is over all judged relevant documents for
    the query, so this is comparable across conditions for the same query."""
    judged = [(doc_id, s) for doc_id, s in ranked if doc_id in rels]
    gains = [rels[doc_id] for doc_id, _ in judged[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains) if g > 0)
    ideal = sorted((g for g in rels.values() if g > 0), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def bpref_per_qid(run, qrels) -> dict[str, float]:
    """Authoritative bpref via pytrec_eval. Falls back to nothing -- if trec_eval is not
    installed this diagnostic refuses rather than substituting a hand-rolled approximation,
    for the same reason evaluate() does."""
    try:
        import pytrec_eval  # type: ignore
    except ImportError:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "pytrec_eval is not installed, so bpref cannot be computed against the "
            "authoritative trec_eval implementation. Install pytrec_eval_terrier."
        )
    scored = {qid: {d: s for d, s in ranked} for qid, ranked in run.items() if qid in qrels}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"bpref"})
    return {qid: m["bpref"] for qid, m in evaluator.evaluate(scored).items()}


def _paired(a: dict[str, float], b: dict[str, float]):
    """Paired t over the qids scored in both, plus the tie count. Same unit of analysis as
    eval/stats.py -- the turn -- so the numbers here are comparable to the ones reported."""
    qids = sorted(set(a) & set(b))
    diffs = [b[q] - a[q] for q in qids]
    n = len(diffs)
    if n < 2:
        return {"n": n, "delta": 0.0, "p": 1.0, "dz": 0.0, "ties": n}
    mean = statistics.fmean(diffs)
    sd = statistics.stdev(diffs)
    ties = sum(1 for d in diffs if d == 0.0)
    if sd == 0:
        return {"n": n, "delta": mean, "p": 1.0, "dz": 0.0, "ties": ties}
    t = mean / (sd / math.sqrt(n))
    try:
        from scipy import stats as sps  # type: ignore
        p = float(sps.t.sf(abs(t), n - 1) * 2)
    except ImportError:  # pragma: no cover
        p = float("nan")
    return {"n": n, "delta": mean, "p": p, "dz": mean / sd, "ties": ties}


def build_rows(runs_dir: str, collection: str, injection: str = "rewrite_only",
               context: str = "memory_only", qrels_path: str | None = None) -> dict:
    with open(os.path.join(runs_dir, "stats_summary.json"), encoding="utf-8") as f:
        stats = json.load(f)
    cell = {rid: row for rid, row in stats.items()
            if row.get("collection") == collection and row.get("injection") == injection
            and row.get("context", "history_plus_memory") == context}
    if not cell:
        raise SystemExit(f"no runs scored for {collection}/{injection}/{context}")
    qrels_path = qrels_path or QRELS.get(collection)
    if not qrels_path or not os.path.exists(qrels_path):
        raise SystemExit(f"no qrels for {collection} at {qrels_path}")
    qrels = load_qrels(qrels_path)

    per_run = {}
    for rid, row in cell.items():
        run_file = os.path.join(runs_dir, "trec", f"{rid}.run")
        if not os.path.exists(run_file):
            continue
        run = load_run(run_file)
        label = row["condition"] + "".join(
            f"[{k}={v}]" for k, v in sorted((row.get("ablation") or {}).items()))
        if row.get("memory_k") not in (None, 5):
            label += f"[k={row['memory_k']}]"
        per_run[label] = {
            "run_id": rid,
            "ndcg_cut_3": row.get("ndcg_cut_3"),
            "judged_10": row.get("judged_10"),
            "bpref": bpref_per_qid(run, qrels),
            "cndcg_3": {qid: condensed_ndcg_at_k(ranked, qrels[qid], 3)
                        for qid, ranked in run.items() if qid in qrels},
        }

    base = next((lab for lab in per_run if lab == BASELINE), None)
    ceiling = next((lab for lab in per_run if lab == "B1_full_history"), None)
    return {"collection": collection, "context": context, "runs": per_run,
            "baseline": base, "ceiling": ceiling}


def render(res: dict) -> str:
    runs, base, ceil = res["runs"], res["baseline"], res["ceiling"]
    out = [f"Pool-robustness check - {res['collection']} / {res['context']}", ""]
    out.append(f"{'condition':26s} {'nDCG@3':>8s} {'judged@10':>10s} {'bpref':>8s} {'cnDCG@3':>8s}")
    out.append("-" * 66)
    for lab in sorted(runs):
        r = runs[lab]
        out.append(f"{lab:26s} {r['ndcg_cut_3']:8.4f} {r['judged_10']:10.4f} "
                   f"{statistics.fmean(r['bpref'].values()):8.4f} "
                   f"{statistics.fmean(r['cndcg_3'].values()):8.4f}")
    if not base:
        out.append("\nno B0_none in this cell -- deltas skipped")
        return "\n".join(out)

    out += ["", f"Paired vs {base}, unit = turn (same as the reported analysis):", ""]
    out.append(f"{'condition':26s} {'measure':>9s} {'delta':>9s} {'dz':>7s} {'p':>9s} {'ties':>5s}")
    out.append("-" * 70)
    for lab in sorted(runs):
        if lab == base:
            continue
        for m, key in (("bpref", "bpref"), ("cnDCG@3", "cndcg_3")):
            s = _paired(runs[base][key], runs[lab][key])
            out.append(f"{lab:26s} {m:>9s} {s['delta']:+9.4f} {s['dz']:+7.3f} "
                       f"{s['p']:9.2e} {s['ties']:5d}")

    if ceil and ceil in runs:
        out += ["", "Headroom delivered, by measure (a condition's share of the B1-B0 gap):", ""]
        for m, key in (("nDCG@3", None), ("bpref", "bpref"), ("cnDCG@3", "cndcg_3")):
            def val(lab):
                return (runs[lab]["ndcg_cut_3"] if key is None
                        else statistics.fmean(runs[lab][key].values()))
            span = val(ceil) - val(base)
            if abs(span) < 0.02:
                out.append(f"  {m:8s} headroom {span:+.4f} -- too small to express a share of")
                continue
            shares = "  ".join(
                f"{lab}={100 * (val(lab) - val(base)) / span:.0f}%"
                for lab in sorted(runs) if lab not in (base, ceil))
            out.append(f"  {m:8s} headroom {span:+.4f}   {shares}")
        out += ["",
                "If a share holds across all three measures, pool coverage does not explain it.",
                "If it collapses under bpref and cnDCG@3, do not report the nDCG gap as an",
                "effectiveness claim. Report which of these two happened."]
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--collection", required=True)
    ap.add_argument("--injection", default="rewrite_only")
    ap.add_argument("--context", default="memory_only",
                    help="memory_only (default here -- it is the cell with the headroom) "
                         "or history_plus_memory")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    res = build_rows(args.runs_dir, args.collection, args.injection, args.context)
    if args.json:
        slim = {lab: {"run_id": r["run_id"], "ndcg_cut_3": r["ndcg_cut_3"],
                      "judged_10": r["judged_10"],
                      "bpref": statistics.fmean(r["bpref"].values()),
                      "cndcg_3": statistics.fmean(r["cndcg_3"].values())}
                for lab, r in res["runs"].items()}
        print(json.dumps({"collection": res["collection"], "context": res["context"],
                          "runs": slim}, indent=2))
    else:
        print(render(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
