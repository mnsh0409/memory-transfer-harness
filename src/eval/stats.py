"""Statistical aggregation over scored runs -- the logic behind scripts/collect_stats.py
Reads what run_eval wrote and nothing else:

    runs/stats_summary.json                 one aggregate row per run_id
    runs/metrics/<run_id>.per_turn.jsonl    per-turn metrics for that run

and produces paired comparisons (Holm-corrected p-values, effect sizes, win/loss counts),
the per-collection metrics tables the frozen scope mandates (nDCG@3, nDCG@5, R@1000, MRR, judged@10),
and per-topic breakdowns. Everything is written under runs/stats/.

Decisions an auditor will ask about -- each is
deliberate and each is recorded in the output so the audit can check it without this docstring:

* PAIRING UNIT IS THE TURN (qid), paired t-test across turns, as pre-registered.
  Turns within a dialogue are not independent, so every
  comparison ALSO carries a dialogue-level test (unit = per-dialogue mean delta) as a
  robustness check. If the two disagree on significance, the paper says so.
* HOLM FAMILIES ARE EXPLICIT. One family per (group, collection, injection, metric).
  "main" = every condition vs B0_none plus every non-baseline condition vs B1_full_history,
  on the un-ablated default-k grid. "ablation" = each ablated or k-swept run vs its
  un-ablated counterpart. Correcting per metric is the convention; the family definition
  lives in `comparison_specs` and changing it is a visible code diff, not an analysis-time
  choice. The family key is stored on every comparison row.
* n IS THE NUMBER OF TURNS SCORED IN BOTH RUNS. run_eval scores only judged turns, so on
  CAsT 2019 that is 173, not 479 (the frozen scope "Frozen: metrics"). If the two runs scored
  different turn sets the comparison is flagged (`n_unpaired > 0`) -- that is a coverage
  anomaly upstream and is surfaced, never silently dropped.
* POOLED ROWS. When the same comparison exists on more than one collection, a "pooled" row
  concatenates the per-turn deltas across them (CAsT 2019 and 2020 turn ids are disjoint, so
  this is a plain union). Pooled rows are their own Holm family.
* STDLIB ONLY, like the rest of the harness. The Student-t CDF is the regularized incomplete
  beta function; tests/test_stats.py pins it against fixed reference values and cross-checks
  against scipy wherever scipy happens to be installed.
* OUTPUTS MAY BE REGENERATED IN PLACE. They are pure functions of (stats_summary.json,
  per-turn files) -- same reasoning as stats_summary.json itself -- so unlike
  run files / manifests they are not append-only.
"""
from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass, field

from .extrinsic import RELEVANCE_MEASURES

DEFAULT_MEMORY_K = 5            # RunConfig's default; un-ablated grid runs all use it
BASELINE_B0 = "B0_none"
BASELINE_B1 = "B1_full_history"
DEFAULT_ALPHA = 0.05
DEFAULT_CONTEXT = "history_plus_memory"   # config.CONTEXTS default
#: judged@k spread across conditions above which the pooling confound is flagged
POOLING_SPREAD_WARN = 0.05
#: margin on the primary metric below which "best system" is a near-tie, not a winner
NEAR_TIE = 0.01
PRIMARY_METRIC = "ndcg_cut_3"    # the frozen scope: the gate and "best system" are decided on nDCG@3

STATS_SUMMARY = "stats_summary.json"


# ---------------------------------------------------------------------------
# Student-t distribution (stdlib)
# ---------------------------------------------------------------------------

def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (modified Lentz's method)."""
    max_iter, eps, fpmin = 300, 3e-15, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
           + a * math.log(x) + b * math.log1p(-x))
    bt = math.exp(lbt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t: float, df: float) -> float:
    """Two-sided p-value P(|T| >= |t|) for Student's t with `df` degrees of freedom."""
    if df <= 0:
        raise ValueError("df must be positive")
    if math.isinf(t):
        return 0.0
    x = df / (df + t * t)
    return min(1.0, max(0.0, betainc(df / 2.0, 0.5, x)))


def t_critical(df: float, alpha: float = DEFAULT_ALPHA) -> float:
    """t such that P(|T| >= t) = alpha (two-sided critical value), by bisection."""
    lo, hi = 0.0, 1e4
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_two_sided_p(mid, df) > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Holm step-down
# ---------------------------------------------------------------------------

def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, same order as input. Monotone (adjusted values are
    non-decreasing in the sorted order) and capped at 1."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adjusted[i] = running
    return adjusted


# ---------------------------------------------------------------------------
# Paired comparison of two per-unit vectors
# ---------------------------------------------------------------------------

def wilcoxon_signed_rank(deltas: list[float]) -> tuple[float | None, int]:
    """Two-sided Wilcoxon signed-rank p over the NON-ZERO deltas, and how many there were.

    Why this exists alongside the t-test. On CAsT most conditions leave the rewrite untouched
    on nearly every judged turn (S1_mem0 changed 5 of 173; S2 changed 2), so the delta vector
    is **zero-inflated by construction**: the ties are not small effects, they are turns where
    the treatment was never delivered. Those structural zeros inflate n, shrink the standard
    deviation, deflate Cohen's d_z, and make the paired t-test answer a question nobody asked
    ("did memory help, averaged over turns that had no memory?").

    Dropping the zeros and ranking the rest is Wilcoxon's own treatment of ties and gives the
    per-protocol counterpart to the t-test's intention-to-treat: *given that the system did
    inject, did ranking move?* Report BOTH -- ITT alone hides the effect, treated-only alone
    hides how rarely it applied.

    Normal approximation with tie correction; exact for tiny samples is not attempted, so with
    very few non-zero deltas the p-value is indicative only -- which is why the count is
    returned with it and printed next to it."""
    nz = [d for d in deltas if d != 0]
    n = len(nz)
    if n == 0:
        return None, 0
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:                      # average ranks within ties on |delta|
        j = i
        while j + 1 < n and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    w_plus = sum(r for r, d in zip(ranks, nz) if d > 0)
    mean_w = n * (n + 1) / 4
    tie_term = 0.0
    counts: dict[float, int] = {}
    for r in ranks:
        counts[r] = counts.get(r, 0) + 1
    for c in counts.values():
        if c > 1:
            tie_term += c ** 3 - c
    var_w = (n * (n + 1) * (2 * n + 1) - tie_term / 2) / 24
    if var_w <= 0:
        return None, n
    z = (w_plus - mean_w) / math.sqrt(var_w)
    # two-sided normal tail, via the error function
    p = math.erfc(abs(z) / math.sqrt(2))
    return min(1.0, max(0.0, p)), n


def paired_stats(baseline: list[float], treatment: list[float],
                 alpha: float = DEFAULT_ALPHA) -> dict:
    """Paired t-test of treatment vs baseline over aligned units, plus effect sizes.

    Returns mean_baseline, mean_treatment, mean_delta (treatment - baseline), sd_delta,
    t, df, p (two-sided), cohen_dz (= mean_delta / sd_delta), ci_low/ci_high (t-based
    (1-alpha) CI on mean_delta), wins/losses/ties (per-unit sign of delta), n.
    With n < 2 or zero variance the test is undefined: t/cohen_dz are None and `degenerate`
    is True (p is 1.0 for an all-zero delta vector, 0.0 for a constant nonzero one)."""
    if len(baseline) != len(treatment):
        raise ValueError("baseline and treatment must be aligned (same length)")
    n = len(baseline)
    deltas = [t - b for b, t in zip(baseline, treatment)]
    out = {
        "n": n,
        "mean_baseline": _mean(baseline),
        "mean_treatment": _mean(treatment),
        "mean_delta": _mean(deltas),
        "wins": sum(1 for d in deltas if d > 0),
        "losses": sum(1 for d in deltas if d < 0),
        "ties": sum(1 for d in deltas if d == 0),
        "sd_delta": None, "t": None, "df": None, "p": None,
        "cohen_dz": None, "ci_low": None, "ci_high": None,
        "degenerate": True,
    }
    # treated-subset (per-protocol) view: the turns where the condition actually changed the
    # rewrite. See wilcoxon_signed_rank for why the ITT view alone is misleading here.
    nonzero = [d for d in deltas if d != 0]
    p_w, n_changed = wilcoxon_signed_rank(deltas)
    out["n_changed"] = n_changed
    out["mean_delta_changed"] = _mean(nonzero)
    out["p_wilcoxon_changed"] = p_w
    if n < 2:
        return out
    mean_d = out["mean_delta"]
    var = sum((d - mean_d) ** 2 for d in deltas) / (n - 1)
    sd = math.sqrt(var)
    out["sd_delta"] = sd
    out["df"] = n - 1
    if sd == 0.0:
        out["p"] = 1.0 if mean_d == 0.0 else 0.0
        out["ci_low"] = out["ci_high"] = mean_d
        return out
    se = sd / math.sqrt(n)
    t = mean_d / se
    tcrit = t_critical(n - 1, alpha)
    out.update({
        "t": t,
        "p": t_two_sided_p(t, n - 1),
        "cohen_dz": mean_d / sd,
        "ci_low": mean_d - tcrit * se,
        "ci_high": mean_d + tcrit * se,
        "degenerate": False,
    })
    return out


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


# ---------------------------------------------------------------------------
# Loading what run_eval wrote
# ---------------------------------------------------------------------------

@dataclass
class RunRecord:
    run_id: str
    condition: str
    collection: str
    injection: str
    context: str
    ablation: dict
    memory_k: int
    aggregate: dict                       # the stats_summary row, verbatim
    per_turn: dict = field(default_factory=dict)   # turn_id -> {metric: value}

    @property
    def label(self) -> str:
        """Condition plus whatever distinguishes this run from the un-ablated default."""
        extras = [f"{k}={v}" for k, v in sorted(self.ablation.items())]
        if self.memory_k != DEFAULT_MEMORY_K:
            extras.append(f"k={self.memory_k}")
        return self.condition + (f"[{','.join(extras)}]" if extras else "")

    @property
    def is_default_config(self) -> bool:
        return not self.ablation and self.memory_k == DEFAULT_MEMORY_K


def cell_name(record) -> str:
    """Grouping key for tables and Holm families. The context axis is a CELL dimension, not an
    ablation: `history_plus_memory` and `memory_only` are different experiments, and comparing
    a condition across them would be meaningless. Default context renders the historical
    "collection/injection" key unchanged so existing keys and families keep their names."""
    base = f"{record.collection}/{record.injection}"
    return base if record.context == DEFAULT_CONTEXT else f"{base}/{record.context}"


def dialogue_id(turn_id: str, collection: str) -> str:
    """CAsT turn ids are `<topic>_<turn>` (31_1); iKAT's are `<topic>-<turn>` where the topic
    itself contains a hyphen (9-1-2 -> dialogue 9-1)."""
    sep = "-" if collection.startswith("ikat") else "_"
    return turn_id.rsplit(sep, 1)[0]


def load_per_turn(path: str) -> dict[str, dict]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            out[row["turn_id"]] = {k: v for k, v in row.items() if k != "turn_id"}
    return out


def _memory_k_for(run_id: str, row: dict, runs_dir: str) -> int:
    """run_eval rows carry memory_k; for rows scored before that field existed, recover it
    from the eval manifest's canonical config. Refuse to guess: a k-sweep run mistaken for
    the default-k run would land in the wrong Holm family."""
    if "memory_k" in row:
        return int(row["memory_k"])
    manifest_path = os.path.join(runs_dir, "manifests", f"{run_id}.eval.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        config = manifest.get("config")
        if isinstance(config, str):
            config = json.loads(config)
        if isinstance(config, dict) and "memory_k" in config:
            return int(config["memory_k"])
    raise ValueError(
        f"{run_id}: stats_summary row has no memory_k and {manifest_path} does not carry it "
        "-- re-run scripts/run_eval for this config (scoring is free) rather than assuming k"
    )


def load_runs(runs_dir: str) -> list[RunRecord]:
    """Every scored run, with its per-turn metrics attached. A stats_summary entry whose
    per-turn file is missing is an error, not a skip -- the comparisons would silently lose
    a run."""
    stats_path = os.path.join(runs_dir, STATS_SUMMARY)
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"{stats_path} not found -- run scripts/run_eval first")
    with open(stats_path, encoding="utf-8") as f:
        stats = json.load(f)
    records = []
    for run_id in sorted(stats):
        row = stats[run_id]
        per_turn_path = os.path.join(runs_dir, "metrics", f"{run_id}.per_turn.jsonl")
        if not os.path.exists(per_turn_path):
            raise FileNotFoundError(
                f"{per_turn_path} missing for run {run_id} listed in {stats_path}"
            )
        records.append(RunRecord(
            run_id=run_id,
            condition=row["condition"],
            collection=row["collection"],
            injection=row["injection"],
            # rows scored before the context axis existed ran under the
            # history_plus_memory design, which is what that default means
            context=row.get("context", "history_plus_memory"),
            ablation=dict(row.get("ablation") or {}),
            memory_k=_memory_k_for(run_id, row, runs_dir),
            aggregate=row,
            per_turn=load_per_turn(per_turn_path),
        ))
    return records


def metric_names(records: list[RunRecord]) -> list[str]:
    """Relevance measures first, then every judged_<k> key present in the per-turn files."""
    judged = sorted({k for r in records for m in r.per_turn.values()
                     for k in m if k.startswith("judged_")})
    return [*RELEVANCE_MEASURES, *judged]


# ---------------------------------------------------------------------------
# Which pairs to compare (the Holm families)
# ---------------------------------------------------------------------------

def comparison_specs(records: list[RunRecord]) -> tuple[list[dict], list[RunRecord]]:
    """Enumerate (baseline_run, treatment_run, group) triples.

    group "main":     on the default-config grid, every condition vs B0 and every condition
                      other than B0/B1 vs B1 -- within a (collection, injection).
    group "ablation": every non-default run vs the default-config run of the same condition
                      in the same (collection, injection).
    Also returns the runs that cannot be paired (an ablation with no un-ablated counterpart)
    so the caller can report them rather than lose them."""
    by_cell: dict[tuple, list[RunRecord]] = {}
    for r in records:
        by_cell.setdefault(cell_name(r), []).append(r)

    specs, unpaired = [], []
    for _cell, runs in sorted(by_cell.items()):
        defaults = {r.condition: r for r in runs if r.is_default_config}
        # main family
        b0, b1 = defaults.get(BASELINE_B0), defaults.get(BASELINE_B1)
        for cond in sorted(defaults):
            r = defaults[cond]
            if b0 is not None and cond != BASELINE_B0:
                specs.append({"group": "main", "baseline": b0, "treatment": r})
            if b1 is not None and cond not in (BASELINE_B0, BASELINE_B1):
                specs.append({"group": "main", "baseline": b1, "treatment": r})
        # systems family: every S-vs-S pair, in its own Holm family.
        #
        # The paper's headline is a system-vs-system contrast ("selective retrieval of raw
        # content beats extractive memory": S3_amem over S1_mem0, +0.2844, p=5e-19), and a
        # comparison that is REPORTED must be inside the family it is corrected in -- a cold
        # auditor's first question is whether the Holm family covers the comparisons actually
        # made. It is deliberately NOT folded into "main": that family answers "does this
        # condition beat the baselines", and adding k(k-1)/2 system pairs to it would inflate
        # the correction on the baseline comparisons for an unrelated question.
        systems = sorted(c for c in defaults if c.startswith("S"))
        for i, a in enumerate(systems):
            for b in systems[i + 1:]:
                specs.append({"group": "systems", "baseline": defaults[a],
                              "treatment": defaults[b]})
        # ablation family
        for r in sorted(runs, key=lambda r: r.label):
            if r.is_default_config:
                continue
            base = defaults.get(r.condition)
            if base is None:
                unpaired.append(r)
                continue
            specs.append({"group": "ablation", "baseline": base, "treatment": r})
    return specs, unpaired


def _aligned(baseline: RunRecord, treatment: RunRecord, metric: str):
    common = sorted(set(baseline.per_turn) & set(treatment.per_turn))
    a = [baseline.per_turn[q][metric] for q in common]
    b = [treatment.per_turn[q][metric] for q in common]
    return common, a, b


def _dialogue_level(common: list[str], a: list[float], b: list[float],
                    collection: str, alpha: float) -> dict:
    """Collapse per-turn deltas to per-dialogue means and test those (robustness check for
    the turns-within-dialogue clustering the audit prompt asks about)."""
    per_dialogue: dict[str, list[tuple[float, float]]] = {}
    for q, x, y in zip(common, a, b):
        per_dialogue.setdefault(dialogue_id(q, collection), []).append((x, y))
    dids = sorted(per_dialogue)
    da = [_mean([x for x, _ in per_dialogue[d]]) for d in dids]
    db = [_mean([y for _, y in per_dialogue[d]]) for d in dids]
    s = paired_stats(da, db, alpha)
    return {"n_dialogues": len(dids), "p_dialogue_level": s["p"],
            "t_dialogue_level": s["t"], "per_dialogue": dict(zip(dids, zip(da, db)))}


def build_comparisons(records: list[RunRecord], alpha: float = DEFAULT_ALPHA) -> dict:
    """All paired comparisons with Holm adjustment applied within each family.
    Returns {"comparisons": [...], "families": {...}, "unpaired_runs": [...], ...}."""
    metrics = metric_names(records)
    # judged@k is a POOLING DIAGNOSTIC, not an effectiveness claim: it measures how much of a
    # run's top-k the assessors happened to judge. Significance-testing it would assert that a
    # condition is "significantly better judged", which is not a claim about retrieval, and it
    # would inflate every Holm family with hypotheses nobody is making. It stays in the metrics
    # TABLES (the frozen scope requires it beside every nDCG column) and out of the inference.
    inferential = [m for m in metrics if not m.startswith("judged_")]
    specs, unpaired = comparison_specs(records)

    rows: list[dict] = []
    pooled_deltas: dict[tuple, dict] = {}   # (group, injection, metric, labels) -> pooled vectors
    for spec in specs:
        base, treat = spec["baseline"], spec["treatment"]
        for metric in inferential:
            common, a, b = _aligned(base, treat, metric)
            s = paired_stats(a, b, alpha)
            dl = _dialogue_level(common, a, b, base.collection, alpha)
            row = {
                "family": f"{spec['group']}/{cell_name(base)}/{metric}",
                "group": spec["group"],
                "collection": base.collection,
                "cell": cell_name(base),
                "injection": base.injection,
                "metric": metric,
                "baseline": base.label, "treatment": treat.label,
                "baseline_run_id": base.run_id, "treatment_run_id": treat.run_id,
                "n_unpaired": len(set(base.per_turn) ^ set(treat.per_turn)),
                **s,
                "n_dialogues": dl["n_dialogues"],
                "p_dialogue_level": dl["p_dialogue_level"],
                "t_dialogue_level": dl["t_dialogue_level"],
                "_per_dialogue": dl["per_dialogue"],
            }
            rows.append(row)
            key = (spec["group"], base.injection, base.context, metric,
                   base.label, treat.label)
            pooled = pooled_deltas.setdefault(key, {"a": [], "b": [], "collections": []})
            pooled["a"].extend(a)
            pooled["b"].extend(b)
            pooled["collections"].append(base.collection)

    # pooled rows, only where the same comparison exists on >1 collection
    for key, pooled in sorted(pooled_deltas.items(), key=lambda kv: kv[0]):
        if len(pooled["collections"]) < 2:
            continue
        group, injection, _ctx, metric, blabel, tlabel = key
        s = paired_stats(pooled["a"], pooled["b"], alpha)
        name = "pooled(" + "+".join(sorted(pooled["collections"])) + ")"
        rows.append({
            "family": f"{group}/{name}/{injection}/{metric}",
            "group": group, "collection": name, "injection": injection, "metric": metric,
            "baseline": blabel, "treatment": tlabel,
            "baseline_run_id": None, "treatment_run_id": None,
            "n_unpaired": None, **s,
            "n_dialogues": None, "p_dialogue_level": None, "t_dialogue_level": None,
            "_per_dialogue": {},
        })

    # Holm within each family (pvals that are None -- n<2 -- are excluded from the family)
    families: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        families.setdefault(row["family"], []).append(i)
    family_info = {}
    for fam, idxs in families.items():
        testable = [i for i in idxs if rows[i]["p"] is not None]
        adjusted = holm([rows[i]["p"] for i in testable]) if testable else []
        for i, p_adj in zip(testable, adjusted):
            rows[i]["p_holm"] = p_adj
            rows[i]["significant"] = p_adj < alpha
        for i in idxs:
            rows[i].setdefault("p_holm", None)
            rows[i].setdefault("significant", None)
            rows[i]["family_size"] = len(testable)
        family_info[fam] = {
            "size": len(testable),
            "comparisons": [f"{rows[i]['baseline']} -> {rows[i]['treatment']}" for i in idxs],
        }
    return {
        "comparisons": rows,
        "families": family_info,
        "unpaired_runs": [r.run_id for r in unpaired],
        "alpha": alpha,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Tables (the frozen scope "Frozen: metrics": nDCG@3, nDCG@5, R@1000, MRR, judged@10, every table)
# ---------------------------------------------------------------------------

METRIC_HEADERS = {
    "ndcg_cut_3": "nDCG@3", "ndcg_cut_5": "nDCG@5", "recall_1000": "R@1000",
    "recip_rank": "MRR",
}


def _header(metric: str) -> str:
    if metric in METRIC_HEADERS:
        return METRIC_HEADERS[metric]
    if metric.startswith("judged_"):
        return "judged@" + metric.split("_", 1)[1]
    return metric


def metrics_table(records: list[RunRecord], comparisons: list[dict], cell: str,
                  metrics: list[str]) -> list[dict]:
    """One row per run in the cell, with significance markers per metric:
    '*' Holm-significant vs B0_none, '+' Holm-significant vs B1_full_history (main family);
    '^' Holm-significant vs its own un-ablated counterpart (ablation family)."""
    sig: dict[tuple, set] = {}
    for c in comparisons:
        if c.get("cell") != cell or not c["significant"]:
            continue
        if c["group"] == "main" and c["baseline"] == BASELINE_B0:
            mark = "*"
        elif c["group"] == "main" and c["baseline"] == BASELINE_B1:
            mark = "+"
        elif c["group"] == "ablation":
            mark = "^"
        else:
            continue
        sig.setdefault((c["treatment"], c["metric"]), set()).add(mark)

    rows = []
    in_cell = [r for r in records if cell_name(r) == cell]
    order = {"B0_none": 0, "B1_full_history": 1, "B2_gold_ptkb": 2}
    b0s = [r for r in in_cell if r.condition == BASELINE_B0]

    def _identical_to_b0(r) -> bool:
        """True when this run's per-turn metrics are bitwise a B0 run's in the same cell.

        Such a run never exercised its mechanism, and presenting it as a measurement is how a
        broken condition becomes the tallest bar in a headroom figure. Flagged here so every
        consumer -- figures, tables, best-system selection -- can drop it on one rule."""
        if r.condition == BASELINE_B0:
            return False
        for b0 in b0s:
            common = set(r.per_turn) & set(b0.per_turn)
            if common and all(r.per_turn[t] == b0.per_turn[t] for t in common):
                return True
        return False

    for r in sorted(in_cell, key=lambda r: (order.get(r.condition, 9), r.label)):
        row = {"run": r.label, "run_id": r.run_id, "n": r.aggregate.get("n_scored_turns"),
               # carried through for the cost figure (RQ1); None on pre-accounting runs
               "usage": r.aggregate.get("usage"),
               "identical_to_B0": _identical_to_b0(r)}
        for m in metrics:
            row[m] = r.aggregate.get(m)
            row[m + "_marks"] = "".join(sorted(sig.get((r.label, m), set())))
        rows.append(row)
    return rows


def render_markdown_table(rows: list[dict], metrics: list[str], title: str) -> str:
    head = ["run", "n", *[_header(m) for m in metrics]]
    lines = [f"**{title}**", "", "| " + " | ".join(head) + " |",
             "|" + "|".join("---" for _ in head) + "|"]
    for r in rows:
        cells = [r["run"], str(r["n"])]
        for m in metrics:
            v = r.get(m)
            cells.append(("-" if v is None else f"{v:.4f}") + r.get(m + "_marks", ""))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("`*` Holm-sig. vs B0_none, `+` vs B1_full_history, `^` vs own un-ablated run "
                 "(alpha in collect_stats.json).")
    return "\n".join(lines) + "\n"


def render_latex_table(rows: list[dict], metrics: list[str], title: str) -> str:
    cols = "l r " + " ".join("r" for _ in metrics)
    lines = [r"\begin{tabular}{" + cols + "}", r"\toprule",
             " & ".join(["Run", "$n$", *[_header(m) for m in metrics]]) + r" \\",
             r"\midrule"]
    for r in rows:
        cells = [r["run"].replace("_", r"\_"), str(r["n"])]
        for m in metrics:
            v = r.get(m)
            marks = r.get(m + "_marks", "")
            marks = marks.replace("^", r"$\wedge$").replace("+", r"$\dagger$")
            cells.append(("-" if v is None else f"{v:.3f}") + marks)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", f"% {title}"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

COMPARISON_COLUMNS = [
    "group", "collection", "injection", "metric", "baseline", "treatment", "n", "n_unpaired",
    "mean_baseline", "mean_treatment", "mean_delta", "ci_low", "ci_high", "sd_delta",
    "cohen_dz", "t", "df", "p", "p_holm", "family_size", "significant", "wins", "losses",
    "ties", "n_changed", "mean_delta_changed", "p_wilcoxon_changed",
    "n_dialogues", "p_dialogue_level", "degenerate", "family", "cell",
    "baseline_run_id", "treatment_run_id",
]


def collect_stats(runs_dir: str = "runs", out_dir: str | None = None,
                  alpha: float = DEFAULT_ALPHA, primary_metric: str = PRIMARY_METRIC) -> dict:
    """Run everything and write {out_dir or runs_dir/stats}/. Returns the summary dict that
    was written as collect_stats.json, plus a "_markdown" key with the rendered tables."""
    out_dir = out_dir or os.path.join(runs_dir, "stats")
    records = load_runs(runs_dir)
    if not records:
        raise ValueError(f"{os.path.join(runs_dir, STATS_SUMMARY)} has no runs")
    result = build_comparisons(records, alpha)
    metrics = result["metrics"]
    if primary_metric not in metrics:
        raise ValueError(f"primary metric {primary_metric!r} not among {metrics}")

    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)

    # comparisons.csv
    comparisons_csv = os.path.join(out_dir, "comparisons.csv")
    with open(comparisons_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COMPARISON_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for row in result["comparisons"]:
            w.writerow(row)

    # per-topic breakdown on the primary metric, main family only
    per_topic_paths = []
    for c in result["comparisons"]:
        if c["group"] != "main" or c["metric"] != primary_metric or not c["_per_dialogue"]:
            continue
        fname = (f"per_topic_{c['collection']}_{c['baseline']}_vs_{c['treatment']}"
                 f"_{primary_metric}.csv")
        path = os.path.join(out_dir, fname)
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dialogue_id", "baseline_mean", "treatment_mean", "delta"])
            for did, (a, b) in sorted(c["_per_dialogue"].items()):
                w.writerow([did, f"{a:.6f}", f"{b:.6f}", f"{b - a:.6f}"])
        per_topic_paths.append(path)

    # metrics tables per (collection, injection)
    tables, table_paths, markdown = {}, [], []
    cells = sorted({cell_name(r) for r in records})
    for cell in cells:
        rows = metrics_table(records, result["comparisons"], cell, metrics)
        title = cell.replace("/", " / ")
        tables[cell] = rows
        stem = os.path.join(out_dir, "tables", "main_" + cell.replace("/", "_"))
        md = render_markdown_table(rows, metrics, title)
        with open(stem + ".md", "w", encoding="utf-8") as f:
            f.write(md)
        with open(stem + ".tex", "w", encoding="utf-8") as f:
            f.write(render_latex_table(rows, metrics, title))
        table_paths += [stem + ".md", stem + ".tex"]
        markdown.append(md)

    # Pooling exposure: judged@k varies ACROSS conditions, and a condition retrieving more
    # unjudged documents is penalised by nDCG regardless of true quality. This is the
    # TREC-veteran attack, so the spread is computed and surfaced rather than left for a
    # reviewer to notice.
    pooling = {}
    judged_metrics = [m for m in metrics if m.startswith("judged_")]
    for cell in sorted({cell_name(r) for r in records}):
        for jm in judged_metrics:
            vals = [(r.label, r.aggregate.get(jm)) for r in records
                    if cell_name(r) == cell and r.aggregate.get(jm) is not None]
            if len(vals) < 2:
                continue
            lo, hi = min(vals, key=lambda t: t[1]), max(vals, key=lambda t: t[1])
            pooling[f"{cell}/{jm}"] = {
                "spread": hi[1] - lo[1], "min": {"run": lo[0], "value": lo[1]},
                "max": {"run": hi[0], "value": hi[1]},
                "warn": (hi[1] - lo[1]) > POOLING_SPREAD_WARN,
            }

    # best system on the primary metric / cast19 (the frozen scope scopes the k-sweep to it).
    # NOT a silent argmax: the margin over the runner-up is reported, a near-tie is flagged,
    # and the pick is recorded as having been made on the same data the sweep is reported on.
    best_system = None
    cast19 = [r for r in records if r.collection == "cast19" and r.is_default_config
              and r.condition.startswith("S")]
    if cast19:
        # A system whose per-turn metrics are identical to B0's never exercised its mechanism;
        # crowning it "best" crowns the no-memory baseline wearing memory machinery. The Aug 24
        # cold stats audit caught exactly that: S5_gam's broken run was bitwise B0 on all 173
        # turns and was still selected. Identical-to-B0 candidates are excluded (flagged, not
        # hidden), and if every candidate is identical the block says so instead of picking.
        def identical_to_b0(r):
            # identical to ANY B0 run in the candidate's own cell (same context -- the cast19
            # candidate list spans both context designs). A cell can legitimately hold more
            # than one B0 (the retired gate run plus the grid run); matching either one means
            # the mechanism never changed retrieval.
            b0s = [b for b in records if b.collection == "cast19" and b.is_default_config
                   and b.condition == "B0_none" and b.context == r.context]
            for b0 in b0s:
                common = set(r.per_turn) & set(b0.per_turn)
                if common and all(
                        r.per_turn[t].get(primary_metric) == b0.per_turn[t].get(primary_metric)
                        for t in common):
                    return True
            return False

        flagged = {r.run_id: identical_to_b0(r) for r in cast19}
        eligible = [r for r in cast19 if not flagged[r.run_id]]
        if not eligible:
            best_system = {
                "condition": None,
                "caveat": ("every candidate system's per-turn metrics are identical to "
                           "B0_none's -- no memory mechanism changed retrieval in this cell, "
                           "so there is no best system to select."),
                "identical_to_B0": sorted(r.condition for r in cast19),
            }
        else:
            ranked = sorted(eligible,
                            key=lambda r: r.aggregate.get(primary_metric, float("-inf")),
                            reverse=True)
            best = ranked[0]
            runner = ranked[1] if len(ranked) > 1 else None
            margin = (best.aggregate.get(primary_metric, 0)
                      - runner.aggregate.get(primary_metric, 0)) if runner else None
            best_system = {
                "condition": best.condition, "run_id": best.run_id,
                primary_metric: best.aggregate.get(primary_metric),
                "runner_up": runner.condition if runner else None,
                "margin": margin,
                "near_tie": (margin is not None and margin < NEAR_TIE),
                "excluded_identical_to_B0": sorted(r.condition for r in cast19
                                                   if flagged[r.run_id]),
                "caveat": ("selected on the same data the sweep is reported on, as the frozen "
                           "scope specifies; say so in the text. Candidates whose per-turn "
                           "metrics are bitwise B0's are excluded and listed."),
            }

    public = [{k: v for k, v in c.items() if not k.startswith("_")}
              for c in result["comparisons"]]
    summary = {
        "source": os.path.join(runs_dir, STATS_SUMMARY),
        "alpha": alpha,
        "primary_metric": primary_metric,
        "metrics": metrics,
        "n_runs": len(records),
        "runs": [{"run_id": r.run_id, "label": r.label, "collection": r.collection,
                  "injection": r.injection, "n_scored_turns": r.aggregate.get("n_scored_turns"),
                  "metrics_source": r.aggregate.get("metrics_source"),
                  "coverage": r.aggregate.get("coverage")} for r in records],
        "families": result["families"],
        "pooling_exposure": pooling,
        "unpaired_runs": result["unpaired_runs"],
        "comparisons": public,
        "tables": tables,
        "best_system_cast19": best_system,
        "outputs": {"comparisons_csv": comparisons_csv,
                    "tables": table_paths, "per_topic": per_topic_paths},
        "notes": [
            "pairing unit = turn (qid); p_dialogue_level is the per-dialogue robustness check",
            "n = turns scored in BOTH runs (judged turns only: 173 on CAsT 2019, not 479)",
            "Holm applied within each family = (group, collection, injection, metric)",
            "metrics_source must read 'pytrec_eval' for any reported number",
        ],
    }
    with open(os.path.join(out_dir, "collect_stats.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    summary["_markdown"] = "\n".join(markdown)
    return summary
