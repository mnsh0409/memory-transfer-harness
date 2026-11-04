#!/usr/bin/env python3
"""Intervals on the delivered share, and the sample size injection rate leaves. Read-only.

    python -m scripts.share_inference
    python -m scripts.share_inference --collection cast19 --reference M1_manual
    python -m scripts.share_inference --json > share_inference.json

**Three things the paper states without inference, answered here.**

1. *The delivered share has no uncertainty attached.* Table 1 reports
   ``(S - B0) / (R - B0)`` as point estimates. It is a ratio of two paired mean differences, so
   its interval comes from Fieller's theorem rather than from a t-test on either difference.
   Turns nest within dialogues, so the variances are cluster-robust by dialogue, with a
   dialogue-level bootstrap alongside as a check that does not lean on the normal approximation.
   Fieller also supplies the principled rule the figures code approximates with an ad hoc
   ``MIN_HEADROOM = 0.02``: the interval is bounded exactly when the denominator (the headroom)
   is itself significantly non-zero, and is reported as unbounded otherwise.

2. *"A null means could not measure" is argued, not derived.* The pipeline is deterministic at
   temperature 0 with every model call cached, so a turn on which the memory system leaves the
   query byte-identical to B0's produces a byte-identical ranking: its paired difference is
   exactly zero. That is checked here, not assumed (``untreated != 0`` must read 0). Given it,
   the noncentrality of the paired mean test over all n turns is

       lambda = sqrt(n_eff) * mu1 / sigma1,  n_eff = n_T * sigma1^2 / (sigma1^2 + (1-p) mu1^2)

   where p is the injection rate, n_T = n p the treated turns, and mu1, sigma1 the mean and
   standard deviation of the paired difference on treated turns. Since n_eff <= n_T, testing on
   all n turns is never more informative than testing on the n_T turns memory actually touched.
   **The effective sample size is the number of treated turns, not n.** The minimum detectable
   conditional effect follows from n_T; where it exceeds the local headroom, even a system that
   delivered all of it would not be reliably distinguished from B0.

3. *The selectivity x quality decomposition is an identity, not an approximation.* With exact
   zeros off-treatment,

       share = p * q_T * c,  q_T = mean(S-B0 | T) / mean(R-B0 | T),  c = mean(R-B0 | T) / mean(R-B0)

   i.e. injection rate, times the share delivered where memory fired, times how much of the
   headroom sits on those turns. The residual is reported and must be zero to float precision.

Reads ``runs/stats_summary.json``, ``runs/metrics/*.per_turn.jsonl`` and
``runs/rewrites/*.jsonl``. Writes nothing; ``--json`` prints to stdout. The submitted paper's
numbers and the committed stats snapshot are untouched.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

BASELINE = "B0_none"
REFERENCES = ("M1_manual", "B1_full_history")
METRIC = "ndcg_cut_3"
MEMORY_K = 5
# Table order: references first, then systems best-to-worst, ablations after their parent.
ORDER = ("M1_manual", "B1_full_history", "S3_amem", "S3_amem[evolution=False]", "S1_mem0",
         "S1_mem0[infer=False]", "S2_graphiti", "S5_gam")


# --------------------------------------------------------------------------- inference

def cluster_cov(a, b, clusters=None):
    """Cluster-robust (CR1) variances and covariance of the two sample means.

    With ``clusters=None`` every unit is its own cluster and this reduces exactly to the usual
    ``s^2 / n``. Returns ``(var_mean_a, var_mean_b, cov_means, n_clusters)``."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n = a.size
    labels = np.arange(n) if clusters is None else np.asarray(clusters)
    _, inv = np.unique(labels, return_inverse=True)
    g = int(inv.max()) + 1
    ua = np.bincount(inv, weights=a - a.mean(), minlength=g)
    ub = np.bincount(inv, weights=b - b.mean(), minlength=g)
    f = g / (g - 1) / n ** 2
    return f * float(ua @ ua), f * float(ub @ ub), f * float(ua @ ub), g


def _t_crit(df: int, level: float) -> float:
    from scipy import stats
    return float(stats.t.ppf(0.5 + level / 2, df))


def fieller(a, b, clusters=None, level: float = 0.95) -> dict:
    """Fieller interval for mean(a) / mean(b), paired, cluster-robust.

    The confidence set is ``{theta : (a_bar - theta b_bar)^2 <= t^2 Var(a_bar - theta b_bar)}``,
    a quadratic ``A theta^2 + B theta + C <= 0`` with ``A = b_bar^2 - t^2 Var(b_bar)``. A > 0
    exactly when the denominator is significantly non-zero, and only then is the set a bounded
    interval. Otherwise it is two rays (``kind="exclusive"``: everything outside ``(lo, hi)``)
    or the whole line -- the honest report when there is no headroom to take a share of."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ma, mb = float(a.mean()), float(b.mean())
    vaa, vbb, vab, g = cluster_cov(a, b, clusters)
    t = _t_crit(g - 1, level)
    qa = mb ** 2 - t ** 2 * vbb
    qb = -2.0 * (ma * mb - t ** 2 * vab)
    qc = ma ** 2 - t ** 2 * vaa
    disc = qb ** 2 - 4 * qa * qc
    out = {"est": ma / mb if mb else float("nan"), "df": g - 1, "t": t,
           "denominator_z": mb / math.sqrt(vbb) if vbb > 0 else math.inf}
    if qa > 0:
        r = math.sqrt(max(disc, 0.0))
        lo, hi = sorted(((-qb - r) / (2 * qa), (-qb + r) / (2 * qa)))
        out.update(lo=lo, hi=hi, kind="bounded")
    elif qa < 0 and disc > 0:
        r = math.sqrt(disc)
        lo, hi = sorted(((-qb - r) / (2 * qa), (-qb + r) / (2 * qa)))
        out.update(lo=lo, hi=hi, kind="exclusive")
    else:
        out.update(lo=-math.inf, hi=math.inf, kind="unbounded")
    return out


def cluster_bootstrap(a, b, clusters, n_boot: int = 10000, seed: int = 0,
                      level: float = 0.95) -> dict:
    """Percentile interval for mean(a)/mean(b), resampling whole clusters with replacement.

    Within one draw both means share the same resampled turns, so their ratio is a ratio of
    sums. Also reports how often the resampled headroom was <= 0, where the ratio stops meaning
    a share at all."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    _, inv = np.unique(np.asarray(clusters), return_inverse=True)
    g = int(inv.max()) + 1
    sa = np.bincount(inv, weights=a, minlength=g)
    sb = np.bincount(inv, weights=b, minlength=g)
    draws = np.random.default_rng(seed).integers(0, g, size=(n_boot, g))
    num, den = sa[draws].sum(axis=1), sb[draws].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = num / den
    lo, hi = np.nanpercentile(r, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return {"lo": float(lo), "hi": float(hi), "frac_headroom_nonpositive": float(np.mean(den <= 0)),
            "n_boot": n_boot, "seed": seed}


def decompose(a, b, treated) -> dict:
    """share = p * q_T * c, plus the exact-zero check that makes it an identity."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    t = np.asarray(treated, bool)
    n, n_t = a.size, int(t.sum())
    p = n_t / n
    share = float(a.mean() / b.mean())
    out = {"n": n, "n_treated": n_t, "p": p, "share": share,
           # The structural assumption, measured: an untreated turn must contribute exactly 0.
           "untreated_nonzero": int(np.count_nonzero(a[~t]))}
    if n_t:
        mu1, h_t = float(a[t].mean()), float(b[t].mean())
        q = mu1 / h_t if h_t else float("nan")
        c = h_t / float(b.mean())
        out.update(mu1=mu1, sigma1=float(a[t].std(ddof=1)) if n_t > 1 else float("nan"),
                   local_headroom=h_t, q_treated=q, composition=c,
                   identity_residual=share - p * q * c)
    return out


def effective_n(n_treated: int, p: float, mu1: float, sigma1: float) -> float:
    """n_eff = n_T sigma1^2 / (sigma1^2 + (1-p) mu1^2) <= n_T (Proposition 1)."""
    if not n_treated or not (sigma1 > 0):
        return 0.0
    return n_treated * sigma1 ** 2 / (sigma1 ** 2 + (1 - p) * mu1 ** 2)


def mde(sigma1: float, n_treated: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Minimum detectable mean paired difference on the treated turns, two-sided.

    t-based, ``(t_{1-alpha/2, df} + t_{power, df}) sigma1 / sqrt(n_T)`` with df = n_T - 1. The z
    version is badly wrong exactly where this number matters: at n_T = 2 it understates the MDE
    roughly fivefold, which made a system that touched two turns look measurable."""
    if n_treated < 2 or not (sigma1 > 0):
        return math.inf
    from scipy.stats import t
    df = n_treated - 1
    return (t.ppf(1 - alpha / 2, df) + t.ppf(power, df)) * sigma1 / math.sqrt(n_treated)


def cr_ttest(d, clusters=None) -> dict:
    """Two-sided cluster-robust t-test of mean(d) = 0, df = G - 1.

    This is the test Fieller inverts: 0 lies in the share interval iff the numerator fails it,
    and 100% iff ``S - R`` does. Reported separately because a share interval that crosses 0 or
    100% is the most common way a turn-level significance claim fails to survive clustering."""
    d = np.asarray(d, float)
    v, _, _, g = cluster_cov(d, d, clusters)
    if v <= 0:
        return {"t": math.inf if d.mean() else 0.0, "df": g - 1, "p": 0.0 if d.mean() else 1.0}
    from scipy import stats
    tv = float(d.mean() / math.sqrt(v))
    return {"t": tv, "df": g - 1, "p": float(2 * stats.t.sf(abs(tv), g - 1))}


# --------------------------------------------------------------------------- artifacts

def _label(condition: str, ablation: dict) -> str:
    if not ablation:
        return condition
    return condition + "[" + ",".join(f"{k}={v}" for k, v in sorted(ablation.items())) + "]"


def _index(runs_dir: str, collection: str) -> dict:
    with open(os.path.join(runs_dir, "stats_summary.json"), encoding="utf-8") as f:
        summary = json.load(f)
    out = {}
    for rid, v in summary.items():
        if (v.get("collection") == collection and v.get("context") == "memory_only"
                and v.get("memory_k") == MEMORY_K):
            out[_label(v["condition"], v.get("ablation") or {})] = rid
    return out


def _jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _per_turn(runs_dir: str, rid: str, metric: str) -> dict:
    return {r["turn_id"]: r[metric]
            for r in _jsonl(os.path.join(runs_dir, "metrics", f"{rid}.per_turn.jsonl"))}


def _rewrites(runs_dir: str, rid: str) -> dict:
    return {r["turn_id"]: r for r in _jsonl(os.path.join(runs_dir, "rewrites", f"{rid}.jsonl"))}


def _dialogue(turn_id: str, rewrite: dict | None) -> str:
    if rewrite and rewrite.get("dialogue_id") is not None:
        return str(rewrite["dialogue_id"])
    return turn_id.rsplit("_", 1)[0]


def analyse(runs_dir: str, collection: str, references=REFERENCES, metric: str = METRIC,
            n_boot: int = 10000, seed: int = 0, level: float = 0.95) -> list[dict]:
    idx = _index(runs_dir, collection)
    if BASELINE not in idx:
        return []
    pt = {lab: _per_turn(runs_dir, rid, metric) for lab, rid in idx.items()}
    rw = {lab: _rewrites(runs_dir, rid) for lab, rid in idx.items()
          if os.path.exists(os.path.join(runs_dir, "rewrites", f"{rid}.jsonl"))}
    base = rw.get(BASELINE, {})
    rows = []
    for ref in references:
        if ref not in idx:
            continue
        for lab in sorted(idx, key=lambda s: ORDER.index(s) if s in ORDER else len(ORDER)):
            if lab in (BASELINE, ref):
                continue
            ids = sorted(set(pt[BASELINE]) & set(pt[ref]) & set(pt[lab]))
            a = [pt[lab][t] - pt[BASELINE][t] for t in ids]
            b = [pt[ref][t] - pt[BASELINE][t] for t in ids]
            clusters = [_dialogue(t, base.get(t)) for t in ids]
            row = {"collection": collection, "reference": ref, "system": lab,
                   "run_id": idx[lab], "metric": metric, "n_dialogues": len(set(clusters)),
                   "fieller": fieller(a, b, clusters, level),
                   "bootstrap": cluster_bootstrap(a, b, clusters, n_boot, seed, level),
                   # S - B0 = 0 (share = 0%) and S - R = 0 (share = 100%), clustered.
                   "cr_vs_baseline": cr_ttest(a, clusters),
                   "cr_vs_reference": cr_ttest([x - y for x, y in zip(a, b)], clusters)}
            if lab in rw and base:
                treated = [rw[lab][t]["query"].strip() != base[t]["query"].strip() for t in ids]
                dec = decompose(a, b, treated)
                if dec["n_treated"]:
                    dec["n_eff"] = effective_n(dec["n_treated"], dec["p"], dec["mu1"],
                                               dec["sigma1"])
                    dec["mde"] = mde(dec["sigma1"], dec["n_treated"])
                    dec["mde_over_local_headroom"] = (dec["mde"] / dec["local_headroom"]
                                                      if dec["local_headroom"] > 0 else math.inf)
                row["decomposition"] = dec
            rows.append(row)
    return rows


# --------------------------------------------------------------------------- report

def _pct(x: float) -> str:
    return "  -inf" if x == -math.inf else "  +inf" if x == math.inf else f"{100 * x:6.1f}"


def _interval(f: dict) -> str:
    if f["kind"] == "bounded":
        return f"[{_pct(f['lo'])}, {_pct(f['hi'])}]"
    if f["kind"] == "exclusive":
        return f"outside ({_pct(f['lo'])}, {_pct(f['hi'])})"
    return "unbounded (no significant headroom)"


def _p(x: float) -> str:
    return f"{x:7.1e}" if x < 1e-3 else f"{x:7.3f}"


def render(rows: list[dict]) -> str:
    out = []
    for coll in dict.fromkeys(r["collection"] for r in rows):
        for ref in dict.fromkeys(r["reference"] for r in rows if r["collection"] == coll):
            sub = [r for r in rows if r["collection"] == coll and r["reference"] == ref]
            d0 = sub[0].get("decomposition", {})
            out.append(f"\n{coll} - share of ({ref} - B0) {sub[0]['metric']}, "
                       f"n={d0.get('n', '?')} turns in {sub[0]['n_dialogues']} dialogues; "
                       f"95% intervals and p-values clustered by dialogue")
            out.append(f"  {'system':26} {'share%':>6}  {'Fieller':>18}  {'bootstrap':>18}  "
                       f"{'p(0%)':>7}  {'p(100%)':>7}")
            for r in sub:
                f, bs = r["fieller"], r["bootstrap"]
                out.append(f"  {r['system']:26} {_pct(f['est'])}  {_interval(f):>18}  "
                           f"[{_pct(bs['lo'])}, {_pct(bs['hi'])}]  "
                           f"{_p(r['cr_vs_baseline']['p'])}  {_p(r['cr_vs_reference']['p'])}")
            out.append(f"  {'':26} {'inj.':>6} {'n_T':>5} {'n_eff':>6} {'q_T%':>6} "
                       f"{'MDE':>7} {'MDE/hT':>7} {'off!=0':>6}")
            for r in sub:
                d = r.get("decomposition")
                if not d:
                    continue
                n_eff = f"{d['n_eff']:6.1f}" if d.get("n_eff") is not None else "     -"
                q_t = _pct(d["q_treated"]) if d.get("q_treated") is not None else "     -"
                m, mh = d.get("mde"), d.get("mde_over_local_headroom")
                m_s = "      -" if m is None else ("    inf" if m == math.inf else f"{m:7.3f}")
                mh_s = "      -" if mh is None else ("    inf" if mh == math.inf else f"{mh:7.2f}")
                out.append(f"  {r['system']:26} {100 * d['p']:5.0f}% {d['n_treated']:5d} {n_eff} "
                           f"{q_t} {m_s} {mh_s} {d['untreated_nonzero']:6d}")
    out.append("\n  p(0%)   S - B0 = 0, i.e. the share is zero; p(100%)  S - R = 0, the share is 100%")
    out.append("  inj.    injection rate: share of turns whose query differs from B0's")
    out.append("  n_eff   effective sample size of the all-turn paired test (<= n_T, Prop. 1)")
    out.append("  q_T     share delivered on the treated turns alone, against their local headroom")
    out.append("  MDE     minimum detectable treated-turn effect (alpha .05, power .8, t-based);")
    out.append("  MDE/hT  the same as a fraction of local headroom -- > 1 means delivering ALL of")
    out.append("          the headroom on those turns would still not be reliably detected")
    out.append("  off!=0  untreated turns with a non-zero difference; must be 0 for the identity")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--collection", action="append",
                    help="repeatable; default cast19 and cast20")
    ap.add_argument("--reference", action="append", choices=REFERENCES,
                    help="repeatable; default both")
    ap.add_argument("--metric", default=METRIC)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true", help="emit JSON on stdout instead of tables")
    args = ap.parse_args(argv)

    rows = []
    for coll in args.collection or ["cast19", "cast20"]:
        rows += analyse(args.runs_dir, coll, tuple(args.reference or REFERENCES), args.metric,
                        args.boot, args.seed)
    if not rows:
        print("no memory_only runs with a B0 baseline found under", args.runs_dir)
        return 1
    print(json.dumps(rows, indent=2, default=float) if args.json else render(rows))
    # The identity in (3) and the bound in (2) both rest on exact zeros off-treatment. If that
    # ever fails, say so loudly rather than print numbers that silently stopped meaning anything.
    bad = [r for r in rows if (r.get("decomposition") or {}).get("untreated_nonzero")]
    if bad:
        print("\nWARNING: untreated turns with non-zero differences in: "
              + ", ".join(f"{r['collection']}/{r['system']}" for r in bad), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
