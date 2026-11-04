"""Tests for eval.stats (the logic behind scripts/collect_stats.py). Offline; builds a fake
runs/ directory in the shape run_eval writes rather than running the pipeline, so the
statistical logic is tested on known numbers.

The t-distribution is the harness's own stdlib implementation, so it is pinned hardest:
fixed reference values (classical table values) always, plus a dense cross-check against
scipy wherever scipy happens to be installed."""
import csv
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from eval.stats import (BASELINE_B0, BASELINE_B1, RunRecord, build_comparisons, collect_stats,
                        comparison_specs, dialogue_id, holm, load_runs, paired_stats,
                        t_critical, t_two_sided_p)

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")

try:
    from scipy import stats as _scipy_stats  # type: ignore
    HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    HAVE_SCIPY = False


# ---------------------------------------------------------------------------
# Student t
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("t, df, expected", [
    (0.0, 5, 1.0),
    (1.0, 1, 0.5),              # t(1) is Cauchy: P(|T|>=1) = 0.5 exactly
    (2.228139, 10, 0.05),       # classical table value t_{0.975,10}
    (2.085963, 20, 0.05),       # t_{0.975,20}
    (1.973852, 172, 0.05),      # t_{0.975,172} -- df for CAsT 2019's 173 judged turns
    (12.7062, 1, 0.05),         # t_{0.975,1}
])
def test_t_two_sided_p_reference_values(t, df, expected):
    assert t_two_sided_p(t, df) == pytest.approx(expected, abs=2e-5)


def test_t_two_sided_p_is_symmetric_and_monotone():
    assert t_two_sided_p(-2.0, 7) == t_two_sided_p(2.0, 7)
    ps = [t_two_sided_p(t, 7) for t in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)]
    assert ps == sorted(ps, reverse=True)
    assert t_two_sided_p(float("inf"), 7) == 0.0


@pytest.mark.parametrize("df, expected", [(1, 12.7062), (10, 2.2281), (20, 2.0860),
                                          (172, 1.9739), (1000, 1.9623)])
def test_t_critical_reference_values(df, expected):
    assert t_critical(df, 0.05) == pytest.approx(expected, abs=2e-4)


@pytest.mark.skipif(not HAVE_SCIPY, reason="scipy not installed -- reference values above still run")
def test_t_distribution_matches_scipy_cross_check():
    for df in (1, 2, 3, 5, 10, 30, 172, 478, 5000):
        for t in (0.0, 0.1, 0.7, 1.3, 2.0, 2.5, 3.7, 6.0, 15.0):
            expected = 2 * _scipy_stats.t.sf(t, df)
            assert t_two_sided_p(t, df) == pytest.approx(expected, abs=1e-9), (t, df)
        assert t_critical(df, 0.05) == pytest.approx(_scipy_stats.t.ppf(0.975, df), abs=1e-6)


# ---------------------------------------------------------------------------
# Holm
# ---------------------------------------------------------------------------

def test_holm_known_vector():
    # sorted: .005*4=.02, .01*3=.03, .03*2=.06, .04*1=.04 -> monotone -> .06
    assert holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_holm_single_and_capped():
    assert holm([0.3]) == [0.3]
    assert holm([0.6, 0.7]) == [1.0, 1.0]
    assert holm([]) == []


def test_holm_matches_statsmodels():
    try:
        from statsmodels.stats.multitest import multipletests  # type: ignore
    except ImportError:
        pytest.skip("statsmodels not installed -- test_holm_known_vector pins the values")
    p = [0.01, 0.2, 0.03, 0.004, 0.5, 0.049]
    expected = multipletests(p, method="holm")[1]
    assert holm(p) == pytest.approx(list(expected))


# ---------------------------------------------------------------------------
# paired_stats
# ---------------------------------------------------------------------------

def test_paired_stats_hand_computed():
    # deltas = [1, 1, 2, 1]: mean 1.25, sd 0.5, se 0.25, t = 5.0 on 3 df
    s = paired_stats([1, 2, 3, 4], [2, 3, 5, 5])
    assert s["n"] == 4
    assert s["mean_delta"] == pytest.approx(1.25)
    assert s["sd_delta"] == pytest.approx(0.5)
    assert s["t"] == pytest.approx(5.0)
    assert s["df"] == 3
    assert s["p"] == pytest.approx(0.01539, abs=2e-4)   # 2*sf(5, 3)
    assert s["cohen_dz"] == pytest.approx(2.5)
    assert (s["wins"], s["losses"], s["ties"]) == (4, 0, 0)
    assert s["ci_low"] == pytest.approx(1.25 - 3.1824 * 0.25, abs=1e-3)
    assert s["ci_high"] == pytest.approx(1.25 + 3.1824 * 0.25, abs=1e-3)
    assert s["degenerate"] is False


def test_paired_stats_sign_convention_is_treatment_minus_baseline():
    assert paired_stats([0.5, 0.5], [0.2, 0.2])["mean_delta"] == pytest.approx(-0.3)


def test_paired_stats_degenerate_cases():
    same = paired_stats([0.1, 0.2, 0.3], [0.1, 0.2, 0.3])
    assert same["degenerate"] and same["p"] == 1.0 and same["t"] is None
    const = paired_stats([1, 2, 3], [2, 3, 4])   # exactly-representable constant delta
    assert const["degenerate"] and const["p"] == 0.0 and const["mean_delta"] == 1.0
    one = paired_stats([0.1], [0.3])
    assert one["n"] == 1 and one["p"] is None
    with pytest.raises(ValueError):
        paired_stats([1, 2], [1])


@pytest.mark.skipif(not HAVE_SCIPY, reason="scipy not installed")
def test_paired_stats_matches_scipy_ttest_rel():
    a = [0.31, 0.10, 0.00, 0.52, 0.27, 0.44, 0.05, 0.61, 0.33, 0.12]
    b = [0.35, 0.08, 0.02, 0.60, 0.27, 0.50, 0.09, 0.58, 0.40, 0.20]
    s = paired_stats(a, b)
    ref = _scipy_stats.ttest_rel(b, a)
    assert s["t"] == pytest.approx(ref.statistic, abs=1e-9)
    assert s["p"] == pytest.approx(ref.pvalue, abs=1e-9)


# ---------------------------------------------------------------------------
# dialogue ids
# ---------------------------------------------------------------------------

def test_dialogue_id_conventions():
    assert dialogue_id("31_1", "cast19") == "31"
    assert dialogue_id("81_11", "cast20") == "81"
    assert dialogue_id("9-1-2", "ikat23") == "9-1"      # topic itself contains a hyphen
    assert dialogue_id("10-2-11", "ikat23") == "10-2"


# ---------------------------------------------------------------------------
# Fake runs/ directory in the shape run_eval writes
# ---------------------------------------------------------------------------

METRICS = ("ndcg_cut_3", "ndcg_cut_5", "recall_1000", "recip_rank", "judged_10")


def _write_run(runs_dir, condition, collection, per_turn, ablation=None, memory_k=5,
               injection="rewrite_only", write_memory_k=True, usage="default",
               context="history_plus_memory"):
    """per_turn: {turn_id: value}; every metric gets the same value (enough for tests)."""
    from harness.config import RunConfig
    cfg = RunConfig(condition=condition, injection=injection, collection=collection,
                    backend="echo", model="smoke-model", memory_k=memory_k,
                    ablation=ablation or {}, context=context)
    os.makedirs(os.path.join(runs_dir, "metrics"), exist_ok=True)
    with open(os.path.join(runs_dir, "metrics", f"{cfg.run_id}.per_turn.jsonl"), "w") as f:
        for qid in sorted(per_turn):
            f.write(json.dumps({"turn_id": qid, **{m: per_turn[qid] for m in METRICS}}) + "\n")
    vals = list(per_turn.values())
    row = {
        "run_id": cfg.run_id, "condition": condition, "collection": collection,
        "injection": injection, "ablation": dict(sorted((ablation or {}).items())),
        "context": context,
        **{m: sum(vals) / len(vals) for m in METRICS},
        "n_scored_turns": len(vals),
        "coverage": {"n_run_qids": len(vals), "n_qrels_qids": len(vals), "n_scored": len(vals),
                     "n_qrels_only": 0, "n_run_only": 0},
        "metrics_source": "pytrec_eval",
    }
    if write_memory_k:
        row["memory_k"] = memory_k
    if usage == "default":
        # plausible per-condition spend: B0 cheapest, systems dearer (the cost figure's x)
        base = {"B0_none": 5_000, "B1_full_history": 60_000}.get(condition, 25_000)
        usage = {"n_calls": len(vals), "n_uncached_calls": len(vals),
                 "prompt_tokens": base, "completion_tokens": base // 10,
                 "total_tokens": base + base // 10}
    if usage is not None:
        row["usage"] = usage
    stats_path = os.path.join(runs_dir, "stats_summary.json")
    stats = json.load(open(stats_path)) if os.path.exists(stats_path) else {}
    stats[cfg.run_id] = row
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
    return cfg


def _turns(topics, turns_per_topic, value_fn):
    return {f"{t}_{i}": value_fn(t, i) for t in topics for i in range(1, turns_per_topic + 1)}


@pytest.fixture
def fake_runs(tmp_path):
    runs_dir = str(tmp_path / "runs")
    topics = (31, 32, 33, 34, 35, 36)
    b0 = _turns(topics, 5, lambda t, i: 0.20 + 0.01 * ((t * 7 + i * 3) % 10))
    b1 = {q: v + 0.10 + 0.005 * (i % 4) for i, (q, v) in enumerate(sorted(b0.items()))}
    s1 = {q: v + 0.12 + 0.004 * (i % 5) for i, (q, v) in enumerate(sorted(b0.items()))}
    s1_noinfer = {q: v - 0.03 for q, v in s1.items()}
    cfgs = {
        "b0": _write_run(runs_dir, BASELINE_B0, "cast19", b0),
        "b1": _write_run(runs_dir, BASELINE_B1, "cast19", b1),
        "s1": _write_run(runs_dir, "S1_mem0", "cast19", s1),
        "s1_noinfer": _write_run(runs_dir, "S1_mem0", "cast19", s1_noinfer,
                                 ablation={"infer": False}),
        "s1_k3": _write_run(runs_dir, "S1_mem0", "cast19", {q: v - 0.01 for q, v in s1.items()},
                            memory_k=3),
        # cast20: same pair so a pooled row exists
        "b0_20": _write_run(runs_dir, BASELINE_B0, "cast20", _turns((81, 82, 83), 4,
                                                                    lambda t, i: 0.3)),
        "b1_20": _write_run(runs_dir, BASELINE_B1, "cast20", _turns((81, 82, 83), 4,
                                                                    lambda t, i: 0.3 + 0.02 * i)),
    }
    return runs_dir, cfgs


def test_load_runs_labels_and_default_flags(fake_runs):
    runs_dir, cfgs = fake_runs
    recs = {r.run_id: r for r in load_runs(runs_dir)}
    assert len(recs) == 7
    assert recs[cfgs["s1"].run_id].label == "S1_mem0" and recs[cfgs["s1"].run_id].is_default_config
    assert recs[cfgs["s1_noinfer"].run_id].label == "S1_mem0[infer=False]"
    assert recs[cfgs["s1_k3"].run_id].label == "S1_mem0[k=3]"
    assert not recs[cfgs["s1_k3"].run_id].is_default_config
    assert len(recs[cfgs["b0"].run_id].per_turn) == 30


def test_load_runs_requires_per_turn_file(fake_runs):
    runs_dir, cfgs = fake_runs
    os.remove(os.path.join(runs_dir, "metrics", f"{cfgs['b1'].run_id}.per_turn.jsonl"))
    with pytest.raises(FileNotFoundError, match=cfgs["b1"].run_id):
        load_runs(runs_dir)


def test_load_runs_recovers_memory_k_from_manifest_or_refuses(tmp_path):
    """Rows without memory_k must not be silently treated as k=5: a k-sweep run in the wrong
    family would corrupt the Holm correction. The eval manifest's config is the fallback."""
    runs_dir = str(tmp_path / "runs")
    cfg = _write_run(runs_dir, "S1_mem0", "cast19", {"31_1": 0.1, "31_2": 0.2}, memory_k=3,
                     write_memory_k=False)
    with pytest.raises(ValueError, match="memory_k"):
        load_runs(runs_dir)
    os.makedirs(os.path.join(runs_dir, "manifests"))
    with open(os.path.join(runs_dir, "manifests", f"{cfg.run_id}.eval.json"), "w") as f:
        json.dump({"run_id": cfg.run_id, "config": cfg.canonical_json()}, f)
    assert load_runs(runs_dir)[0].memory_k == 3


def test_comparison_specs_define_the_holm_families(fake_runs):
    runs_dir, cfgs = fake_runs
    specs, unpaired = comparison_specs(load_runs(runs_dir))
    main = {(s["baseline"].label, s["treatment"].label, s["baseline"].collection)
            for s in specs if s["group"] == "main"}
    assert main == {
        (BASELINE_B0, BASELINE_B1, "cast19"), (BASELINE_B0, "S1_mem0", "cast19"),
        (BASELINE_B1, "S1_mem0", "cast19"), (BASELINE_B0, BASELINE_B1, "cast20"),
    }
    abl = {(s["baseline"].label, s["treatment"].label) for s in specs if s["group"] == "ablation"}
    assert abl == {("S1_mem0", "S1_mem0[infer=False]"), ("S1_mem0", "S1_mem0[k=3]")}
    assert unpaired == []


def test_unpaired_ablation_is_reported_not_dropped(tmp_path):
    runs_dir = str(tmp_path / "runs")
    cfg = _write_run(runs_dir, "S3_amem", "cast19", {"31_1": 0.1, "31_2": 0.2},
                     ablation={"evolution": False})
    result = build_comparisons(load_runs(runs_dir))
    assert result["comparisons"] == []
    assert result["unpaired_runs"] == [cfg.run_id]


def test_build_comparisons_numbers_and_families(fake_runs):
    runs_dir, cfgs = fake_runs
    result = build_comparisons(load_runs(runs_dir))
    rows = result["comparisons"]
    by = {(c["group"], c["collection"], c["metric"], c["baseline"], c["treatment"]): c
          for c in rows}

    c = by[("main", "cast19", "ndcg_cut_3", BASELINE_B0, BASELINE_B1)]
    assert c["n"] == 30 and c["n_unpaired"] == 0
    assert c["mean_delta"] == pytest.approx(0.10 + 0.005 * 43 / 30)   # i%4 over i<30 sums to 43
    assert c["wins"] == 30 and c["losses"] == 0
    assert c["p"] < 1e-6 and c["significant"] is True
    assert c["n_dialogues"] == 6 and c["p_dialogue_level"] < 0.01
    assert c["family"] == "main/cast19/rewrite_only/ndcg_cut_3"
    # main family on cast19: B1-B0, S1-B0, S1-B1 = 3 comparisons per metric
    assert c["family_size"] == 3
    assert result["families"]["main/cast19/rewrite_only/ndcg_cut_3"]["size"] == 3

    # Holm is applied within the family: every p_holm >= its raw p, max ratio <= family size
    fam = [r for r in rows if r["family"] == "main/cast19/rewrite_only/ndcg_cut_3"]
    for r in fam:
        assert r["p_holm"] >= r["p"] and r["p_holm"] <= min(1.0, 3 * r["p"]) + 1e-12

    # ablation: infer=False is ~0.03 worse than S1 on every turn (float noise keeps it just
    # off the degenerate path) -> 30 losses, tiny p, negative effect
    a = by[("ablation", "cast19", "ndcg_cut_3", "S1_mem0", "S1_mem0[infer=False]")]
    assert a["mean_delta"] == pytest.approx(-0.03)
    assert a["losses"] == 30 and a["wins"] == 0
    assert a["p"] is not None and a["p"] < 1e-9 and a["significant"] is True
    assert a["family"] == "ablation/cast19/rewrite_only/ndcg_cut_3" and a["family_size"] == 2

    # pooled row exists for B0->B1 (present on cast19 and cast20) and only for that pair
    pooled = [r for r in rows if r["collection"].startswith("pooled(")]
    assert {(r["baseline"], r["treatment"]) for r in pooled} == {(BASELINE_B0, BASELINE_B1)}
    p3 = next(r for r in pooled if r["metric"] == "ndcg_cut_3")
    assert p3["collection"] == "pooled(cast19+cast20)" and p3["n"] == 30 + 12
    assert p3["family"] == "main/pooled(cast19+cast20)/rewrite_only/ndcg_cut_3"

    assert result["metrics"] == ["ndcg_cut_3", "ndcg_cut_5", "recall_1000", "recip_rank",
                                 "judged_10"]


def test_n_is_the_intersection_and_unpaired_is_flagged(tmp_path):
    """A turn scored in one run but not the other must not be silently dropped: n shrinks to
    the intersection and n_unpaired says by how much (coverage anomaly upstream)."""
    runs_dir = str(tmp_path / "runs")
    _write_run(runs_dir, BASELINE_B0, "cast19", {"31_1": 0.1, "31_2": 0.2, "31_3": 0.3})
    _write_run(runs_dir, BASELINE_B1, "cast19", {"31_1": 0.2, "31_2": 0.3, "32_1": 0.9})
    c = build_comparisons(load_runs(runs_dir))["comparisons"][0]
    assert c["n"] == 2 and c["n_unpaired"] == 2


def test_collect_stats_writes_every_output(fake_runs):
    runs_dir, cfgs = fake_runs
    summary = collect_stats(runs_dir)
    out = os.path.join(runs_dir, "stats")
    assert os.path.exists(os.path.join(out, "collect_stats.json"))
    with open(os.path.join(out, "comparisons.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(summary["comparisons"])
    assert {"p_holm", "cohen_dz", "family", "n_unpaired", "p_dialogue_level"} <= set(rows[0])

    # the frozen scope: every metrics table reports nDCG@3, nDCG@5, R@1000, MRR, judged@10
    md = open(os.path.join(out, "tables", "main_cast19_rewrite_only.md"), encoding="utf-8").read()
    for h in ("nDCG@3", "nDCG@5", "R@1000", "MRR", "judged@10"):
        assert h in md
    assert "S1_mem0[infer=False]" in md and "S1_mem0[k=3]" in md
    tex = open(os.path.join(out, "tables", "main_cast19_rewrite_only.tex"), encoding="utf-8").read()
    assert r"\begin{tabular}" in tex and r"B0\_none" in tex
    assert os.path.exists(os.path.join(out, "tables", "main_cast20_rewrite_only.md"))

    # significance markers: B1 and S1 beat B0 on every metric in the fixture -> '*'
    t19 = {r["run"]: r for r in summary["tables"]["cast19/rewrite_only"]}
    assert "*" in t19[BASELINE_B1]["ndcg_cut_3_marks"]
    assert "*" in t19["S1_mem0"]["ndcg_cut_3_marks"]
    assert t19[BASELINE_B0]["ndcg_cut_3_marks"] == ""
    assert [r["run"] for r in summary["tables"]["cast19/rewrite_only"]][:2] == [BASELINE_B0,
                                                                               BASELINE_B1]

    # per-topic CSVs for the main family on the primary metric
    per_topic = summary["outputs"]["per_topic"]
    assert len(per_topic) == 4   # 3 on cast19 + 1 on cast20
    with open(per_topic[0], newline="") as f:
        pt = list(csv.DictReader(f))
    assert {"dialogue_id", "baseline_mean", "treatment_mean", "delta"} == set(pt[0])

    assert summary["best_system_cast19"]["condition"] == "S1_mem0"
    assert summary["primary_metric"] == "ndcg_cut_3"
    # regenerable in place: a second call must not raise
    collect_stats(runs_dir)


def test_collect_stats_requires_stats_summary(tmp_path):
    with pytest.raises(FileNotFoundError, match="run_eval"):
        collect_stats(str(tmp_path / "runs"))


def test_collect_stats_cli(fake_runs):
    runs_dir, _ = fake_runs
    result = subprocess.run(
        [sys.executable, "-m", "scripts.collect_stats", "--runs-dir", runs_dir],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "nDCG@3" in result.stdout
    assert "B0_none -> B1_full_history" in result.stdout
    assert "best system on cast19/ndcg_cut_3: S1_mem0" in result.stdout
    assert "7 runs" in result.stdout


# ---------------------------------------------------------------------------
# treated-subset (per-protocol) view -- the zero-inflation problem
# ---------------------------------------------------------------------------

def test_wilcoxon_matches_scipy():
    from eval.stats import wilcoxon_signed_rank
    if not HAVE_SCIPY:
        pytest.skip("scipy not installed")
    from scipy.stats import wilcoxon as sw
    for d in ([1.0, 2, -1, 3, 4, -2, 5, 6, 7, -3],
              [0.5, -0.5, 1.5, 2.5, -1.0, 3.0, 0, 0, 0],
              [0.1] * 12 + [-0.2] * 4):
        nz = [x for x in d if x]
        mine, n = wilcoxon_signed_rank(d)
        assert n == len(nz)
        assert mine == pytest.approx(sw(nz, correction=False, mode="approx").pvalue, abs=1e-9)


def test_wilcoxon_ignores_structural_zeros():
    """The whole point: 168 tied turns must not dilute the 5 that moved."""
    from eval.stats import wilcoxon_signed_rank
    moved = [0.4, 0.5, 0.6, 0.3, 0.45]
    p_alone, n_alone = wilcoxon_signed_rank(moved)
    p_padded, n_padded = wilcoxon_signed_rank(moved + [0.0] * 168)
    assert (p_alone, n_alone) == (p_padded, n_padded)


def test_wilcoxon_degenerate_cases():
    from eval.stats import wilcoxon_signed_rank
    assert wilcoxon_signed_rank([0.0, 0.0]) == (None, 0)
    p, n = wilcoxon_signed_rank([0.3])
    assert n == 1


def test_paired_stats_carries_the_treated_subset_view():
    """ITT and per-protocol answer different questions and both must be reported: the t-test
    averages over turns where no memory was injected, which is not the question."""
    base = [0.5] * 10
    treat = [0.5] * 8 + [0.9, 0.1]          # 8 ties, one big win, one big loss
    s = paired_stats(base, treat)
    assert s["n"] == 10
    assert s["n_changed"] == 2
    assert s["mean_delta_changed"] == pytest.approx(0.0)
    assert s["mean_delta"] == pytest.approx(0.0)
    # the ITT mean is diluted by the ties; the treated mean is over the 2 turns that moved
    s2 = paired_stats([0.5] * 10, [0.5] * 8 + [0.9, 0.9])
    assert s2["mean_delta"] == pytest.approx(0.08)
    assert s2["mean_delta_changed"] == pytest.approx(0.4)


def test_system_vs_system_contrasts_exist_in_their_own_family(tmp_path):
    """The paper's headline is a system-vs-system contrast, so it must live INSIDE a Holm
    family -- but not inside `main`, which answers "does this beat the baselines" and would be
    over-corrected by k(k-1)/2 unrelated pairs."""
    runs_dir = str(tmp_path / "runs")
    turns = {f"31_{i}": 0.2 for i in range(1, 9)}
    _write_run(runs_dir, BASELINE_B0, "cast19", turns)
    _write_run(runs_dir, BASELINE_B1, "cast19", {k: v + 0.3 for k, v in turns.items()})
    _write_run(runs_dir, "S1_mem0", "cast19", {k: v + 0.01 for k, v in turns.items()})
    _write_run(runs_dir, "S3_amem", "cast19", {k: v + 0.29 for k, v in turns.items()})

    specs, _ = comparison_specs(load_runs(runs_dir))
    sys_pairs = {(s["baseline"].label, s["treatment"].label)
                 for s in specs if s["group"] == "systems"}
    assert sys_pairs == {("S1_mem0", "S3_amem")}
    # baselines are not dragged into the systems family
    assert not any(s["baseline"].label.startswith("B") or s["treatment"].label.startswith("B")
                   for s in specs if s["group"] == "systems")

    rows = build_comparisons(load_runs(runs_dir))["comparisons"]
    head = [r for r in rows if r["group"] == "systems" and r["metric"] == "ndcg_cut_3"]
    assert len(head) == 1
    assert head[0]["family"] == "systems/cast19/rewrite_only/ndcg_cut_3"
    assert head[0]["mean_delta"] == pytest.approx(0.28)
    # main family must be unchanged in size by the addition (B1,S1,S3 vs B0; S1,S3 vs B1)
    main = [r for r in rows if r["group"] == "main" and r["metric"] == "ndcg_cut_3"]
    assert len(main) == 5
