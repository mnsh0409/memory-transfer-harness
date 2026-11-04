"""Tests for scripts/share_inference -- intervals on the delivered share, and effective n.

What these pin, because the paper's claims lean on each:

* Proposition 1 holds *exactly* in population moments, not approximately: with zero differences
  off-treatment, the all-turn noncentrality equals sqrt(n_eff) * mu1 / sigma1, and n_eff <= n_T.
* share = p * q_T * c is an identity under exact zeros, and the script notices when the zeros
  are not exact rather than reporting a decomposition that silently stopped holding.
* Fieller's interval is bounded only when the headroom is significant, and 0 sits inside it
  exactly when the clustered test of the numerator fails -- the two outputs cannot disagree.
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

pytest.importorskip("scipy")

from scripts.share_inference import (analyse, cluster_bootstrap, cluster_cov, cr_ttest,  # noqa: E402
                                     decompose, effective_n, fieller, main, mde)


# --------------------------------------------------------------------------- variance

def test_cluster_cov_reduces_to_iid_when_every_unit_is_its_own_cluster():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=40), rng.normal(size=40)
    vaa, vbb, vab, g = cluster_cov(a, b)
    assert g == 40
    assert vaa == pytest.approx(a.var(ddof=1) / 40)
    assert vbb == pytest.approx(b.var(ddof=1) / 40)
    assert vab == pytest.approx(np.cov(a, b)[0, 1] / 40)


def test_clustering_inflates_variance_when_units_within_a_cluster_move_together():
    """Turns in one dialogue share a topic; if their differences co-move, the iid variance is
    too small. This is the whole reason the paper's turn-level p-values can be optimistic."""
    rng = np.random.default_rng(2)
    shift = np.repeat(rng.normal(size=10), 8)          # one shared draw per dialogue
    a = shift + 0.1 * rng.normal(size=80)
    clusters = np.repeat(np.arange(10), 8)
    v_iid = cluster_cov(a, a)[0]
    v_cr = cluster_cov(a, a, clusters)[0]
    assert v_cr > 3 * v_iid


# --------------------------------------------------------------------------- Fieller

def test_fieller_is_bounded_and_close_to_delta_method_when_headroom_is_large():
    rng = np.random.default_rng(3)
    b = 0.3 + 0.05 * rng.normal(size=200)
    a = 0.5 * b + 0.02 * rng.normal(size=200)
    f = fieller(a, b)
    assert f["kind"] == "bounded"
    assert f["lo"] < f["est"] < f["hi"]
    vaa, vbb, vab, _ = cluster_cov(a, b)
    th = f["est"]
    se = math.sqrt(vaa - 2 * th * vab + th ** 2 * vbb) / abs(b.mean())
    assert f["hi"] - f["lo"] == pytest.approx(2 * f["t"] * se, rel=0.02)


def test_fieller_refuses_a_bounded_interval_when_there_is_no_headroom():
    """The principled replacement for the figures code's MIN_HEADROOM = 0.02 cutoff."""
    rng = np.random.default_rng(4)
    b = 0.01 * rng.normal(size=30)                     # headroom indistinguishable from 0
    a = 0.01 * rng.normal(size=30)
    f = fieller(a, b)
    assert f["kind"] in ("exclusive", "unbounded")
    assert abs(f["denominator_z"]) < f["t"]


def _co_moving(signal):
    """Twelve dialogues of six turns whose differences share a per-dialogue shift -- the
    structure that makes turn-level inference optimistic. With seed 5 the shifts happen to
    average below zero, so signal=0.0 is a true null that an iid analysis calls significant."""
    rng = np.random.default_rng(5)
    clusters = np.repeat(np.arange(12), 6)
    b = 0.3 + 0.05 * rng.normal(size=72)
    a = signal + np.repeat(0.1 * rng.normal(size=12), 6) + 0.01 * rng.normal(size=72)
    return a, b, clusters


@pytest.mark.parametrize("signal", [0.0, 0.3])
def test_zero_is_in_the_interval_exactly_when_the_clustered_numerator_test_fails(signal):
    a, b, clusters = _co_moving(signal)
    f = fieller(a, b, clusters)
    assert (f["lo"] <= 0 <= f["hi"]) == (cr_ttest(a, clusters)["p"] >= 0.05)


def test_ignoring_dialogue_clustering_manufactures_a_false_positive():
    """The mechanism behind the 2019 claims that do not survive clustering: a true null, called
    significant when turns are treated as independent."""
    a, b, clusters = _co_moving(0.0)
    iid, clustered = fieller(a, b), fieller(a, b, clusters)
    assert not (iid["lo"] <= 0 <= iid["hi"])           # iid: "significantly non-zero"
    assert clustered["lo"] <= 0 <= clustered["hi"]     # clustered: cannot exclude zero
    assert cr_ttest(a)["p"] < 0.001 < 0.05 <= cr_ttest(a, clusters)["p"]


# --------------------------------------------------------------------------- Proposition 1

def _population(n=200, n_t=30, seed=6):
    rng = np.random.default_rng(seed)
    d = np.zeros(n)
    d[:n_t] = 0.2 + 0.3 * rng.normal(size=n_t)
    return d, n_t


def test_proposition_1_noncentrality_identity_is_exact_in_population_moments():
    d, n_t = _population()
    n = d.size
    p = n_t / n
    mu1, sigma1 = d[:n_t].mean(), d[:n_t].std(ddof=0)
    lam_all = math.sqrt(n) * d.mean() / d.std(ddof=0)
    lam_prop = math.sqrt(effective_n(n_t, p, mu1, sigma1)) * mu1 / sigma1
    assert lam_all == pytest.approx(lam_prop, rel=1e-12)


def test_effective_n_never_exceeds_the_treated_count_and_equals_it_under_the_null():
    for n_t, p, mu1, sigma1 in [(16, 0.09, 0.2, 0.3), (37, 0.18, 0.4, 0.2), (5, 0.5, 2.0, 0.1)]:
        assert effective_n(n_t, p, mu1, sigma1) <= n_t
    assert effective_n(16, 0.09, 0.0, 0.3) == pytest.approx(16)
    assert effective_n(0, 0.0, 0.0, 0.0) == 0.0


def test_mde_is_t_based_so_a_two_turn_system_is_not_called_measurable():
    """The z approximation at n_T = 2 understates the MDE roughly fivefold."""
    from scipy.stats import norm
    z_version = (norm.ppf(0.975) + norm.ppf(0.8)) * 1.0 / math.sqrt(2)
    assert mde(1.0, 2) > 4 * z_version
    assert mde(1.0, 400) == pytest.approx(z_version * math.sqrt(2) / math.sqrt(400), rel=0.01)
    assert mde(1.0, 1) == math.inf


# --------------------------------------------------------------------------- decomposition

def test_share_decomposes_exactly_into_injection_times_quality_times_composition():
    rng = np.random.default_rng(7)
    n = 120
    treated = np.zeros(n, bool)
    treated[rng.choice(n, 25, replace=False)] = True
    b = 0.3 + 0.2 * rng.normal(size=n)
    a = np.where(treated, 0.5 * b + 0.05 * rng.normal(size=n), 0.0)
    d = decompose(a, b, treated)
    assert d["untreated_nonzero"] == 0
    assert d["p"] == pytest.approx(25 / 120)
    assert abs(d["identity_residual"]) < 1e-12


def test_decomposition_reports_when_the_exact_zeros_fail():
    """If retrieval were nondeterministic, an untreated turn could move -- and the identity, the
    proposition and the effective-n figures would all stop meaning what the paper says."""
    a = np.array([0.4, 0.0, 0.0, 0.1, 0.0])
    b = np.array([0.5, 0.3, 0.2, 0.2, 0.3])
    treated = [True, False, False, False, False]      # turn 4 moved without being treated
    d = decompose(a, b, treated)
    assert d["untreated_nonzero"] == 1
    assert abs(d["identity_residual"]) > 1e-3


# --------------------------------------------------------------------------- bootstrap

def test_cluster_bootstrap_is_reproducible_with_a_seed():
    rng = np.random.default_rng(8)
    a, b = rng.normal(0.1, 0.1, 60), rng.normal(0.3, 0.1, 60)
    cl = np.repeat(np.arange(15), 4)
    assert cluster_bootstrap(a, b, cl, 2000, 11) == cluster_bootstrap(a, b, cl, 2000, 11)


# --------------------------------------------------------------------------- end to end

def _write_runs(tmp_path, break_zeros=False):
    """Four dialogues x three turns. S1 changes the query on two turns only."""
    runs = tmp_path / "runs"
    (runs / "metrics").mkdir(parents=True)
    (runs / "rewrites").mkdir()
    turns = [f"{d}_{t}" for d in (31, 32, 33, 34) for t in (1, 2, 3)]
    rng = np.random.default_rng(9)
    b0 = {t: float(rng.uniform(0.1, 0.3)) for t in turns}
    treated = {"31_2", "33_3"}
    conds = {
        "B0_none": {t: b0[t] for t in turns},
        "B1_full_history": {t: b0[t] + float(rng.uniform(0.2, 0.4)) for t in turns},
        "M1_manual": {t: b0[t] + float(rng.uniform(0.3, 0.5)) for t in turns},
        "S1_mem0": {t: b0[t] + (0.2 if t in treated else 0.0) for t in turns},
    }
    if break_zeros:
        conds["S1_mem0"]["32_1"] += 0.05
    summary = {}
    for cond, scores in conds.items():
        rid = f"cast19.{cond}.rewrite_only.test"
        summary[rid] = {"collection": "cast19", "condition": cond, "context": "memory_only",
                        "memory_k": 5, "ablation": {}}
        with open(runs / "metrics" / f"{rid}.per_turn.jsonl", "w", encoding="utf-8") as f:
            for t in turns:
                f.write(json.dumps({"turn_id": t, "ndcg_cut_3": scores[t]}) + "\n")
        with open(runs / "rewrites" / f"{rid}.jsonl", "w", encoding="utf-8") as f:
            for t in turns:
                q = f"q {t}" + (" +memory" if cond == "S1_mem0" and t in treated else "")
                q = q if cond in ("B0_none", "S1_mem0") else f"{cond} {t}"
                f.write(json.dumps({"turn_id": t, "dialogue_id": t.split("_")[0],
                                    "query": q}) + "\n")
    (runs / "stats_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return str(runs)


def test_end_to_end_reports_injection_and_clusters_by_dialogue(tmp_path):
    rows = analyse(_write_runs(tmp_path), "cast19", n_boot=500)
    s1 = next(r for r in rows if r["system"] == "S1_mem0" and r["reference"] == "M1_manual")
    assert s1["n_dialogues"] == 4
    assert s1["decomposition"]["n_treated"] == 2
    assert s1["decomposition"]["untreated_nonzero"] == 0
    assert abs(s1["decomposition"]["identity_residual"]) < 1e-12


def test_main_exits_nonzero_when_untreated_turns_are_not_exactly_zero(tmp_path, capsys):
    assert main(["--runs-dir", _write_runs(tmp_path), "--collection", "cast19",
                 "--boot", "200"]) == 0
    broken = tmp_path / "b"
    broken.mkdir()
    assert main(["--runs-dir", _write_runs(broken, break_zeros=True), "--collection", "cast19",
                 "--boot", "200"]) == 2
    assert "untreated turns with non-zero differences" in capsys.readouterr().err
