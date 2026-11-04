"""Figures are built from collect_stats output only; these tests reuse test_stats' fake
runs/ directory, run collect_stats, then make_figures, and check that every figure that has
rows is written and the manifest names its inputs. Skipped entirely without matplotlib."""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from eval.stats import collect_stats
from tests.test_stats import fake_runs  # noqa: F401  (fixture)

mpl = pytest.importorskip("matplotlib")
from eval.figures import make_figures  # noqa: E402

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")


def _write_intrinsic_summary(runs_dir):
    """runs/intrinsic_summary.json in the shape scripts/run_intrinsic writes, so the transfer
    figure has something to correlate against."""
    rows = {
        "ikat23.S1_mem0.x": {"axis": "ptkb_reconstruction", "condition": "S1_mem0",
                             "collection": "ikat23", "ablation": {}, "mean_f1": 0.42},
        "ikat23.B1.x": {"axis": "ptkb_reconstruction", "condition": "B1_full_history",
                        "collection": "ikat23", "ablation": {}, "mean_f1": 0.20},
        "ikat23.B0.x": {"axis": "ptkb_reconstruction", "condition": "B0_none",
                        "collection": "ikat23", "ablation": {}, "mean_f1": 0.0},
        "lme.S1_mem0.x": {"axis": "longmemeval_recall", "condition": "S1_mem0",
                          "collection": "longmemeval", "ablation": {},
                          "mean_evidence_recall": 0.61},
        "lme.B1.x": {"axis": "longmemeval_recall", "condition": "B1_full_history",
                     "collection": "longmemeval", "ablation": {},
                     "mean_evidence_recall": 0.90},
        "lme.B0.x": {"axis": "longmemeval_recall", "condition": "B0_none",
                     "collection": "longmemeval", "ablation": {},
                     "mean_evidence_recall": 0.0},
    }
    path = os.path.join(runs_dir, "intrinsic_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f)
    return path


def test_make_figures_end_to_end(fake_runs):  # noqa: F811
    runs_dir, _ = fake_runs
    _write_intrinsic_summary(runs_dir)
    collect_stats(runs_dir)
    out_dir = os.path.join(runs_dir, "figures")
    result = make_figures(os.path.join(runs_dir, "stats"), out_dir)

    assert set(result["figures"]) == {"headroom", "ablations", "pooling", "cost", "transfer"}
    for name, paths in result["figures"].items():
        assert paths, name  # fixture has default runs, ablations, judged_10, usage, intrinsic
        for p in paths:
            assert os.path.exists(p) and os.path.getsize(p) > 0, p
        assert {os.path.splitext(p)[1] for p in paths} == {".pdf", ".png"}
    manifest = json.load(open(result["manifest"], encoding="utf-8"))
    assert manifest["source"].endswith("collect_stats.json")
    assert manifest["skipped"] == []
    assert manifest["n_runs"] == 7
    assert manifest["source_intrinsic_summary"].endswith("intrinsic_summary.json")
    # regenerated in place: a second call must not raise
    make_figures(os.path.join(runs_dir, "stats"), out_dir)


def test_cost_figure_skipped_when_no_usage_recorded(tmp_path):
    """Runs scored before token accounting existed have no usage; the figure must skip and
    SAY so rather than plot a misleading zero."""
    from tests.test_stats import BASELINE_B0, BASELINE_B1, _write_run

    runs_dir = str(tmp_path / "runs")
    _write_run(runs_dir, BASELINE_B0, "cast19", {"31_1": 0.1, "31_2": 0.2}, usage=None)
    _write_run(runs_dir, BASELINE_B1, "cast19", {"31_1": 0.2, "31_2": 0.3}, usage=None)
    collect_stats(runs_dir)
    result = make_figures(os.path.join(runs_dir, "stats"), str(tmp_path / "figs"),
                          formats=("png",))
    assert result["figures"]["cost"] == []
    assert "cost" in result["skipped"]
    manifest = json.load(open(result["manifest"], encoding="utf-8"))
    assert "usage.total_tokens" in manifest["skip_reasons"]["cost"]


def test_transfer_figure_refuses_unvalidated_judge_rows(fake_runs):  # noqa: F811
    """`run_intrinsic --lexical-judge` writes rows labelled UNVALIDATED (a plumbing stand-in
    for the frozen LLM judge). Plotting them would launder a smoke run into a paper figure --
    the same mistake run_eval refuses for native metrics."""
    runs_dir, _ = fake_runs
    unvalidated = {
        f"ikat23.{c}.x": {"axis": "ptkb_reconstruction", "condition": c, "collection": "ikat23",
                          "ablation": {}, "mean_f1": v,
                          "judge_source": "lexical(UNVALIDATED -- not for reported numbers)"}
        for c, v in [("S1_mem0", 0.42), ("B1_full_history", 0.20), ("B0_none", 0.01)]
    }
    with open(os.path.join(runs_dir, "intrinsic_summary.json"), "w", encoding="utf-8") as f:
        json.dump(unvalidated, f)
    collect_stats(runs_dir)
    result = make_figures(os.path.join(runs_dir, "stats"), os.path.join(runs_dir, "f_unval"),
                          formats=("png",))
    assert result["figures"]["transfer"] == [], "unvalidated rows must not produce a figure"
    assert "transfer" in result["skipped"]

    # the same rows with a real judge DO plot -- proving it is the label doing the work
    validated = {k: {**v, "judge_source": "judge_v1 on gpt-4o-mini-2024-07-18"}
                 for k, v in unvalidated.items()}
    with open(os.path.join(runs_dir, "intrinsic_summary.json"), "w", encoding="utf-8") as f:
        json.dump(validated, f)
    result = make_figures(os.path.join(runs_dir, "stats"), os.path.join(runs_dir, "f_val"),
                          formats=("png",))
    assert result["figures"]["transfer"], "validated rows should plot"


def test_transfer_figure_skipped_without_intrinsic_summary(fake_runs):  # noqa: F811
    runs_dir, _ = fake_runs
    collect_stats(runs_dir)
    result = make_figures(os.path.join(runs_dir, "stats"), os.path.join(runs_dir, "f2"),
                          formats=("png",))
    assert result["figures"]["transfer"] == []
    assert result["figures"]["cost"], "cost should still render -- the fixture has usage"


def test_spearman_matches_known_values():
    from eval.figures import _spearman

    assert _spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert _spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert _spearman([1, 2], [1, 2]) is None            # n < 3 is undefined, not 1.0
    assert _spearman([1, 1, 1], [1, 2, 3]) is None      # zero variance in ranks


def test_make_figures_requires_collect_stats_first(tmp_path):
    with pytest.raises(FileNotFoundError, match="collect_stats"):
        make_figures(str(tmp_path / "stats"), str(tmp_path / "figs"))


def test_ablation_figure_skipped_when_no_ablations(tmp_path):
    from tests.test_stats import BASELINE_B0, BASELINE_B1, _write_run
    runs_dir = str(tmp_path / "runs")
    _write_run(runs_dir, BASELINE_B0, "cast19", {"31_1": 0.1, "31_2": 0.2, "32_1": 0.3})
    _write_run(runs_dir, BASELINE_B1, "cast19", {"31_1": 0.2, "31_2": 0.3, "32_1": 0.5})
    collect_stats(runs_dir)
    result = make_figures(os.path.join(runs_dir, "stats"), str(tmp_path / "figs"))
    assert result["figures"]["ablations"] == []
    assert result["figures"]["headroom"] and result["figures"]["pooling"]
    # transfer also skips here: no intrinsic_summary.json in this fixture
    assert result["skipped"] == ["ablations", "transfer"]


def test_make_figures_cli(fake_runs):  # noqa: F811
    runs_dir, _ = fake_runs
    collect_stats(runs_dir)
    result = subprocess.run(
        [sys.executable, "-m", "scripts.make_figures",
         "--stats-dir", os.path.join(runs_dir, "stats"),
         "--out-dir", os.path.join(runs_dir, "figures"), "--formats", "pdf"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert "headroom:" in result.stdout and "figures_manifest.json" in result.stdout
