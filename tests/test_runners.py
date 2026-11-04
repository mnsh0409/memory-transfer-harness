"""Unit tests for harness.runners (the logic behind scripts/run_replay.py and
scripts/run_retrieval.py). All offline: echo backend, no real API key/pyserini/model download.
See tests/test_cli_scripts.py for a subprocess-level test of the actual `python -m
scripts.run_replay` entry point."""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from eval.extrinsic import load_qrels
from harness.config import RunConfig
from harness.manifest import read_manifest
from harness.replay import read_rewrites
from harness.runners import (resolve_model, run_eval_config, run_replay_config,
                             run_retrieval_config)

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")

ECHO_QUERY = "[echo:smoke-model] Standalone query:"  # the echo backend's fixed output for
# rewrite_v1.txt specifically -- its last line has nothing substituted after it, so every turn
# produces this exact same string regardless of input. Real (openai-backend) queries obviously
# vary; this is a property of the echo stand-in used for plumbing tests only.


class FakeBM25:
    def __init__(self, corpus):
        self.corpus = corpus

    def search(self, query, k=1000):
        return self.corpus.get(query, [])


class FakeCE:
    def rerank(self, query, candidates, depth=100, batch_size=32):
        return [(doc_id, 1.0 - i * 0.01) for i, (doc_id, _, _) in enumerate(candidates[:depth])]


@pytest.fixture
def cast_data_dir(tmp_path):
    d = tmp_path / "data" / "cast"
    d.mkdir(parents=True)
    shutil.copy(os.path.join(FIXTURES, "cast_smoke.json"), d / "cast19_topics.json")
    return str(tmp_path / "data")


def make_cast19_config(**overrides):
    fields = dict(
        condition="B0_none", injection="rewrite_only", collection="cast19",
        backend="echo", model="smoke-model",
    )
    fields.update(overrides)
    return RunConfig(**fields)


def test_run_replay_config_end_to_end(tmp_path, cast_data_dir):
    config = make_cast19_config()
    runs_dir = str(tmp_path / "runs")
    manifest = run_replay_config(
        config, data_dir=cast_data_dir, runs_dir=runs_dir,
        cache_path=os.path.join(runs_dir, "cache.jsonl"),
    )
    assert manifest["n_dialogues"] == 2  # topics 31 + 32 in the fixture
    assert manifest["n_rewrites"] == 9 + 11  # turn counts from the real fixture

    rewrites = read_rewrites(manifest["rewrites_path"])
    assert len(rewrites) == manifest["n_rewrites"]
    assert all(r.query == ECHO_QUERY for r in rewrites)
    assert {r.turn_id for r in rewrites if r.dialogue_id == "31"} == {f"31_{i}" for i in range(1, 10)}

    on_disk_manifest = read_manifest(
        os.path.join(runs_dir, "manifests", f"{config.run_id}.replay.json")
    )
    assert on_disk_manifest == manifest


def test_run_replay_config_is_append_only(tmp_path, cast_data_dir):
    config = make_cast19_config()
    runs_dir = str(tmp_path / "runs")
    kwargs = dict(data_dir=cast_data_dir, runs_dir=runs_dir,
                  cache_path=os.path.join(runs_dir, "cache.jsonl"))
    run_replay_config(config, **kwargs)
    with pytest.raises(FileExistsError):
        run_replay_config(config, **kwargs)


def test_run_replay_config_resolves_pinned_model_from_env(tmp_path, cast_data_dir, monkeypatch):
    monkeypatch.setenv("EXPERIMENT_MODEL", "resolved-model-xyz")
    config = make_cast19_config(model="PINNED_IN_DOTENV")
    runs_dir = str(tmp_path / "runs")
    manifest = run_replay_config(
        config, data_dir=cast_data_dir, runs_dir=runs_dir,
        cache_path=os.path.join(runs_dir, "cache.jsonl"),
    )
    assert "resolved-model-xyz" in manifest["config"]
    assert "PINNED_IN_DOTENV" not in manifest["config"]


def test_run_replay_config_missing_experiment_model_raises(tmp_path, cast_data_dir, monkeypatch):
    monkeypatch.delenv("EXPERIMENT_MODEL", raising=False)
    config = make_cast19_config(model="PINNED_IN_DOTENV")
    with pytest.raises(RuntimeError, match="EXPERIMENT_MODEL"):
        run_replay_config(config, data_dir=cast_data_dir, runs_dir=str(tmp_path / "runs"))


def test_run_retrieval_config_end_to_end(tmp_path, cast_data_dir):
    config = make_cast19_config()
    runs_dir = str(tmp_path / "runs")
    run_replay_config(
        config, data_dir=cast_data_dir, runs_dir=runs_dir,
        cache_path=os.path.join(runs_dir, "cache.jsonl"),
    )
    candidates = [("dA", "text a", 0.5), ("dB", "text b", 0.9)]
    bm25 = FakeBM25({ECHO_QUERY: candidates})
    manifest = run_retrieval_config(config, runs_dir=runs_dir, bm25=bm25, ce=FakeCE())

    assert manifest["n_queries"] == 20  # 9 + 11 turns, matches run_replay's n_rewrites
    with open(manifest["run_path"], encoding="utf-8") as f:
        lines = [ln.split() for ln in f if ln.strip()]
    assert len(lines) == 20 * 2  # 2 candidates ranked per query
    assert {ln[0] for ln in lines} == {f"31_{i}" for i in range(1, 10)} | {f"32_{i}" for i in range(1, 12)}
    assert all(ln[1] == "Q0" and ln[5] == config.run_id for ln in lines)


def test_all_three_stages_agree_on_run_id_for_a_pinned_model_config(tmp_path, cast_data_dir,
                                                                    monkeypatch):
    """The grid's configs carry model="PINNED_IN_DOTENV"; the resolved snapshot is part of run
    identity. If only replay resolved it, retrieval would look for the hash of the placeholder
    while replay wrote the hash of the real model -- the run dies between stages, after the
    rewrites have already been paid for. Caught on the real gate configs before the gate run."""
    monkeypatch.setenv("EXPERIMENT_MODEL", "smoke-model")
    config = make_cast19_config(model="PINNED_IN_DOTENV")
    assert config.run_id != resolve_model(config).run_id, "fixture must actually exercise this"
    runs_dir = str(tmp_path / "runs")

    run_replay_config(config, data_dir=cast_data_dir, runs_dir=runs_dir,
                      cache_path=os.path.join(runs_dir, "cache.jsonl"))
    qrels = load_qrels(os.path.join(FIXTURES, "cast_smoke_qrels.txt"))
    candidates = [(d, f"passage {i}", 1.0) for i, d in enumerate(list(qrels["31_1"])[:5])]
    # would raise FileNotFoundError("run scripts/run_replay.py ... first") before the fix
    run_retrieval_config(config, runs_dir=runs_dir,
                         bm25=FakeBM25({ECHO_QUERY: candidates}), ce=FakeCE())
    shutil.copy(os.path.join(FIXTURES, "cast_smoke_qrels.txt"),
                os.path.join(cast_data_dir, "cast", "cast19_qrels.txt"))
    m = run_eval_config(config, data_dir=cast_data_dir, runs_dir=runs_dir,
                        allow_native_metrics=True)
    assert m["run_id"] == resolve_model(config).run_id
    assert "PINNED_IN_DOTENV" not in m["run_id"]


def test_run_retrieval_config_requires_replay_first(tmp_path):
    config = make_cast19_config()
    with pytest.raises(FileNotFoundError, match="run_replay"):
        run_retrieval_config(config, runs_dir=str(tmp_path / "runs"), bm25=FakeBM25({}), ce=FakeCE())


# ---------------------------------------------------------------------------
# run_eval
# ---------------------------------------------------------------------------

def _stage_full_pipeline(tmp_path, cast_data_dir, config):
    """replay -> retrieval, so there's a real TREC run file on disk to score. Returns runs_dir.
    Docs retrieved are drawn from the real qrels fixture so the scored numbers aren't all 0."""
    runs_dir = str(tmp_path / "runs")
    run_replay_config(
        config, data_dir=cast_data_dir, runs_dir=runs_dir,
        cache_path=os.path.join(runs_dir, "cache.jsonl"),
    )
    qrels = load_qrels(os.path.join(FIXTURES, "cast_smoke_qrels.txt"))
    judged_docs = list(qrels["31_1"])[:5]
    candidates = [(d, f"passage text {i}", 1.0) for i, d in enumerate(judged_docs)]
    run_retrieval_config(
        config, runs_dir=runs_dir, bm25=FakeBM25({ECHO_QUERY: candidates}), ce=FakeCE()
    )
    shutil.copy(
        os.path.join(FIXTURES, "cast_smoke_qrels.txt"),
        os.path.join(cast_data_dir, "cast", "cast19_qrels.txt"),
    )
    return runs_dir


def test_run_eval_config_end_to_end(tmp_path, cast_data_dir):
    config = make_cast19_config()
    runs_dir = _stage_full_pipeline(tmp_path, cast_data_dir, config)
    m = run_eval_config(
        config, data_dir=cast_data_dir, runs_dir=runs_dir, allow_native_metrics=True
    )

    # the fixture qrels only cover 31_1..31_3, but the run covers all 20 fixture turns
    assert m["coverage"]["n_scored"] == 3
    assert m["coverage"]["n_run_only"] == 17

    # allow_native_metrics is a PERMISSION, not a preference: when trec_eval is available it
    # is still used, and metrics_source records which one actually ran so any number computed
    # off the fallback stays traceable in stats_summary.json.
    from eval.extrinsic import HAVE_PYTREC_EVAL
    if HAVE_PYTREC_EVAL:
        assert m["metrics_source"] == "pytrec_eval"
    else:
        assert "UNVALIDATED" in m["metrics_source"]

    per_turn = {json.loads(ln)["turn_id"]: json.loads(ln)
                for ln in open(m["per_turn_path"], encoding="utf-8")}
    assert set(per_turn) == {"31_1", "31_2", "31_3"}
    assert all("judged_10" in r and "ndcg_cut_3" in r for r in per_turn.values())

    # judged@k is per-qid against THAT qid's own judgments, not a global judged-doc pool.
    # Every retrieved doc here is judged for 31_1, so 31_1 scores 1.0 -- while 31_2/31_3,
    # which were handed the same documents, score lower because those docs were not judged
    # for *their* turn. A global-pool implementation would wrongly give all three 1.0.
    assert per_turn["31_1"]["judged_10"] == pytest.approx(1.0)
    assert per_turn["31_2"]["judged_10"] < 1.0
    assert 0.0 < m["judged_10"] < 1.0

    stats = json.load(open(m["stats_path"], encoding="utf-8"))
    assert config.run_id in stats
    assert stats[config.run_id]["condition"] == "B0_none"
    assert stats[config.run_id]["memory_k"] == 5   # collect_stats reads it (k-sweep family)


def test_run_eval_config_refuses_unvalidated_metrics_by_default(tmp_path, cast_data_dir):
    from eval.extrinsic import HAVE_PYTREC_EVAL
    if HAVE_PYTREC_EVAL:
        pytest.skip("pytrec_eval installed -- the refusal path can't trigger here")
    config = make_cast19_config()
    runs_dir = _stage_full_pipeline(tmp_path, cast_data_dir, config)
    with pytest.raises(RuntimeError, match="pytrec_eval"):
        run_eval_config(config, data_dir=cast_data_dir, runs_dir=runs_dir)


def test_run_eval_config_per_turn_jsonl_is_append_only(tmp_path, cast_data_dir):
    config = make_cast19_config()
    runs_dir = _stage_full_pipeline(tmp_path, cast_data_dir, config)
    kwargs = dict(data_dir=cast_data_dir, runs_dir=runs_dir, allow_native_metrics=True)
    run_eval_config(config, **kwargs)
    with pytest.raises(FileExistsError):
        run_eval_config(config, **kwargs)


def test_run_eval_config_accumulates_stats_across_run_ids(tmp_path, cast_data_dir):
    """stats_summary.json is the one deliberately-not-append-only artifact: it's an index
    keyed by run_id that collect_stats/figures read whole. A second config must ADD to it,
    not replace it."""
    runs_dir = None
    b0 = make_cast19_config(condition="B0_none")
    runs_dir = _stage_full_pipeline(tmp_path, cast_data_dir, b0)
    run_eval_config(b0, data_dir=cast_data_dir, runs_dir=runs_dir, allow_native_metrics=True)

    b1 = make_cast19_config(condition="B1_full_history")
    run_replay_config(b1, data_dir=cast_data_dir, runs_dir=runs_dir,
                      cache_path=os.path.join(runs_dir, "cache.jsonl"))
    qrels = load_qrels(os.path.join(FIXTURES, "cast_smoke_qrels.txt"))
    candidates = [(d, f"passage {i}", 1.0) for i, d in enumerate(list(qrels["31_1"])[:5])]
    run_retrieval_config(b1, runs_dir=runs_dir,
                         bm25=FakeBM25({ECHO_QUERY: candidates}), ce=FakeCE())
    m1 = run_eval_config(b1, data_dir=cast_data_dir, runs_dir=runs_dir,
                         allow_native_metrics=True)

    stats = json.load(open(m1["stats_path"], encoding="utf-8"))
    assert set(stats) == {b0.run_id, b1.run_id}


def test_run_eval_config_requires_retrieval_first(tmp_path, cast_data_dir):
    config = make_cast19_config()
    shutil.copy(
        os.path.join(FIXTURES, "cast_smoke_qrels.txt"),
        os.path.join(cast_data_dir, "cast", "cast19_qrels.txt"),
    )
    with pytest.raises(FileNotFoundError, match="run_retrieval"):
        run_eval_config(config, data_dir=cast_data_dir, runs_dir=str(tmp_path / "runs"),
                        allow_native_metrics=True)


def test_run_eval_config_missing_qrels_points_at_fetch_script(tmp_path, cast_data_dir):
    config = make_cast19_config()
    with pytest.raises(FileNotFoundError, match="fetch_free_data"):
        run_eval_config(config, data_dir=cast_data_dir, runs_dir=str(tmp_path / "runs"),
                        allow_native_metrics=True)


def test_run_eval_config_unknown_collection_qrels_raises(tmp_path, cast_data_dir):
    config = make_cast19_config(collection="longmemeval")
    with pytest.raises(ValueError, match="no qrels registered"):
        run_eval_config(config, data_dir=cast_data_dir, runs_dir=str(tmp_path / "runs"))
