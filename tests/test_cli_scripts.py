"""Subprocess-level test of the actual Fixed CLI Contract entry points:

    python -m scripts.run_replay    --config CFG.json
    python -m scripts.run_retrieval --config CFG.json

tests/test_runners.py already covers the underlying logic in-process; this file exists
separately to catch wiring bugs a direct function call can't -- sys.path bootstrap, argparse,
`python -m scripts.X` module resolution -- by actually invoking the command as documented."""
import json
import os
import shutil
import subprocess
import sys

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def run_cli(module, args, cwd_extra_env=None):
    env = dict(os.environ)
    if cwd_extra_env:
        env.update(cwd_extra_env)
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=60,
    )


def test_run_replay_cli_end_to_end(tmp_path):
    data_dir = tmp_path / "data" / "cast"
    data_dir.mkdir(parents=True)
    shutil.copy(os.path.join(FIXTURES, "cast_smoke.json"), data_dir / "cast19_topics.json")

    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps({
        "condition": "B0_none", "injection": "rewrite_only", "collection": "cast19",
        "backend": "echo", "model": "smoke-model",
    }))
    runs_dir = tmp_path / "runs"

    result = run_cli("scripts.run_replay", [
        "--config", str(config_path),
        "--data-dir", str(tmp_path / "data"),
        "--runs-dir", str(runs_dir),
        "--cache", str(runs_dir / "cache.jsonl"),
        "--env", str(tmp_path / "nonexistent.env"),
    ])

    assert result.returncode == 0, result.stderr
    assert "run_replay" in result.stdout
    assert "20 rewrites" in result.stdout  # 9 + 11 turns in the fixture
    rewrite_files = list((runs_dir / "rewrites").glob("*.jsonl"))
    assert len(rewrite_files) == 1


def test_run_replay_cli_missing_config_file_fails_clearly(tmp_path):
    result = run_cli("scripts.run_replay", ["--config", str(tmp_path / "nope.json")])
    assert result.returncode != 0
    assert "nope.json" in result.stderr or "No such file" in result.stderr
