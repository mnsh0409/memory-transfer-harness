"""GAM stores nothing on Windows unless its shell commands go through bash.

This is why the first GAM run produced a TREC file byte-identical to B0 after 5,379 model
calls and 3.35M tokens. GAM's `TextGAMAgent` edits its knowledge base with Unix commands via
`subprocess.run(code, shell=True)`, which on Windows is `cmd.exe`:

  * `mkdir -p "..."` half-works — cmd's `mkdir` treats `-p` as another directory to create and
    still exits 0, so GAM's exit-code check passes and a junk `-p` directory appears.
  * every file is written with a bash heredoc, `cat > "..." << 'EOF_FILE_CONTENT'`, which
    cmd.exe cannot run at all.

`add()` then returns `BatchProcessingResult(success=True, errors=[], created_files=[])` — GAM
reports success having stored nothing. These tests pin the failure mode itself, not just the
fix, so that the reason cannot be lost.
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

import adapters.gam_adapter as gam_mod

HEREDOC = "cat > \"{path}\" << 'EOF_FILE_CONTENT'\nhello\nworld\nEOF_FILE_CONTENT"


def test_the_failure_mode_is_real_on_this_platform(tmp_path):
    """cmd.exe cannot run GAM's file write. Skipped off Windows, where the bug cannot occur."""
    if os.name != "nt":
        pytest.skip("cmd.exe failure mode is Windows-only")
    target = (tmp_path / "sub" / "f.md").as_posix()
    subprocess.run(f'mkdir -p "{(tmp_path / "sub").as_posix()}"', shell=True, cwd=tmp_path,
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    r = subprocess.run(HEREDOC.format(path=target), shell=True, cwd=tmp_path,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert r.returncode != 0, "cmd.exe unexpectedly ran a heredoc"
    assert not (tmp_path / "sub" / "f.md").exists()
    # ...and the tell-tale junk directory `mkdir -p` leaves behind
    assert (tmp_path / "-p").exists(), "expected cmd's mkdir to create a literal '-p' dir"


def test_bash_runs_the_same_commands_correctly(tmp_path):
    bash = __import__("shutil").which("bash")
    if not bash:
        pytest.skip("no bash on PATH")
    target = (tmp_path / "sub" / "f.md").as_posix()
    subprocess.run([bash, "-c", f'mkdir -p "{(tmp_path / "sub").as_posix()}"'], cwd=tmp_path,
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    r = subprocess.run([bash, "-c", HEREDOC.format(path=target)], cwd=tmp_path,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert r.returncode == 0
    assert (tmp_path / "sub" / "f.md").read_text().strip() == "hello\nworld"
    assert not (tmp_path / "-p").exists()


@pytest.mark.skipif(not gam_mod.HAVE_GAM, reason="gam not installed")
def test_patch_is_applied_on_windows():
    from gam.workspaces.local_workspace import LocalWorkspace
    if os.name != "nt":
        assert gam_mod.POSIX_SHELL_PATCHED is False, "patch must be a no-op on POSIX"
        return
    assert gam_mod.POSIX_SHELL_PATCHED, "GAM would store nothing without this"
    assert getattr(LocalWorkspace, "_harness_posix_patch", False)


@pytest.mark.skipif(not gam_mod.HAVE_GAM, reason="gam not installed")
def test_patched_workspace_actually_writes_the_file(tmp_path):
    """End-to-end through GAM's own workspace object: the heredoc must produce a file."""
    if os.name != "nt":
        pytest.skip("patch only applies on Windows")
    from gam.workspaces.local_workspace import LocalWorkspace

    ws = LocalWorkspace(str(tmp_path))
    out, rc = ws.run(f'mkdir -p "{(tmp_path / "sub").as_posix()}"')
    assert rc == "0", out
    out, rc = ws.run(HEREDOC.format(path=(tmp_path / "sub" / "f.md").as_posix()))
    assert rc == "0", out
    assert (tmp_path / "sub" / "f.md").exists(), "GAM would silently store nothing"


@pytest.mark.skipif(not gam_mod.HAVE_GAM, reason="gam not installed")
def test_patched_run_reports_failure_rather_than_swallowing_it(tmp_path):
    """A non-zero exit must surface as a non-'0' code -- GAM checks it, and swallowing it is
    how the original bug stayed invisible."""
    if os.name != "nt":
        pytest.skip("patch only applies on Windows")
    from gam.workspaces.local_workspace import LocalWorkspace

    out, rc = LocalWorkspace(str(tmp_path)).run("exit 3")
    assert rc == "3", (out, rc)


@pytest.mark.skipif(not gam_mod.HAVE_GAM, reason="gam not installed")
def test_patch_is_idempotent():
    assert gam_mod._posix_shell_patch_applied() == gam_mod.POSIX_SHELL_PATCHED
