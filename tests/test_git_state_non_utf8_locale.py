"""git_state() must not depend on the locale codec: a dirty UTF-8 diff under a CP950 console
used to raise UnicodeDecodeError before any manifest was written (PR #14 review)."""
import hashlib
import subprocess
from pathlib import Path

import pytest

from script import create_artifact_manifest
from script.create_artifact_manifest import git_state
from script.merit_source_provenance import repo_git_state

CHINESE = "研究用 runner 的 manifest 記錄 git commit SHA 與 dirty 狀態\n"


def _git(cwd: Path, *args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True).stdout


@pytest.fixture
def dirty_utf8_repo(tmp_path: Path) -> Path:
    try:
        _git(tmp_path, "init", "-q")
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("git not available")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "test")
    _git(tmp_path, "config", "core.autocrlf", "false")
    note = tmp_path / "note.txt"
    note.write_text("ascii only\n", encoding="utf-8")
    _git(tmp_path, "add", "note.txt")
    _git(tmp_path, "commit", "-q", "-m", "init")
    note.write_text(CHINESE, encoding="utf-8")  # uncommitted UTF-8 change -> dirty diff
    return tmp_path


@pytest.fixture
def cp950_console(monkeypatch):
    """Simulate a CP950 locale: any text-mode subprocess call without an explicit
    encoding decodes stdout with cp950 (strict), as Python 3.10 does on a zh-TW console."""
    real_run = subprocess.run

    def run(args, **kwargs):
        if kwargs.get("text") and not kwargs.get("encoding"):
            kwargs = {k: v for k, v in kwargs.items() if k != "text"}
            proc = real_run(args, **kwargs)
            proc.stdout = proc.stdout.decode("cp950")  # raises on UTF-8 Chinese bytes
            proc.stderr = proc.stderr.decode("cp950") if proc.stderr else proc.stderr
            return proc
        return real_run(args, **kwargs)

    monkeypatch.setattr(create_artifact_manifest.subprocess, "run", run)
    return run


def test_simulated_locale_would_reject_the_diff(dirty_utf8_repo):
    raw = _git(dirty_utf8_repo, "diff", "--binary", "HEAD")
    assert CHINESE.encode("utf-8").splitlines()[0] in raw
    with pytest.raises(UnicodeDecodeError):
        raw.decode("cp950")


def test_git_state_dirty_utf8_diff_under_cp950_console(dirty_utf8_repo, cp950_console):
    state = git_state(dirty_utf8_repo)
    assert state["available"] is True
    assert state["dirty"] is True
    assert any("note.txt" in line for line in state["status_short"])
    expected = hashlib.sha256(_git(dirty_utf8_repo, "diff", "--binary", "HEAD")).hexdigest()
    assert state["diff_sha256"] == expected
    assert state["sha"] == _git(dirty_utf8_repo, "rev-parse", "HEAD").decode().strip()


def test_repo_git_state_uses_the_same_path(dirty_utf8_repo, cp950_console):
    state = repo_git_state(dirty_utf8_repo)
    assert state["available"] is True and state["dirty"] is True


def test_clean_tree_diff_hash_is_sha256_of_empty_bytes(dirty_utf8_repo):
    _git(dirty_utf8_repo, "checkout", "--", "note.txt")
    state = git_state(dirty_utf8_repo)
    assert state["dirty"] is False
    assert state["diff_sha256"] == hashlib.sha256(b"").hexdigest()


def test_unavailable_repository_falls_back(tmp_path):
    state = git_state(tmp_path)
    assert state["available"] is False and state["sha"] is None and state["dirty"] is True
