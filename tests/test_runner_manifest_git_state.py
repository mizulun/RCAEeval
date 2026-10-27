"""Research-run manifests must record the git commit and dirty state they were produced from."""
import subprocess
from pathlib import Path

from script.merit_source_provenance import repo_git_state

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_repo_git_state_matches_checkout():
    state = repo_git_state()
    assert set(state) >= {"available", "sha", "branch", "dirty", "status_short", "diff_sha256"}
    if not state["available"]:
        return  # source tarball without .git: fields are None/True, never missing
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()
    assert state["sha"] == head
    assert isinstance(state["dirty"], bool)
    assert isinstance(state["status_short"], list)


def test_repo_git_state_outside_repository(tmp_path):
    state = repo_git_state(tmp_path)
    assert state["available"] is False
    assert state["sha"] is None
    assert state["dirty"] is True
