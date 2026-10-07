import hashlib
from pathlib import Path

import pytest

from script.create_artifact_manifest import git_state, sha256_file, tree_checksum


def test_sha256_file_matches_known_content(tmp_path):
    path = tmp_path / "value.bin"
    path.write_bytes(b"MERIT\n")

    assert sha256_file(path) == hashlib.sha256(b"MERIT\n").hexdigest()


def test_tree_checksum_is_stable_and_content_sensitive(tmp_path):
    root = tmp_path / "dataset"
    (root / "b").mkdir(parents=True)
    (root / "a.txt").write_text("one", encoding="utf-8")
    (root / "b" / "c.txt").write_text("two", encoding="utf-8")

    first = tree_checksum(root, "full")
    second = tree_checksum(root, "full")
    assert first == second
    assert first["file_count"] == 2
    assert first["cryptographic_content_checksum"] is True

    (root / "b" / "c.txt").write_text("changed", encoding="utf-8")
    assert tree_checksum(root, "full")["tree_sha256"] != first["tree_sha256"]


def test_metadata_checksum_is_explicitly_non_cryptographic(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "a.txt").write_text("one", encoding="utf-8")

    result = tree_checksum(root, "metadata")

    assert result["checksum_mode"] == "metadata"
    assert result["cryptographic_content_checksum"] is False


def test_tree_checksum_rejects_unknown_mode(tmp_path):
    with pytest.raises(ValueError, match="unsupported checksum mode"):
        tree_checksum(Path(tmp_path), "quick")


def test_archive_without_vcs_metadata_is_explicitly_dirty(tmp_path):
    state = git_state(tmp_path)

    assert state == {
        "available": False,
        "sha": None,
        "branch": None,
        "dirty": True,
        "status_short": ["VCS metadata unavailable"],
        "diff_sha256": None,
    }
