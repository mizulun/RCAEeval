from pathlib import Path

import pytest

from script.merit_source_provenance import (
    MERIT_SOURCE_FILES, merit_source_hashes, require_merit_sources,
    snapshot_merit_sources,
)


@pytest.mark.parametrize("changed", MERIT_SOURCE_FILES)
def test_each_module_change_invalidates_reuse(tmp_path, changed):
    for name in MERIT_SOURCE_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    manifest = {"method_source_sha256": merit_source_hashes(tmp_path)}
    assert require_merit_sources(manifest, tmp_path) == manifest["method_source_sha256"]
    (tmp_path / changed).write_text("changed")
    with pytest.raises(ValueError, match="reuse forbidden"):
        require_merit_sources(manifest, tmp_path)


def test_old_single_file_manifest_is_not_current_evidence():
    hashes = merit_source_hashes()
    with pytest.raises(ValueError, match="incomplete"):
        require_merit_sources({"source_sha256": hashes[MERIT_SOURCE_FILES[0]]})


def test_snapshot_contains_all_seven_identical_sources(tmp_path):
    snapshot_merit_sources(tmp_path)
    root = Path(__file__).resolve().parents[1]
    for name in MERIT_SOURCE_FILES:
        assert (tmp_path / Path(name).name).read_bytes() == (root / name).read_bytes()


def test_formal_manifest_tracks_all_split_modules(tmp_path, monkeypatch):
    import json
    from script import create_artifact_manifest as creator
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / "experiment/protocol.json").read_text())
    protocol.update(datasets=[], baseline_provenance={})
    monkeypatch.setattr(creator, "_load_json", lambda path: protocol)
    monkeypatch.setattr(creator, "git_state", lambda root: {"sha": "test", "dirty": False})
    for name in MERIT_SOURCE_FILES:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / name).read_bytes())
    result = creator.create_manifest(tmp_path, Path("experiment/protocol.json"), "full", "test")
    expected = merit_source_hashes(tmp_path)
    assert {name: result["file_sha256"][name] for name in MERIT_SOURCE_FILES} == expected
