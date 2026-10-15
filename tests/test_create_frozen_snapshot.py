from pathlib import Path

from script.create_frozen_snapshot import (
    SMALL_TEST_FIXTURES,
    _content_sha256,
    _copy_ignore,
    parse_args,
)


def test_snapshot_copy_excludes_generated_and_cache_files():
    ignored = _copy_ignore(
        "/tmp/source",
        [
            "module.py",
            "module.pyc",
            "paper.pdf",
            "paper",
            "__pycache__",
            "data",
            "README.md",
        ],
    )

    assert ignored == {"module.pyc", "paper.pdf", "paper", "__pycache__", "data"}


def test_snapshot_copy_excludes_private_thesis_documents():
    ignored = _copy_ignore(
        "/tmp/source/docs",
        ["method-theory.md", "thesis-workplan.md", "submission-checklist.md"],
    )

    assert ignored == {"thesis-workplan.md", "submission-checklist.md"}


def test_snapshot_cli_requires_an_independent_output_directory():
    args = parse_args(["--output-dir", "/tmp/frozen"])

    assert args.output_dir == Path("/tmp/frozen")
    assert SMALL_TEST_FIXTURES == (Path("tests/data/carts.toml"),)


def test_snapshot_content_hash_does_not_follow_external_environment_links(tmp_path):
    external = tmp_path / "external"
    external.mkdir()
    payload = external / "large.bin"
    payload.write_bytes(b"first")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "source.py").write_text("value = 1\n", encoding="utf-8")
    (snapshot / "data").symlink_to(external, target_is_directory=True)

    before = _content_sha256(snapshot)
    payload.write_bytes(b"changed outside snapshot")

    assert _content_sha256(snapshot) == before
