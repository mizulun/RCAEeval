"""Create an isolated, clean Git snapshot for formal MERIT experiments.

The source repository may contain unrelated user work.  This builder never
touches its ``.git`` directory: it copies the research tree to a new directory,
links the ignored dataset and Python 3.10 environment, initializes a local
anonymous Git repository, and verifies that the resulting snapshot is clean.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


COPY_DIRECTORIES = (
    ".github",
    "RCAEval",
    "LICENSES",
    "docs",
    "experiment",
    "script",
    "tests",
)
ROOT_SUFFIXES = {".py", ".sh", ".md", ".txt", ".lock"}
ROOT_NAMES = {".gitignore", "LICENSE", "pytest.ini", "requirements.txt", "setup.py"}
SMALL_TEST_FIXTURES = (Path("tests/data/carts.toml"),)
IGNORED_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".DS_Store",
    # The repository's tests/data directory is a second 39 GB dataset copy.
    # Formal experiments use the separately linked, ignored root data/ tree.
    "data",
    "paper",
}
PRIVATE_DOC_NAMES = {
    "private-submission-metadata.md",
    "submission-checklist.md",
    "paper-content-map.md",
    "paper-writing-workflow.md",
    "thesis-design-rationale-ledger.md",
    "thesis-workplan.md",
    "merit-thesis-english-blueprint-20260930.md",
}
PRIVATE_ROOT_NAMES = {"TASK.md"}
IGNORED_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".aux",
    ".bbl",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".out",
    ".xdv",
    ".pdf",
}


def _copy_ignore(_directory: str, names: list[str]) -> set[str]:
    in_docs = Path(_directory).name == "docs"
    return {
        name
        for name in names
        if name in IGNORED_NAMES
        or (in_docs and name in PRIVATE_DOC_NAMES)
        or Path(name).suffix in IGNORED_SUFFIXES
    }


def _run(root: Path, *command: str) -> str:
    process = subprocess.run(
        list(command),
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    )
    return process.stdout.strip()


def _content_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        parts = path.relative_to(root).parts
        if ".git" in parts or parts[0] in {"data", "py310"} or path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\n")
    return digest.hexdigest()


def create_snapshot(source_root: Path, output_dir: Path) -> dict:
    source_root = source_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)

    for name in COPY_DIRECTORIES:
        source = source_root / name
        if source.is_dir():
            shutil.copytree(source, output_dir / name, ignore=_copy_ignore)
    for source in source_root.iterdir():
        if not source.is_file():
            continue
        if source.name in PRIVATE_ROOT_NAMES:
            continue
        if source.name in ROOT_NAMES or source.suffix in ROOT_SUFFIXES:
            shutil.copy2(source, output_dir / source.name)

    for relative in SMALL_TEST_FIXTURES:
        source = source_root / relative
        destination = output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    for name in ("data", "py310"):
        source = source_root / name
        if not source.exists():
            raise FileNotFoundError(source)
        os.symlink(source, output_dir / name, target_is_directory=True)

    protocol = json.loads(
        (output_dir / "experiment/protocol.json").read_text(encoding="utf-8")
    )
    provenance = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": protocol["protocol_version"],
        "source_content_sha256": _content_sha256(output_dir),
        "data_link_target_recorded": False,
        "python_link_target_recorded": False,
    }
    (output_dir / "FROZEN_SNAPSHOT.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )

    _run(output_dir, "git", "init", "-b", "merit-frozen")
    _run(output_dir, "git", "config", "user.name", "Anonymous Research Snapshot")
    _run(output_dir, "git", "config", "user.email", "anonymous@invalid")
    _run(output_dir, "git", "add", "--all")
    _run(
        output_dir,
        "git",
        "commit",
        "-m",
        f"Freeze MERIT protocol {protocol['protocol_version']}",
    )
    status = _run(output_dir, "git", "status", "--porcelain")
    if status:
        raise RuntimeError(f"frozen snapshot is dirty:\n{status}")
    provenance["git_sha"] = _run(output_dir, "git", "rev-parse", "HEAD")
    provenance["output_dir"] = str(output_dir)
    return provenance


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    result = create_snapshot(args.source_root, args.output_dir)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
