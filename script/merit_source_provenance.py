"""Fingerprint the complete split MERIT implementation for new research runs."""

import hashlib
from pathlib import Path


MERIT_SOURCE_FILES = (
    "RCAEval/e2e/merit.py",
    "RCAEval/e2e/merit_config.py",
    "RCAEval/e2e/merit_ablation.py",
)


def merit_source_hashes(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in MERIT_SOURCE_FILES}


def require_merit_sources(manifest, root=None):
    """Never qualify an old single-file manifest as a current split-source run."""
    current = merit_source_hashes(root)
    if manifest.get("method_source_sha256") != current:
        raise ValueError("MERIT source set differs or is incomplete; reuse forbidden")
    return current


def snapshot_merit_sources(destination, root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name in MERIT_SOURCE_FILES:
        (destination / Path(name).name).write_bytes((root / name).read_bytes())
