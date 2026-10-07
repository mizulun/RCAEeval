"""Verify the frozen Python environment used by paper experiments."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import sys
from pathlib import Path


PIN = re.compile(
    r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;]+)(?:\s*;.*)?$"
)


def canonical_name(name: str) -> str:
    """Apply the package-name normalization used by Python indexes."""

    return re.sub(r"[-_.]+", "-", name).lower()


def parse_lock(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("-e "):
            continue
        match = PIN.fullmatch(line)
        if match is None:
            raise ValueError(
                f"unsupported non-exact requirement at {path}:{line_number}: {line}"
            )
        name = canonical_name(match.group(1))
        version = match.group(2)
        previous = pins.get(name)
        if previous is not None and previous != version:
            raise ValueError(f"conflicting pins for {name}: {previous}, {version}")
        pins[name] = version
    if not pins:
        raise ValueError(f"lock file contains no exact package pins: {path}")
    return pins


def installed_versions() -> dict[str, str]:
    versions = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            versions[canonical_name(name)] = distribution.version
    return versions


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_environment(
    lock_path: Path,
    expected_python: str,
    *,
    actual_packages: dict[str, str] | None = None,
    actual_python: str | None = None,
) -> dict:
    expected = parse_lock(lock_path)
    actual = (
        {canonical_name(name): value for name, value in actual_packages.items()}
        if actual_packages is not None
        else installed_versions()
    )
    interpreter_version = actual_python or platform.python_version()
    missing = sorted(name for name in expected if name not in actual)
    mismatched = {
        name: {"expected": expected[name], "actual": actual[name]}
        for name in sorted(expected)
        if name in actual and actual[name] != expected[name]
    }
    python_matches = interpreter_version == expected_python
    return {
        "schema_version": 1,
        "ok": python_matches and not missing and not mismatched,
        "python": {
            "expected": expected_python,
            "actual": interpreter_version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
            "matches": python_matches,
        },
        "lock_file": lock_path.name,
        "lock_sha256": sha256_file(lock_path),
        "pinned_package_count": len(expected),
        "installed_package_count": len(actual),
        "missing": missing,
        "version_mismatches": mismatched,
        "extra_packages_allowed": True,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--protocol", type=Path, default=Path("experiment/protocol.json")
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    root = args.repo_root.resolve()
    protocol_path = (
        args.protocol if args.protocol.is_absolute() else root / args.protocol
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    frozen = protocol["python_environment"]
    lock_path = root / frozen["lock_file"]
    report = audit_environment(lock_path, frozen["version"])
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
