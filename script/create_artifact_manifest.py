"""Create a provenance manifest before a MERIT paper experiment.

Formal runs use ``--checksum-mode full``.  ``metadata`` exists only for quick
smoke validation and is labeled non-cryptographic in the output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from script.merit_source_provenance import MERIT_SOURCE_FILES
except ModuleNotFoundError:  # direct ``python script/create_artifact_manifest.py``
    from merit_source_provenance import MERIT_SOURCE_FILES


DATASET_PATHS = {
    "re2-ob": Path("data/RE2/RE2-OB"),
    "re2-ss": Path("data/RE2/RE2-SS"),
    "re2-tt": Path("data/RE2/RE2-TT"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_checksum(root: Path, mode: str = "full") -> dict:
    if mode not in {"full", "metadata"}:
        raise ValueError(f"unsupported checksum mode: {mode}")
    if not root.is_dir():
        raise FileNotFoundError(root)

    digest = hashlib.sha256()
    count = 0
    total_bytes = 0
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        file_digest = sha256_file(path) if mode == "full" else "not-computed"
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\n")
        count += 1
        total_bytes += size
    return {
        "path": root.as_posix(),
        "file_count": count,
        "total_bytes": total_bytes,
        "checksum_mode": mode,
        "tree_sha256": digest.hexdigest(),
        "cryptographic_content_checksum": mode == "full",
    }


def _git(root: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    )
    return proc.stdout.strip()


def git_state(root: Path) -> dict:
    try:
        status = _git(root, "status", "--short")
        return {
            "available": True,
            "sha": _git(root, "rev-parse", "HEAD"),
            "branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(status),
            "status_short": status.splitlines(),
            "diff_sha256": hashlib.sha256(
                _git(root, "diff", "--binary", "HEAD").encode("utf-8")
            ).hexdigest(),
        }
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {
            "available": False,
            "sha": None,
            "branch": None,
            "dirty": True,
            "status_short": ["VCS metadata unavailable"],
            "diff_sha256": None,
        }


def _machine_memory_bytes() -> int | None:
    try:
        if sys.platform == "darwin":
            return int(
                subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    check=True,
                    text=True,
                    capture_output=True,
                ).stdout.strip()
            )
        if sys.platform == "win32":
            import psutil

            return int(psutil.virtual_memory().total)
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        return int(page_size * pages)
    except (
        AttributeError,
        ImportError,
        KeyError,
        OSError,
        ValueError,
        subprocess.SubprocessError,
    ):
        return None


def hardware_state() -> dict:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "logical_cpu_count": os.cpu_count(),
        "memory_bytes": _machine_memory_bytes(),
    }


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _case_count(root: Path) -> int:
    data_files = list(root.glob("*/*/data.csv"))
    if data_files:
        return len(data_files)
    return len(list(root.glob("*/*/simple_metrics.csv")))


def create_manifest(
    root: Path,
    protocol_path: Path,
    checksum_mode: str,
    command: str,
) -> dict:
    protocol = _load_json(root / protocol_path)
    datasets = {}
    for dataset in protocol["datasets"]:
        dataset_root = root / DATASET_PATHS[dataset]
        info = tree_checksum(dataset_root, checksum_mode)
        info["case_count"] = _case_count(dataset_root)
        info["expected_case_count"] = protocol["expected_cases_per_dataset"]
        if info["case_count"] != info["expected_case_count"]:
            raise ValueError(
                f"{dataset}: expected {info['expected_case_count']} cases, "
                f"found {info['case_count']}"
            )
        datasets[dataset] = info

    tracked_files = [
        protocol_path,
        *(Path(name) for name in MERIT_SOURCE_FILES),
        Path("RCAEval/e2e/__init__.py"),
        Path("RCAEval/benchmark/paper_evaluation.py"),
        Path("script/run_paper_dataset.py"),
        Path("script/run_paper_experiments.py"),
        Path("script/evaluate_paper_results.py"),
        Path("script/benchmark_banzhaf_equivalence.py"),
        Path("script/reproduce_paper.py"),
        Path("script/select_global_sweep.py"),
        Path("script/generate_paper_tables.py"),
        Path("script/crosscheck_baro_reference.py"),
        Path("script/verify_environment_lock.py"),
        Path("script/create_frozen_snapshot.py"),
        Path("requirements_py310.lock"),
        Path("pytest.ini"),
    ]
    for baseline in protocol["baseline_provenance"].values():
        tracked_files.extend(Path(item) for item in baseline["source_files"])
    tracked_files = list(dict.fromkeys(tracked_files))
    file_checksums = {}
    for relative in tracked_files:
        path = root / relative
        file_checksums[relative.as_posix()] = (
            sha256_file(path) if path.exists() else None
        )

    threads = {
        key: str(protocol["threads"][key])
        for key in sorted(protocol["threads"])
    }
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_manifest": checksum_mode == "full",
        "protocol_version": protocol["protocol_version"],
        "protocol_path": protocol_path.as_posix(),
        "git": git_state(root),
        "datasets": datasets,
        "dataset_release": protocol["dataset_release"],
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "file_sha256": file_checksums,
        "baseline_revisions": {
            method: {
                **metadata,
                "local_git_sha": git_state(root)["sha"],
                "source_sha256": {
                    source: file_checksums[source]
                    for source in metadata["source_files"]
                },
            }
            for method, metadata in protocol["baseline_provenance"].items()
        },
        "hardware": hardware_state(),
        "threads": threads,
        "seeds": protocol["seeds"],
        "per_case_timeout_seconds": protocol["per_case_timeout_seconds"],
        "per_case_timeout_scope": protocol["per_case_timeout_scope"],
        "dataset_worker_policy": protocol["dataset_worker_policy"],
        "dataset_budget_seconds": protocol["dataset_budget_seconds"],
        "workers": protocol["workers"],
        "command": command,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--protocol", type=Path, default=Path("experiment/protocol.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checksum-mode", choices=["full", "metadata"], default="full"
    )
    parser.add_argument("--command", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite manifest: {output}")
    manifest = create_manifest(
        root=root,
        protocol_path=args.protocol,
        checksum_mode=args.checksum_mode,
        command=args.command,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
