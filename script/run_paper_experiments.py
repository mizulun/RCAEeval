"""One-entry orchestration for frozen MERIT paper experiments."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic

from RCAEval.benchmark.input_boundaries import (
    BOUNDARY_SCHEMA_VERSION,
    inspect_case_exposure,
)

try:
    from script.create_artifact_manifest import create_manifest, git_state, sha256_file
except ModuleNotFoundError:  # direct ``python script/...py`` execution
    from create_artifact_manifest import create_manifest, git_state, sha256_file


@dataclass(frozen=True)
class MethodSpec:
    label: str
    base_method: str
    environment: dict[str, str]
    seed: int | None = None
    case_order_seed: int | None = None
    warmup_method_call: bool = False


FIXED_VARIANTS = {
    "merit_full": MethodSpec("merit_full", "merit", {}),
    "merit_no_trace": MethodSpec(
        "merit_no_trace", "merit", {"GABANZHAF_DISABLE_TRACE": "1"}
    ),
    "merit_no_metric": MethodSpec(
        "merit_no_metric", "merit", {"GABANZHAF_DISABLE_METRICS": "1"}
    ),
    "merit_no_trace_topology": MethodSpec(
        "merit_no_trace_topology",
        "merit",
        {"GABANZHAF_DISABLE_TRACE_TOPOLOGY": "1"},
    ),
    "merit_no_coverage": MethodSpec(
        "merit_no_coverage", "merit", {"GABANZHAF_DISABLE_COVERAGE": "1"}
    ),
    "merit_no_victim_demotion": MethodSpec(
        "merit_no_victim_demotion",
        "merit",
        {"GABANZHAF_DISABLE_VICTIM_DEMOTION": "1"},
    ),
    "merit_no_coverage_no_victim": MethodSpec(
        "merit_no_coverage_no_victim",
        "merit",
        {
            "GABANZHAF_DISABLE_COVERAGE": "1",
            "GABANZHAF_DISABLE_VICTIM_DEMOTION": "1",
        },
    ),
    "merit_no_coverage_parent_protection": MethodSpec(
        "merit_no_coverage_parent_protection",
        "merit",
        {"GABANZHAF_DISABLE_COVERAGE_PARENT_PROTECTION": "1"},
    ),
    "merit_no_metric_type_dominance": MethodSpec(
        "merit_no_metric_type_dominance",
        "merit",
        {"GABANZHAF_DISABLE_METRIC_TYPE_DOMINANCE": "1"},
    ),
    "merit_no_early_full_fusion": MethodSpec(
        "merit_no_early_full_fusion",
        "merit",
        {"GABANZHAF_DISABLE_EARLY_FULL_FUSION": "1"},
    ),
    "merit_trace_latency_only": MethodSpec(
        "merit_trace_latency_only",
        "merit",
        {"GABANZHAF_TRACE_SIGNAL": "latency"},
    ),
    "merit_trace_error_only": MethodSpec(
        "merit_trace_error_only",
        "merit",
        {"GABANZHAF_TRACE_SIGNAL": "error"},
    ),
    "merit_trace_lag0": MethodSpec(
        "merit_trace_lag0",
        "merit",
        {"GABANZHAF_TRACE_MAX_LAG": "0"},
    ),
    "merit_trace_time_shuffle": MethodSpec(
        "merit_trace_time_shuffle",
        "merit",
        {"GABANZHAF_TRACE_CONTROL": "time_shuffle"},
    ),
    "merit_trace_random_edges": MethodSpec(
        "merit_trace_random_edges",
        "merit",
        {"GABANZHAF_TRACE_CONTROL": "degree_matched_random_edges"},
    ),
    "merit_direct_service_evidence": MethodSpec(
        "merit_direct_service_evidence",
        "merit",
        {"GABANZHAF_RANK_MODE": "service_evidence"},
    ),
    "merit_direct_service_trace_evidence": MethodSpec(
        "merit_direct_service_trace_evidence",
        "merit",
        {"GABANZHAF_RANK_MODE": "service_evidence_trace"},
    ),
    "merit_trace_only": MethodSpec(
        "merit_trace_only", "merit", {"GABANZHAF_RANK_MODE": "trace_evidence"}
    ),
    "merit_individual_attribution": MethodSpec(
        "merit_individual_attribution",
        "merit",
        {"GABANZHAF_BANZHAF_MODE": "individual"},
    ),
    "merit_graph_per_case_full": MethodSpec(
        "merit_graph_per_case_full",
        "merit",
        {"GABANZHAF_GRAPH_SOURCE": "per_case_full"},
    ),
    "merit_graph_first_case_pre_fault": MethodSpec(
        "merit_graph_first_case_pre_fault",
        "merit",
        {"GABANZHAF_GRAPH_SOURCE": "first_case_pre_fault"},
    ),
    "merit_graph_union_pre_fault": MethodSpec(
        "merit_graph_union_pre_fault",
        "merit",
        {"GABANZHAF_GRAPH_SOURCE": "union_pre_fault"},
    ),
    "merit_graph_first_case_order1": MethodSpec(
        "merit_graph_first_case_order1",
        "merit",
        {"GABANZHAF_GRAPH_SOURCE": "first_case_pre_fault"},
        case_order_seed=20260713,
    ),
    "merit_graph_first_case_order2": MethodSpec(
        "merit_graph_first_case_order2",
        "merit",
        {"GABANZHAF_GRAPH_SOURCE": "first_case_pre_fault"},
        case_order_seed=20260714,
    ),
    "merit_graph_first_case_order3": MethodSpec(
        "merit_graph_first_case_order3",
        "merit",
        {"GABANZHAF_GRAPH_SOURCE": "first_case_pre_fault"},
        case_order_seed=20260715,
    ),
    "merit_bruteforce": MethodSpec(
        "merit_bruteforce", "merit", {"GABANZHAF_BANZHAF_MODE": "bruteforce"}
    ),
}


def resolve_method(label: str) -> MethodSpec:
    if label in FIXED_VARIANTS:
        return FIXED_VARIANTS[label]
    repeated = re.fullmatch(r"merit_full_rep([1-9]\d*)", label)
    if repeated:
        return MethodSpec(label=label, base_method="merit", environment={})
    warm_repeated = re.fullmatch(r"merit_warm_rep([1-9]\d*)", label)
    if warm_repeated:
        return MethodSpec(
            label=label,
            base_method="merit",
            environment={},
            warmup_method_call=True,
        )
    seeded = re.fullmatch(r"(rcd|causalrca)_seed(\d+)", label)
    if seeded:
        return MethodSpec(
            label=label,
            base_method=seeded.group(1),
            environment={},
            seed=int(seeded.group(2)),
        )
    patterns = [
        (r"merit_victim_(0[1-9])$", "GABANZHAF_VICTIM_FLOOR"),
        (r"merit_tconf_(0[1-9])$", "GABANZHAF_TRACE_TOPOLOGY_CONF"),
        (r"merit_cconf_(0[1-9])$", "GABANZHAF_COVERAGE_CONF"),
    ]
    for pattern, variable in patterns:
        match = re.fullmatch(pattern, label)
        if match:
            value = str(int(match.group(1)) / 10)
            return MethodSpec(label, "merit", {variable: value})
    match = re.fullmatch(r"merit_k(3|6|9|12|15)", label)
    if match:
        return MethodSpec(
            label,
            "merit",
            {"GABANZHAF_EXACT_CANDIDATE_K": match.group(1)},
        )
    match = re.fullmatch(r"merit_trace_dropout_(10|30|50|70|100)", label)
    if match:
        return MethodSpec(
            label,
            "merit",
            {"GABANZHAF_TRACE_DROPOUT": str(int(match.group(1)) / 100)},
        )
    match = re.fullmatch(r"merit_trace_(edge_)?noise_(10|30|50)", label)
    if match:
        variable = (
            "GABANZHAF_TRACE_EDGE_NOISE"
            if match.group(1)
            else "GABANZHAF_TRACE_NOISE"
        )
        return MethodSpec(
            label,
            "merit",
            {variable: str(int(match.group(2)) / 100)},
        )
    return MethodSpec(label, label, {})


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument(
        "--datasets", nargs="+", default=["re2-ob", "re2-ss", "re2-tt"]
    )
    parser.add_argument("--protocol", type=Path, default=Path("experiment/protocol.json"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--checksum-mode", choices=["full", "metadata"], default="full")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--manifest-template", type=Path)
    parser.add_argument(
        "--evaluation-mode",
        choices=["full_record_retrospective", "window_sensitivity"],
    )
    parser.add_argument("--length", type=int)
    parser.add_argument("--tdelta", type=int, default=0)
    parser.add_argument(
        "--graph-source",
        choices=[
            "per_case_pre_fault",
            "per_case_full",
            "first_case_pre_fault",
            "union_pre_fault",
        ],
    )
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--limit", type=int)
    return parser.parse_args(argv)


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _infeasible_task_record(
    label: str, base_method: str, dataset: str, declaration: dict
) -> dict:
    details = dict(declaration)
    declared_status = details.pop("status", "N/A")
    return {
        "method": label,
        "base_method": base_method,
        "dataset": dataset,
        **details,
        "status": "infeasible",
        "declared_status": declared_status,
    }


def _inapplicable_task_record(
    label: str, base_method: str, dataset: str, declaration: dict
) -> dict:
    details = dict(declaration)
    declared_status = details.pop("status", "N/A")
    return {
        "method": label,
        "base_method": base_method,
        "dataset": dataset,
        **details,
        "status": "inapplicable",
        "declared_status": declared_status,
    }


def _all_tasks_succeeded(task_records: list[dict]) -> bool:
    return all(
        record["status"] in {"completed", "infeasible", "inapplicable"}
        for record in task_records
    )


def _validate_dataset_budget(protocol: dict, limit: int | None = None) -> None:
    expected_cases = int(protocol["expected_cases_per_dataset"])
    attempted_cases = min(limit, expected_cases) if limit is not None else expected_cases
    minimum = attempted_cases * int(protocol["per_case_timeout_seconds"])
    actual = int(protocol["dataset_budget_seconds"])
    if actual < minimum:
        raise ValueError(
            "dataset budget can truncate required case artifacts: "
            f"{actual}s < {attempted_cases} cases x "
            f"{protocol['per_case_timeout_seconds']}s = {minimum}s"
        )


def _resolve_python(repo_root: Path, python_rel: str) -> Path:
    """Resolve a POSIX venv path such as ./py310/bin/python, falling back to the
    Windows venv layout (py310/Scripts/python.exe) when only that one exists."""

    python = repo_root / python_rel.removeprefix("./")
    if not python.exists() and python.parent.name == "bin":
        windows = python.parent.parent / "Scripts" / "python.exe"
        if windows.exists():
            return windows
    return python


def _validate_method_environments(
    repo_root: Path, protocol_path: Path, protocol: dict
) -> list[dict]:
    """Audit each distinct configured interpreter before any run artifact exists."""

    reports = []
    for python_rel in sorted(set(protocol["method_environments"].values())):
        python = _resolve_python(repo_root, python_rel)
        if not python.exists():
            raise FileNotFoundError(f"frozen environment is absent: {python}")
        process = subprocess.run(
            [
                str(python),
                "script/verify_environment_lock.py",
                "--repo-root",
                str(repo_root),
                "--protocol",
                str(protocol_path),
            ],
            cwd=repo_root,
            text=True,
            capture_output=True,
        )
        if process.returncode != 0:
            details = "\n".join(
                item for item in (process.stdout.strip(), process.stderr.strip()) if item
            )
            raise RuntimeError(
                f"frozen environment validation failed for {python_rel}:\n{details}"
            )
        try:
            report = json.loads(process.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"environment validator returned invalid JSON for {python_rel}"
            ) from exc
        if not report.get("ok"):
            raise RuntimeError(f"environment validator rejected {python_rel}")
        report["configured_environment"] = python_rel
        reports.append(report)
    return reports


def _requested_run_config(args, protocol: dict) -> dict:
    return {
        "schema_version": 2,
        "protocol_version": protocol["protocol_version"],
        "methods": list(args.methods),
        "datasets": list(args.datasets),
        "checksum_mode": args.checksum_mode,
        "allow_dirty": bool(args.allow_dirty),
        "length": args.length,
        "tdelta": args.tdelta,
        "evaluation_mode": args.evaluation_mode,
        "graph_source": args.graph_source,
        "seed": args.seed,
        "limit": args.limit,
        "manifest_template": (
            str(args.manifest_template.resolve()) if args.manifest_template else None
        ),
    }


def _validate_resume_manifest(repo_root: Path, manifest: dict, protocol: dict) -> None:
    if manifest.get("protocol_version") != protocol["protocol_version"]:
        raise RuntimeError("resume manifest/protocol version mismatch")
    current_git = git_state(repo_root)
    saved_git = manifest.get("git", {})
    for key in ("sha", "dirty", "diff_sha256"):
        if saved_git.get(key) != current_git.get(key):
            raise RuntimeError(f"resume source snapshot changed: git.{key}")
    for relative, expected in manifest.get("file_sha256", {}).items():
        path = repo_root / relative
        actual = sha256_file(path) if path.exists() else None
        if actual != expected:
            raise RuntimeError(f"resume source file changed: {relative}")


def _validate_manifest_output_path(manifest: dict, run_root: Path) -> None:
    if manifest.get("output_path") != str(run_root):
        raise RuntimeError("resume manifest/output path mismatch")


def _completed_resume_record(
    combination_root: Path, expected_cases: int
) -> dict | None:
    task_path = combination_root / "task.json"
    combination_path = combination_root / "combination.json"
    if not task_path.is_file() or not combination_path.is_file():
        return None
    task = json.loads(task_path.read_text(encoding="utf-8"))
    combination = json.loads(combination_path.read_text(encoding="utf-8"))
    if task.get("status") == "infeasible":
        return task
    if task.get("status") == "inapplicable":
        return task
    if task.get("status") != "completed":
        return None
    if combination.get("cases_attempted") != expected_cases:
        return None
    metadata_dir = combination_root / "results" / "case_metadata"
    if not metadata_dir.is_dir():
        return None
    return task if len(list(metadata_dir.glob("*.json"))) == expected_cases else None


def _archive_interrupted_combination(
    run_root: Path, combination_root: Path, label: str, dataset: str
) -> Path:
    archive_root = run_root / "interrupted-attempts" / label / dataset
    archive_root.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while (archive_root / f"attempt-{attempt}").exists():
        attempt += 1
    destination = archive_root / f"attempt-{attempt}"
    shutil.move(str(combination_root), destination)
    return destination


def _manifest_from_template(
    repo_root: Path,
    template_path: Path,
    protocol: dict,
    checksum_mode: str,
    command: str,
) -> dict:
    template = json.loads(template_path.read_text(encoding="utf-8"))
    _validate_resume_manifest(repo_root, template, protocol)
    expected_formal = checksum_mode == "full"
    if bool(template.get("formal_manifest")) != expected_formal:
        raise RuntimeError("manifest template checksum mode mismatch")
    for dataset in protocol["datasets"]:
        metadata = template.get("datasets", {}).get(dataset, {})
        if metadata.get("case_count") != protocol["expected_cases_per_dataset"]:
            raise RuntimeError(f"manifest template has invalid {dataset} case count")
        if expected_formal and not metadata.get("cryptographic_content_checksum"):
            raise RuntimeError(f"manifest template lacks full {dataset} checksum")
    manifest = json.loads(json.dumps(template))
    manifest["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["command"] = command
    manifest["derived_from_manifest"] = str(template_path.resolve())
    return manifest


def _formal_output_path_is_safe(repo_root: Path, run_root: Path) -> bool:
    try:
        relative = run_root.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return True
    process = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", relative.as_posix()],
        cwd=repo_root,
        check=False,
    )
    return process.returncode == 0


DATASET_PATHS = {
    "re2-ob": Path("data/RE2/RE2-OB"),
    "re2-ss": Path("data/RE2/RE2-SS"),
    "re2-tt": Path("data/RE2/RE2-TT"),
}


def _dataset_case_paths(repo_root: Path, dataset: str) -> list[Path]:
    root = repo_root / DATASET_PATHS[dataset]
    paths = sorted(root.glob("*/*/data.csv"))
    return paths or sorted(root.glob("*/*/simple_metrics.csv"))


def _build_exposure_manifest(
    repo_root: Path,
    dataset: str,
    mode: str,
    length: int | None,
    tdelta: int,
    limit: int | None,
) -> dict:
    paths = _dataset_case_paths(repo_root, dataset)
    if limit is not None:
        paths = paths[:limit]
    cases = [
        inspect_case_exposure(
            path,
            mode=mode,
            length_minutes=length,
            tdelta_seconds=tdelta,
        )
        for path in paths
    ]
    return {
        "schema_version": BOUNDARY_SCHEMA_VERSION,
        "dataset": dataset,
        "evaluation_mode": mode,
        "length_minutes": length,
        "tdelta_seconds": tdelta,
        "case_count": len(cases),
        "cases": {case["case_id"]: case for case in cases},
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    protocol_path = args.protocol if args.protocol.is_absolute() else repo_root / args.protocol
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    args.evaluation_mode = args.evaluation_mode or protocol.get(
        "evaluation_mode", "full_record_retrospective"
    )
    args.graph_source = args.graph_source or protocol.get(
        "graph_policy", {}
    ).get("primary_source", "per_case_pre_fault")
    if args.evaluation_mode == "full_record_retrospective":
        if args.length is not None or args.tdelta != 0:
            raise ValueError(
                "full-record evaluation forbids --length and nonzero --tdelta"
            )
    elif args.length is None:
        raise ValueError("window sensitivity requires --length")
    _validate_dataset_budget(protocol, args.limit)
    environment_validation = _validate_method_environments(
        repo_root, protocol_path, protocol
    )
    run_root = args.artifact_root / args.run_id
    if not run_root.is_absolute():
        run_root = repo_root / run_root
    requested_config = _requested_run_config(args, protocol)
    if run_root.exists() and not args.resume:
        raise FileExistsError(f"independent run directory already exists: {run_root}")
    if args.resume and not run_root.is_dir():
        raise FileNotFoundError(f"resume run directory is absent: {run_root}")
    formal = args.checksum_mode == "full" and not args.allow_dirty
    if formal and git_state(repo_root)["dirty"]:
        raise RuntimeError("formal runs require a clean frozen commit; use --allow-dirty only for diagnostics")
    if formal and not _formal_output_path_is_safe(repo_root, run_root):
        raise RuntimeError(
            "formal output inside the repository must be Git-ignored so the run "
            f"cannot dirty its own source snapshot: {run_root}"
        )
    command = " ".join(["./py310/bin/python", *sys.argv])
    config_path = run_root / "run-config.json"
    manifest_path = run_root / "manifest.json"
    if args.resume:
        if not config_path.is_file() or not manifest_path.is_file():
            raise RuntimeError("resume run is missing run-config.json or manifest.json")
        saved_config = json.loads(config_path.read_text(encoding="utf-8"))
        if saved_config != requested_config:
            raise RuntimeError("resume request does not match the frozen run configuration")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_resume_manifest(repo_root, manifest, protocol)
        _validate_manifest_output_path(manifest, run_root)
    else:
        if args.manifest_template:
            template_path = (
                args.manifest_template
                if args.manifest_template.is_absolute()
                else repo_root / args.manifest_template
            )
            manifest = _manifest_from_template(
                repo_root,
                template_path,
                protocol,
                args.checksum_mode,
                command,
            )
        else:
            manifest_protocol_path = protocol_path.relative_to(repo_root)
            manifest = create_manifest(
                root=repo_root,
                protocol_path=manifest_protocol_path,
                checksum_mode=args.checksum_mode,
                command=command,
            )
        manifest["environment_validation"] = environment_validation
        manifest["output_path"] = str(run_root)
        if len(environment_validation) == 1:
            validated_python = environment_validation[0]["python"]
            manifest["python"] = {
                "executable": validated_python["executable"],
                "version": validated_python["actual"],
                "implementation": validated_python["implementation"],
            }
        run_root.mkdir(parents=True)
        _write_json(manifest_path, manifest)
        _write_json(config_path, requested_config)
    runtime_cache = run_root / "runtime-cache"
    runtime_cache.mkdir(exist_ok=args.resume)
    exposure_root = run_root / "input-boundaries"
    exposure_root.mkdir(exist_ok=args.resume)
    exposure_paths = {}
    for dataset in args.datasets:
        exposure_path = exposure_root / f"{dataset}.json"
        expected_exposure = _build_exposure_manifest(
            repo_root,
            dataset,
            args.evaluation_mode,
            args.length,
            args.tdelta,
            args.limit,
        )
        if exposure_path.exists():
            existing_exposure = json.loads(
                exposure_path.read_text(encoding="utf-8")
            )
            if existing_exposure != expected_exposure:
                raise RuntimeError(
                    f"resume input-boundary manifest changed for {dataset}"
                )
        else:
            _write_json(exposure_path, expected_exposure)
        exposure_paths[dataset] = exposure_path

    task_records = []
    for label in args.methods:
        spec = resolve_method(label)
        for dataset in args.datasets:
            combination_root = run_root / label / dataset
            result_dir = combination_root / "results"
            expected_cases = min(
                args.limit
                if args.limit is not None
                else protocol["expected_cases_per_dataset"],
                protocol["expected_cases_per_dataset"],
            )
            if args.resume and combination_root.exists():
                completed = _completed_resume_record(
                    combination_root, expected_cases
                )
                if completed is not None:
                    task_records.append(completed)
                    _write_json(run_root / "tasks.json", task_records)
                    continue
                _archive_interrupted_combination(
                    run_root, combination_root, label, dataset
                )
            combination_root.mkdir(parents=True)
            infeasible = protocol.get("infeasible", {}).get(
                f"{spec.base_method}:{dataset}"
            )
            if infeasible:
                record = _infeasible_task_record(
                    label, spec.base_method, dataset, infeasible
                )
                _write_json(combination_root / "combination.json", record)
                _write_json(combination_root / "task.json", record)
                task_records.append(record)
                _write_json(run_root / "tasks.json", task_records)
                continue
            inapplicable = protocol.get("modality_inapplicable", {}).get(
                f"{label}:{dataset}"
            ) or protocol.get("modality_inapplicable", {}).get(
                f"{spec.base_method}:{dataset}"
            )
            if inapplicable:
                record = _inapplicable_task_record(
                    label, spec.base_method, dataset, inapplicable
                )
                _write_json(combination_root / "combination.json", record)
                _write_json(combination_root / "task.json", record)
                task_records.append(record)
                _write_json(run_root / "tasks.json", task_records)
                continue

            python_rel = protocol["method_environments"][spec.base_method]
            python = _resolve_python(repo_root, python_rel)
            if not python.exists():
                raise FileNotFoundError(
                    f"frozen environment for {spec.base_method} is absent: {python}"
                )
            cmd = [
                str(python),
                "script/run_paper_dataset.py",
                "--repo-root", str(repo_root),
                "--method", spec.base_method,
                "--dataset", dataset,
                "--result-dir", str(result_dir),
                "--sli", protocol["fixed_sli"][dataset],
                "--input-kind", protocol["method_inputs"][spec.base_method],
                "--evaluation-mode", args.evaluation_mode,
                "--tdelta", str(args.tdelta),
                "--exposure-manifest", str(exposure_paths[dataset]),
                "--graph-source", spec.environment.get(
                    "GABANZHAF_GRAPH_SOURCE", args.graph_source
                ),
                "--seed", str(spec.seed if spec.seed is not None else args.seed),
                "--per-case-timeout", str(protocol["per_case_timeout_seconds"]),
            ]
            if args.length is not None:
                cmd.extend(["--length", str(args.length)])
            if args.limit is not None:
                cmd.extend(["--limit", str(args.limit)])
            if spec.case_order_seed is not None:
                cmd.extend(["--case-order-seed", str(spec.case_order_seed)])
            if spec.warmup_method_call:
                cmd.append("--warmup-method-call")
            env = os.environ.copy()
            env.update({key: str(value) for key, value in protocol["threads"].items()})
            env["MPLCONFIGDIR"] = str(runtime_cache / "matplotlib")
            env["XDG_CACHE_HOME"] = str(runtime_cache)
            if protocol.get("strict_baseline_failures", False):
                env["RCAEVAL_STRICT_FAILURES"] = "1"
            env.update(spec.environment)
            if spec.base_method == "merit":
                env["GABANZHAF_WRITE_DIAGNOSTICS"] = "1"
            task_started = monotonic()
            try:
                process = subprocess.run(
                    cmd,
                    cwd=repo_root,
                    env=env,
                    text=True,
                    capture_output=True,
                    timeout=protocol["dataset_budget_seconds"],
                )
                status = "completed" if process.returncode == 0 else "failed"
                record = {
                    "method": label,
                    "base_method": spec.base_method,
                    "dataset": dataset,
                    "status": status,
                    "returncode": process.returncode,
                    "variant_environment": spec.environment,
                    "seed": spec.seed if spec.seed is not None else args.seed,
                    "case_order_seed": spec.case_order_seed,
                    "warmup_method_call": spec.warmup_method_call,
                    "strict_baseline_failures": protocol.get(
                        "strict_baseline_failures", False
                    ),
                    "observed_task_wall_seconds": monotonic() - task_started,
                    "command": cmd,
                }
                (combination_root / "stdout.log").write_text(
                    process.stdout, encoding="utf-8"
                )
                (combination_root / "stderr.log").write_text(
                    process.stderr, encoding="utf-8"
                )
            except subprocess.TimeoutExpired as exc:
                record = {
                    "method": label,
                    "base_method": spec.base_method,
                    "dataset": dataset,
                    "status": "dataset_budget_timeout",
                    "dataset_budget_seconds": protocol["dataset_budget_seconds"],
                    "observed_task_wall_seconds": monotonic() - task_started,
                    "command": cmd,
                }
                (combination_root / "stdout.log").write_text(
                    exc.stdout or "", encoding="utf-8"
                )
                (combination_root / "stderr.log").write_text(
                    exc.stderr or "", encoding="utf-8"
                )
            _write_json(combination_root / "task.json", record)
            task_records.append(record)
            _write_json(run_root / "tasks.json", task_records)

    _write_json(run_root / "tasks.json", task_records)
    return 0 if _all_tasks_succeeded(task_records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
