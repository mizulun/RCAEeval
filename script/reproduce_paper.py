"""Build and execute the artifact-traceable MERIT paper workflow.

The smoke profile exercises one case per method and dataset.  The retrospective
profile expands the frozen method matrix under the full-record v14 protocol;
window and delay jobs are explicitly separate sensitivity analyses.  The legacy
``confirmatory`` spelling remains a CLI alias for old automation but does not
change the estimand recorded by the protocol.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    from script.run_paper_experiments import (
        _resolve_python,
        _validate_method_environments,
        _validate_resume_manifest,
    )
except ModuleNotFoundError:  # direct script execution
    from run_paper_experiments import (
        _resolve_python,
        _validate_method_environments,
        _validate_resume_manifest,
    )


@dataclass(frozen=True)
class ExperimentJob:
    name: str
    methods: tuple[str, ...]
    evaluation_mode: str = "full_record_retrospective"
    length: int | None = None
    tdelta: int = 0


VARIANT_LABELS = {
    "no_trace": "merit_no_trace",
    "no_metric": "merit_no_metric",
    "no_trace_topology": "merit_no_trace_topology",
    "no_coverage": "merit_no_coverage",
    "no_victim_demotion": "merit_no_victim_demotion",
    "no_coverage_no_victim": "merit_no_coverage_no_victim",
    "no_coverage_parent_protection": "merit_no_coverage_parent_protection",
    "no_metric_type_dominance": "merit_no_metric_type_dominance",
    "no_early_full_fusion": "merit_no_early_full_fusion",
    "trace_latency_only": "merit_trace_latency_only",
    "trace_error_only": "merit_trace_error_only",
    "trace_lag0": "merit_trace_lag0",
    "trace_time_shuffle": "merit_trace_time_shuffle",
    "trace_random_edges": "merit_trace_random_edges",
    "trace_only": "merit_trace_only",
    "direct_service_evidence": "merit_direct_service_evidence",
    "direct_service_trace_evidence": "merit_direct_service_trace_evidence",
    "alternative_attribution": "merit_individual_attribution",
    "graph_per_case_full": "merit_graph_per_case_full",
    "graph_first_case_pre_fault": "merit_graph_first_case_pre_fault",
    "graph_union_pre_fault": "merit_graph_union_pre_fault",
    "graph_first_case_order1": "merit_graph_first_case_order1",
    "graph_first_case_order2": "merit_graph_first_case_order2",
    "graph_first_case_order3": "merit_graph_first_case_order3",
    "trace_dropout_10": "merit_trace_dropout_10",
    "trace_dropout_30": "merit_trace_dropout_30",
    "trace_dropout_50": "merit_trace_dropout_50",
    "trace_dropout_70": "merit_trace_dropout_70",
    "trace_dropout_100": "merit_trace_dropout_100",
    "trace_noise_10": "merit_trace_noise_10",
    "trace_noise_30": "merit_trace_noise_30",
    "trace_noise_50": "merit_trace_noise_50",
    "trace_edge_noise_10": "merit_trace_edge_noise_10",
    "trace_edge_noise_30": "merit_trace_edge_noise_30",
    "trace_edge_noise_50": "merit_trace_edge_noise_50",
    "bruteforce": "merit_bruteforce",
}


def _parameter_label(parameter: str, value: float | int) -> str:
    if parameter == "exact_candidate_k":
        return f"merit_k{int(value)}"
    prefix = {
        "victim_floor": "victim",
        "trace_topology_conf": "tconf",
        "coverage_conf": "cconf",
    }[parameter]
    return f"merit_{prefix}_{round(float(value) * 10):02d}"


def _baseline_labels(protocol: dict, seeded: bool) -> list[str]:
    labels = []
    stochastic = set(protocol["stochastic_methods"])
    for method in protocol["baselines"]:
        if seeded and method in stochastic:
            labels.extend(f"{method}_seed{seed}" for seed in protocol["seeds"])
        else:
            labels.append(method)
    return labels


def _ordered_baseline_labels(protocol: dict) -> tuple[list[str], list[str]]:
    """Return quick and expensive labels without changing the frozen method set."""

    configured = set(protocol["baselines"])
    preferred = (
        "baro",
        "tracerca",
        "circa",
        "microrank",
        "rcd",
        "pc_pagerank",
        "causalrca",
    )
    ordered = [method for method in preferred if method in configured]
    ordered.extend(method for method in protocol["baselines"] if method not in ordered)
    stochastic = set(protocol["stochastic_methods"])
    labels = []
    for method in ordered:
        if method in stochastic:
            labels.extend(f"{method}_seed{seed}" for seed in protocol["seeds"])
        else:
            labels.append(method)
    expensive = {
        label
        for label in labels
        if label.startswith(("rcd_seed", "causalrca_seed"))
        or label == "pc_pagerank"
    }
    return (
        [label for label in labels if label not in expensive],
        [label for label in labels if label in expensive],
    )


def build_jobs(protocol: dict, profile: str) -> list[ExperimentJob]:
    if profile == "smoke":
        return [
            ExperimentJob(
                name="main",
                methods=tuple(["merit_full", *_baseline_labels(protocol, False)]),
            )
        ]
    if profile not in {"retrospective", "confirmatory"}:
        raise ValueError(f"unsupported profile: {profile}")

    quick_baselines, expensive_baselines = _ordered_baseline_labels(protocol)
    methods = ["merit_full", *quick_baselines]
    methods.extend(
        VARIANT_LABELS[name]
        for name in protocol["controlled_variants"]
        if name in VARIANT_LABELS
    )
    methods.extend(
        f"merit_full_rep{repetition}"
        for repetition in range(2, protocol["runtime_repetitions"] + 1)
    )
    methods.extend(
        f"merit_warm_rep{repetition}"
        for repetition in range(1, protocol["runtime_repetitions"] + 1)
    )
    for parameter in (
        "victim_floor",
        "trace_topology_conf",
        "coverage_conf",
        "exact_candidate_k",
    ):
        methods.extend(
            _parameter_label(parameter, value)
            for value in protocol["sweeps"][parameter]
        )
    methods.extend(expensive_baselines)

    # A duplicated label would overwrite results inside a run directory.
    if len(methods) != len(set(methods)):
        raise ValueError("retrospective method labels are not unique")

    jobs = [ExperimentJob(name="main", methods=tuple(methods))]
    jobs.extend(
        ExperimentJob(
            name=f"delay-{tdelta}",
            methods=("merit_full",),
            evaluation_mode="window_sensitivity",
            length=20,
            tdelta=tdelta,
        )
        for tdelta in protocol["sweeps"]["tdelta_seconds"]
        if tdelta != 0
    )
    jobs.extend(
        ExperimentJob(
            name=f"window-{length}",
            methods=("merit_full",),
            evaluation_mode="window_sensitivity",
            length=length,
        )
        for length in protocol["sweeps"]["observation_window_total_minutes"]
    )
    return jobs


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--profile",
        choices=("smoke", "retrospective", "confirmatory"),
        default="smoke",
    )
    parser.add_argument(
        "--protocol", type=Path, default=Path("experiment/protocol.json")
    )
    parser.add_argument(
        "--artifact-root", type=Path, default=Path("artifacts/reproductions")
    )
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume an identical reproduction, preserving and auditing prior attempts",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("experiments", "evaluation", "crosscheck", "selection", "banzhaf", "tables"),
        default=None,
    )
    return parser.parse_args(argv)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _execute(command: list[str], repo_root: Path, log_root: Path) -> dict:
    process = subprocess.run(
        command,
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    log_root.mkdir(parents=True, exist_ok=True)
    (log_root / "stdout.log").write_text(process.stdout, encoding="utf-8")
    (log_root / "stderr.log").write_text(process.stderr, encoding="utf-8")
    record = {"command": command, "returncode": process.returncode}
    _write_json(log_root / "command.json", record)
    if process.returncode != 0:
        raise RuntimeError(
            f"reproduction command failed ({process.returncode}); see {log_root}"
        )
    return record


def _validate_post_experiment_state(
    repo_root: Path,
    protocol_path: Path,
    protocol: dict,
    manifest_path: Path,
) -> None:
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"main formal manifest is absent before analysis: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("formal_manifest"):
        raise RuntimeError("post-experiment stages require a full formal manifest")
    _validate_resume_manifest(repo_root, manifest, protocol)
    reports = _validate_method_environments(repo_root, protocol_path, protocol)
    if not reports or not all(report.get("ok") for report in reports):
        raise RuntimeError("post-experiment environment validation failed")


def main(argv=None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    protocol_path = (
        args.protocol if args.protocol.is_absolute() else repo_root / args.protocol
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    jobs = build_jobs(protocol, args.profile)
    serialized_jobs = json.loads(json.dumps([asdict(job) for job in jobs]))
    stages = list(args.stages) if args.stages else [
        "experiments",
        "evaluation",
        "banzhaf",
        *([] if args.profile == "smoke" else ["crosscheck", "selection", "tables"]),
    ]
    artifact_root = (
        args.artifact_root
        if args.artifact_root.is_absolute()
        else repo_root / args.artifact_root
    )
    reproduction_root = artifact_root / args.run_id
    if "experiments" in stages:
        if reproduction_root.exists() and not args.resume:
            raise FileExistsError(
                f"independent reproduction directory already exists: {reproduction_root}"
            )
        if args.resume and not reproduction_root.is_dir():
            raise FileNotFoundError(reproduction_root)
        reproduction_root.mkdir(parents=True, exist_ok=args.resume)
    elif not reproduction_root.is_dir():
        raise FileNotFoundError(reproduction_root)

    plan_path = reproduction_root / "reproduction-plan.json"
    if plan_path.exists():
        record = json.loads(plan_path.read_text(encoding="utf-8"))
        if (
            record.get("protocol_version") != protocol["protocol_version"]
            or record.get("profile") != args.profile
            or record.get("jobs") != serialized_jobs
        ):
            raise RuntimeError("existing reproduction plan does not match this request")
        record["stages"] = sorted(set(record.get("stages", ())) | set(stages))
    else:
        if args.resume:
            raise RuntimeError("resume reproduction is missing reproduction-plan.json")
        record = {
            "protocol_version": protocol["protocol_version"],
            "profile": args.profile,
            "stages": stages,
            "jobs": serialized_jobs,
            "commands": [],
        }
    _write_json(plan_path, record)

    python = _resolve_python(repo_root, "./py310/bin/python")
    if not python.exists():
        raise FileNotFoundError(f"frozen Python 3.10 environment is absent: {python}")

    if "experiments" in stages:
        for job_index, job in enumerate(jobs):
            command = [
                str(python),
                "script/run_paper_experiments.py",
                "--run-id",
                job.name,
                "--methods",
                *job.methods,
                "--artifact-root",
                str(reproduction_root / "runs"),
                "--protocol",
                str(protocol_path),
                "--checksum-mode",
                "metadata" if args.profile == "smoke" else "full",
                "--evaluation-mode",
                job.evaluation_mode,
                "--graph-source",
                protocol.get("graph_policy", {}).get(
                    "primary_source", "per_case_pre_fault"
                ),
            ]
            if job.length is not None:
                command.extend(("--length", str(job.length)))
            if job.tdelta != 0:
                command.extend(("--tdelta", str(job.tdelta)))
            if args.profile == "smoke":
                command.extend(("--limit", "1"))
            if args.allow_dirty:
                command.append("--allow-dirty")
            if job_index > 0:
                manifest_template = reproduction_root / "runs" / "main" / "manifest.json"
                if not manifest_template.is_file():
                    raise FileNotFoundError(
                        f"shared full manifest is absent: {manifest_template}"
                    )
                command.extend(("--manifest-template", str(manifest_template)))
            if args.resume and (reproduction_root / "runs" / job.name).is_dir():
                command.append("--resume")
            result = _execute(
                command,
                repo_root,
                reproduction_root / "logs" / "experiments" / job.name,
            )
            record["commands"].append(result)
            _write_json(plan_path, record)

    post_experiment_stages = set(stages) - {"experiments"}
    if post_experiment_stages:
        _validate_post_experiment_state(
            repo_root,
            protocol_path,
            protocol,
            reproduction_root / "runs" / "main" / "manifest.json",
        )

    if "evaluation" in stages:
        for job in jobs:
            command = [
                str(python),
                "script/evaluate_paper_results.py",
                "--repo-root",
                str(repo_root),
                "--runs-root",
                str(reproduction_root / "runs" / job.name),
                "--methods",
                *job.methods,
                "--reference-method",
                "merit_full",
                "--protocol",
                str(protocol_path),
                "--output-dir",
                str(reproduction_root / "analysis" / job.name),
            ]
            if args.profile == "smoke":
                command.append("--allow-incomplete")
            result = _execute(
                command,
                repo_root,
                reproduction_root / "logs" / "evaluation" / job.name,
            )
            record["commands"].append(result)
            _write_json(plan_path, record)

    if "crosscheck" in stages:
        command = [
            str(python),
            "script/crosscheck_baro_reference.py",
            "--dataset-summaries",
            str(reproduction_root / "analysis" / "main" / "dataset_summaries.json"),
            "--protocol",
            str(protocol_path),
            "--output",
            str(reproduction_root / "analysis" / "crosschecks" / "baro.json"),
        ]
        result = _execute(
            command,
            repo_root,
            reproduction_root / "logs" / "crosscheck" / "baro",
        )
        record["commands"].append(result)
        _write_json(plan_path, record)

    if "selection" in stages:
        summaries = reproduction_root / "analysis" / "main" / "dataset_summaries.json"
        for parameter in (
            "victim_floor",
            "trace_topology_conf",
            "coverage_conf",
            "exact_candidate_k",
        ):
            command = [
                str(python),
                "script/select_global_sweep.py",
                "--dataset-summaries",
                str(summaries),
                "--parameter",
                parameter,
                "--protocol",
                str(protocol_path),
                "--output-dir",
                str(reproduction_root / "analysis" / "selections" / parameter),
            ]
            result = _execute(
                command,
                repo_root,
                reproduction_root / "logs" / "selection" / parameter,
            )
            record["commands"].append(result)
            _write_json(plan_path, record)

    if "banzhaf" in stages:
        command = [
            str(python),
            "script/benchmark_banzhaf_equivalence.py",
            "--protocol",
            str(protocol_path),
            "--output-dir",
            str(reproduction_root / "analysis" / "banzhaf-equivalence"),
        ]
        case_results = reproduction_root / "analysis" / "main" / "case_results.json"
        if case_results.is_file():
            command.extend(("--case-results", str(case_results)))
        result = _execute(
            command,
            repo_root,
            reproduction_root / "logs" / "banzhaf-equivalence",
        )
        record["commands"].append(result)
        _write_json(plan_path, record)

    if "tables" in stages:
        command = [
            str(python),
            "script/generate_paper_tables.py",
            "--reproduction-root",
            str(reproduction_root),
            "--protocol",
            str(protocol_path),
            "--output",
            str(reproduction_root / "generated" / "results-summary.tex"),
        ]
        result = _execute(
            command,
            repo_root,
            reproduction_root / "logs" / "tables",
        )
        record["commands"].append(result)
        _write_json(plan_path, record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
