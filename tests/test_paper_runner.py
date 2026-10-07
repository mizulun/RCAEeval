import multiprocessing
import time
import json
from pathlib import Path

import pandas as pd
import pytest

from script.run_paper_dataset import (
    _stop_worker,
    _wait_for_worker_case,
    case_artifact_name,
    prepare_case,
    prepare_trace_case,
)
from script.run_paper_experiments import (
    _all_tasks_succeeded,
    _archive_interrupted_combination,
    _completed_resume_record,
    _formal_output_path_is_safe,
    _infeasible_task_record,
    _validate_dataset_budget,
    _validate_method_environments,
    _validate_manifest_output_path,
    resolve_method,
)


def _sleeping_worker(input_queue, _output_queue):
    input_queue.get()
    time.sleep(5)


def test_prepare_case_uses_window_and_renames_latency_without_selecting_sli(tmp_path):
    case_dir = tmp_path / "answer_cpu" / "1"
    case_dir.mkdir(parents=True)
    path = case_dir / "simple_metrics.csv"
    pd.DataFrame({
        "time": list(range(8)),
        "frontend_latency-50": list(range(8)),
        "frontend_latency-90": list(range(8)),
        "answer_latency-90": list(range(8)),
    }).to_csv(path, index=False)
    (case_dir / "inject_time.txt").write_text("4\n", encoding="utf-8")

    data, inject_time = prepare_case(path, length=1, tdelta=1)

    assert inject_time == 5
    assert "frontend_latency-50" not in data
    assert "frontend_latency" in data
    assert "answer_latency" in data


def test_case_artifact_name_is_evaluation_only_label(tmp_path):
    path = tmp_path / "service_delay" / "3" / "simple_metrics.csv"
    assert case_artifact_name(path) == "service_delay_3.json"


def test_prepare_trace_case_uses_same_total_window_and_microseconds(tmp_path):
    case_dir = tmp_path / "answer_cpu" / "1"
    case_dir.mkdir(parents=True)
    metric_path = case_dir / "simple_metrics.csv"
    pd.DataFrame({"time": [100]}).to_csv(metric_path, index=False)
    pd.DataFrame(
        {
            "traceID": ["before", "left", "right", "after"],
            "spanID": ["1", "2", "3", "4"],
            "serviceName": ["svc"] * 4,
            "methodName": ["op"] * 4,
            "operationName": ["op"] * 4,
            "startTime": [69, 70, 129, 130],
            "duration": [0, 0, 0, 0],
            "parentSpanID": [""] * 4,
        }
    ).assign(startTime=lambda frame: frame.startTime * 1_000_000).to_csv(
        case_dir / "traces.csv", index=False
    )

    spans, inject_us = prepare_trace_case(metric_path, inject_time=100, length=1)

    assert inject_us == 100_000_000
    assert spans.traceID.tolist() == ["left", "right"]


def test_prepare_trace_case_reports_missing_trace_input(tmp_path):
    metric_path = tmp_path / "service_cpu" / "1" / "simple_metrics.csv"
    metric_path.parent.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="trace input is absent"):
        prepare_trace_case(metric_path, inject_time=100, length=20)


@pytest.mark.parametrize(
    "label,variable,value",
    [
        ("merit_victim_01", "GABANZHAF_VICTIM_FLOOR", "0.1"),
        ("merit_tconf_09", "GABANZHAF_TRACE_TOPOLOGY_CONF", "0.9"),
        ("merit_cconf_05", "GABANZHAF_COVERAGE_CONF", "0.5"),
        ("merit_k15", "GABANZHAF_EXACT_CANDIDATE_K", "15"),
    ],
)
def test_resolve_method_exposes_complete_preregistered_sweeps(label, variable, value):
    spec = resolve_method(label)

    assert spec.base_method == "merit"
    assert spec.environment == {variable: value}


def test_alternative_attribution_variant_changes_only_attribution_mode():
    spec = resolve_method("merit_individual_attribution")

    assert spec.base_method == "merit"
    assert spec.environment == {"GABANZHAF_BANZHAF_MODE": "individual"}


def test_no_metric_variant_removes_metric_values_without_changing_rank_mode():
    spec = resolve_method("merit_no_metric")

    assert spec.base_method == "merit"
    assert spec.environment == {"GABANZHAF_DISABLE_METRICS": "1"}


def test_direct_metric_trace_sum_variant_bypasses_banzhaf():
    spec = resolve_method("merit_direct_service_trace_evidence")

    assert spec.base_method == "merit"
    assert spec.environment == {
        "GABANZHAF_RANK_MODE": "service_evidence_trace"
    }


@pytest.mark.parametrize(
    "label,variable,value",
    [
        ("merit_trace_dropout_10", "GABANZHAF_TRACE_DROPOUT", "0.1"),
        ("merit_trace_dropout_100", "GABANZHAF_TRACE_DROPOUT", "1.0"),
        ("merit_trace_noise_30", "GABANZHAF_TRACE_NOISE", "0.3"),
        ("merit_trace_edge_noise_50", "GABANZHAF_TRACE_EDGE_NOISE", "0.5"),
    ],
)
def test_trace_robustness_variants_change_one_frozen_variable(
    label, variable, value
):
    spec = resolve_method(label)

    assert spec.base_method == "merit"
    assert spec.environment == {variable: value}


@pytest.mark.parametrize("method", ["rcd", "causalrca"])
def test_seeded_stochastic_method_keeps_distinct_label_and_base_method(method):
    spec = resolve_method(f"{method}_seed20260714")

    assert spec.label == f"{method}_seed20260714"
    assert spec.base_method == method
    assert spec.environment == {}
    assert spec.seed == 20260714


def test_merit_runtime_repetition_has_an_independent_artifact_label():
    spec = resolve_method("merit_full_rep5")

    assert spec.label == "merit_full_rep5"
    assert spec.base_method == "merit"
    assert spec.environment == {}


def test_declared_na_keeps_internal_infeasible_status_for_exit_gate():
    record = _infeasible_task_record(
        "circa",
        "circa",
        "re2-tt",
        {
            "status": "N/A",
            "basis": "measured probe and projected full-run cost",
            "projection_only": True,
        },
    )

    assert record["status"] == "infeasible"
    assert record["declared_status"] == "N/A"
    assert _all_tasks_succeeded([{"status": "completed"}, record])


def test_dataset_budget_cannot_truncate_required_case_artifacts():
    protocol_path = Path(__file__).resolve().parents[1] / "experiment/protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    _validate_dataset_budget(protocol)

    protocol["dataset_budget_seconds"] = (
        protocol["expected_cases_per_dataset"]
        * protocol["per_case_timeout_seconds"]
        - 1
    )
    with pytest.raises(ValueError, match="truncate required case artifacts"):
        _validate_dataset_budget(protocol)


def test_frozen_python310_environment_matches_protocol_and_lock():
    root = Path(__file__).resolve().parents[1]
    if not (root / "py310/bin/python").exists():
        pytest.skip("source artifact does not bundle a Python environment")
    protocol_path = root / "experiment/protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

    reports = _validate_method_environments(root, protocol_path, protocol)

    assert len(reports) == 1
    assert reports[0]["ok"] is True
    assert reports[0]["python"]["actual"] == "3.10.18"
    assert reports[0]["configured_environment"] == "./py310/bin/python"


def test_resume_accepts_only_a_complete_case_metadata_set(tmp_path):
    combination = tmp_path / "run" / "rcd" / "re2-ob"
    metadata = combination / "results" / "case_metadata"
    metadata.mkdir(parents=True)
    (combination / "task.json").write_text(
        json.dumps({"status": "completed", "method": "rcd"}),
        encoding="utf-8",
    )
    (combination / "combination.json").write_text(
        json.dumps({"cases_attempted": 2}), encoding="utf-8"
    )
    (metadata / "case-1.json").write_text("{}", encoding="utf-8")

    assert _completed_resume_record(combination, expected_cases=2) is None

    (metadata / "case-2.json").write_text("{}", encoding="utf-8")
    assert _completed_resume_record(combination, expected_cases=2)["status"] == "completed"


def test_resume_archives_partial_combination_without_deleting_it(tmp_path):
    run_root = tmp_path / "run"
    combination = run_root / "rcd" / "re2-ob"
    combination.mkdir(parents=True)
    (combination / "partial.txt").write_text("kept", encoding="utf-8")

    archived = _archive_interrupted_combination(
        run_root, combination, "rcd", "re2-ob"
    )

    assert not combination.exists()
    assert (archived / "partial.txt").read_text(encoding="utf-8") == "kept"


def test_resume_manifest_is_bound_to_its_exact_output_path(tmp_path):
    run_root = tmp_path / "runs" / "frozen"

    _validate_manifest_output_path({"output_path": str(run_root)}, run_root)

    with pytest.raises(RuntimeError, match="output path mismatch"):
        _validate_manifest_output_path(
            {"output_path": str(tmp_path / "other")},
            run_root,
        )


def test_runner_cli_accepts_a_shared_manifest_template():
    from script.run_paper_experiments import parse_args

    args = parse_args(
        [
            "--run-id",
            "window-5",
            "--methods",
            "merit_full",
            "--manifest-template",
            "main/manifest.json",
        ]
    )

    assert args.manifest_template == Path("main/manifest.json")


def test_default_formal_artifact_root_is_git_ignored():
    root = Path(__file__).resolve().parents[1]
    if not (root / ".git").exists():
        pytest.skip("anonymous source artifact intentionally excludes VCS metadata")

    assert _formal_output_path_is_safe(
        root, root / "artifacts" / "reproductions" / "frozen"
    )


def test_parent_enforces_hard_case_deadline(tmp_path):
    context = multiprocessing.get_context("fork")
    input_queue = context.Queue(maxsize=1)
    output_queue = context.Queue(maxsize=1)
    worker = context.Process(
        target=_sleeping_worker, args=(input_queue, output_queue)
    )
    worker.start()
    try:
        result, _peak, elapsed = _wait_for_worker_case(
            worker,
            input_queue,
            output_queue,
            tmp_path / "case.csv",
            timeout_seconds=0.1,
        )
        assert result is None
        assert 0.09 <= elapsed < 1.0
    finally:
        _stop_worker(worker)
    assert not worker.is_alive()
