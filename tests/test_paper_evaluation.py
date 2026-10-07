import json

import pytest

from script.evaluate_paper_results import (
    build_diagnostic_checks,
    build_paired_statistics,
    comparison_family,
)

from RCAEval.benchmark.paper_evaluation import (
    CaseOutcome,
    ExpectedCase,
    evaluate_method_dataset,
    exact_mcnemar,
    holm_adjust,
    paired_bootstrap,
)


def _expected(index, root="svc"):
    return ExpectedCase(
        dataset="re2-test",
        case_id=f"{root}_cpu_{index}",
        root_service=root,
        fault_type="cpu",
        repetition=str(index),
    )


def _write_result(result_dir, expected, payload):
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / f"{expected.case_id}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_failure_accounting_keeps_error_timeout_and_missing_in_denominator(tmp_path):
    result_dir = tmp_path / "results"
    expected = [_expected(i) for i in range(1, 5)]
    _write_result(result_dir, expected[0], {"0": ["svc_cpu"]})
    _write_result(result_dir, expected[1], {"0": ["other_cpu"]})
    _write_result(result_dir, expected[2], {"status": "timeout", "error": "900s"})
    # expected[3] is deliberately missing

    outcomes, summary = evaluate_method_dataset(
        "method", "re2-test", expected, result_dir
    )

    assert [outcome.status for outcome in outcomes] == [
        "success", "success", "timeout", "missing"
    ]
    assert summary["n_expected"] == 4
    assert summary["n_success"] == 2
    assert summary["n_timeout"] == 1
    assert summary["n_missing"] == 1
    assert summary["execution_success_rate"] == pytest.approx(0.5)
    assert summary["overall"]["AC@1"] == pytest.approx(0.25)
    assert summary["successful_cases_only"]["AC@1"] == pytest.approx(0.5)
    assert summary["combination_status"] == "incomplete"


def test_candidate_recall_is_computed_offline_from_answer_and_diagnostics(tmp_path):
    result_dir = tmp_path / "results"
    expected = _expected(1, root="root")
    _write_result(result_dir, expected, {"0": ["root_cpu"]})
    diagnostics = result_dir / "diagnostics"
    diagnostics.mkdir()
    (diagnostics / f"{expected.case_id}.json").write_text(
        json.dumps({"game_candidates": ["other", "root"]}), encoding="utf-8"
    )

    outcomes, summary = evaluate_method_dataset(
        "method", "re2-test", [expected], result_dir
    )

    assert outcomes[0].candidate_recall is True
    assert summary["candidate_recall"] == 1.0


def test_case_metadata_is_joined_and_efficiency_is_summarized(tmp_path):
    result_dir = tmp_path / "results"
    expected = [_expected(1), _expected(2)]
    for item in expected:
        _write_result(result_dir, item, {"0": ["svc_cpu"]})
    metadata_dir = result_dir / "case_metadata"
    metadata_dir.mkdir()
    for item, io_time, method_time, observed, peak in [
        (expected[0], 1.0, 3.0, 4.5, 100),
        (expected[1], 2.0, 4.0, 6.5, 200),
    ]:
        (metadata_dir / f"{item.case_id}.json").write_text(
            json.dumps(
                {
                    "io_seconds": io_time,
                    "method_seconds": method_time,
                    "observed_case_seconds": observed,
                    "peak_rss_bytes": peak,
                    "input_modality": "metrics",
                    "input_rows": 1200,
                }
            ),
            encoding="utf-8",
        )
    (result_dir.parent / "task.json").write_text(
        json.dumps({"observed_task_wall_seconds": 12.5}),
        encoding="utf-8",
    )

    outcomes, summary = evaluate_method_dataset(
        "method", "re2-test", expected, result_dir
    )

    assert outcomes[0].input_modality == "metrics"
    assert outcomes[0].input_rows == 1200
    assert summary["timing_seconds"]["io"]["median"] == pytest.approx(1.5)
    assert summary["timing_seconds"]["method"]["p95"] == pytest.approx(3.95)
    assert outcomes[0].observed_case_seconds == pytest.approx(4.5)
    assert summary["timing_seconds"]["end_to_end"]["median"] == pytest.approx(5.5)
    assert summary["timing_seconds"]["first_case_end_to_end"] == pytest.approx(4.5)
    assert summary["timing_seconds"]["remaining_cases_end_to_end"]["median"] == pytest.approx(6.5)
    assert summary["dataset_observed_wall_seconds"] == pytest.approx(11.0)
    assert summary["task_observed_wall_seconds"] == pytest.approx(12.5)
    assert summary["peak_rss_bytes"] == 200


def test_timeout_observed_duration_is_reported_separately(tmp_path):
    result_dir = tmp_path / "results"
    expected = _expected(1)
    _write_result(
        result_dir,
        expected,
        {"status": "timeout", "error": "per-case hard timeout: 900s"},
    )
    metadata_dir = result_dir / "case_metadata"
    metadata_dir.mkdir()
    (metadata_dir / f"{expected.case_id}.json").write_text(
        json.dumps(
            {
                "status": "timeout",
                "observed_case_seconds": 900.01,
                "peak_rss_bytes": 100,
            }
        ),
        encoding="utf-8",
    )

    _, summary = evaluate_method_dataset(
        "method", "re2-test", [expected], result_dir
    )

    assert summary["timing_seconds"]["timeouts_observed"]["n"] == 1
    assert summary["timing_seconds"]["timeouts_observed"]["median"] == pytest.approx(
        900.01
    )


def test_merit_stage_timings_are_summarized_by_stage(tmp_path):
    result_dir = tmp_path / "results"
    expected = [_expected(1), _expected(2)]
    for item, graph, banzhaf in [
        (expected[0], 2.0, 0.1),
        (expected[1], 0.0, 0.3),
    ]:
        _write_result(result_dir, item, {"0": ["svc_cpu"]})
        diagnostics = result_dir / "diagnostics"
        diagnostics.mkdir(exist_ok=True)
        (diagnostics / f"{item.case_id}.json").write_text(
            json.dumps(
                {"stage_timing": {"call_graph": graph, "banzhaf": banzhaf}}
            ),
            encoding="utf-8",
        )

    _, summary = evaluate_method_dataset(
        "merit_full", "re2-test", expected, result_dir
    )

    assert summary["stage_timing_seconds"]["call_graph"]["median"] == 1.0
    assert summary["stage_timing_seconds"]["banzhaf"]["p95"] == pytest.approx(0.29)


def _outcome(case_id, hit):
    root = "root"
    ranks = ["root_cpu"] if hit else ["other_cpu"]
    return CaseOutcome(
        method="m",
        dataset="d",
        case_id=case_id,
        root_service=root,
        fault_type="cpu",
        repetition=case_id,
        status="success",
        ranks=ranks,
        ranked_services=["root" if hit else "other"],
        error=None,
        candidate_recall=None,
        stage_timing={},
    )


def test_exact_mcnemar_reports_discordant_pairs_and_effect():
    a = [_outcome("1", False), _outcome("2", False), _outcome("3", True)]
    b = [_outcome("1", True), _outcome("2", True), _outcome("3", False)]

    result = exact_mcnemar(a, b)

    assert result["a_only"] == 1
    assert result["b_only"] == 2
    assert result["p_value"] == pytest.approx(1.0)
    assert result["effect_pp_b_minus_a"] == pytest.approx(100 / 3)


def test_paired_bootstrap_is_deterministic_and_reports_percentage_points():
    a = [_outcome(str(i), False) for i in range(4)]
    b = [_outcome(str(i), i < 2) for i in range(4)]

    first = paired_bootstrap(a, b, "AC@1", resamples=1000, seed=7)
    second = paired_bootstrap(a, b, "AC@1", resamples=1000, seed=7)

    assert first == second
    assert first["effect_pp_b_minus_a"] == pytest.approx(50.0)
    assert first["ci95_pp"][0] <= 50.0 <= first["ci95_pp"][1]


def test_holm_adjust_is_monotone_in_sorted_p_values():
    adjusted = holm_adjust([0.04, 0.01, 0.03])

    assert adjusted == pytest.approx([0.06, 0.03, 0.06])


def test_comparison_families_exclude_repetitions_and_sweeps():
    protocol = {"baselines": ["baro", "rcd"]}

    assert comparison_family("baro", protocol) == "rq1_baselines"
    assert comparison_family("rcd_seed7", protocol) == "rq1_baselines"
    assert comparison_family("merit_no_trace", protocol) == "rq2_ablations"
    assert comparison_family("merit_full_rep2", protocol) is None
    assert comparison_family("merit_victim_01", protocol) is None


def test_paired_statistics_add_compatible_pool_and_separate_holm_families():
    protocol = {
        "baselines": ["baro"],
        "bootstrap_resamples": 100,
        "bootstrap_seed": 7,
    }
    all_outcomes = {
        ("merit_full", "ob"): [_outcome("ob-1", True)],
        ("merit_full", "ss"): [_outcome("ss-1", True)],
        ("baro", "ob"): [_outcome("ob-1", False)],
        # baro x ss is absent, modeling a declared N/A combination.
        ("merit_no_trace", "ob"): [_outcome("ob-1", True)],
        ("merit_no_trace", "ss"): [_outcome("ss-1", False)],
    }

    records = build_paired_statistics(
        all_outcomes,
        ["merit_full", "baro", "merit_no_trace", "merit_full_rep2"],
        ["ob", "ss"],
        "merit_full",
        protocol,
    )

    keys = {(row["method"], row["scope"]) for row in records}
    assert keys == {
        ("baro", "ob"),
        ("baro", "pooled"),
        ("merit_no_trace", "ob"),
        ("merit_no_trace", "ss"),
        ("merit_no_trace", "pooled"),
    }
    baro_pool = next(
        row for row in records if row["method"] == "baro" and row["scope"] == "pooled"
    )
    assert baro_pool["datasets"] == ["ob"]
    assert baro_pool["mcnemar_ac1"]["n"] == 1
    assert all(
        "holm_adjusted_p_value" in row["mcnemar_ac1"] for row in records
    )


def test_missing_trace_diagnostic_compares_exact_metric_and_service_ranks():
    full = [_outcome("1", True), _outcome("2", True)]
    no_trace = [_outcome("1", True), _outcome("2", True)]
    no_trace[1].ranks = ["root_mem"]
    no_trace[1].ranked_services = ["root"]

    check = build_diagnostic_checks(
        {
            ("merit_full", "re2-ss"): full,
            ("merit_no_trace", "re2-ss"): no_trace,
        }
    )["missing_trace"]

    assert check["n"] == 2
    assert check["execution_status_matches"] == 2
    assert check["exact_metric_rank_matches"] == 1
    assert check["exact_service_rank_matches"] == 2
    assert check["all_metric_ranks_equal"] is False
    assert check["all_service_ranks_equal"] is True
    assert check["metric_rank_mismatch_case_ids"] == ["2"]
