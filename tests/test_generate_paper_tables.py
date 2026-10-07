import json

import pytest

from script.generate_paper_tables import (
    _equivalence_table,
    _paired_statistics_tables,
    _stochastic_seed_variation_table,
    _validate_diagnostic_checks,
    _validate_job,
    _validate_paired_statistics,
    render_fault_breakdown,
    render_extended_results,
    render_headline_macros,
    render_statistical_details,
    render_tables,
)
from script.reproduce_paper import ExperimentJob


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _minimal_job_tree(tmp_path, dirty=False):
    job = ExperimentJob("main", ("baro",))
    protocol = {
        "protocol_version": "test-v1",
        "datasets": ["d"],
        "expected_cases_per_dataset": 1,
        "infeasible": {},
    }
    _write(
        tmp_path / "runs/main/manifest.json",
        {
            "formal_manifest": True,
            "protocol_version": "test-v1",
            "git": {"dirty": dirty},
            "output_path": str(tmp_path / "runs/main"),
            "datasets": {
                "d": {
                    "case_count": 1,
                    "expected_case_count": 1,
                    "checksum_mode": "full",
                    "cryptographic_content_checksum": True,
                    "tree_sha256": "abc",
                }
            },
        },
    )
    _write(
        tmp_path / "analysis/main/dataset_summaries.json",
        [
            {
                "method": "baro",
                "dataset": "d",
                "combination_status": "completed",
                "n_expected": 1,
                "n_missing": 0,
            }
        ],
    )
    _write(tmp_path / "analysis/main/aggregate_summaries.json", [])
    return job, protocol


def test_table_gate_accepts_only_complete_clean_full_job(tmp_path):
    job, protocol = _minimal_job_tree(tmp_path)

    summaries, aggregates = _validate_job(tmp_path, job, protocol)

    assert len(summaries) == 1
    assert aggregates == []


def test_table_gate_rejects_dirty_manifest(tmp_path):
    job, protocol = _minimal_job_tree(tmp_path, dirty=True)

    with pytest.raises(RuntimeError, match="clean full manifest"):
        _validate_job(tmp_path, job, protocol)


def test_equivalence_table_contains_measured_error_and_speedup():
    latex = _equivalence_table(
        {
            "summaries": [
                {
                    "k": 3,
                    "max_absolute_error": 1e-12,
                    "modes": {
                        "closed_form": {"seconds": {"median": 0.001}},
                        "bruteforce": {"seconds": {"median": 0.1}},
                    },
                    "median_speedup_bruteforce_over_closed_form": 100.0,
                }
            ]
        }
    )

    assert "1.00e-12" in latex
    assert "100.0$\\times$" in latex


def test_headline_macros_use_only_complete_three_system_baselines():
    validated = {
        "jobs": {
            "main": {
                "aggregates": [
                    {
                        "method": "merit_full",
                        "all_datasets_completed": True,
                        "macro_equal_system_weight": {"AC@1": 0.6},
                    },
                    {
                        "method": "baro",
                        "all_datasets_completed": True,
                        "macro_equal_system_weight": {"AC@1": 0.5},
                    },
                    {
                        "method": "tracerca",
                        "all_datasets_completed": False,
                        "macro_equal_system_weight": None,
                    },
                ]
            }
        }
    }
    protocol = {
        "baselines": ["baro", "tracerca"],
        "stochastic_methods": [],
        "seeds": [],
    }

    latex = render_headline_macros(validated, protocol)

    assert r"\MeritMacroACOne}{60.0\%}" in latex
    assert r"\MeritBestCompleteBaseline}{BARO}" in latex
    assert r"\MeritHeadlineDelta}{+10.0}" in latex


def _paired_record(method="baro", scope="ob", datasets=None, family="rq1_baselines"):
    datasets = datasets or ["ob"]
    return {
        "comparison_family": family,
        "reference": "merit_full",
        "method": method,
        "dataset": scope,
        "scope": scope,
        "datasets": datasets,
        "case_ids": ["case-1"],
        "mcnemar_ac1": {
            "n": 1,
            "p_value": 0.04,
            "holm_adjusted_p_value": 0.08,
            "effect_pp_b_minus_a": -10.0,
        },
        "cluster_permutation_ac1": {
            "n": 1,
            "n_clusters": 1,
            "p_value": 0.04,
            "holm_adjusted_p_value": 0.08,
            "effect_pp_b_minus_a": -10.0,
        },
        "bootstrap": [
            {
                "metric": metric,
                "n": 1,
                "effect_pp_b_minus_a": -10.0,
                "ci95_pp": [-15.0, -5.0],
            }
            for metric in ("AC@1", "AC@3", "AC@5", "Avg@5")
        ],
        "cluster_bootstrap": [
            {
                "metric": metric,
                "n": 1,
                "n_clusters": 1,
                "effect_pp_b_minus_a": -10.0,
                "ci95_pp": [-15.0, -5.0],
            }
            for metric in ("AC@1", "AC@3", "AC@5", "Avg@5")
        ],
    }


def test_paired_gate_requires_per_system_and_compatible_pool():
    job = ExperimentJob("main", ("baro",))
    protocol = {
        "baselines": ["baro"],
        "datasets": ["ob", "ss"],
        "expected_cases_per_dataset": 1,
        "infeasible": {"baro:ss": {"status": "N/A"}},
    }
    records = [
        _paired_record(),
        _paired_record(scope="pooled", datasets=["ob"]),
    ]

    _validate_paired_statistics(job, records, protocol)

    with pytest.raises(RuntimeError, match="coverage mismatch"):
        _validate_paired_statistics(job, records[:1], protocol)


def test_paired_table_reports_merit_minus_comparator_direction():
    main = {
        "paired_statistics": [
            _paired_record(scope="pooled"),
        ]
    }

    latex = _paired_statistics_tables(
        main,
        pooled_only=True,
        families=("rq1_baselines",),
    )

    assert "+10.0 [+5.0, +15.0]" in latex
    assert "0.040" in latex
    assert "0.080" in latex
    assert "Pool(OB)" in latex


def test_stochastic_table_retains_each_seed_and_sample_variation():
    labels = ["rcd_seed1", "rcd_seed2", "rcd_seed3"]
    values = [0.4, 0.5, 0.6]
    main = {
        "summaries": [
            {
                "method": label,
                "dataset": "d",
                "overall": {"AC@1": value},
            }
            for label, value in zip(labels, values)
        ],
        "aggregates": [
            {
                "method": label,
                "macro_equal_system_weight": {"AC@1": value},
            }
            for label, value in zip(labels, values)
        ],
    }
    protocol = {
        "stochastic_methods": ["rcd"],
        "seeds": [1, 2, 3],
        "datasets": ["d"],
    }

    latex = _stochastic_seed_variation_table(main, protocol)

    assert "40.0 & 50.0 & 60.0 & 50.0 & 10.0" in latex
    assert "averages each case metric across frozen seeds" in latex


def _render_summary(method, dataset, *, failed=False):
    n_success = 0 if failed else 90
    value = 0.0 if failed else 0.5
    stages = {
        name: {"n": 90, "median": 0.001, "iqr": [0.001, 0.001], "p95": 0.001}
        for name in ("trace", "preprocess", "metric_evidence", "candidates", "banzhaf", "ranking")
    }
    return {
        "method": method,
        "dataset": dataset,
        "combination_status": "completed",
        "n_expected": 90,
        "n_success": n_success,
        "n_error": 90 if failed else 0,
        "n_timeout": 0,
        "n_missing": 0,
        "execution_success_rate": n_success / 90,
        "overall": {"AC@1": value, "AC@3": value, "AC@5": value, "Avg@5": value},
        "successful_cases_only": (
            None
            if failed
            else {"AC@1": value, "AC@3": value, "AC@5": value, "Avg@5": value}
        ),
        "candidate_recall": 0.8,
        "candidate_recall_n": 90,
        "dataset_observed_wall_seconds": 10.0,
        "task_observed_wall_seconds": 10.5,
        "timing_seconds": {
            "io": {"median": 0.01, "p95": 0.02},
            "method": {"median": 0.09, "p95": 0.18},
            "end_to_end": {"median": 0.1, "p95": 0.2},
            "measured_end_to_end": {"median": 0.1, "p95": 0.2},
            "first_case_end_to_end": 0.3,
            "remaining_cases_end_to_end": {"median": 0.09},
            "timeouts_observed": None,
        },
        "stage_timing_seconds": stages,
        "peak_rss_bytes": 1024 * 1024,
        "candidate_stage_k": {
            "base_candidate_k": {"n": 90, "median": 3.0, "iqr": [3.0, 3.0], "p95": 3.0},
            "coverage_rescued_k": {"n": 90, "median": 1.0, "iqr": [1.0, 1.0], "p95": 1.0},
            "final_candidate_k": {"n": 90, "median": 4.0, "iqr": [4.0, 4.0], "p95": 4.0},
            "target_count": {"n": 90, "median": 20.0, "iqr": [20.0, 20.0], "p95": 20.0},
        },
        "explanation_faithfulness": {
            "edge_count": {"n": 90, "median": 12.0, "iqr": [10.0, 14.0], "p95": 16.0},
            "pearson_correlation": {"n": 90, "median": 0.9, "iqr": [0.8, 0.95], "p95": 0.98},
            "mean_absolute_difference": {"n": 90, "median": 0.01, "iqr": [0.0, 0.02], "p95": 0.03},
            "sign_agreement": {"n": 90, "median": 1.0, "iqr": [1.0, 1.0], "p95": 1.0},
        },
        "fault_breakdown": {
            "cpu": {
                "n": 15,
                "successes": 0 if failed else 15,
                "AC@1": value,
                "AC@3": value,
                "AC@5": value,
                "Avg@5": value,
            }
        },
    }


def _render_aggregate(method):
    metrics = {"AC@1": 0.5, "AC@3": 0.5, "AC@5": 0.5, "Avg@5": 0.5}
    return {
        "method": method,
        "all_datasets_completed": True,
        "macro_equal_system_weight": metrics,
        "pooled": metrics,
    }


def test_complete_synthetic_reproduction_renders_main_and_supplements(tmp_path):
    datasets = ["re2-ob", "re2-ss", "re2-tt"]
    methods = [
        "merit_full",
        "baro",
        "merit_no_trace",
        "merit_graph_first_case_pre_fault",
        "merit_warm_rep1",
        "merit_victim_01",
        "merit_tconf_01",
        "merit_cconf_01",
        "merit_k3",
    ]
    protocol = {
        "datasets": datasets,
        "expected_cases_per_dataset": 90,
        "baselines": ["baro"],
        "stochastic_methods": [],
        "seeds": [1, 2, 3],
        "controlled_variants": ["no_trace", "graph_first_case_pre_fault"],
        "runtime_repetitions": 1,
        "sweeps": {
            "victim_floor": [0.1],
            "trace_topology_conf": [0.1],
            "coverage_conf": [0.1],
            "exact_candidate_k": [3],
            "tdelta_seconds": [0, 60],
            "observation_window_total_minutes": [20, 30],
        },
    }
    summaries = [
        _render_summary(
            method,
            dataset,
            failed=(method == "baro" and dataset == "re2-ss"),
        )
        for method in methods
        for dataset in datasets
    ]
    baseline_pool = _paired_record(scope="pooled", datasets=datasets)
    baseline_pool["case_ids"] = [f"case-{index}" for index in range(270)]
    baseline_pool["mcnemar_ac1"]["n"] = 270
    baseline_pool["cluster_permutation_ac1"]["n"] = 270
    baseline_pool["cluster_permutation_ac1"]["n_clusters"] = 18
    for row in baseline_pool["cluster_bootstrap"]:
        row["n"] = 270
        row["n_clusters"] = 18
    ablation_pool = _paired_record(
        method="merit_no_trace",
        scope="pooled",
        datasets=datasets,
        family="rq2_ablations",
    )
    ablation_pool["case_ids"] = [f"case-{index}" for index in range(270)]
    ablation_pool["mcnemar_ac1"]["n"] = 270
    ablation_pool["cluster_permutation_ac1"]["n"] = 270
    ablation_pool["cluster_permutation_ac1"]["n_clusters"] = 18
    for row in ablation_pool["cluster_bootstrap"]:
        row["n"] = 270
        row["n_clusters"] = 18
    robustness_pool = _paired_record(
        method="merit_graph_first_case_pre_fault",
        scope="pooled",
        datasets=datasets,
        family="rq4_robustness",
    )
    robustness_pool["case_ids"] = [f"case-{index}" for index in range(270)]
    robustness_pool["cluster_permutation_ac1"]["n"] = 270
    robustness_pool["cluster_permutation_ac1"]["n_clusters"] = 18
    for row in robustness_pool["cluster_bootstrap"]:
        row["n"] = 270
        row["n_clusters"] = 18
    main = {
        "summaries": summaries,
        "aggregates": [_render_aggregate(method) for method in methods],
        "paired_statistics": [baseline_pool, ablation_pool, robustness_pool],
        "case_results": [
            {
                "method": "merit_full",
                "dataset": "re2-ob",
                "case_id": "svc_cpu_1",
                "root_service": "svc",
                "status": "success",
                "ranked_services": ["svc", "other"],
                "final_candidate_k": 4,
                "top_explanation_player": "svc",
                "top_explanation_target": "other",
                "top_explanation_contribution": 0.2,
                "top_explanation_utility_drop": 0.18,
            },
            {
                "method": "merit_full",
                "dataset": "re2-tt",
                "case_id": "root_delay_2",
                "root_service": "root",
                "status": "success",
                "ranked_services": ["other", "root"],
                "final_candidate_k": 5,
                "top_explanation_player": "other",
                "top_explanation_target": "root",
                "top_explanation_contribution": 0.1,
                "top_explanation_utility_drop": 0.08,
            },
        ],
        "diagnostic_checks": {
                "missing_trace": {
                "status": "completed",
                "n": 90,
                "execution_status_matches": 90,
                "exact_metric_rank_matches": 90,
                "exact_service_rank_matches": 90,
                "all_execution_status_equal": True,
                "all_metric_ranks_equal": True,
                "all_service_ranks_equal": True,
                "metric_rank_mismatch_case_ids": [],
                    "service_rank_mismatch_case_ids": [],
                },
                "graph_sensitivity": {
                    "status": "completed",
                    "comparisons": [
                        {
                            "method": "merit_graph_first_case_pre_fault",
                            "dataset": dataset,
                            "n": 90,
                            "edge_jaccard_mean": 0.8,
                            "edge_jaccard_min": 0.5,
                            "ac1_effect_pp_vs_primary": -1.0,
                            "exact_service_rank_matches": 80,
                        }
                        for dataset in datasets
                    ],
                },
                "explanation_stability": {
                    "status": "completed",
                    "comparisons": [
                        {
                            "method": method,
                            "dataset": dataset,
                            "n": 90,
                            "n_explanations": 90,
                            "top_player_matches": 80,
                            "top_edge_matches": 75,
                            "top_player_agreement": 80 / 90,
                            "top_edge_agreement": 75 / 90,
                            "mean_absolute_top_contribution_change": 0.01,
                        }
                        for method in (
                            "merit_no_trace",
                            "merit_graph_first_case_pre_fault",
                        )
                        for dataset in datasets
                    ],
                },
            },
    }
    robustness_job = {
        "summaries": [
            _render_summary("merit_full", dataset) for dataset in datasets
        ],
        "aggregates": [_render_aggregate("merit_full")],
        "paired_statistics": [],
        "diagnostic_checks": {
            "missing_trace": {"status": "not_applicable"}
        },
    }
    validated = {
        "jobs": {
                "main": main,
                "window-20": robustness_job,
                "delay-60": robustness_job,
            "window-30": robustness_job,
        },
        "equivalence": {
            "summaries": [
                {
                    "k": 3,
                    "max_absolute_error": 0.0,
                    "modes": {
                        "closed_form": {"seconds": {"median": 0.001}},
                        "bruteforce": {"seconds": {"median": 0.1}},
                    },
                    "median_speedup_bruteforce_over_closed_form": 100.0,
                }
            ]
        },
        "selections": {
            parameter: {"selected_value": 0.1, "strategy": "joint"}
            for parameter in (
                "victim_floor",
                "trace_topology_conf",
                "coverage_conf",
                "exact_candidate_k",
            )
        },
    }

    main_tex = render_tables(validated, protocol, tmp_path)
    extended_tex = render_extended_results(validated, protocol)
    stats_tex = render_statistical_details(validated)
    faults_tex = render_fault_breakdown(validated, protocol)

    for marker in (
        "tab:effectiveness",
        "tab:missing-trace-exact",
        "tab:explanation-stability",
        "tab:core-inference",
    ):
        assert marker in main_tex
    for marker in (
        "tab:secondary-metrics",
        "tab:failure-sensitivity",
        "tab:candidate-recall",
        "tab:parameter-curves",
        "tab:delay-window",
        "tab:candidate-k-recall",
        "tab:explanation-stability-complete",
    ):
        assert marker in extended_tex
    assert "RQ1 baseline comparisons" in stats_tex
    assert "RQ2 controlled-ablation comparisons" in stats_tex
    assert "Descriptive fault-type breakdown" in faults_tex


def test_missing_trace_gate_requires_exact_equivalence():
    job = ExperimentJob("main", ("merit_full", "merit_no_trace"))
    protocol = {
        "expected_cases_per_dataset": 2,
        "datasets": ["re2-ss"],
    }
    diagnostics = {
        "missing_trace": {
            "status": "completed",
            "dataset": "re2-ss",
            "n": 2,
            "execution_status_matches": 2,
            "exact_metric_rank_matches": 2,
            "exact_service_rank_matches": 2,
            "all_execution_status_equal": True,
            "all_metric_ranks_equal": True,
            "all_service_ranks_equal": True,
            "metric_rank_mismatch_case_ids": [],
            "service_rank_mismatch_case_ids": [],
        },
        "explanation_stability": {
            "status": "completed",
            "comparisons": [
                {
                    "method": "merit_no_trace",
                    "dataset": "re2-ss",
                    "n": 2,
                    "n_explanations": 2,
                    "top_player_matches": 2,
                    "top_edge_matches": 2,
                    "top_player_agreement": 1.0,
                    "top_edge_agreement": 1.0,
                    "mean_absolute_top_contribution_change": 0.0,
                }
            ],
        },
    }

    _validate_diagnostic_checks(job, diagnostics, protocol)

    diagnostics["missing_trace"]["exact_metric_rank_matches"] = 1
    diagnostics["missing_trace"]["all_metric_ranks_equal"] = False
    diagnostics["missing_trace"]["metric_rank_mismatch_case_ids"] = ["case-2"]
    with pytest.raises(RuntimeError, match="exact-equivalence"):
        _validate_diagnostic_checks(job, diagnostics, protocol)
