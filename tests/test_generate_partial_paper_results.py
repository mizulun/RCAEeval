from script.generate_partial_paper_results import (
    BARO_BENCHMARK_FAULTS,
    ZH_EFFECTIVENESS_METRICS,
    _baro_benchmark_overall,
    _expected_partial_status,
    _partial_protocol,
    _rq1_draft_text,
    _rq2_draft_text,
    _zh_ablation_table,
    _zh_effectiveness_table,
    _zh_rq1_draft_text,
    _zh_rq2_draft_text,
)


def test_partial_protocol_excludes_requested_baselines():
    protocol = {
        "baselines": ["baro", "rcd", "causalrca", "circa"],
        "stochastic_methods": ["rcd", "causalrca"],
    }

    partial = _partial_protocol(protocol)

    assert partial["baselines"] == ["baro", "rcd"]
    assert partial["stochastic_methods"] == ["rcd"]
    assert protocol["baselines"][-2:] == ["causalrca", "circa"]


def test_expected_partial_status_honors_frozen_na_declarations():
    protocol = {
        "modality_inapplicable": {
            "tracerca:re2-ss": {"status": "N/A"},
        },
        "infeasible": {
            "circa:re2-tt": {"status": "N/A"},
        },
    }

    assert _expected_partial_status("tracerca", "re2-ss", protocol) == "inapplicable"
    assert _expected_partial_status("circa", "re2-tt", protocol) == "infeasible"
    assert _expected_partial_status("baro", "re2-ob", protocol) == "completed"


def _aggregate(method, value, complete=True):
    return {
        "method": method,
        "all_datasets_completed": complete,
        "macro_equal_system_weight": {"AC@1": value} if complete else None,
    }


def _baro_results(datasets, values=None):
    values = values or {
        "AC@1": 0.3,
        "AC@3": 0.4,
        "AC@5": 0.5,
        "Avg@5": 0.45,
    }
    return [
        {
            "method": "baro",
            "dataset": dataset,
            "status": "OK",
            **{
                f"{metric}-{fault}": value
                for metric, value in values.items()
                for fault in BARO_BENCHMARK_FAULTS
            },
        }
        for dataset in datasets
    ]


def test_baro_benchmark_overall_averages_six_fault_types():
    protocol = {"datasets": ["re2-ob"]}
    results = _baro_results(protocol["datasets"])
    for index, fault in enumerate(BARO_BENCHMARK_FAULTS):
        results[0][f"AC@1-{fault}"] = 0.1 * index

    overall = _baro_benchmark_overall(results, protocol)

    assert overall[("re2-ob", "AC@1")] == 0.25


def test_rq1_text_marks_excluded_baselines_and_subset_holm_scope():
    protocol = {
        "baselines": ["baro", "rcd"],
        "stochastic_methods": ["rcd"],
        "seeds": [1, 2, 3],
    }
    main = {
        "aggregates": [
            _aggregate("merit_full", 0.6),
            _aggregate("baro", 0.3),
            _aggregate("rcd_seed1", 0.4),
            _aggregate("rcd_seed2", 0.5),
            _aggregate("rcd_seed3", 0.45),
        ],
        "summaries": [],
    }

    text = _rq1_draft_text(
        main,
        protocol,
        {
            "passed": False,
            "rows": [{"passed": True}, {"passed": False}],
        },
    )

    assert "60.0\\% macro AC@1" in text
    assert "RCD at 45.0\\%" in text
    assert "CausalRCA and CIRCA are excluded" in text
    assert "not the final pre-registered RQ1 correction" in text
    assert "passes only 1/2" in text


def test_rq2_text_reports_negative_and_inconclusive_attribution_result():
    methods = {
        "merit_no_trace": -1.0,
        "merit_no_trace_topology": 0.0,
        "merit_no_coverage": -2.0,
        "merit_no_victim_demotion": -0.5,
        "merit_trace_only": -20.0,
        "merit_individual_attribution": 1.0,
    }
    main = {
        "paired_statistics": [
            {
                "method": method,
                "scope": "pooled",
                "comparison_family": "rq2_ablations",
                "cluster_permutation_ac1": {
                    "holm_adjusted_p_value": 0.01 if method == "merit_trace_only" else 1.0
                },
                "cluster_bootstrap": [
                    {
                        "metric": "AC@1",
                        "effect_pp_b_minus_a": effect,
                        "ci95_pp": [effect - 1.0, effect + 1.0],
                    }
                ],
            }
            for method, effect in methods.items()
        ]
    }

    text = _rq2_draft_text(main)

    assert "and -1.0 points, respectively" in text
    assert "cluster-bootstrap 95\\% CI" in text
    assert "Trace only" in text


def test_chinese_effectiveness_table_groups_four_ranking_metrics_by_method():
    datasets = ["re2-ob", "re2-ss", "re2-tt"]
    protocol = {
        "datasets": datasets,
        "baselines": ["baro"],
        "stochastic_methods": [],
        "seeds": [1, 2, 3],
        "infeasible": {},
    }
    summaries = [
        {
            "method": method,
            "dataset": dataset,
            "combination_status": "completed",
            "execution_success_rate": 1.0,
            "overall": {
                "AC@1": value,
                "AC@3": value + 0.1,
                "AC@5": value + 0.2,
                "Avg@5": value + 0.15,
            },
        }
        for method, value in (("merit_full", 0.6), ("baro", 0.3))
        for dataset in datasets
    ]
    aggregates = [
        {
            "method": method,
            "all_datasets_completed": True,
            "macro_equal_system_weight": {"AC@1": value},
            "pooled": {"AC@1": value},
        }
        for method, value in (("merit_full", 0.6), ("baro", 0.3))
    ]

    latex = _zh_effectiveness_table(
        {"summaries": summaries, "aggregates": aggregates},
        protocol,
        _baro_results(datasets),
    )

    assert "AC@1 & AC@3 & AC@5 & AVG@5" in latex
    assert r"\multicolumn{4}{c}{RE2-SS}" in latex
    assert (
        r"\method{} & \textbf{60.0} & \textbf{70.0} & "
        r"\textbf{80.0} & \textbf{75.0}" in latex
    )
    assert (
        r"BARO & \underline{30.0} & \underline{40.0} & \underline{50.0} & "
        r"\underline{45.0}" in latex
    )
    assert r"\resizebox" not in latex
    assert r"\centering\scriptsize" not in latex
    assert r"\fontsize{10}{12}\selectfont" in latex
    assert r"\begin{table*}[!t]" in latex
    assert r"\caption{" in latex
    assert "成功率" not in latex
    assert "CausalRCA" not in latex


def test_chinese_ablation_table_separates_modality_removal_from_direct_ranking():
    datasets = ["re2-ob", "re2-ss", "re2-tt"]
    protocol = {"datasets": datasets}
    values = {
        "merit_full": 0.50,
        "merit_no_trace": 0.49,
        "merit_no_metric": 0.20,
        "merit_direct_service_trace_evidence": 0.61,
    }
    summaries = []
    for method, value in values.items():
        for dataset in datasets:
            inapplicable = dataset == "re2-ss" and method in {
                "merit_no_metric",
            }
            summaries.append(
                {
                    "method": method,
                    "dataset": dataset,
                    "combination_status": (
                        "inapplicable" if inapplicable else "completed"
                    ),
                    "overall": (
                        None
                        if inapplicable
                        else {
                            "AC@1": value,
                            "AC@3": value + 0.10,
                            "AC@5": value + 0.20,
                            "Avg@5": value + 0.15,
                        }
                    ),
                }
            )

    latex = _zh_ablation_table({"summaries": summaries}, protocol)

    assert "無 trace" in latex
    assert "無 metric" in latex
    assert "直接分數和" in latex
    assert "直接 trace 分數排序" not in latex
    assert "直接 metric 分數排序" not in latex
    assert r"\begin{table*}[!t]" in latex
    assert r"\caption{" in latex
    assert r"\fontsize{10}{12}\selectfont" in latex
    assert "直接分數和只執行圖一前兩階段" in latex
    assert "無 trace 保留圖一完整流程" in latex
    assert "第二階段只擷取 metric evidence" in latex
    assert "trace score 或其他 trace-derived 資訊" in latex
    assert r"\multicolumn{4}{c}{RE2-SS}" in latex
    assert "AC@1 & AC@3 & AC@5 & AVG@5" in latex
    assert (
        r"無 metric & N/A & N/A & N/A & N/A"
        in latex
    )
    assert "Macro" not in latex


def test_chinese_rq2_text_explains_comparator_contracts_and_na_scope():
    main = {
        "summaries": [
            {
                "method": method,
                "dataset": dataset,
                "combination_status": (
                    "inapplicable"
                    if dataset == "re2-ss"
                    and method == "merit_no_metric"
                    else "completed"
                ),
                "overall": (
                    None
                    if dataset == "re2-ss"
                    and method == "merit_no_metric"
                    else {
                        "AC@1": value,
                        "AC@3": (
                            0.75
                            if method == "merit_direct_service_trace_evidence"
                            else value + 0.30
                        ),
                        "AC@5": (
                            0.85
                            if method == "merit_direct_service_trace_evidence"
                            else value + 0.40
                        ),
                        "Avg@5": (
                            0.80
                            if method == "merit_direct_service_trace_evidence"
                            else value + 0.35
                        ),
                    }
                ),
            }
            for method, value in {
                "merit_full": 0.50,
                "merit_no_trace": 0.49,
                "merit_no_metric": 0.20,
                "merit_direct_service_trace_evidence": 0.61,
            }.items()
            for dataset in ("re2-ob", "re2-ss", "re2-tt")
        ],
        "paired_statistics": [
            {
                "method": "merit_direct_service_trace_evidence",
                "scope": "pooled",
                "comparison_family": "rq2_ablations",
                "cluster_permutation_ac1": {"holm_adjusted_p_value": 1.0},
                "cluster_bootstrap": [
                    {
                        "metric": "AC@1",
                        "effect_pp_b_minus_a": 11.9,
                        "ci95_pp": [4.4, 19.6],
                    }
                ],
            }
        ]
    }

    text = _zh_rq2_draft_text(main)

    assert "仍保留 MERIT 的候選建構、causal game 與 Banzhaf 歸因" in text
    assert "無 trace 在 evidence extraction 只產生 metric evidence" in text
    assert "trace edge、trace score 或任何 trace-derived 資訊" in text
    assert "但整體方法流程不變" in text
    assert "只執行圖一前兩階段" in text
    assert "metric 與 trace evidence 等權相加" in text
    assert "50.0\\%、49.0\\% 與 61.0\\%" in text
    assert "Top-1 與前五名排序品質的取捨" in text
    assert "直接分數和退化為 metric evidence 排序" in text


def test_chinese_rq1_uses_benchmark_baro_without_failed_crosscheck_claim():
    datasets = ["re2-ob", "re2-ss", "re2-tt"]
    protocol = {
        "datasets": datasets,
        "baselines": ["baro"],
        "stochastic_methods": [],
        "seeds": [1, 2, 3],
    }
    main = {
        "aggregates": [
            {
                "method": "merit_full",
                "all_datasets_completed": True,
                "macro_equal_system_weight": {"AC@1": 0.5, "Avg@5": 0.855},
            }
        ],
        "summaries": [
            {
                "method": "merit_full",
                "dataset": dataset,
                "overall": {
                    "AC@1": 0.5,
                    "AC@3": 0.8,
                    "AC@5": 0.9,
                    "Avg@5": 0.855,
                },
            }
            for dataset in datasets
        ],
    }

    text = _zh_rq1_draft_text(main, protocol, _baro_results(datasets))

    assert "BARO 為 30.0\\%" in text
    assert "benchmark\\_results.json" in text
    assert "不能宣稱已重現" not in text
