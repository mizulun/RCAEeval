import importlib.util
import json
import pathlib


_spec = importlib.util.spec_from_file_location(
    "analyze_gabanzhaf_ablation",
    pathlib.Path(__file__).resolve().parent.parent
    / "script"
    / "analyze_gabanzhaf_ablation.py",
)
analysis_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analysis_mod)


def _write_case(root, method, dataset, name, ranks, diagnostics=None):
    result_dir = root / method / dataset / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / name).write_text(
        json.dumps({"0": ranks}),
        encoding="utf-8",
    )
    if diagnostics is not None:
        diagnostics_dir = result_dir / "diagnostics"
        diagnostics_dir.mkdir(exist_ok=True)
        (diagnostics_dir / name).write_text(
            json.dumps(diagnostics),
            encoding="utf-8",
        )


def test_compare_method_results_reports_paired_wins_losses_and_flips(tmp_path):
    _write_case(
        tmp_path,
        "base",
        "re2-ob",
        "svc-a_cpu_1.json",
        ["svc-b_cpu", "svc-a_cpu"],
        diagnostics={
            "game_candidates": ["svc-a", "svc-b"],
            "root_candidate_recall": True,
        },
    )
    _write_case(
        tmp_path,
        "variant",
        "re2-ob",
        "svc-a_cpu_1.json",
        ["svc-a_cpu", "svc-b_cpu"],
        diagnostics={
            "game_candidates": ["svc-a", "svc-c"],
            "root_candidate_recall": True,
        },
    )
    _write_case(
        tmp_path,
        "base",
        "re2-ob",
        "svc-b_cpu_1.json",
        ["svc-b_cpu", "svc-a_cpu"],
    )
    _write_case(
        tmp_path,
        "variant",
        "re2-ob",
        "svc-b_cpu_1.json",
        ["svc-a_cpu", "svc-b_cpu"],
    )

    comparison = analysis_mod.compare_method_results(
        tmp_path,
        "base",
        "variant",
        "re2-ob",
        k=1,
    )

    assert comparison["cases"] == 2
    assert comparison["variant_wins"] == 1
    assert comparison["variant_losses"] == 1
    assert comparison["ties"] == 0
    assert comparison["same_top1_rate"] == 0.0
    assert comparison["candidate_jaccard"] == 1 / 3
    assert comparison["baseline_candidate_recall"] == 1.0
    assert comparison["variant_candidate_recall"] == 1.0


def test_summarize_diagnostics_averages_trace_overlap_fields(tmp_path):
    for case, value in [("svc-a_cpu_1.json", 0.2), ("svc-b_cpu_1.json", 0.8)]:
        _write_case(
            tmp_path,
            "with_trace_full",
            "re2-ob",
            case,
            ["svc-a_cpu"],
            diagnostics={
                "trace_metric_spearman": value,
                "trace_metric_top_k_overlap": value,
                "trace_added_seed_count": 1,
                "accepted_call_graph_edge_count": 0,
                "effective_call_graph_edge_count": 0,
                "trace_extraction_sec": 0.1,
            },
        )

    summary = analysis_mod.summarize_diagnostics(
        tmp_path,
        "with_trace_full",
        "re2-ob",
    )

    assert summary["cases"] == 2
    assert summary["trace_metric_spearman"] == 0.5
    assert summary["trace_metric_top_k_overlap"] == 0.5
    assert summary["accepted_call_graph_edge_count"] == 0.0
    assert summary["has_trace_fields"] is True


def test_render_report_uses_with_trace_full_for_controlled_trace_variants(tmp_path):
    for method, ranks in [
        ("gabanzhaf", ["svc-b_cpu", "svc-a_cpu"]),
        ("with_trace_full", ["svc-a_cpu", "svc-b_cpu"]),
        ("with_trace_no_scores", ["svc-b_cpu", "svc-a_cpu"]),
    ]:
        _write_case(
            tmp_path,
            method,
            "re2-ob",
            "svc-a_cpu_1.json",
            ranks,
        )

    report = analysis_mod.render_report(
        tmp_path,
        benchmark_path=None,
        baseline="gabanzhaf",
        trace_baseline="with_trace_full",
        variants=["with_trace_full", "with_trace_no_scores"],
        datasets=["re2-ob"],
    )

    assert "| gabanzhaf | with_trace_full | re2-ob |" in report
    assert "| with_trace_full | with_trace_no_scores | re2-ob |" in report
    assert "## Experiment-supported conclusions" in report
    assert "## Acceptable method limitations" in report
    assert "## Issues requiring follow-up" in report
    assert "## Redundant stages and measured cost" in report


def test_render_report_distinguishes_low_k_from_stable_k_plateau(tmp_path):
    _write_case(
        tmp_path,
        "gabanzhaf",
        "re2-ob",
        "svc-a_cpu_1.json",
        ["svc-a_cpu", "svc-b_cpu"],
    )
    _write_case(
        tmp_path,
        "gabanzhaf_k1",
        "re2-ob",
        "svc-a_cpu_1.json",
        ["svc-a_cpu", "svc-c_cpu"],
    )
    _write_case(
        tmp_path,
        "gabanzhaf_k6",
        "re2-ob",
        "svc-a_cpu_1.json",
        ["svc-a_cpu", "svc-b_cpu"],
    )

    report = analysis_mod.render_report(
        tmp_path,
        benchmark_path=None,
        baseline="gabanzhaf",
        trace_baseline="with_trace_full",
        variants=["gabanzhaf_k1", "gabanzhaf_k6"],
        datasets=["re2-ob"],
    )

    assert "Candidate K=1 changed at least one paired ranking" in report
    assert "Candidate K=6 produced identical paired top-1 and ordered top-5" in report


def test_k_variant_number_accepts_merit_renamed_variants():
    assert analysis_mod._k_variant_number("merit_k6") == 6
    assert analysis_mod._k_variant_number("gabanzhaf_k6") == 6
    assert analysis_mod._k_variant_number("merit_evidence_metric") is None


def test_render_report_uses_benchmark_metrics_for_k_stability(tmp_path):
    _write_case(
        tmp_path,
        "gabanzhaf",
        "re2-ob",
        "svc-a_cpu_1.json",
        ["svc-a_cpu", "svc-b_cpu"],
    )
    _write_case(
        tmp_path,
        "gabanzhaf_k2",
        "re2-ob",
        "svc-a_cpu_1.json",
        ["svc-a_cpu", "svc-b_cpu"],
    )
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(
        json.dumps([
            {
                "method": "gabanzhaf",
                "dataset": "re2-ob",
                "AC@1-CPU": 1.0,
                "AC@3-CPU": 1.0,
                "AC@5-CPU": 1.0,
                "Avg@5-CPU": 1.0,
            },
            {
                "method": "gabanzhaf_k2",
                "dataset": "re2-ob",
                "AC@1-CPU": 1.0,
                "AC@3-CPU": 1.0,
                "AC@5-CPU": 0.0,
                "Avg@5-CPU": 0.0,
            },
        ]),
        encoding="utf-8",
    )

    report = analysis_mod.render_report(
        tmp_path,
        benchmark_path=benchmark,
        baseline="gabanzhaf",
        trace_baseline="with_trace_full",
        variants=["gabanzhaf_k2"],
        datasets=["re2-ob"],
    )

    assert "Candidate K=2 changed at least one benchmark metric" in report


def test_render_report_lag_conclusion_accepts_merit_variant_name(tmp_path):
    _write_case(
        tmp_path,
        "merit",
        "re2-ob",
        "svc-a_cpu_1.json",
        ["svc-b_cpu", "svc-a_cpu"],
    )
    _write_case(
        tmp_path,
        "merit_no_lag_1",
        "re2-ob",
        "svc-a_cpu_1.json",
        ["svc-a_cpu", "svc-b_cpu"],
    )

    report = analysis_mod.render_report(
        tmp_path,
        benchmark_path=None,
        baseline="merit",
        trace_baseline="with_trace_full",
        variants=["merit_no_lag_1"],
        datasets=["re2-ob"],
    )

    assert "Removing lag-1 produced 1 wins and 0 losses across 1 paired cases." in report


def test_render_report_includes_delta_vs_baseline_table(tmp_path):
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(
        json.dumps([
            {
                "method": "gabanzhaf",
                "dataset": "re2-ob",
                "wall_time_sec": 10.0,
                "AC@1-CPU": 0.5,
                "AC@1-MEM": 1.0,
                "Avg@5-CPU": 0.8,
                "Avg@5-MEM": 1.0,
            },
            {
                "method": "variant",
                "dataset": "re2-ob",
                "wall_time_sec": 7.5,
                "AC@1-CPU": 0.25,
                "AC@1-MEM": 1.0,
                "Avg@5-CPU": 0.7,
                "Avg@5-MEM": 1.0,
            },
        ]),
        encoding="utf-8",
    )

    report = analysis_mod.render_report(
        tmp_path,
        benchmark_path=benchmark,
        baseline="gabanzhaf",
        trace_baseline="with_trace_full",
        variants=["variant"],
        datasets=["re2-ob"],
    )

    assert "## Delta vs Baseline" in report
    assert (
        "| variant | re2-ob | 0.625 | -0.125 | 0.850 | -0.050 | "
        "7.50 | -2.50 |"
    ) in report
