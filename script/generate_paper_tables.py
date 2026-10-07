"""Generate gated LaTeX RQ tables from a complete v14 reproduction."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean, median

import numpy as np

try:
    from script.evaluate_paper_results import comparison_family
    from script.reproduce_paper import VARIANT_LABELS, _parameter_label, build_jobs
    from script.run_paper_experiments import resolve_method
except ModuleNotFoundError:  # direct ``python script/...py`` execution
    from evaluate_paper_results import comparison_family
    from reproduce_paper import VARIANT_LABELS, _parameter_label, build_jobs
    from run_paper_experiments import resolve_method


DISPLAY_NAMES = {
    "merit_full": r"\method{}",
    "baro": "BARO",
    "microrank": "MicroRank",
    "tracerca": "TraceRCA",
    "rcd": "RCD",
    "causalrca": "CausalRCA",
    "pc_pagerank": "PC+PageRank",
    "circa": "CIRCA",
    "merit_no_trace": "No trace",
    "merit_no_metric": "No metric",
    "merit_no_trace_topology": "No trace topology",
    "merit_no_coverage": "No coverage",
    "merit_no_victim_demotion": "No victim demotion",
    "merit_no_coverage_no_victim": "No coverage/victim demotion",
    "merit_no_coverage_parent_protection": "No parent protection",
    "merit_no_metric_type_dominance": "No metric-type dominance",
    "merit_no_early_full_fusion": "No early/full fusion",
    "merit_trace_latency_only": "Trace latency only",
    "merit_trace_error_only": "Trace error only",
    "merit_trace_lag0": "Trace lag 0",
    "merit_trace_time_shuffle": "Trace time shuffle",
    "merit_trace_random_edges": "Random trace edges",
    "merit_trace_only": "Trace only",
    "merit_direct_service_evidence": "Direct service evidence",
    "merit_direct_service_trace_evidence": "Direct metric+trace sum",
    "merit_individual_attribution": "Singleton attribution",
    "merit_bruteforce": "Brute-force Banzhaf",
    "merit_graph_per_case_full": "Per-case full graph",
    "merit_graph_first_case_pre_fault": "First-case pre-fault graph",
    "merit_graph_union_pre_fault": "Union pre-fault graph",
    "merit_graph_first_case_order1": "First-case graph (order 1)",
    "merit_graph_first_case_order2": "First-case graph (order 2)",
    "merit_graph_first_case_order3": "First-case graph (order 3)",
    "merit_trace_dropout_50": r"50\% trace dropout",
    "merit_trace_noise_30": r"30\% trace-series noise",
    "merit_trace_edge_noise_30": r"30\% trace-edge noise",
}


CORE_ABLATION_METHODS = (
    "merit_full",
    "merit_no_trace",
    "merit_no_metric",
    "merit_no_coverage",
    "merit_no_victim_demotion",
    "merit_no_coverage_no_victim",
    "merit_no_metric_type_dominance",
    "merit_no_early_full_fusion",
    "merit_trace_only",
    "merit_direct_service_evidence",
    "merit_direct_service_trace_evidence",
    "merit_individual_attribution",
)

CORE_INFERENCE_METHODS = {
    "rq1_baselines": {
        "baro", "microrank", "tracerca", "rcd", "causalrca",
        "pc_pagerank", "circa",
    },
    "rq2_ablations": set(CORE_ABLATION_METHODS) - {"merit_full"},
    "rq4_robustness": {
        "merit_no_trace",
        "merit_graph_per_case_full",
        "merit_graph_first_case_pre_fault",
        "merit_graph_union_pre_fault",
        "merit_trace_dropout_50",
        "merit_trace_noise_30",
        "merit_trace_edge_noise_30",
    },
}


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_percent(value) -> str:
    return "--" if value is None else f"{100.0 * float(value):.1f}"


def _fmt_seconds(value) -> str:
    return "--" if value is None else f"{float(value):.2f}"


def _fmt_p(value: float) -> str:
    value = float(value)
    return r"$<.001$" if value < 0.001 else f"{value:.3f}"


def render_headline_macros(validated: dict, protocol: dict) -> str:
    """Render the only numerical claims allowed before the Results section."""

    aggregates = {
        row["method"]: row for row in validated["jobs"]["main"]["aggregates"]
    }
    merit = aggregates["merit_full"]
    if not merit.get("all_datasets_completed"):
        raise RuntimeError("MERIT lacks a complete three-system headline aggregate")
    merit_value = float(merit["macro_equal_system_weight"]["AC@1"])
    eligible = []
    for method in protocol["baselines"]:
        labels = _method_labels(method, protocol)
        rows = [aggregates[label] for label in labels]
        if not all(row.get("all_datasets_completed") for row in rows):
            continue
        value = mean(
            float(row["macro_equal_system_weight"]["AC@1"])
            for row in rows
        )
        eligible.append((value, method))
    if not eligible:
        raise RuntimeError("no complete three-system baseline supports the headline")
    baseline_value, baseline = max(eligible, key=lambda item: (item[0], item[1]))
    delta = 100.0 * (merit_value - baseline_value)
    return "\n".join(
        [
            "% Generated and gated headline values; do not edit manually.",
            rf"\newcommand{{\MeritMacroACOne}}{{{100.0 * merit_value:.1f}\%}}",
            rf"\newcommand{{\MeritBestCompleteBaseline}}{{{_display_name(baseline)}}}",
            rf"\newcommand{{\MeritBestCompleteBaselineACOne}}{{{100.0 * baseline_value:.1f}\%}}",
            rf"\newcommand{{\MeritHeadlineDelta}}{{{delta:+.1f}}}",
            "",
        ]
    )


def _display_name(method: str) -> str:
    if method in DISPLAY_NAMES:
        return DISPLAY_NAMES[method]
    for base in ("rcd", "causalrca"):
        prefix = f"{base}_seed"
        if method.startswith(prefix):
            return f"{DISPLAY_NAMES[base]} (s{method.removeprefix(prefix)})"
    perturbation = re.fullmatch(
        r"merit_trace_(dropout|noise|edge_noise)_(\d+)", method
    )
    if perturbation:
        kind = {
            "dropout": "trace dropout",
            "noise": "trace-series noise",
            "edge_noise": "trace-edge noise",
        }[perturbation.group(1)]
        return rf"{perturbation.group(2)}\% {kind}"
    return method.replace("_", r"\_")


def _index_summaries(summaries: list[dict]) -> dict[tuple[str, str], dict]:
    result = {}
    for summary in summaries:
        key = (summary["method"], summary["dataset"])
        if key in result:
            raise RuntimeError(f"duplicate evaluated cell: {key}")
        result[key] = summary
    return result


def _na_declaration(method: str, dataset: str, protocol: dict) -> dict | None:
    """Return an exact-label or base-method N/A declaration."""

    base = resolve_method(method).base_method
    for key in (f"{method}:{dataset}", f"{base}:{dataset}"):
        if key in protocol.get("modality_inapplicable", {}):
            return protocol["modality_inapplicable"][key]
        if key in protocol.get("infeasible", {}):
            return protocol["infeasible"][key]
    return None


def _validate_job(
    reproduction_root: Path, job, protocol: dict
) -> tuple[list[dict], list[dict]]:
    manifest = _load(reproduction_root / "runs" / job.name / "manifest.json")
    if not manifest.get("formal_manifest") or manifest.get("git", {}).get("dirty"):
        raise RuntimeError(f"{job.name}: table generation requires a clean full manifest")
    if manifest.get("protocol_version") != protocol["protocol_version"]:
        raise RuntimeError(f"{job.name}: manifest/protocol version mismatch")
    if not manifest.get("output_path"):
        raise RuntimeError(f"{job.name}: manifest output path is absent")
    for dataset in protocol["datasets"]:
        dataset_manifest = manifest.get("datasets", {}).get(dataset, {})
        if (
            dataset_manifest.get("case_count")
            != protocol["expected_cases_per_dataset"]
            or dataset_manifest.get("expected_case_count")
            != protocol["expected_cases_per_dataset"]
            or dataset_manifest.get("checksum_mode") != "full"
            or not dataset_manifest.get("cryptographic_content_checksum")
            or not dataset_manifest.get("tree_sha256")
        ):
            raise RuntimeError(
                f"{job.name}: {dataset} lacks a complete full-checksum manifest"
            )
    if "python_environment" in protocol:
        environment = manifest.get("environment_validation", [])
        if not environment or not all(row.get("ok") for row in environment):
            raise RuntimeError(
                f"{job.name}: frozen environment validation is absent or failed"
            )
    analysis = reproduction_root / "analysis" / job.name
    summaries = _load(analysis / "dataset_summaries.json")
    aggregates = _load(analysis / "aggregate_summaries.json")
    indexed = _index_summaries(summaries)
    expected_keys = {
        (method, dataset)
        for method in job.methods
        for dataset in protocol["datasets"]
    }
    if set(indexed) != expected_keys:
        missing = sorted(expected_keys - set(indexed))
        extra = sorted(set(indexed) - expected_keys)
        raise RuntimeError(f"{job.name}: evaluated cell mismatch; missing={missing}, extra={extra}")
    for (method, dataset), summary in indexed.items():
        declared_na = _na_declaration(method, dataset, protocol)
        if declared_na:
            if summary["combination_status"] not in {"infeasible", "inapplicable"}:
                raise RuntimeError(f"{method} x {dataset}: expected declared N/A")
            continue
        if summary["combination_status"] != "completed":
            raise RuntimeError(f"{method} x {dataset}: incomplete retrospective cell")
        if summary["n_expected"] != protocol["expected_cases_per_dataset"]:
            raise RuntimeError(f"{method} x {dataset}: invalid denominator")
        if summary.get("n_missing") != 0:
            raise RuntimeError(f"{method} x {dataset}: missing artifacts")
    return summaries, aggregates


def _validate_paired_statistics(job, paired: list[dict], protocol: dict) -> None:
    expected = {}
    stochastic = set(protocol.get("stochastic_methods", []))
    seeded_groups: dict[str, list[str]] = {}
    for method in job.methods:
        match = re.fullmatch(r"(.+)_seed\d+", method)
        if match and match.group(1) in stochastic:
            seeded_groups.setdefault(match.group(1), []).append(method)
            continue
        family = comparison_family(method, protocol)
        if family is None:
            continue
        compatible = []
        for dataset in protocol["datasets"]:
            if _na_declaration(method, dataset, protocol):
                continue
            expected[(method, dataset)] = {
                "family": family,
                "datasets": [dataset],
                "n": protocol["expected_cases_per_dataset"],
            }
            compatible.append(dataset)
        if compatible:
            expected[(method, "pooled")] = {
                "family": family,
                "datasets": compatible,
                "n": len(compatible) * protocol["expected_cases_per_dataset"],
            }

    # Stochastic baselines are one pre-registered seed-mean estimand, so the
    # evaluator emits one record under the base label rather than three tests.
    for method, labels in seeded_groups.items():
        family = comparison_family(method, protocol)
        if family is None:
            continue
        compatible = []
        for dataset in protocol["datasets"]:
            if _na_declaration(method, dataset, protocol):
                continue
            expected[(method, dataset)] = {
                "family": family,
                "datasets": [dataset],
                "n": protocol["expected_cases_per_dataset"],
                "n_seeds": len(labels),
            }
            compatible.append(dataset)
        if compatible:
            expected[(method, "pooled")] = {
                "family": family,
                "datasets": compatible,
                "n": len(compatible) * protocol["expected_cases_per_dataset"],
                "n_seeds": len(labels),
            }

    indexed = {}
    for record in paired:
        key = (record.get("method"), record.get("scope", record.get("dataset")))
        if key in indexed:
            raise RuntimeError(f"duplicate paired statistic: {key}")
        indexed[key] = record
    if set(indexed) != set(expected):
        missing = sorted(set(expected) - set(indexed))
        extra = sorted(set(indexed) - set(expected))
        raise RuntimeError(
            f"paired statistic coverage mismatch; missing={missing}, extra={extra}"
        )

    for key, requirement in expected.items():
        record = indexed[key]
        if record.get("comparison_family") != requirement["family"]:
            raise RuntimeError(f"{key}: incorrect Holm family")
        if record.get("datasets") != requirement["datasets"]:
            raise RuntimeError(f"{key}: incorrect compatible datasets")
        permutation = record.get("cluster_permutation_ac1", {})
        if permutation.get("n") != requirement["n"]:
            raise RuntimeError(f"{key}: incorrect paired denominator")
        for field in ("p_value", "holm_adjusted_p_value"):
            if field not in permutation:
                raise RuntimeError(
                    f"{key}: missing cluster-permutation field {field}"
                )
        if not {
            "effect_pp_b_minus_a",
            "effect_pp_seed_mean_minus_reference",
        }.intersection(permutation):
            raise RuntimeError(f"{key}: cluster-permutation effect is absent")
        cluster_rows = record.get("cluster_bootstrap", [])
        metrics = {row.get("metric") for row in cluster_rows}
        if metrics != {"AC@1", "AC@3", "AC@5", "Avg@5"}:
            raise RuntimeError(f"{key}: incomplete paired cluster bootstrap metrics")
        if any(row.get("n") != requirement["n"] for row in cluster_rows):
            raise RuntimeError(f"{key}: incorrect cluster-bootstrap denominator")
        if requirement.get("n_seeds") is not None:
            if record.get("seed_labels") is None or any(
                row.get("n_seeds") != requirement["n_seeds"]
                for row in cluster_rows
            ):
                raise RuntimeError(f"{key}: incorrect frozen-seed aggregation")
        if len(record.get("case_ids", [])) != requirement["n"]:
            raise RuntimeError(f"{key}: incorrect paired case-id count")


def _validate_diagnostic_checks(
    job, diagnostics: dict, protocol: dict
) -> None:
    missing_trace = diagnostics.get("missing_trace", {})
    requires_check = {
        "merit_full",
        "merit_no_trace",
    }.issubset(set(job.methods))
    if not requires_check:
        if missing_trace.get("status") != "not_applicable":
            raise RuntimeError(
                f"{job.name}: unexpected missing-trace diagnostic status"
            )
        return
    expected = protocol["expected_cases_per_dataset"]
    if (
        missing_trace.get("status") != "completed"
        or missing_trace.get("dataset") != "re2-ss"
        or missing_trace.get("n") != expected
    ):
        raise RuntimeError(f"{job.name}: incomplete RE2-SS exact-rank diagnostic")
    for count_field, mismatch_field, all_field in (
        (
            "exact_metric_rank_matches",
            "metric_rank_mismatch_case_ids",
            "all_metric_ranks_equal",
        ),
        (
            "exact_service_rank_matches",
            "service_rank_mismatch_case_ids",
            "all_service_ranks_equal",
        ),
    ):
        count = missing_trace.get(count_field)
        mismatches = missing_trace.get(mismatch_field, [])
        if (
            not isinstance(count, int)
            or not 0 <= count <= expected
            or len(mismatches) != expected - count
            or bool(missing_trace.get(all_field)) != (count == expected)
        ):
            raise RuntimeError(
                f"{job.name}: inconsistent RE2-SS diagnostic field {count_field}"
            )
    status_matches = missing_trace.get("execution_status_matches")
    if (
        not isinstance(status_matches, int)
        or not 0 <= status_matches <= expected
        or bool(missing_trace.get("all_execution_status_equal"))
        != (status_matches == expected)
    ):
        raise RuntimeError(f"{job.name}: invalid execution-status comparison")
    if not all(
        missing_trace.get(field)
        for field in (
            "all_execution_status_equal",
            "all_metric_ranks_equal",
            "all_service_ranks_equal",
        )
    ):
        raise RuntimeError(
            f"{job.name}: RE2-SS full/no-trace exact-equivalence gate failed"
        )

    graph_variants = {
        "merit_graph_per_case_full",
        "merit_graph_first_case_pre_fault",
        "merit_graph_union_pre_fault",
        "merit_graph_first_case_order1",
        "merit_graph_first_case_order2",
        "merit_graph_first_case_order3",
    }
    graph = diagnostics.get("graph_sensitivity", {})
    graph_records = graph.get("comparisons", [])
    expected_graph_pairs = {
        (method, dataset)
        for method in graph_variants & set(job.methods)
        for dataset in protocol["datasets"]
    }
    actual_graph_pairs = {
        (record.get("method"), record.get("dataset"))
        for record in graph_records
    }
    if expected_graph_pairs:
        if (
            graph.get("status") != "completed"
            or actual_graph_pairs != expected_graph_pairs
            or any(record.get("n") != expected for record in graph_records)
        ):
            raise RuntimeError(
                f"{job.name}: graph-sensitivity evidence is incomplete"
            )
    elif graph.get("status", "not_applicable") != "not_applicable":
        raise RuntimeError(f"{job.name}: unexpected graph-sensitivity evidence")

    stability_variants = graph_variants | {
        "merit_no_trace",
        "merit_trace_dropout_50",
        "merit_trace_noise_30",
        "merit_trace_edge_noise_30",
    }
    stability = diagnostics.get("explanation_stability", {})
    stability_records = stability.get("comparisons", [])
    expected_stability_pairs = {
        (method, dataset)
        for method in stability_variants & set(job.methods)
        for dataset in protocol["datasets"]
    }
    actual_stability_pairs = {
        (record.get("method"), record.get("dataset"))
        for record in stability_records
    }
    if expected_stability_pairs:
        if (
            stability.get("status") != "completed"
            or actual_stability_pairs != expected_stability_pairs
        ):
            raise RuntimeError(
                f"{job.name}: explanation-stability evidence is incomplete"
            )
    elif stability.get("status", "not_applicable") != "not_applicable":
        raise RuntimeError(f"{job.name}: unexpected explanation-stability evidence")
    for record in stability_records:
        n_explanations = record.get("n_explanations")
        if (
            record.get("n") != expected
            or not isinstance(n_explanations, int)
            or not 0 < n_explanations <= expected
            or not 0.0 <= float(record.get("top_player_agreement", -1)) <= 1.0
            or not 0.0 <= float(record.get("top_edge_agreement", -1)) <= 1.0
            or float(record.get("mean_absolute_top_contribution_change", -1)) < 0.0
        ):
            raise RuntimeError(
                f"{job.name}: invalid explanation-stability record"
            )


def validate_reproduction(reproduction_root: Path, protocol: dict) -> dict:
    jobs = build_jobs(protocol, "retrospective")
    evaluated = {}
    for job in jobs:
        summaries, aggregates = _validate_job(reproduction_root, job, protocol)
        case_results = _load(
            reproduction_root / "analysis" / job.name / "case_results.json"
        )
        paired = _load(
            reproduction_root / "analysis" / job.name / "paired_statistics.json"
        )
        _validate_paired_statistics(job, paired, protocol)
        diagnostics = _load(
            reproduction_root / "analysis" / job.name / "diagnostic_checks.json"
        )
        _validate_diagnostic_checks(job, diagnostics, protocol)
        evaluated[job.name] = {
            "summaries": summaries,
            "aggregates": aggregates,
            "case_results": case_results,
            "paired_statistics": paired,
            "diagnostic_checks": diagnostics,
        }

    equivalence = _load(
        reproduction_root
        / "analysis"
        / "banzhaf-equivalence"
        / "banzhaf_equivalence.json"
    )
    if equivalence.get("protocol_version") != protocol["protocol_version"]:
        raise RuntimeError("Banzhaf benchmark/protocol version mismatch")
    if not equivalence.get("summaries") or not all(
        row.get("equivalent") for row in equivalence["summaries"]
    ):
        raise RuntimeError("Banzhaf equivalence gate did not pass")

    baro_crosscheck = _load(
        reproduction_root / "analysis" / "crosschecks" / "baro.json"
    )
    if baro_crosscheck.get("protocol_version") != protocol["protocol_version"]:
        raise RuntimeError("BARO cross-check/protocol version mismatch")
    if not baro_crosscheck.get("passed"):
        raise RuntimeError("self-run BARO does not reproduce the RCAEval reference")

    selections = {}
    for parameter in (
        "victim_floor",
        "trace_topology_conf",
        "coverage_conf",
        "exact_candidate_k",
    ):
        selection = _load(
            reproduction_root
            / "analysis"
            / "selections"
            / parameter
            / "selection.json"
        )
        if selection.get("protocol_version") != protocol["protocol_version"]:
            raise RuntimeError(f"{parameter}: selection/protocol version mismatch")
        selections[parameter] = selection
    return {
        "jobs": evaluated,
        "equivalence": equivalence,
        "baro_crosscheck": baro_crosscheck,
        "selections": selections,
    }


def _method_cells(
    indexed: dict[tuple[str, str], dict],
    aggregates: dict[str, dict],
    method: str,
    protocol: dict,
) -> tuple[list[str], str, str, str]:
    labels = _method_labels(method, protocol)
    dataset_cells = []
    execution = []
    for dataset in protocol["datasets"]:
        summaries = [indexed[(label, dataset)] for label in labels]
        if any(
            summary["combination_status"] in {"infeasible", "inapplicable"}
            for summary in summaries
        ):
            dataset_cells.append("N/A")
            continue
        dataset_cells.append(
            _fmt_percent(mean(summary["overall"]["AC@1"] for summary in summaries))
        )
        execution.extend(summary["execution_success_rate"] for summary in summaries)
    method_aggregates = [aggregates[label] for label in labels]
    if all(row["all_datasets_completed"] for row in method_aggregates):
        macro = _fmt_percent(
            mean(row["macro_equal_system_weight"]["AC@1"] for row in method_aggregates)
        )
        pooled = _fmt_percent(mean(row["pooled"]["AC@1"] for row in method_aggregates))
    else:
        macro = pooled = "N/A"
    return dataset_cells, macro, pooled, _fmt_percent(mean(execution) if execution else None)


def _method_labels(method: str, protocol: dict) -> list[str]:
    if method in protocol["stochastic_methods"]:
        return [f"{method}_seed{seed}" for seed in protocol["seeds"]]
    return [method]


def _secondary_metrics_table(main: dict, protocol: dict) -> str:
    aggregates = {row["method"]: row for row in main["aggregates"]}
    rows = []
    for method in ["merit_full", *protocol["baselines"]]:
        labels = _method_labels(method, protocol)
        summaries = [aggregates[label] for label in labels]
        values = []
        for metric in ("AC@3", "AC@5", "Avg@5"):
            if all(row["all_datasets_completed"] for row in summaries):
                values.append(
                    _fmt_percent(
                        mean(row["macro_equal_system_weight"][metric] for row in summaries)
                    )
                )
            else:
                values.append("N/A")
        rows.append(f"{DISPLAY_NAMES[method]} & {' & '.join(values)} \\\\")
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{RQ1 secondary service-level metrics (\%), macro-averaged with equal system weight. A method lacking a complete system receives no three-system macro.}",
            r"\label{tab:secondary-metrics}",
            r"\begin{tabular}{lrrr}",
            r"\toprule Method & AC@3 & AC@5 & Avg@5 \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )


def _stochastic_seed_variation_table(main: dict, protocol: dict) -> str:
    if not protocol["stochastic_methods"]:
        return "% No stochastic methods in this protocol."
    indexed = _index_summaries(main["summaries"])
    aggregates = {row["method"]: row for row in main["aggregates"]}
    rows = []
    for method in protocol["stochastic_methods"]:
        labels = _method_labels(method, protocol)
        for scope in [*protocol["datasets"], "macro"]:
            if scope == "macro":
                values = [
                    aggregates[label]["macro_equal_system_weight"]["AC@1"]
                    for label in labels
                ]
                scope_name = "Macro"
            else:
                values = [
                    indexed[(label, scope)]["overall"]["AC@1"]
                    for label in labels
                ]
                scope_name = scope.upper()
            rows.append(
                f"{DISPLAY_NAMES[method]} & {scope_name} & "
                f"{' & '.join(_fmt_percent(value) for value in values)} & "
                f"{_fmt_percent(mean(values))} & "
                f"{100.0 * float(np.std(values, ddof=1)):.1f} \\\\"
            )
    return "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{RQ1 stochastic AC@1 variation (\%) across the three pre-registered seeds. SD is the sample standard deviation. Primary inference averages each case metric across frozen seeds before cluster resampling; individual seed values are descriptive.}",
            r"\label{tab:seed-variation}",
            r"\begin{tabular}{llrrrrr}",
            r"\toprule Method & Scope & Seed 1 & Seed 2 & Seed 3 & Mean & SD \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )


def _candidate_recall_table(main: dict, protocol: dict) -> str:
    indexed = _index_summaries(main["summaries"])
    methods = [
        "merit_full",
        *(VARIANT_LABELS[name] for name in protocol["controlled_variants"] if name in VARIANT_LABELS),
    ]
    rows = []
    for method in methods:
        recalls = []
        rendered = []
        for dataset in protocol["datasets"]:
            summary = indexed[(method, dataset)]
            if summary["combination_status"] in {"infeasible", "inapplicable"}:
                rendered.append("N/A")
                continue
            if (
                summary.get("candidate_recall") is None
                or summary.get("candidate_recall_n") != protocol["expected_cases_per_dataset"]
            ):
                raise RuntimeError(f"{method} x {dataset}: incomplete candidate recall")
            recalls.append(summary["candidate_recall"])
            rendered.append(_fmt_percent(summary["candidate_recall"]))
        rows.append(
            f"{DISPLAY_NAMES[method]} & "
            f"{' & '.join(rendered)} & "
            f"{_fmt_percent(mean(recalls))} \\\\")
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Offline candidate recall (\%). Ground truth is joined only by the evaluator after ranking. Each applicable system has 90 cases; macro weights compatible systems equally.}",
            r"\label{tab:candidate-recall}",
            r"\begin{tabular}{lrrrr}",
            r"\toprule Variant & OB & SS & TT & Macro \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )


def _failure_sensitivity_table(main: dict, protocol: dict) -> str:
    indexed = _index_summaries(main["summaries"])
    rows = []
    for method in ["merit_full", *protocol["baselines"]]:
        labels = _method_labels(method, protocol)
        for dataset in protocol["datasets"]:
            summaries = [indexed[(label, dataset)] for label in labels]
            if any(
                row["combination_status"] in {"infeasible", "inapplicable"}
                for row in summaries
            ):
                rows.append(
                    f"{DISPLAY_NAMES[method]} & {dataset.upper()} & N/A & N/A & "
                    r"N/A & N/A & N/A & N/A \\")
                continue
            total = sum(row["n_expected"] for row in summaries)
            successes = sum(row["n_success"] for row in summaries)
            if successes == total:
                continue
            successful_values = [
                row["successful_cases_only"]["AC@1"]
                for row in summaries
                if row.get("successful_cases_only") is not None
            ]
            timeout_medians = [
                row["timing_seconds"]["timeouts_observed"]["median"]
                for row in summaries
                if row["timing_seconds"].get("timeouts_observed") is not None
            ]
            rows.append(
                f"{DISPLAY_NAMES[method]} & {dataset.upper()} & {successes}/{total} & "
                f"{_fmt_percent(mean(row['overall']['AC@1'] for row in summaries))} & "
                f"{_fmt_percent(mean(successful_values) if successful_values else None)} & "
                f"{sum(row['n_error'] for row in summaries)} & "
                f"{sum(row['n_timeout'] for row in summaries)} & "
                f"{_fmt_seconds(mean(timeout_medians) if timeout_medians else None)} \\\\")
    if not rows:
        raise RuntimeError("failure-sensitivity table unexpectedly has no rows")
    return "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Failure sensitivity for method--system cells with failures or a pre-declared N/A. Overall AC@1 retains every expected case; success-only AC@1 is diagnostic and never replaces it. Timeout observation is the mean of seed-level median parent-observed seconds. Stochastic counts include all three seeds.}",
            r"\label{tab:failure-sensitivity}",
            r"\begin{tabular}{llrrrrrr}",
            r"\toprule Method & System & Success & Overall & Success-only & Errors & Timeouts & Timeout obs. \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )


def _paired_statistics_tables(
    main: dict,
    *,
    pooled_only: bool,
    families: tuple[str, ...] = (
        "rq1_baselines",
        "rq2_ablations",
        "rq4_robustness",
    ),
) -> str:
    grouped = {
        "rq1_baselines": [],
        "rq2_ablations": [],
        "rq4_robustness": [],
    }
    for record in main["paired_statistics"]:
        if pooled_only and record["scope"] != "pooled":
            continue
        grouped[record["comparison_family"]].append(record)
    captions = {
        "rq1_baselines": "RQ1 baseline comparisons",
        "rq2_ablations": "RQ2 controlled-ablation comparisons",
        "rq4_robustness": "RQ4 robustness comparisons",
    }
    tables = []
    for family in families:
        records = grouped[family]
        if not records:
            raise RuntimeError(f"missing paired-statistics family: {family}")
        chunks = [
            records[index : index + 18]
            for index in range(0, len(records), 18)
        ]
        for chunk_index, chunk in enumerate(chunks, start=1):
            rows = []
            for record in chunk:
                ac1 = next(
                    row
                    for row in record["cluster_bootstrap"]
                    if row["metric"] == "AC@1"
                )
                # Evaluator stores comparator minus MERIT; the paper displays
                # the more readable MERIT minus comparator direction.
                raw_effect = ac1.get(
                    "effect_pp_b_minus_a",
                    ac1.get("effect_pp_seed_mean_minus_reference"),
                )
                effect = -float(raw_effect)
                low, high = ac1["ci95_pp"]
                merit_low, merit_high = -float(high), -float(low)
                scope = record["scope"].upper()
                if record["scope"] == "pooled":
                    scope = "Pool(" + "+".join(
                        item.removeprefix("re2-").upper()
                        for item in record["datasets"]
                    ) + ")"
                rows.append(
                    f"{_display_name(record['method'])} & {scope} & "
                    f"{ac1['n']} & {ac1['n_clusters']} & {effect:+.1f} "
                    f"[{merit_low:+.1f}, {merit_high:+.1f}] & "
                    f"{_fmt_p(record['cluster_permutation_ac1']['p_value'])} & "
                    f"{_fmt_p(record['cluster_permutation_ac1']['holm_adjusted_p_value'])} \\\\"
                )
            part = (
                ""
                if len(chunks) == 1
                else f" (part {chunk_index}/{len(chunks)})"
            )
            tables.append(
                "\n".join(
                    [
                        r"\begin{table*}[t]",
                        r"\centering\scriptsize",
                        rf"\caption{{{captions[family]}{part}. Effects are MERIT minus comparator AC@1 percentage points with paired cluster-bootstrap 95\% intervals. $p$ is a cluster-level paired sign-flip permutation test; $p_H$ is Holm-adjusted within this RQ family. Clusters are system$\times$root service$\times$fault type; failures remain misses.}}",
                        rf"\label{{tab:{family.replace('_', '-')}-{chunk_index}}}",
                        r"\begin{tabular}{llrrrrr}",
                        r"\toprule Comparator & Scope & $n$ & Clusters & $\Delta$AC@1 [95\% CI] & $p$ & $p_H$ \\",
                        r"\midrule",
                        *rows,
                        r"\bottomrule",
                        r"\end{tabular}",
                        r"\end{table*}",
                    ]
                )
            )
    return "\n\n".join(tables)


def _compact_paired_statistics_table(main: dict) -> str:
    """Render the pre-declared core comparisons that fit in the main paper."""

    family_names = {
        "rq1_baselines": "RQ1",
        "rq2_ablations": "RQ2",
        "rq4_robustness": "RQ4",
    }
    rows = []
    for record in main["paired_statistics"]:
        family = record["comparison_family"]
        if record["scope"] != "pooled":
            continue
        if record["method"] not in CORE_INFERENCE_METHODS.get(family, set()):
            continue
        ac1 = next(
            item for item in record["cluster_bootstrap"]
            if item["metric"] == "AC@1"
        )
        raw_effect = ac1.get(
            "effect_pp_b_minus_a",
            ac1.get("effect_pp_seed_mean_minus_reference"),
        )
        effect = -float(raw_effect)
        low, high = ac1["ci95_pp"]
        included = "+".join(
            dataset.removeprefix("re2-").upper()
            for dataset in record["datasets"]
        )
        permutation = record["cluster_permutation_ac1"]
        rows.append(
            f"{family_names[family]} & {_display_name(record['method'])} & "
            f"{included} & {ac1['n_clusters']} & {effect:+.1f} "
            f"[{-float(high):+.1f}, {-float(low):+.1f}] & "
            f"{_fmt_p(permutation['holm_adjusted_p_value'])} \\\\"
        )
    if not rows:
        raise RuntimeError("core paired-inference records are absent")
    return "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering\scriptsize",
            r"\caption{Pre-declared core paired inference. Effects are MERIT minus comparator AC@1 percentage points with paired cluster-bootstrap 95\% intervals. $p_H$ is the cluster sign-flip permutation value after Holm correction within each RQ family. Complete per-system and all-variant results are in the generated companion tables.}",
            r"\label{tab:core-inference}",
            r"\begin{tabular}{lllrrr}",
            r"\toprule RQ & Comparator & Systems & Clusters & $\Delta$AC@1 [95\% CI] & $p_H$ \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )


def _effectiveness_table(main: dict, protocol: dict) -> str:
    indexed = _index_summaries(main["summaries"])
    aggregates = {row["method"]: row for row in main["aggregates"]}
    methods = ["merit_full", *protocol["baselines"]]
    rows = []
    for method in methods:
        cells, macro, pooled, execution = _method_cells(
            indexed, aggregates, method, protocol
        )
        rows.append(
            f"{DISPLAY_NAMES[method]} & {' & '.join(cells)} & {macro} & {pooled} & {execution} \\\\"
        )
    return "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{RQ1 service-level AC@1 and execution success (\%). OB, SS, and TT each use 90 cases; macro weights systems equally and pooled uses 270 cases. Crashes and timeouts are misses. Stochastic methods are means over three pre-registered seeds. N/A is a pre-declared infeasible or modality-inapplicable method--dataset combination.}",
            r"\label{tab:effectiveness}",
            r"\begin{tabular}{lrrrrrr}",
            r"\toprule",
            r"Method & OB & SS & TT & Macro & Pooled & Exec. \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )


def _ablation_table(
    main: dict,
    protocol: dict,
    *,
    methods: tuple[str, ...] | None = None,
    label: str = "tab:ablation",
    caption: str = (
        "RQ2 controlled variants, service AC@1 (\\%). Each cell retains all "
        "90 cases; macro weights the three systems equally."
    ),
) -> str:
    indexed = _index_summaries(main["summaries"])
    if methods is None:
        methods = tuple([
            "merit_full",
            *(
                VARIANT_LABELS[name]
                for name in protocol["controlled_variants"]
                if name in VARIANT_LABELS
            ),
        ])
    rows = []
    for method in methods:
        cells = []
        applicable_values = []
        for dataset in protocol["datasets"]:
            summary = indexed[(method, dataset)]
            if summary["combination_status"] in {"infeasible", "inapplicable"}:
                cells.append("N/A")
            else:
                value = summary["overall"]["AC@1"]
                applicable_values.append(value)
                cells.append(_fmt_percent(value))
        macro = (
            _fmt_percent(mean(applicable_values)) if applicable_values else "N/A"
        )
        rows.append(
            f"{DISPLAY_NAMES[method]} & {' & '.join(cells)} & {macro} \\\\"
        )
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\begin{tabular}{lrrrr}",
            r"\toprule Variant & OB & SS & TT & Macro \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )


def _equivalence_table(equivalence: dict) -> str:
    rows = []
    for summary in equivalence["summaries"]:
        closed = 1000 * summary["modes"]["closed_form"]["seconds"]["median"]
        brute = 1000 * summary["modes"]["bruteforce"]["seconds"]["median"]
        rows.append(
            f"{summary['k']} & {summary['max_absolute_error']:.2e} & "
            f"{closed:.3f} & {brute:.3f} & "
            f"{summary['median_speedup_bruteforce_over_closed_form']:.1f}$\\times$ \\\\"
        )
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{RQ3 numerical equivalence and median measured cost over five repetitions with 50 targets. Times are milliseconds.}",
            r"\label{tab:banzhaf-equivalence}",
            r"\begin{tabular}{rrrrr}",
            r"\toprule $K$ & Max error & Closed & Brute & Speedup \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )


def _robustness_table(selections: dict) -> str:
    names = {
        "victim_floor": "Victim floor",
        "trace_topology_conf": "Trace confidence",
        "coverage_conf": "Coverage confidence",
        "exact_candidate_k": "Candidate $K$",
    }
    rows = [
        f"{names[parameter]} & {selection['selected_value']} & "
        f"{selection['strategy'].replace('_', ' ')} \\\\"
        for parameter, selection in selections.items()
    ]
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{RQ4 global values selected by the frozen joint/LOSO rule. Complete curves are retained in the artifact.}",
            r"\label{tab:robustness-selection}",
            r"\begin{tabular}{lrl}",
            r"\toprule Parameter & Value & Rule \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )


def _missing_trace_diagnostic_table(main: dict) -> str:
    check = main["diagnostic_checks"]["missing_trace"]
    if check.get("status") != "completed":
        raise RuntimeError("RE2-SS missing-trace diagnostic is incomplete")
    rows = [
        (
            "Execution status",
            check["execution_status_matches"],
            check["all_execution_status_equal"],
        ),
        (
            "Metric-rank sequence",
            check["exact_metric_rank_matches"],
            check["all_metric_ranks_equal"],
        ),
        (
            "Service-rank sequence",
            check["exact_service_rank_matches"],
            check["all_service_ranks_equal"],
        ),
    ]
    rendered = [
        f"{name} & {matches}/{check['n']} & "
        f"{'Yes' if all_equal else 'No'} \\\\"
        for name, matches, all_equal in rows
    ]
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{RQ4 exact full versus no-trace gate on all 90 RE2-SS cases. Execution status and complete metric/service rank sequences must match; mismatch case IDs remain in the diagnostic artifact.}",
            r"\label{tab:missing-trace-exact}",
            r"\begin{tabular}{lrr}",
            r"\toprule Compared artifact & Exact matches & All equal \\",
            r"\midrule",
            *rendered,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )


def _graph_sensitivity_table(main: dict) -> str:
    check = main["diagnostic_checks"].get("graph_sensitivity", {})
    if check.get("status") != "completed":
        raise RuntimeError("graph-sensitivity diagnostic is incomplete")
    rows = []
    for record in check.get("comparisons", []):
        n = record["n"]
        rows.append(
            f"{_display_name(record['method'])} & {record['dataset'].upper()} & "
            f"{n} & {record['edge_jaccard_mean']:.3f} & "
            f"{record['edge_jaccard_min']:.3f} & "
            f"{100.0 * record['exact_service_rank_matches'] / n:.1f} & "
            f"{record['ac1_effect_pp_vs_primary']:+.1f} \\\\"
        )
    if not rows:
        raise RuntimeError("graph-sensitivity diagnostic has no comparisons")
    return "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering\scriptsize",
            r"\caption{RQ4 graph-source and case-order sensitivity relative to the per-case pre-fault primary graph. Edge columns report call-graph Jaccard similarity; rank agreement is exact service-ranking agreement. $\Delta$AC@1 is variant minus primary.}",
            r"\label{tab:graph-sensitivity}",
            r"\begin{tabular}{llrrrrr}",
            r"\toprule Variant & System & $n$ & Edge Jaccard mean & Min. & Rank agree (\%) & $\Delta$AC@1 \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )


def _faithfulness_table(main: dict, protocol: dict) -> str:
    indexed = _index_summaries(main["summaries"])
    rows = []
    for dataset in protocol["datasets"]:
        summary = indexed[("merit_full", dataset)]
        faithfulness = summary.get("explanation_faithfulness", {})
        edges = faithfulness.get("edge_count")
        correlation = faithfulness.get("pearson_correlation")
        difference = faithfulness.get("mean_absolute_difference")
        agreement = faithfulness.get("sign_agreement")
        if not all((edges, difference, agreement)):
            raise RuntimeError(f"{dataset}: explanation-faithfulness evidence is absent")
        correlation_text = (
            "--" if correlation is None else f"{correlation['median']:.3f}"
        )
        rows.append(
            f"{dataset.upper()} & {int(edges['n'])} & {edges['median']:.1f} & "
            f"{correlation_text} & {difference['median']:.4f} & "
            f"{100.0 * agreement['median']:.1f} \\\\"
        )
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Local explanation faithfulness. For each nonzero player--target coverage edge, we compare its local Banzhaf contribution with the full-coalition utility drop after removing that edge. Values are case medians; correlation may be undefined for constant cases.}",
            r"\label{tab:explanation-faithfulness}",
            r"\begin{tabular}{lrrrrr}",
            r"\toprule System & Cases & Edges/case & Pearson $r$ & MAE & Sign agree (\%) \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )


def _explanation_stability_table(
    main: dict,
    *,
    aggregate: bool,
    methods: set[str] | None = None,
) -> str:
    check = main["diagnostic_checks"].get("explanation_stability", {})
    if check.get("status") != "completed":
        raise RuntimeError("explanation-stability diagnostic is incomplete")
    records = [
        record for record in check.get("comparisons", [])
        if methods is None or record["method"] in methods
    ]
    if not records:
        raise RuntimeError("explanation-stability diagnostic has no comparisons")
    rows = []
    if aggregate:
        for method in sorted({record["method"] for record in records}):
            group = [record for record in records if record["method"] == method]
            n = sum(record["n_explanations"] for record in group)
            player_matches = sum(record["top_player_matches"] for record in group)
            edge_matches = sum(record["top_edge_matches"] for record in group)
            change = sum(
                record["mean_absolute_top_contribution_change"]
                * record["n_explanations"]
                for record in group
            ) / n
            rows.append(
                f"{_display_name(method)} & {n} & "
                f"{100.0 * player_matches / n:.1f} & "
                f"{100.0 * edge_matches / n:.1f} & {change:.4f} \\\\"
            )
        columns = r"lrrrr"
        header = (
            r"Variant & Explanations & Top player agree (\%) & "
            r"Top edge agree (\%) & $|\Delta\phi_{\max}|$ \\"
        )
        caption = (
            "Explanation stability pooled over compatible systems. Agreement "
            "compares the largest local contribution against the primary run; "
            "the last column is its mean absolute contribution change."
        )
        label = "tab:explanation-stability"
    else:
        for record in records:
            rows.append(
                f"{_display_name(record['method'])} & "
                f"{record['dataset'].upper()} & {record['n_explanations']} & "
                f"{100.0 * record['top_player_agreement']:.1f} & "
                f"{100.0 * record['top_edge_agreement']:.1f} & "
                f"{record['mean_absolute_top_contribution_change']:.4f} \\\\"
            )
        columns = r"llrrrr"
        header = (
            r"Variant & System & Explanations & Player agree (\%) & "
            r"Edge agree (\%) & $|\Delta\phi_{\max}|$ \\"
        )
        caption = (
            "Complete per-system explanation stability under pre-registered "
            "graph and trace perturbations."
        )
        label = "tab:explanation-stability-complete"
    return "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering\scriptsize",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            rf"\begin{{tabular}}{{{columns}}}",
            r"\toprule",
            header,
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )


def _tex_escape(value: object) -> str:
    text_value = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in text_value)


def _case_study_table(main: dict) -> str:
    cases = [
        row
        for row in main.get("case_results", [])
        if row.get("method") == "merit_full" and row.get("status") == "success"
    ]
    if not cases:
        raise RuntimeError("MERIT case-study records are absent")
    hits = [
        row
        for row in cases
        if row.get("root_service") in (row.get("ranked_services") or [])[:1]
    ]
    misses = [
        row
        for row in cases
        if row.get("root_service") not in (row.get("ranked_services") or [])[:1]
    ]
    selected = []
    if hits:
        selected.append(("Success", sorted(hits, key=lambda row: row["case_id"])[0]))
    if misses:
        selected.append(("Failure", sorted(misses, key=lambda row: row["case_id"])[0]))
    else:
        selected.append(("Hardest", sorted(cases, key=lambda row: row["case_id"])[-1]))
    rows = []
    for kind, row in selected:
        ranked = row.get("ranked_services") or []
        top1 = ranked[0] if ranked else "--"
        player = row.get("top_explanation_player") or "--"
        target = row.get("top_explanation_target") or "--"
        contribution = row.get("top_explanation_contribution")
        utility_drop = row.get("top_explanation_utility_drop")
        explanation = f"{_tex_escape(player)}$\\to${_tex_escape(target)}"
        rows.append(
            f"{kind} & {_tex_escape(row['dataset'].upper())} & "
            f"{_tex_escape(row['case_id'])} & {_tex_escape(row['root_service'])} & "
            f"{_tex_escape(top1)} & {row.get('final_candidate_k', '--')} & "
            f"{explanation} & "
            f"{'--' if contribution is None else f'{float(contribution):.3f}'} & "
            f"{'--' if utility_drop is None else f'{float(utility_drop):.3f}'} \\\\"
        )
    return "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering\scriptsize",
            r"\caption{Deterministically selected success and failure (or hardest if no AC@1 failure exists) from the primary run. The local edge is the largest player--target contribution; drop is the full-coalition utility reduction when that edge is removed. Full decompositions remain in case diagnostics.}",
            r"\label{tab:case-study}",
            r"\begin{tabular}{lllllrlrr}",
            r"\toprule Type & System & Case & Root & Top-1 & $K$ & Local edge & Contribution & Drop \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )


def _efficiency_table(main: dict, protocol: dict) -> str:
    indexed = _index_summaries(main["summaries"])
    cold_labels = [
        "merit_full",
        *(f"merit_full_rep{i}" for i in range(2, protocol["runtime_repetitions"] + 1)),
    ]
    warm_labels = [
        f"merit_warm_rep{i}"
        for i in range(1, protocol["runtime_repetitions"] + 1)
    ]
    rows = []
    for dataset in protocol["datasets"]:
        for condition, labels in (("Cold", cold_labels), ("Warm", warm_labels)):
            summaries = [indexed[(method, dataset)] for method in labels]
            totals = [summary["task_observed_wall_seconds"] for summary in summaries]
            measured = [
                summary["timing_seconds"]["measured_end_to_end"]["median"]
                for summary in summaries
            ]
            p95 = [
                summary["timing_seconds"]["measured_end_to_end"]["p95"]
                for summary in summaries
            ]
            method_times = [
                summary["timing_seconds"]["method"]["median"]
                for summary in summaries
            ]
            if any(value is None for value in [*totals, *measured, *p95, *method_times]):
                raise RuntimeError(
                    f"{dataset}/{condition}: complete efficiency measurements are absent"
                )
            execution = mean(
                summary["execution_success_rate"] for summary in summaries
            )
            memory_mib = max(
                summary["peak_rss_bytes"] for summary in summaries
            ) / 2**20
            rows.append(
                f"{dataset.upper()} & {condition} & {median(totals):.1f} & "
                f"{median(measured):.2f} "
                f"[{np.percentile(measured, 25):.2f}, {np.percentile(measured, 75):.2f}] & "
                f"{median(p95):.2f} & {median(method_times):.2f} & "
                f"{memory_mib:.1f} & {_fmt_percent(execution)} \\\\"
            )
    return "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{MERIT efficiency over five independent repetitions per cache condition. Cold runs start a fresh process; warm runs measure a second invocation of the same case after an untimed-for-score warm-up. Task wall includes warm-up overhead, whereas measured end-to-end is input plus the measured invocation. Values are seconds; brackets are the IQR across repetition medians. RSS is maximum MiB.}",
            r"\label{tab:efficiency}",
            r"\begin{tabular}{llrrrrrr}",
            r"\toprule System & Cache & Task wall & Measured E2E [IQR] & E2E p95 & Method med. & RSS & Exec. \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )


def _stage_table(main: dict, protocol: dict) -> str:
    indexed = _index_summaries(main["summaries"])
    conditions = {
        "Cold": [
            "merit_full",
            *(f"merit_full_rep{i}" for i in range(2, protocol["runtime_repetitions"] + 1)),
        ],
        "Warm": [
            f"merit_warm_rep{i}"
            for i in range(1, protocol["runtime_repetitions"] + 1)
        ],
    }
    stages = ("trace", "preprocess", "metric_evidence", "candidates", "banzhaf", "ranking")
    rows = []
    for dataset in protocol["datasets"]:
        for condition, labels in conditions.items():
            summaries = [indexed[(method, dataset)] for method in labels]
            io_ms = 1000 * median(
                summary["timing_seconds"]["io"]["median"] for summary in summaries
            )
            values = []
            for stage in stages:
                samples = [
                    summary["stage_timing_seconds"].get(stage, {}).get("median")
                    for summary in summaries
                ]
                samples = [sample for sample in samples if sample is not None]
                values.append(
                    "--" if not samples else f"{1000 * median(samples):.3f}"
                )
            rows.append(
                f"{dataset.upper()} & {condition} & {io_ms:.3f} & "
                f"{' & '.join(values)} \\\\"
            )
    return "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{MERIT per-case stage medians (ms), reported across five independent repetitions for cold and same-case warm conditions. Trace includes trace extraction and call-graph work; input loading is measured by the harness.}",
            r"\label{tab:stage-efficiency}",
            r"\begin{tabular}{llrrrrrrr}",
            r"\toprule System & Cache & Input & Trace & Preproc. & Metric & Candidates & Banzhaf & Ranking \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )


def _stage_share_table(main: dict, protocol: dict) -> str:
    indexed = _index_summaries(main["summaries"])
    conditions = {
        "Cold": [
            "merit_full",
            *(f"merit_full_rep{i}" for i in range(2, protocol["runtime_repetitions"] + 1)),
        ],
        "Warm": [
            f"merit_warm_rep{i}"
            for i in range(1, protocol["runtime_repetitions"] + 1)
        ],
    }
    stages = ("trace", "preprocess", "metric_evidence", "candidates", "banzhaf", "ranking")
    rows = []
    for dataset in protocol["datasets"]:
        for condition, labels in conditions.items():
            summaries = [indexed[(label, dataset)] for label in labels]
            stage_values = []
            for stage in stages:
                samples = [
                    summary["stage_timing_seconds"].get(stage, {}).get("median")
                    for summary in summaries
                ]
                samples = [sample for sample in samples if sample is not None]
                stage_values.append(median(samples) if samples else 0.0)
            total = sum(stage_values)
            if total <= 0:
                raise RuntimeError(f"{dataset}/{condition}: stage-share evidence is absent")
            rows.append(
                f"{dataset.upper()} & {condition} & "
                f"{' & '.join(f'{100.0 * value / total:.1f}' for value in stage_values)} \\\\"
            )
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering\scriptsize",
            r"\caption{Share of instrumented MERIT method time (\%) by stage. Shares use the median stage duration across five repetitions and exclude harness input loading.}",
            r"\label{tab:stage-shares}",
            r"\begin{tabular}{llrrrrrr}",
            r"\toprule System & Cache & Trace & Preproc. & Metric & Candidates & Banzhaf & Ranking \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )


def _robustness_curves(validated: dict, protocol: dict) -> str:
    rows = []
    for setting, values, default, prefix in (
        ("Delay (s)", protocol["sweeps"]["tdelta_seconds"], 0, "delay"),
        (
            "Window (min)",
            protocol["sweeps"]["observation_window_total_minutes"],
            20,
            "window",
        ),
    ):
        for value in values:
            # The v14 main job is full-record and cannot stand in for a
            # fixed-window sensitivity point. Delay zero is the same explicit
            # 20-minute window used by the window sweep, so reuse that job.
            job_name = (
                "window-20"
                if prefix == "delay" and value == default
                else f"{prefix}-{value}"
            )
            job = validated["jobs"][job_name]
            indexed = _index_summaries(job["summaries"])
            aggregate = next(
                row for row in job["aggregates"] if row["method"] == "merit_full"
            )
            cells = [
                _fmt_percent(indexed[("merit_full", dataset)]["overall"]["AC@1"])
                for dataset in protocol["datasets"]
            ]
            macro = _fmt_percent(aggregate["macro_equal_system_weight"]["AC@1"])
            rows.append(
                f"{setting} & {value} & {' & '.join(cells)} & {macro} \\\\"
            )
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{RQ4 delay and total-window sensitivity, service AC@1 (\%). Every cell retains 90 cases; macro weights systems equally.}",
            r"\label{tab:delay-window}",
            r"\begin{tabular}{lrrrrr}",
            r"\toprule Setting & Value & OB & SS & TT & Macro \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )


def _parameter_sweep_curves(validated: dict, protocol: dict) -> str:
    main = validated["jobs"]["main"]
    indexed = _index_summaries(main["summaries"])
    aggregates = {row["method"]: row for row in main["aggregates"]}
    titles = {
        "victim_floor": "Victim floor",
        "trace_topology_conf": "Trace confidence",
        "coverage_conf": "Coverage confidence",
        "exact_candidate_k": "Candidate $K$",
    }
    blocks = []
    for parameter in (
        "victim_floor",
        "trace_topology_conf",
        "coverage_conf",
        "exact_candidate_k",
    ):
        rows = []
        for value in protocol["sweeps"][parameter]:
            label = _parameter_label(parameter, value)
            cells = [
                _fmt_percent(indexed[(label, dataset)]["overall"]["AC@1"])
                for dataset in protocol["datasets"]
            ]
            macro = _fmt_percent(
                aggregates[label]["macro_equal_system_weight"]["AC@1"]
            )
            rows.append(f"{value} & {' & '.join(cells)} & {macro} \\\\")
        blocks.append(
            "\n".join(
                [
                    r"\begin{minipage}[t]{0.48\textwidth}",
                    r"\centering\scriptsize",
                    rf"\textbf{{{titles[parameter]}}}\\[1mm]",
                    r"\begin{tabular}{rrrrr}",
                    r"\toprule Value & OB & SS & TT & Macro \\",
                    r"\midrule",
                    *rows,
                    r"\bottomrule",
                    r"\end{tabular}",
                    r"\end{minipage}",
                ]
            )
        )
    return "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{RQ4 complete pre-registered parameter-sensitivity curves, service AC@1 (\%). Every cell retains 90 cases and macro weights systems equally. Selection follows the frozen joint/LOSO rule rather than a single best test cell.}",
            r"\label{tab:parameter-curves}",
            blocks[0] + r"\hfill" + blocks[1],
            r"\par\medskip",
            blocks[2] + r"\hfill" + blocks[3],
            r"\end{table*}",
        ]
    )


def _candidate_k_recall_table(validated: dict, protocol: dict) -> str:
    main = validated["jobs"]["main"]
    indexed = _index_summaries(main["summaries"])
    rows = []
    for value in protocol["sweeps"]["exact_candidate_k"]:
        label = _parameter_label("exact_candidate_k", value)
        for dataset in protocol["datasets"]:
            summary = indexed[(label, dataset)]
            if (
                summary.get("candidate_recall") is None
                or summary.get("candidate_recall_n")
                != protocol["expected_cases_per_dataset"]
            ):
                raise RuntimeError(
                    f"{label} x {dataset}: incomplete candidate-K recall"
                )
            stages = summary.get("candidate_stage_k", {})
            required = {
                key: stages.get(key)
                for key in (
                    "base_candidate_k",
                    "coverage_rescued_k",
                    "final_candidate_k",
                    "target_count",
                )
            }
            if any(distribution is None for distribution in required.values()):
                raise RuntimeError(
                    f"{label} x {dataset}: observed candidate-shape evidence is absent"
                )
            rows.append(
                f"{value} & {dataset.upper()} & "
                f"{_fmt_percent(summary['candidate_recall'])} & "
                f"{required['base_candidate_k']['median']:.1f} & "
                f"{required['coverage_rescued_k']['median']:.1f} & "
                f"{required['final_candidate_k']['median']:.1f} & "
                f"{required['target_count']['median']:.1f} \\\\"
            )
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{RQ4 offline candidate recall and observed game shape across the pre-registered candidate cap. Base, rescued, final $K$, and target $N$ are medians over 90 cases. Recall is an offline upper bound and remains separate from attribution accuracy.}",
            r"\label{tab:candidate-k-recall}",
            r"\begin{tabular}{rlrrrrr}",
            r"\toprule Cap & System & Recall (\%) & Base $K$ & Rescued & Final $K$ & Targets $N$ \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )


def _fault_breakdown_tables(main: dict, protocol: dict) -> str:
    indexed = _index_summaries(main["summaries"])
    tables = []
    for method in ["merit_full", *protocol["baselines"]]:
        rows = []
        for dataset in protocol["datasets"]:
            summaries = [
                indexed[(label, dataset)]
                for label in _method_labels(method, protocol)
            ]
            if any(
                row["combination_status"] in {"infeasible", "inapplicable"}
                for row in summaries
            ):
                rows.append(
                    f"{dataset.upper()} & N/A & N/A & N/A & N/A & N/A & N/A \\\\"
                )
                continue
            fault_names = sorted(summaries[0]["fault_breakdown"])
            for fault in fault_names:
                cells = [row["fault_breakdown"][fault] for row in summaries]
                n = cells[0]["n"]
                rows.append(
                    f"{dataset.upper()} & {fault} & {n} & "
                    f"{_fmt_percent(mean(cell['successes'] / cell['n'] for cell in cells))} & "
                    f"{_fmt_percent(mean(cell['AC@1'] for cell in cells))} & "
                    f"{_fmt_percent(mean(cell['AC@3'] for cell in cells))} & "
                    f"{_fmt_percent(mean(cell['Avg@5'] for cell in cells))} \\\\"
                )
        tables.append(
            "\n".join(
                [
                    r"\begin{table*}[t]",
                    r"\centering\scriptsize",
                    rf"\caption{{Descriptive fault-type breakdown for {_display_name(method)}. Each system--fault cell has 15 cases per seed; no fault-level inferential test is performed. Stochastic values are means across three seeds.}}",
                    rf"\label{{tab:fault-{method.replace('_', '-')}}}",
                    r"\begin{tabular}{llrrrrr}",
                    r"\toprule System & Fault & $n$/seed & Exec. & AC@1 & AC@3 & Avg@5 \\",
                    r"\midrule",
                    *rows,
                    r"\bottomrule",
                    r"\end{tabular}",
                    r"\end{table*}",
                ]
            )
        )
    return "\n\n".join(tables)


def render_tables(validated: dict, protocol: dict, source: Path) -> str:
    main = validated["jobs"]["main"]
    available_methods = {summary["method"] for summary in main["summaries"]}
    core_ablation_methods = tuple(
        method for method in CORE_ABLATION_METHODS if method in available_methods
    )
    core_stability_methods = {
        method
        for method in {
            "merit_no_trace",
            "merit_graph_first_case_pre_fault",
            "merit_trace_dropout_50",
            "merit_trace_noise_30",
            "merit_trace_edge_noise_30",
        }
        if method in available_methods
    }
    return "\n\n".join(
        [
            "% Generated core results from the gated reproduction root",
            r"\subsection{RQ1: Effectiveness}",
            _effectiveness_table(main, protocol),
            r"\subsection{RQ2: Component Contributions}",
            _ablation_table(
                main,
                protocol,
                methods=core_ablation_methods,
                caption=(
                    "RQ2 core controlled variants, service AC@1 (\\%). "
                    "No trace keeps the complete Figure~1 pipeline but extracts "
                    "metric evidence only and supplies no trace-derived input "
                    "to downstream stages. "
                    "Each cell retains all 90 cases; macro weights systems equally."
                ),
            ),
            _faithfulness_table(main, protocol),
            _explanation_stability_table(
                main,
                aggregate=True,
                methods=core_stability_methods,
            ),
            _case_study_table(main),
            r"\subsection{RQ3: Equivalence and Efficiency}",
            _equivalence_table(validated["equivalence"]),
            _efficiency_table(main, protocol),
            r"\subsection{RQ4: Robustness}",
            _missing_trace_diagnostic_table(main),
            _graph_sensitivity_table(main),
            _compact_paired_statistics_table(main),
            r"\noindent Complete secondary metrics, failure sensitivity, "
            r"per-system inference, stage timing, candidate recall, and all "
            r"pre-registered robustness curves are retained in the generated "
            r"companion file \texttt{extended-results.tex}.",
        ]
    ) + "\n"


def render_extended_results(validated: dict, protocol: dict) -> str:
    """Render complete gated evidence for the anonymous companion artifact."""

    main = validated["jobs"]["main"]
    return "\n\n".join(
        [
            "% Generated complete evidence supplement; do not edit manually.",
            r"\section{Complete Gated Results}",
            _secondary_metrics_table(main, protocol),
            _stochastic_seed_variation_table(main, protocol),
            _failure_sensitivity_table(main, protocol),
            _paired_statistics_tables(main, pooled_only=False),
            _ablation_table(
                main,
                protocol,
                label="tab:all-ablations",
                caption=(
                    "All controlled variants, service AC@1 (\\%). Each cell "
                    "retains all 90 cases; macro weights systems equally."
                ),
            ),
            _candidate_recall_table(main, protocol),
            _explanation_stability_table(main, aggregate=False),
            _stage_table(main, protocol),
            _stage_share_table(main, protocol),
            _robustness_table(validated["selections"]),
            _parameter_sweep_curves(validated, protocol),
            _candidate_k_recall_table(validated, protocol),
            _robustness_curves(validated, protocol),
        ]
    ) + "\n"


def render_statistical_details(validated: dict) -> str:
    return "\n\n".join(
        [
            "% Generated full per-system and compatible-pooled paired statistics",
            _paired_statistics_tables(
                validated["jobs"]["main"], pooled_only=False
            ),
        ]
    ) + "\n"


def render_fault_breakdown(validated: dict, protocol: dict) -> str:
    return "\n\n".join(
        [
            "% Generated descriptive fault-type supplement",
            _fault_breakdown_tables(validated["jobs"]["main"], protocol),
        ]
    ) + "\n"


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--reproduction-root", type=Path, required=True)
    parser.add_argument(
        "--protocol", type=Path, default=Path("experiment/protocol.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _write_idempotent(path: Path, content: str) -> None:
    """Create a gated output, or accept an identical prior retry output."""

    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(
                f"existing generated file differs from fresh gated render: {path}"
            )
        return
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def main(argv=None) -> int:
    args = parse_args(argv)
    protocol = _load(args.protocol)
    validated = validate_reproduction(args.reproduction_root, protocol)
    statistics_output = args.output.with_name("paired-statistics-details.tex")
    fault_output = args.output.with_name("fault-breakdown.tex")
    headline_output = args.output.with_name("headline-macros.tex")
    extended_output = args.output.with_name("extended-results.tex")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_idempotent(
        args.output,
        render_tables(validated, protocol, args.reproduction_root),
    )
    _write_idempotent(
        statistics_output,
        render_statistical_details(validated),
    )
    _write_idempotent(
        fault_output,
        render_fault_breakdown(validated, protocol),
    )
    _write_idempotent(
        headline_output,
        render_headline_macros(validated, protocol),
    )
    _write_idempotent(
        extended_output,
        render_extended_results(validated, protocol),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
