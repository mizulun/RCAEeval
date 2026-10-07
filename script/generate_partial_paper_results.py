"""Generate a traceable working-paper result section from a baseline subset.

This command is deliberately separate from ``generate_paper_tables.py``.  It
accepts only complete cells after the declared baseline exclusions, labels the
output as provisional, and does not relax or replace the final reporting gate.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from statistics import mean, median

import numpy as np

try:
    from script.generate_paper_tables import (
        DISPLAY_NAMES,
        _ablation_table,
        _efficiency_table,
        _effectiveness_table,
        _equivalence_table,
        _index_summaries,
        _method_cells,
        _method_labels,
        _paired_statistics_tables,
        _parameter_sweep_curves,
        _stage_table,
        _validate_diagnostic_checks,
        _validate_paired_statistics,
    )
    from script.reproduce_paper import ExperimentJob, build_jobs
    from script.run_paper_experiments import resolve_method
except ModuleNotFoundError:  # direct script execution
    from generate_paper_tables import (
        DISPLAY_NAMES,
        _ablation_table,
        _efficiency_table,
        _effectiveness_table,
        _equivalence_table,
        _index_summaries,
        _method_cells,
        _method_labels,
        _paired_statistics_tables,
        _parameter_sweep_curves,
        _stage_table,
        _validate_diagnostic_checks,
        _validate_paired_statistics,
    )
    from reproduce_paper import ExperimentJob, build_jobs
    from run_paper_experiments import resolve_method


PARTIAL_BASELINE_EXCLUSIONS = frozenset({"causalrca", "circa"})

BARO_BENCHMARK_FAULTS = ("CPU", "MEM", "DISK", "SOCKET", "DELAY", "LOSS")
ZH_EFFECTIVENESS_METRICS = ("AC@1", "AC@3", "AC@5", "Avg@5")

PARTIAL_RQ2_METHODS = (
    "merit_full",
    "merit_no_trace",
    "merit_no_metric",
    "merit_no_trace_topology",
    "merit_no_coverage",
    "merit_no_victim_demotion",
    "merit_no_coverage_no_victim",
    "merit_no_coverage_parent_protection",
    "merit_no_metric_type_dominance",
    "merit_no_early_full_fusion",
    "merit_trace_latency_only",
    "merit_trace_error_only",
    "merit_trace_lag0",
    "merit_trace_time_shuffle",
    "merit_trace_random_edges",
    "merit_trace_only",
    "merit_direct_service_evidence",
    "merit_direct_service_trace_evidence",
    "merit_individual_attribution",
)


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _baro_benchmark_overall(
    results: list[dict], protocol: dict
) -> dict[tuple[str, str], float]:
    """Aggregate the accepted BARO fault-type results for the Chinese main table."""

    rows = [row for row in results if row.get("method") == "baro"]
    indexed = {row.get("dataset"): row for row in rows}
    if len(indexed) != len(rows) or set(indexed) != set(protocol["datasets"]):
        raise RuntimeError("the BARO aggregate JSON must contain one row per dataset")

    overall = {}
    for dataset in protocol["datasets"]:
        row = indexed[dataset]
        if row.get("status") != "OK":
            raise RuntimeError(f"{dataset}: BARO benchmark result is not complete")
        for metric in ZH_EFFECTIVENESS_METRICS:
            values = []
            for fault in BARO_BENCHMARK_FAULTS:
                value = row.get(f"{metric}-{fault}")
                if not isinstance(value, (int, float)):
                    raise RuntimeError(
                        f"{dataset}: BARO benchmark lacks {metric}-{fault}"
                    )
                values.append(float(value))
            overall[(dataset, metric)] = mean(values)
    return overall


def _partial_protocol(protocol: dict) -> dict:
    result = deepcopy(protocol)
    result["baselines"] = [
        method
        for method in protocol["baselines"]
        if method not in PARTIAL_BASELINE_EXCLUSIONS
    ]
    result["stochastic_methods"] = [
        method
        for method in protocol["stochastic_methods"]
        if method not in PARTIAL_BASELINE_EXCLUSIONS
    ]
    return result


def _partial_methods(protocol: dict) -> tuple[str, ...]:
    main = build_jobs(protocol, "confirmatory")[0]
    return tuple(
        method
        for method in main.methods
        if resolve_method(method).base_method not in PARTIAL_BASELINE_EXCLUSIONS
    )


def _expected_partial_status(method: str, dataset: str, protocol: dict) -> str:
    """Return the frozen status for a retained method--dataset cell."""

    base_method = resolve_method(method).base_method
    for key in (f"{method}:{dataset}", f"{base_method}:{dataset}"):
        if key in protocol.get("modality_inapplicable", {}):
            return "inapplicable"
        if key in protocol.get("infeasible", {}):
            return "infeasible"
    return "completed"


def _validate_manifest(reproduction_root: Path, protocol: dict) -> None:
    manifest = _load(reproduction_root / "runs" / "main" / "manifest.json")
    if not manifest.get("formal_manifest") or manifest.get("git", {}).get("dirty"):
        raise RuntimeError("partial draft requires the clean formal manifest")
    if manifest.get("protocol_version") != protocol["protocol_version"]:
        raise RuntimeError("partial draft manifest/protocol version mismatch")
    if not manifest.get("output_path"):
        raise RuntimeError("partial draft manifest output path is absent")
    for dataset in protocol["datasets"]:
        record = manifest.get("datasets", {}).get(dataset, {})
        if (
            record.get("case_count") != protocol["expected_cases_per_dataset"]
            or record.get("expected_case_count")
            != protocol["expected_cases_per_dataset"]
            or record.get("checksum_mode") != "full"
            or not record.get("cryptographic_content_checksum")
            or not record.get("tree_sha256")
        ):
            raise RuntimeError(f"{dataset}: incomplete formal dataset manifest")


def validate_partial_analysis(
    reproduction_root: Path, analysis_dir: Path, protocol: dict
) -> dict:
    """Validate completed cells without pretending that the full gate passed."""

    _validate_manifest(reproduction_root, protocol)
    methods = _partial_methods(protocol)
    summaries = _load(analysis_dir / "dataset_summaries.json")
    aggregates = _load(analysis_dir / "aggregate_summaries.json")
    paired = _load(analysis_dir / "paired_statistics.json")
    diagnostics = _load(analysis_dir / "diagnostic_checks.json")

    indexed = _index_summaries(summaries)
    expected = {
        (method, dataset)
        for method in methods
        for dataset in protocol["datasets"]
    }
    if set(indexed) != expected:
        missing = sorted(expected - set(indexed))
        extra = sorted(set(indexed) - expected)
        raise RuntimeError(
            f"partial evaluated-cell mismatch; missing={missing}, extra={extra}"
        )
    for (method, dataset), summary in indexed.items():
        expected_status = _expected_partial_status(method, dataset, protocol)
        if summary.get("combination_status") != expected_status:
            raise RuntimeError(
                f"{method} x {dataset}: expected {expected_status} partial cell"
            )
        if expected_status in {"infeasible", "inapplicable"}:
            continue
        if (
            summary.get("n_expected") != protocol["expected_cases_per_dataset"]
            or summary.get("n_missing") != 0
        ):
            raise RuntimeError(f"{method} x {dataset}: incomplete case artifacts")

    aggregate_methods = {row["method"] for row in aggregates}
    if aggregate_methods != set(methods):
        raise RuntimeError("partial aggregate-method coverage mismatch")

    partial_protocol = _partial_protocol(protocol)
    partial_job = ExperimentJob("main", methods)
    _validate_paired_statistics(partial_job, paired, partial_protocol)
    _validate_diagnostic_checks(partial_job, diagnostics, partial_protocol)

    selections = {}
    for parameter in (
        "victim_floor",
        "trace_topology_conf",
        "coverage_conf",
        "exact_candidate_k",
    ):
        selection = _load(analysis_dir / "selections" / parameter / "selection.json")
        if selection.get("protocol_version") != protocol["protocol_version"]:
            raise RuntimeError(f"{parameter}: selection/protocol version mismatch")
        selections[parameter] = selection

    equivalence = _load(
        analysis_dir / "banzhaf-equivalence" / "banzhaf_equivalence.json"
    )
    if equivalence.get("protocol_version") != protocol["protocol_version"]:
        raise RuntimeError("Banzhaf benchmark/protocol version mismatch")
    if not equivalence.get("summaries") or not all(
        row.get("equivalent") for row in equivalence["summaries"]
    ):
        raise RuntimeError("Banzhaf equivalence gate did not pass")

    baro_crosscheck = _load(analysis_dir / "crosschecks" / "baro.json")
    if baro_crosscheck.get("protocol_version") != protocol["protocol_version"]:
        raise RuntimeError("BARO cross-check/protocol version mismatch")

    return {
        "jobs": {
            "main": {
                "summaries": summaries,
                "aggregates": aggregates,
                "paired_statistics": paired,
                "diagnostic_checks": diagnostics,
            }
        },
        "equivalence": equivalence,
        "selections": selections,
        "baro_crosscheck": baro_crosscheck,
    }


def _method_macro_metric(
    main: dict, method: str, protocol: dict, metric: str
) -> float | None:
    aggregates = {row["method"]: row for row in main["aggregates"]}
    labels = _method_labels(method, protocol)
    records = [aggregates[label] for label in labels]
    if not all(record["all_datasets_completed"] for record in records):
        return None
    return mean(record["macro_equal_system_weight"][metric] for record in records)


def _method_macro(main: dict, method: str, protocol: dict) -> float | None:
    return _method_macro_metric(main, method, protocol, "AC@1")


def _merit_minus_comparator(record: dict) -> tuple[float, float, float, float]:
    """Return MERIT-minus-comparator AC@1 effect, CI, and adjusted p-value."""

    ac1 = next(
        row for row in record["cluster_bootstrap"] if row["metric"] == "AC@1"
    )
    raw_effect = ac1.get(
        "effect_pp_b_minus_a",
        ac1.get("effect_pp_seed_mean_minus_reference"),
    )
    low, high = ac1["ci95_pp"]
    return (
        -float(raw_effect),
        -float(high),
        -float(low),
        float(record["cluster_permutation_ac1"]["holm_adjusted_p_value"]),
    )


def _rq1_draft_text(
    main: dict, protocol: dict, baro_crosscheck: dict | None = None
) -> str:
    merit = _method_macro(main, "merit_full", protocol)
    complete = [
        (method, value)
        for method in protocol["baselines"]
        if (value := _method_macro(main, method, protocol)) is not None
    ]
    best_method, best_value = max(complete, key=lambda item: item[1])
    crosscheck_note = ""
    if baro_crosscheck is not None:
        passed = sum(bool(row.get("passed")) for row in baro_crosscheck["rows"])
        total = len(baro_crosscheck["rows"])
        crosscheck_note = (
            " The local BARO rankings are retained, but their RCAEval reference "
            f"cross-check passes only {passed}/{total} RE2-TT fault-type rows; "
            "this blocks a formal BARO reproducibility claim."
            if not baro_crosscheck.get("passed")
            else " The local BARO reference cross-check passes."
        )
    paired_note = ""
    paired_records = [
        record
        for record in main.get("paired_statistics", [])
        if record.get("method") == best_method
    ]
    if paired_records:
        pooled = next(record for record in paired_records if record["scope"] == "pooled")
        effect, low, high, adjusted_p = _merit_minus_comparator(pooled)
        system_effects = {
            record["scope"]: _merit_minus_comparator(record)[0]
            for record in paired_records
            if record["scope"] != "pooled"
        }
        p_text = "<.001" if adjusted_p < 0.001 else f"={adjusted_p:.3f}"
        paired_note = (
            f" Against {DISPLAY_NAMES[best_method]}, the paired pooled effect is "
            f"{effect:+.1f} points (95\\% CI [{low:+.1f}, {high:+.1f}], "
            f"$p_H{p_text}$). The system effects are "
            f"{system_effects['re2-ob']:+.1f} (OB), "
            f"{system_effects['re2-ss']:+.1f} (SS), and "
            f"{system_effects['re2-tt']:+.1f} (TT), so the retained evidence "
            "does not support a system-consistent superiority claim."
        )
    return (
        f"Across the currently complete three-system cells, \\method{{}} obtains "
        f"{100 * merit:.1f}\\% macro AC@1. The strongest retained baseline with "
        f"complete three-system coverage is {DISPLAY_NAMES[best_method]} at "
        f"{100 * best_value:.1f}\\%, an "
        f"{100 * (merit - best_value):.1f}-percentage-point difference. The "
        "per-system table and paired intervals determine whether this aggregate "
        "trend is consistent across compatible cases. CausalRCA and CIRCA are "
        "excluded from this interim analysis; its RQ1 Holm adjustment therefore "
        "applies only to the retained comparator family and is not the final "
        "pre-registered RQ1 correction."
        + paired_note
        + crosscheck_note
    )


def _rq2_draft_text(main: dict) -> str:
    pooled = {
        record["method"]: record
        for record in main["paired_statistics"]
        if record["comparison_family"] == "rq2_ablations"
        and record["scope"] == "pooled"
    }

    def merit_minus(method: str) -> tuple[float, float, float]:
        row = next(
            item
            for item in pooled[method]["cluster_bootstrap"]
            if item["metric"] == "AC@1"
        )
        low, high = row["ci95_pp"]
        return (
            -float(row["effect_pp_b_minus_a"]),
            -float(high),
            -float(low),
        )

    significant = [
        method
        for method, record in pooled.items()
        if record["cluster_permutation_ac1"]["holm_adjusted_p_value"] < 0.05
    ]
    significant_text = (
        ", ".join(DISPLAY_NAMES[method] for method in significant)
        if significant
        else "none"
    )
    trace_effect, trace_low, trace_high = merit_minus("merit_trace_only")
    trace_adjusted_p = float(
        pooled["merit_trace_only"]["cluster_permutation_ac1"][
            "holm_adjusted_p_value"
        ]
    )
    trace_p_text = (
        "<.001" if trace_adjusted_p < 0.001 else f"={trace_adjusted_p:.3f}"
    )
    named_effects = [
        merit_minus(method)[0]
        for method in (
            "merit_no_trace",
            "merit_no_trace_topology",
            "merit_no_coverage",
            "merit_no_victim_demotion",
            "merit_individual_attribution",
        )
    ]
    return (
        "The full current-default configuration exceeds trace-only ranking by "
        f"{trace_effect:.1f} percentage points pooled (cluster-bootstrap 95\\% "
        f"CI [{trace_low:+.1f}, {trace_high:+.1f}], $p_H{trace_p_text}$). "
        "This descriptive contrast does not survive the frozen family-wise "
        "correction. Its "
        "differences from no trace, no trace topology, no coverage, no victim "
        "demotion, and singleton attribution are "
        f"{named_effects[0]:+.1f}, {named_effects[1]:+.1f}, "
        f"{named_effects[2]:+.1f}, {named_effects[3]:+.1f}, and "
        f"{named_effects[4]:+.1f} points, respectively. After the frozen RQ2 "
        f"cluster sign-flip tests and Holm correction, comparisons below "
        f"$0.05$ are: {significant_text}."
    )


def _rq3_draft_text(validated: dict) -> str:
    summaries = validated["equivalence"]["summaries"]
    max_error = max(float(row["max_absolute_error"]) for row in summaries)
    largest = max(summaries, key=lambda row: row["k"])
    return (
        f"Closed form and enumeration agree to a maximum absolute error of "
        f"{max_error:.2e} over the frozen validation grid. At K={largest['k']}, "
        "the median measured speedup is "
        f"{largest['median_speedup_bruteforce_over_closed_form']:.1f}$\\times$. "
        "This speedup applies only to the attribution kernel and does not imply "
        "the same end-to-end acceleration. "
        "The repeated end-to-end and stage measurements below use the same "
        "current-default configuration as RQ1 and RQ2."
    )


def _rq4_draft_text(validated: dict) -> str:
    main = validated["jobs"]["main"]
    check = main["diagnostic_checks"]["missing_trace"]
    selections = validated["selections"]
    indexed = _index_summaries(main["summaries"])
    k_recalls = {
        k: mean(
            indexed[(f"merit_k{k}", dataset)]["candidate_recall"]
            for dataset in ("re2-ob", "re2-ss", "re2-tt")
        )
        for k in (3, 6, 9, 12, 15)
    }
    selected_k = int(selections["exact_candidate_k"]["selected_value"])
    return (
        f"On RE2-SS, full and no-trace runs match exactly for all "
        f"{check['exact_service_rank_matches']}/{check['n']} service rankings "
        "and all metric rankings, supporting graceful degradation for this "
        "specific missing-trace dataset. The frozen selection rule chooses "
        f"victim floor {selections['victim_floor']['selected_value']}, trace "
        f"confidence {selections['trace_topology_conf']['selected_value']}, "
        f"coverage confidence {selections['coverage_conf']['selected_value']}, "
        f"and candidate K={selected_k}. Candidate recall at the selected K is "
        f"{100 * k_recalls[selected_k]:.1f}\\%, compared with a maximum of "
        f"{100 * max(k_recalls.values()):.1f}\\% on the grid; this distinction "
        "prevents attributing candidate-set effects to the Banzhaf score. "
        "Because these differ from the development defaults and their combined "
        "configuration has not yet been rerun, the current effectiveness and "
        "ablation values remain working-draft evidence. Delay and window "
        "sensitivity are also pending."
    )


def _zh_rq1_draft_text(
    main: dict, protocol: dict, baro_results: list[dict]
) -> str:
    indexed = _index_summaries(main["summaries"])
    baro_overall = _baro_benchmark_overall(baro_results, protocol)
    merit_ac1 = _method_macro_metric(main, "merit_full", protocol, "AC@1")
    merit_avg5 = _method_macro_metric(main, "merit_full", protocol, "Avg@5")
    baro_ac1 = mean(
        baro_overall[(dataset, "AC@1")] for dataset in protocol["datasets"]
    )
    baro_avg5 = mean(
        baro_overall[(dataset, "Avg@5")] for dataset in protocol["datasets"]
    )

    def metric(method: str, dataset: str, name: str) -> float:
        if method == "baro":
            return baro_overall[(dataset, name)]
        labels = _method_labels(method, protocol)
        return mean(indexed[(label, dataset)]["overall"][name] for label in labels)

    return (
        "表~\\ref{tab:effectiveness-zh} 先比較各方法在三個 RE2 系統上的排序"
        f"準確率。\\method{{}} 的三系統 macro AC@1 為 {100 * merit_ac1:.1f}\\%，"
        f"BARO 為 {100 * baro_ac1:.1f}\\%，相差 "
        f"{100 * (merit_ac1 - baro_ac1):.1f} 個百分點；macro AVG@5 則分別為 "
        f"{100 * merit_avg5:.1f}\\% 與 {100 * baro_avg5:.1f}\\%，相差 "
        f"{100 * (merit_avg5 - baro_avg5):.1f} 個百分點。逐系統來看，"
        f"\\method{{}} 在 SS、OB、TT 的 AVG@5 分別為 "
        f"{100 * metric('merit_full', 're2-ss', 'Avg@5'):.1f}\\%、"
        f"{100 * metric('merit_full', 're2-ob', 'Avg@5'):.1f}\\% 與 "
        f"{100 * metric('merit_full', 're2-tt', 'Avg@5'):.1f}\\%，皆為表中最高。"
        "AC@1 則呈現系統差異：RCD 在兩個小型系統領先，\\method{} 在複雜的 "
        f"TT 達 {100 * metric('merit_full', 're2-tt', 'AC@1'):.1f}\\%，高於"
        f"次佳 BARO 的 {100 * metric('baro', 're2-tt', 'AC@1'):.1f}\\%。"
        "因此目前證據支持較穩定的前五名排序品質，但不支持在每個系統的 Top-1 "
        "都優於所有 baseline。BARO 數值取自 \\texttt{benchmark\\_results.json}，"
        "各系統皆對六種 fault type 等權平均。"
    )


def _zh_effectiveness_table(
    main: dict, protocol: dict, baro_results: list[dict]
) -> str:
    indexed = _index_summaries(main["summaries"])
    baro_overall = _baro_benchmark_overall(baro_results, protocol)
    methods = ["merit_full", *protocol["baselines"]]
    metrics = ZH_EFFECTIVENESS_METRICS
    datasets = [
        dataset
        for dataset in ("re2-ss", "re2-ob", "re2-tt", *protocol["datasets"])
        if dataset in protocol["datasets"]
    ]
    datasets = list(dict.fromkeys(datasets))

    values: dict[tuple[str, str, str], float | None] = {}
    for dataset in datasets:
        for method in methods:
            summaries = [
                indexed[(label, dataset)] for label in _method_labels(method, protocol)
            ]
            for metric in metrics:
                if method == "baro":
                    value = baro_overall[(dataset, metric)]
                elif any(
                    summary["combination_status"] in {"infeasible", "inapplicable"}
                    for summary in summaries
                ):
                    value = None
                else:
                    value = mean(summary["overall"][metric] for summary in summaries)
                values[(dataset, method, metric)] = value

    def formatted(dataset: str, method: str, metric: str) -> str:
        value = values[(dataset, method, metric)]
        if value is None:
            return "--"
        ordered = sorted(
            {
                candidate
                for candidate_method in methods
                if (
                    candidate := values[(dataset, candidate_method, metric)]
                )
                is not None
            },
            reverse=True,
        )
        rendered = f"{100 * value:.1f}"
        if np.isclose(value, ordered[0]):
            return rf"\textbf{{{rendered}}}"
        if len(ordered) > 1 and np.isclose(value, ordered[1]):
            return rf"\underline{{{rendered}}}"
        return rendered

    dataset_names = {
        "re2-ss": "RE2-SS",
        "re2-ob": "RE2-OB",
        "re2-tt": "RE2-TT",
    }
    method_names = {
        "merit_full": r"\method{}",
        "baro": "BARO",
        "microrank": "MicroRank",
        "tracerca": "TraceRCA",
        "rcd": "RCD",
        "pc_pagerank": "PC+PR",
    }
    rows = []
    for index, method in enumerate(methods):
        cells = [
            formatted(dataset, method, metric)
            for dataset in datasets
            for metric in metrics
        ]
        rows.append(
            f"{method_names.get(method, DISPLAY_NAMES[method])} & "
            f"{' & '.join(cells)} \\\\"
        )
        if index == 0:
            rows.append(r"\midrule")

    group_header = " & ".join(
        rf"\multicolumn{{4}}{{c}}{{{dataset_names.get(dataset, dataset.upper())}}}"
        for dataset in datasets
    )
    cmidrules = " ".join(
        rf"\cmidrule(lr){{{2 + 4 * index}-{5 + 4 * index}}}"
        for index in range(len(datasets))
    )
    metric_header = " & ".join("AC@1 & AC@3 & AC@5 & AVG@5" for _ in datasets)
    return "\n".join(
        [
            r"\begin{table*}[!t]",
            r"\centering",
            r"\fontsize{10}{12}\selectfont",
            r"\setlength{\tabcolsep}{1.2pt}",
            r"\caption{各方法在 RCAEval RE2 的服務層級準確率（\%）。\textbf{粗體}與\underline{底線}分別表示同一資料集、同一指標的最佳與次佳結果。BARO 對六種 fault type 等權平均；RCD 為三個 seeds 的平均；PC+PR 表示 PC+PageRank。RE2-SS 未提供原始 traces，因此 MicroRank 與 TraceRCA 不適用。}",
            r"\label{tab:effectiveness-zh}",
            rf"\begin{{tabular}}{{l*{{{len(datasets)}}}{{rrrr}}}}",
            r"\toprule",
            f"方法 & {group_header} \\\\ ",
            cmidrules,
            f" & {metric_header} \\\\ ",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )


def _zh_rq2_draft_text(main: dict) -> str:
    pooled = {
        record["method"]: record
        for record in main["paired_statistics"]
        if record["comparison_family"] == "rq2_ablations"
        and record["scope"] == "pooled"
    }

    def comparator_minus_merit(method: str) -> tuple[float, float, float, float]:
        row = next(
            item
            for item in pooled[method]["cluster_bootstrap"]
            if item["metric"] == "AC@1"
        )
        low, high = row["ci95_pp"]
        return (
            float(row["effect_pp_b_minus_a"]),
            float(low),
            float(high),
            float(
                pooled[method]["cluster_permutation_ac1"][
                    "holm_adjusted_p_value"
                ]
            ),
        )

    direct_effect, direct_low, direct_high, direct_p = comparator_minus_merit(
        "merit_direct_service_trace_evidence"
    )
    direct_p_text = "<.001" if direct_p < 0.001 else f"={direct_p:.3f}"
    direct_inference = (
        "且通過 family-wise Holm 校正。"
        if direct_p < 0.05
        else "但未通過 family-wise Holm 校正，故只作描述性解讀。"
    )

    def compatible_macro(method: str, metric: str = "AC@1") -> float:
        values = [
            100 * record["overall"][metric]
            for record in main["summaries"]
            if record["method"] == method
            and record["combination_status"] == "completed"
        ]
        return mean(values)

    full_macro = compatible_macro("merit_full")
    no_trace_macro = compatible_macro("merit_no_trace")
    no_metric_macro = compatible_macro("merit_no_metric")
    direct_sum_macro = compatible_macro("merit_direct_service_trace_evidence")
    full_topk_macros = [
        compatible_macro("merit_full", metric)
        for metric in ("AC@3", "AC@5", "Avg@5")
    ]
    direct_topk_macros = [
        compatible_macro("merit_direct_service_trace_evidence", metric)
        for metric in ("AC@3", "AC@5", "Avg@5")
    ]
    return (
        "表~\\ref{tab:ablation-zh} 將兩類對照分開：無 trace／無 metric "
        "仍保留 MERIT 的候選建構、causal game 與 Banzhaf 歸因。其中無 trace "
        "在 evidence extraction 只產生 metric evidence，後續步驟不使用 call graph、"
        "trace edge、trace score 或任何 trace-derived 資訊，但整體方法流程不變；"
        "直接分數和排序則"
        "只執行圖一前兩階段，在 evidence extraction 後將跨服務正規化的 metric "
        "與 trace evidence 等權相加，再直接產生服務排名。"
        f"三系統 macro AC@1 中，完整組態、無 trace 與直接分數和排序分別為 "
        f"{full_macro:.1f}\\%、{no_trace_macro:.1f}\\% 與 {direct_sum_macro:.1f}\\%；"
        f"無 metric 在相容 OB／TT 的 macro 為 {no_metric_macro:.1f}\\%。"
        f"直接分數和的三系統 macro AC@3、AC@5 與 AVG@5 分別為 "
        f"{direct_topk_macros[0]:.1f}\\%、{direct_topk_macros[1]:.1f}\\% 與 "
        f"{direct_topk_macros[2]:.1f}\\%，低於完整組態的 "
        f"{full_topk_macros[0]:.1f}\\%、{full_topk_macros[1]:.1f}\\% 與 "
        f"{full_topk_macros[2]:.1f}\\%；因此點估計呈現 Top-1 與前五名排序品質的取捨。"
        f"相較於完整組態，直接分數和排序之 pooled AC@1 提高 "
        f"{direct_effect:.1f} 個百分點（95\\% CI [{direct_low:+.1f}, "
        f"{direct_high:+.1f}]，cluster sign-flip Holm-adjusted "
        f"$p{direct_p_text}$），{direct_inference}"
        "此結果不支持將 AC@1 增益歸因於 causal game 或 Banzhaf。RE2-SS 無 "
        "trace-derived evidence，因此直接分數和退化為 metric evidence 排序；"
        "無 metric 列為 N/A，其 macro 只對 OB／TT 等權平均。"
    )


def _zh_ablation_table(main: dict, protocol: dict) -> str:
    indexed = _index_summaries(main["summaries"])
    names = {
        "merit_full": r"完整 \method{}",
        "merit_no_trace": "無 trace",
        "merit_no_metric": "無 metric",
        "merit_direct_service_trace_evidence": "直接分數和",
    }
    metrics = ZH_EFFECTIVENESS_METRICS
    datasets = [
        dataset
        for dataset in ("re2-ss", "re2-ob", "re2-tt", *protocol["datasets"])
        if dataset in protocol["datasets"]
    ]
    datasets = list(dict.fromkeys(datasets))
    values: dict[tuple[str, str, str], float | None] = {}
    for dataset in datasets:
        for method in names:
            summary = indexed[(method, dataset)]
            for metric in metrics:
                values[(dataset, method, metric)] = (
                    None
                    if summary["combination_status"] in {"infeasible", "inapplicable"}
                    else summary["overall"][metric]
                )

    def formatted(dataset: str, method: str, metric: str) -> str:
        value = values[(dataset, method, metric)]
        if value is None:
            return "N/A"
        ordered = sorted(
            {
                candidate
                for candidate_method in names
                if (
                    candidate := values[(dataset, candidate_method, metric)]
                )
                is not None
            },
            reverse=True,
        )
        rendered = f"{100 * value:.1f}"
        if np.isclose(value, ordered[0]):
            return rf"\textbf{{{rendered}}}"
        if len(ordered) > 1 and np.isclose(value, ordered[1]):
            return rf"\underline{{{rendered}}}"
        return rendered

    rows = []
    for method in names:
        cells = [
            formatted(dataset, method, metric)
            for dataset in datasets
            for metric in metrics
        ]
        rows.append(f"{names[method]} & {' & '.join(cells)} \\\\")

    dataset_names = {
        "re2-ss": "RE2-SS",
        "re2-ob": "RE2-OB",
        "re2-tt": "RE2-TT",
    }
    group_header = " & ".join(
        rf"\multicolumn{{4}}{{c}}{{{dataset_names.get(dataset, dataset.upper())}}}"
        for dataset in datasets
    )
    cmidrules = " ".join(
        rf"\cmidrule(lr){{{2 + 4 * index}-{5 + 4 * index}}}"
        for index in range(len(datasets))
    )
    metric_header = " & ".join("AC@1 & AC@3 & AC@5 & AVG@5" for _ in datasets)
    return "\n".join(
        [
            r"\begin{table*}[!t]",
            r"\centering",
            r"\fontsize{10}{12}\selectfont",
            r"\setlength{\tabcolsep}{1.2pt}",
            r"\caption{\method{} 輸入模態移除與直接證據分數和排序的服務層級準確率（\%）。\textbf{粗體}與\underline{底線}分別表示同一資料集、同一指標的最佳與次佳結果。無 trace 保留圖一完整流程，但第二階段只擷取 metric evidence，後續不使用 call graph、trace edge、trace score 或其他 trace-derived 資訊。直接分數和只執行圖一前兩階段，將跨服務正規化後的 metric 與 trace evidence 等權相加並直接排序；各適用 cell 均含 90 cases，RE2-SS 的無 metric 組態因缺少 traces 而不適用。}",
            r"\label{tab:ablation-zh}",
            rf"\begin{{tabular}}{{l*{{{len(datasets)}}}{{rrrr}}}}",
            r"\toprule",
            f"組態 & {group_header} \\\\ ",
            cmidrules,
            f" & {metric_header} \\\\ ",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )


def _zh_rq3_draft_text(validated: dict, protocol: dict) -> str:
    equivalence = validated["equivalence"]["summaries"]
    max_error = max(float(row["max_absolute_error"]) for row in equivalence)
    largest = max(equivalence, key=lambda row: row["k"])
    indexed = _index_summaries(validated["jobs"]["main"]["summaries"])
    repetitions = [
        "merit_full",
        *(f"merit_full_rep{i}" for i in range(2, protocol["runtime_repetitions"] + 1)),
    ]
    dataset_parts = []
    for dataset in protocol["datasets"]:
        rows = [indexed[(method, dataset)] for method in repetitions]
        totals = [row["task_observed_wall_seconds"] for row in rows]
        per_case = [row["timing_seconds"]["end_to_end"]["median"] for row in rows]
        peak_mib = max(row["peak_rss_bytes"] for row in rows) / 2**20
        dataset_parts.append(
            f"{dataset.upper()} 為 {median(totals):.1f} s "
            f"[{np.percentile(totals, 25):.1f}, {np.percentile(totals, 75):.1f}]、"
            f"per-case median {median(per_case):.2f} s、peak RSS {peak_mib:.1f} MiB"
        )
    return (
        f"閉式解與窮舉在凍結驗證網格上的最大絕對誤差為 {max_error:.2e}；"
        f"於 $K={largest['k']}$ 時，閉式解的 median speedup 為 "
        f"{largest['median_speedup_bruteforce_over_closed_form']:.1f}$\\times$；"
        "此數字只代表 attribution kernel，不外推為同幅度 end-to-end 加速。"
        "五次重複執行的 dataset wall-clock median [IQR]、per-case median 與"
        "記憶體峰值分別為：" + "；".join(dataset_parts) + "。"
    )




def render_partial_results(validated: dict, protocol: dict) -> str:
    partial_protocol = _partial_protocol(protocol)
    main = validated["jobs"]["main"]
    retained_labels = len(_partial_methods(protocol))
    return "\n\n".join(
        [
            "% Generated by script/generate_partial_paper_results.py",
            "% Not a substitute for the complete confirmatory reporting gate.",
            (
                r"\noindent\textit{Working-result boundary: the tables below use "
                f"{retained_labels} complete frozen-v14 main-job labels after "
                r"excluding CausalRCA and CIRCA at the author's request. The RQ1 "
                r"Holm family is therefore subset-specific, not the final frozen "
                r"family. Detection-delay and observation-window experiments are "
                r"pending. The "
                r"official generated result file replaces this section only after "
                r"the complete confirmatory gate passes.}"
            ),
            r"\subsection{RQ1: Effectiveness on the Interim Baseline Set}",
            _rq1_draft_text(
                main, partial_protocol, validated["baro_crosscheck"]
            ),
            _effectiveness_table(main, partial_protocol),
            _paired_statistics_tables(
                main, pooled_only=True, families=("rq1_baselines",)
            ),
            r"\subsection{RQ2: Component Contributions}",
            _rq2_draft_text(main),
            _ablation_table(
                main,
                partial_protocol,
                methods=PARTIAL_RQ2_METHODS,
                caption=(
                    "RQ2 controlled variants, service AC@1 (\\%). No trace "
                    "keeps the complete Figure~1 pipeline but extracts metric "
                    "evidence only; no downstream stage receives a call graph, "
                    "trace edge, trace score, or other trace-derived input. Each "
                    "applicable cell retains all 90 cases; macro weights "
                    "compatible systems equally."
                ),
            ),
            _paired_statistics_tables(
                main, pooled_only=True, families=("rq2_ablations",)
            ),
            r"\subsection{RQ3: Equivalence and Efficiency}",
            _rq3_draft_text(validated),
            _equivalence_table(validated["equivalence"]),
            _efficiency_table(main, partial_protocol),
            _stage_table(main, partial_protocol),
            r"\subsection{RQ4: Partial Robustness Evidence}",
            _rq4_draft_text(validated),
            _paired_statistics_tables(
                main, pooled_only=True, families=("rq4_robustness",)
            ),
            _parameter_sweep_curves(validated, partial_protocol),
        ]
    ) + "\n"


def render_partial_results_zh(
    validated: dict,
    protocol: dict,
    baro_results: list[dict],
    *,
    include_wide_tables: bool = True,
) -> str:
    partial_protocol = _partial_protocol(protocol)
    main = validated["jobs"]["main"]
    parts = [
        "% 由 script/generate_partial_paper_results.py 自動產生",
        "% 本檔案不取代完整 confirmatory reporting gate。",
        (
            r"\noindent\textit{以下僅呈現 frozen v14 中已完成且通過"
            r"結果完整性檢查的實驗。數值在正式結果稽核前視為可追溯的 "
            r"interim evidence。}"
        ),
        r"\subsection{根因定位效果}",
    ]
    if include_wide_tables:
        parts.append(_zh_effectiveness_table(main, partial_protocol, baro_results))
    parts.extend(
        [
            _zh_rq1_draft_text(main, partial_protocol, baro_results),
            r"\subsection{Multi-source 證據貢獻}",
            _zh_rq2_draft_text(main),
        ]
    )
    if include_wide_tables:
        parts.append(_zh_ablation_table(main, partial_protocol))
    parts.extend(
        [
            r"\subsection{等價性與效率}",
            _zh_rq3_draft_text(validated, partial_protocol),
        ]
    )
    return "\n\n".join(parts) + "\n"


def render_partial_results_zh_tables(
    validated: dict, protocol: dict, baro_results: list[dict]
) -> str:
    partial_protocol = _partial_protocol(protocol)
    main = validated["jobs"]["main"]
    return "\n\n".join(
        [
            "% 由 script/generate_partial_paper_results.py 自動產生",
            "% 為使雙欄寬表固定在實驗結果頁頂端，本檔案須在實驗設計章前載入。",
            _zh_effectiveness_table(main, partial_protocol, baro_results),
            "% RE1 韌性表插在此處，使其成為表 2、消融表成為表 3。\n"
            r"\IfFileExists{generated/re1-robustness-table-zh.tex}{%" "\n"
            r"  \input{generated/re1-robustness-table-zh.tex}" "\n"
            r"}{}",
            _zh_ablation_table(main, partial_protocol),
        ]
    ) + "\n"


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--reproduction-root", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument(
        "--protocol", type=Path, default=Path("experiment/protocol.json")
    )
    parser.add_argument("--language", choices=("en", "zh"), default="en")
    parser.add_argument(
        "--baro-results", type=Path, default=Path("evidence/data/baro-re2.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tables-output",
        type=Path,
        help="Chinese-only output for wide tables declared ahead of the results body",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    protocol = _load(args.protocol)
    validated = validate_partial_analysis(
        args.reproduction_root, args.analysis_dir, protocol
    )
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.tables_output is not None and args.tables_output.exists():
        raise FileExistsError(args.tables_output)
    if args.tables_output is not None and args.language != "zh":
        raise ValueError("--tables-output is supported only with --language zh")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.language == "zh":
        baro_results = _load(args.baro_results)
        rendered = render_partial_results_zh(
            validated,
            protocol,
            baro_results,
            include_wide_tables=args.tables_output is None,
        )
        if args.tables_output is not None:
            args.tables_output.parent.mkdir(parents=True, exist_ok=True)
            args.tables_output.write_text(
                render_partial_results_zh_tables(
                    validated, protocol, baro_results
                ),
                encoding="utf-8",
            )
    else:
        rendered = render_partial_results(validated, protocol)
    args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
