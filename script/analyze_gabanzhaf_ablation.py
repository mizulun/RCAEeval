"""Summarize GA-Banzhaf ablation metrics, paired cases, and trace diagnostics."""

import argparse
import json
from pathlib import Path
from statistics import mean


FAULTS = ["CPU", "MEM", "DISK", "SOCKET", "DELAY", "LOSS"]
METRICS = ["AC@1", "AC@3", "AC@5", "Avg@5"]
DIAGNOSTIC_FIELDS = [
    "trace_metric_spearman",
    "trace_metric_top_k_overlap",
    "trace_added_seed_count",
    "raw_call_graph_edge_count",
    "accepted_call_graph_edge_count",
    "nonduplicate_call_graph_edge_count",
    "effective_call_graph_edge_count",
    "lag_edge_count",
    "trace_extraction_sec",
]


def _load_rank_file(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    ranks = data.get("0", data.get(0, []))
    return ranks if isinstance(ranks, list) else []


def _service(metric_name):
    return str(metric_name).split("_", 1)[0].replace("-db", "")


def _root_service(case_name):
    return case_name.split("_", 1)[0]


def _root_in_top_k(ranks, root, k):
    services = []
    for rank in ranks:
        service = _service(rank)
        if service not in services:
            services.append(service)
    return root in services[:k]


def _result_files(runs_root, method, dataset):
    result_dir = Path(runs_root) / method / dataset / "results"
    if not result_dir.exists():
        return {}
    return {
        path.name: path
        for path in result_dir.glob("*.json")
        if not path.name.endswith("_failed.json")
    }


def _diagnostic_file(runs_root, method, dataset, name):
    return (
        Path(runs_root)
        / method
        / dataset
        / "results"
        / "diagnostics"
        / name
    )


def _load_diagnostics(path):
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def compare_method_results(runs_root, baseline, variant, dataset, k=1):
    baseline_files = _result_files(runs_root, baseline, dataset)
    variant_files = _result_files(runs_root, variant, dataset)
    common = sorted(set(baseline_files) & set(variant_files))
    wins = losses = ties = same_top1 = same_top5 = same_top5_set = 0
    candidate_jaccards = []
    baseline_recalls = []
    variant_recalls = []

    for name in common:
        baseline_ranks = _load_rank_file(baseline_files[name])
        variant_ranks = _load_rank_file(variant_files[name])
        root = _root_service(name)
        baseline_hit = _root_in_top_k(baseline_ranks, root, k)
        variant_hit = _root_in_top_k(variant_ranks, root, k)
        if variant_hit and not baseline_hit:
            wins += 1
        elif baseline_hit and not variant_hit:
            losses += 1
        else:
            ties += 1
        same_top1 += baseline_ranks[:1] == variant_ranks[:1]
        same_top5 += baseline_ranks[:5] == variant_ranks[:5]
        same_top5_set += set(baseline_ranks[:5]) == set(variant_ranks[:5])
        baseline_diag = _load_diagnostics(
            _diagnostic_file(runs_root, baseline, dataset, name)
        )
        variant_diag = _load_diagnostics(
            _diagnostic_file(runs_root, variant, dataset, name)
        )
        if baseline_diag is not None and variant_diag is not None:
            baseline_candidates = set(baseline_diag.get("game_candidates", []))
            variant_candidates = set(variant_diag.get("game_candidates", []))
            union = baseline_candidates | variant_candidates
            candidate_jaccards.append(
                len(baseline_candidates & variant_candidates) / len(union)
                if union
                else 1.0
            )
            if baseline_diag.get("root_candidate_recall") is not None:
                baseline_recalls.append(
                    bool(baseline_diag["root_candidate_recall"])
                )
            if variant_diag.get("root_candidate_recall") is not None:
                variant_recalls.append(
                    bool(variant_diag["root_candidate_recall"])
                )

    count = len(common)
    divisor = count or 1
    return {
        "baseline": baseline,
        "variant": variant,
        "dataset": dataset,
        "k": k,
        "cases": count,
        "variant_wins": wins,
        "variant_losses": losses,
        "ties": ties,
        "same_top1_rate": same_top1 / divisor,
        "same_ordered_top5_rate": same_top5 / divisor,
        "same_top5_set_rate": same_top5_set / divisor,
        "candidate_jaccard": (
            mean(candidate_jaccards) if candidate_jaccards else None
        ),
        "baseline_candidate_recall": (
            sum(baseline_recalls) / len(baseline_recalls)
            if baseline_recalls
            else None
        ),
        "variant_candidate_recall": (
            sum(variant_recalls) / len(variant_recalls)
            if variant_recalls
            else None
        ),
    }


def summarize_diagnostics(runs_root, method, dataset):
    diagnostics_dir = (
        Path(runs_root) / method / dataset / "results" / "diagnostics"
    )
    records = []
    if diagnostics_dir.exists():
        for path in sorted(diagnostics_dir.glob("*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(record, dict):
                records.append(record)

    summary = {"method": method, "dataset": dataset, "cases": len(records)}
    summary["has_trace_fields"] = any(
        any(field in record for field in DIAGNOSTIC_FIELDS)
        for record in records
    )
    for field in DIAGNOSTIC_FIELDS:
        values = [
            float(record[field])
            for record in records
            if isinstance(record.get(field), (int, float))
        ]
        if values:
            summary[field] = mean(values)
    recalls = [
        bool(record["root_candidate_recall"])
        for record in records
        if record.get("root_candidate_recall") is not None
    ]
    if recalls:
        summary["root_candidate_recall_rate"] = sum(recalls) / len(recalls)
    candidate_pool_sizes = [
        len(record["candidate_pool"])
        for record in records
        if isinstance(record.get("candidate_pool"), list)
    ]
    game_candidate_sizes = [
        len(record["game_candidates"])
        for record in records
        if isinstance(record.get("game_candidates"), list)
    ]
    if candidate_pool_sizes:
        summary["candidate_pool_size"] = mean(candidate_pool_sizes)
    if game_candidate_sizes:
        summary["game_candidate_count"] = mean(game_candidate_sizes)
    return summary


def _macro(record, metric):
    values = [
        float(record[f"{metric}-{fault}"])
        for fault in FAULTS
        if isinstance(record.get(f"{metric}-{fault}"), (int, float))
    ]
    return mean(values) if values else None


def _benchmark_rows(paths):
    if paths is None:
        return []
    if isinstance(paths, (str, Path)):
        paths = [paths]
    records = []
    for path in paths:
        if Path(path).exists():
            records.extend(json.loads(Path(path).read_text(encoding="utf-8")))
    deduped = {}
    for record in records:
        row = {
            "method": record.get("method"),
            "dataset": record.get("dataset"),
            "wall_time_sec": record.get("wall_time_sec"),
            "wall_time_min_sec": record.get("wall_time_min_sec"),
            "wall_time_max_sec": record.get("wall_time_max_sec"),
            "Avg speed": record.get("Avg speed"),
        }
        for metric in METRICS:
            row[f"{metric} macro"] = _macro(record, metric)
        deduped[(row["method"], row["dataset"])] = row
    return list(deduped.values())


def _k_variant_number(variant):
    prefix = "gabanzhaf_k"
    if not str(variant).startswith(prefix):
        return None
    suffix = str(variant)[len(prefix):]
    return int(suffix) if suffix.isdigit() else None


def _format_k_values(values):
    values = sorted(set(values))
    if not values:
        return ""
    ranges = []
    start = prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append((start, prev))
        start = prev = value
    ranges.append((start, prev))
    return ", ".join(
        f"{start}" if start == end else f"{start}–{end}"
        for start, end in ranges
    )


def _same_benchmark_metrics(baseline_row, variant_row):
    if baseline_row is None or variant_row is None:
        return None
    compared = []
    for metric in METRICS:
        key = f"{metric} macro"
        if baseline_row.get(key) is not None and variant_row.get(key) is not None:
            compared.append(abs(baseline_row[key] - variant_row[key]) <= 1e-12)
    return all(compared) if compared else None


def render_report(
    runs_root,
    benchmark_path,
    baseline,
    trace_baseline,
    variants,
    datasets,
):
    benchmark_rows = _benchmark_rows(benchmark_path)
    benchmark_index = {
        (row["method"], row["dataset"]): row
        for row in benchmark_rows
    }
    lines = [
        "# GA-Banzhaf Ablation and Trace Diagnostics",
        "",
        "## Aggregate benchmark",
        "",
        "| Method | Dataset | AC@1 macro | Avg@5 macro | Wall median (s) | Range (s) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in benchmark_rows:
        wall_range = "-"
        if row["wall_time_min_sec"] is not None and row["wall_time_max_sec"] is not None:
            wall_range = (
                f"{row['wall_time_min_sec']:.2f}–"
                f"{row['wall_time_max_sec']:.2f}"
            )
        lines.append(
            "| {method} | {dataset} | {ac1} | {avg5} | {wall} | {wall_range} |".format(
                method=row["method"],
                dataset=row["dataset"],
                ac1=(
                    f"{row['AC@1 macro']:.3f}"
                    if row["AC@1 macro"] is not None
                    else "-"
                ),
                avg5=(
                    f"{row['Avg@5 macro']:.3f}"
                    if row["Avg@5 macro"] is not None
                    else "-"
                ),
                wall=(
                    f"{row['wall_time_sec']:.2f}"
                    if isinstance(row["wall_time_sec"], (int, float))
                    else "-"
                ),
                wall_range=wall_range,
            )
        )

    lines.extend([
        "",
        "## Paired case comparison",
        "",
        "| Baseline | Variant | Dataset | Cases | Wins | Losses | Ties | Same top-1 | Same ordered top-5 | Candidate Jaccard | Candidate recall |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    paired_results = []
    for variant in variants:
        for dataset in datasets:
            comparison_baseline = (
                trace_baseline
                if variant.startswith("with_trace_") and variant != trace_baseline
                else baseline
            )
            result = compare_method_results(
                runs_root,
                comparison_baseline,
                variant,
                dataset,
                k=1,
            )
            paired_results.append(result)
            lines.append(
                f"| {comparison_baseline} | {variant} | {dataset} | "
                f"{result['cases']} | "
                f"{result['variant_wins']} | {result['variant_losses']} | "
                f"{result['ties']} | {result['same_top1_rate']:.3f} | "
                f"{result['same_ordered_top5_rate']:.3f} | "
                f"{result['candidate_jaccard']:.3f} | "
                f"{result['variant_candidate_recall']:.3f} |"
                if result["candidate_jaccard"] is not None
                and result["variant_candidate_recall"] is not None
                else (
                    f"| {comparison_baseline} | {variant} | {dataset} | "
                    f"{result['cases']} | {result['variant_wins']} | "
                    f"{result['variant_losses']} | {result['ties']} | "
                    f"{result['same_top1_rate']:.3f} | "
                    f"{result['same_ordered_top5_rate']:.3f} | - | - |"
                )
            )

    lines.extend([
        "",
        "## Trace overlap diagnostics",
        "",
        "| Method | Dataset | Cases | Spearman | Top-K overlap | Added seeds | Accepted call edges | Effective call edges | Trace seconds |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    diagnostic_summaries = []
    for method in variants:
        for dataset in datasets:
            summary = summarize_diagnostics(runs_root, method, dataset)
            if not summary["cases"] or not summary["has_trace_fields"]:
                continue
            diagnostic_summaries.append(summary)
            lines.append(
                f"| {method} | {dataset} | {summary['cases']} | "
                f"{summary.get('trace_metric_spearman', 0.0):.3f} | "
                f"{summary.get('trace_metric_top_k_overlap', 0.0):.3f} | "
                f"{summary.get('trace_added_seed_count', 0.0):.2f} | "
                f"{summary.get('accepted_call_graph_edge_count', 0.0):.2f} | "
                f"{summary.get('effective_call_graph_edge_count', 0.0):.2f} | "
                f"{summary.get('trace_extraction_sec', 0.0):.3f} |"
            )

    by_variant = {}
    for result in paired_results:
        by_variant.setdefault(result["variant"], []).append(result)

    k_results_by_k = {}
    for variant, results in by_variant.items():
        k_value = _k_variant_number(variant)
        if k_value is not None:
            k_results_by_k[k_value] = results
    stable_ks = []
    changed_ks = []
    changed_by_benchmark_ks = []
    for k_value, results in k_results_by_k.items():
        if not results:
            continue
        benchmark_checks = []
        for result in results:
            benchmark_same = _same_benchmark_metrics(
                benchmark_index.get((baseline, result["dataset"])),
                benchmark_index.get((result["variant"], result["dataset"])),
            )
            if benchmark_same is not None:
                benchmark_checks.append(benchmark_same)
        if benchmark_checks:
            if all(benchmark_checks):
                stable_ks.append(k_value)
            else:
                changed_ks.append(k_value)
                changed_by_benchmark_ks.append(k_value)
            continue
        if all(
            result["same_top1_rate"] == 1.0
            and result["same_ordered_top5_rate"] == 1.0
            for result in results
        ):
            stable_ks.append(k_value)
        else:
            changed_ks.append(k_value)
    k_conclusions = []
    if stable_ks:
        k_conclusions.append(
            "- Candidate K="
            f"{_format_k_values(stable_ks)} produced identical paired top-1 "
            "and ordered top-5 rankings across the evaluated datasets."
        )
    if changed_ks:
        if changed_by_benchmark_ks:
            k_conclusions.append(
                "- Candidate K="
                f"{_format_k_values(changed_by_benchmark_ks)} changed at least "
                "one benchmark metric; low-K effects must be interpreted as "
                "candidate truncation effects before attributing them to "
                "Banzhaf scoring."
            )
        paired_only_changed = sorted(set(changed_ks) - set(changed_by_benchmark_ks))
        if paired_only_changed:
            k_conclusions.append(
                "- Candidate K="
                f"{_format_k_values(paired_only_changed)} changed at least one "
                "paired ranking; low-K effects must be interpreted as candidate "
                "truncation effects before attributing them to Banzhaf scoring."
            )
    if not k_conclusions:
        k_conclusions.append(
            "- Candidate K variants were not available in this report run."
        )

    def _paired_totals(variant):
        results = by_variant.get(variant, [])
        return (
            sum(result["variant_wins"] for result in results),
            sum(result["variant_losses"] for result in results),
            sum(result["cases"] for result in results),
        )

    no_lag_wins, no_lag_losses, no_lag_cases = _paired_totals(
        "gabanzhaf_no_lag_1"
    )
    no_score_wins, no_score_losses, no_score_cases = _paired_totals(
        "with_trace_no_scores"
    )
    no_residual_wins, no_residual_losses, no_residual_cases = _paired_totals(
        "with_trace_no_residual"
    )
    no_call_wins, no_call_losses, no_call_cases = _paired_totals(
        "with_trace_no_call_graph"
    )

    overhead_lines = []
    for dataset in datasets:
        full = benchmark_index.get(("with_trace_full", dataset))
        no_trace = benchmark_index.get(("with_trace_no_trace", dataset))
        if (
            full
            and no_trace
            and isinstance(full.get("wall_time_sec"), (int, float))
            and isinstance(no_trace.get("wall_time_sec"), (int, float))
            and no_trace["wall_time_sec"] > 0
        ):
            saved = full["wall_time_sec"] - no_trace["wall_time_sec"]
            overhead_lines.append(
                f"- {dataset}: removing trace reduced median wall time by "
                f"{saved:.2f}s ({full['wall_time_sec'] / no_trace['wall_time_sec']:.2f}×)."
            )

    lines.extend([
        "",
        "## Experiment-supported conclusions",
        "",
        *k_conclusions,
        (
            f"- Removing lag-1 produced {no_lag_wins} wins and "
            f"{no_lag_losses} losses across {no_lag_cases} paired cases."
        ),
        (
            f"- Removing trace scores produced {no_score_wins} wins and "
            f"{no_score_losses} losses across {no_score_cases} paired cases."
        ),
        (
            f"- Removing residual suppression produced {no_residual_wins} wins "
            f"and {no_residual_losses} losses across {no_residual_cases} paired cases."
        ),
        (
            f"- Removing the call graph produced {no_call_wins} wins and "
            f"{no_call_losses} losses across {no_call_cases} paired cases."
        ),
        "",
        "## Acceptable method limitations",
        "",
        "- Closed-form and brute-force Banzhaf are equivalent implementations of the same noisy-OR game; runtime, not accuracy, is the intended comparison.",
        "- Deterministic name ordering makes ties reproducible, but it is not evidence-based metric identification.",
        "- Aggregate AC metrics can hide ranking changes that do not cross the evaluation cutoff; paired top-1/top-5 rates must be reported with them.",
        "",
        "## Issues requiring follow-up",
        "",
        "- Lag topology currently takes a raw maximum across metrics with different units and may be dominated by the largest-scale metric.",
        "- Concatenating normal and anomalous regimes can create shared step-change correlations that are not propagation.",
        "- Self-explanation probability 1 can make the Banzhaf score nearly identical to service evidence when off-diagonal probabilities are weak.",
        "- The baseline ranks metrics within a service by column name rather than local metric evidence.",
        "",
        "## Redundant stages and measured cost",
        "",
        (
            f"- Trace scores did not change top-1 correctness in "
            f"{no_score_cases} controlled paired cases."
        ),
        (
            f"- Residual suppression did not change top-1 correctness in "
            f"{no_residual_cases} controlled paired cases."
        ),
    ])
    lines.extend(overhead_lines or [
        "- Runtime overhead could not be computed because benchmark timing rows were unavailable."
    ])

    lines.extend([
        "",
        "## Interpretation rules",
        "",
        "- Treat direct comparisons between the two source files as version comparisons, not isolated trace effects.",
        "- Attribute trace effects only from controlled variants that hold residual, ranking, and candidate logic constant.",
        "- If accepted call edges are non-zero but effective edges are zero, the call graph is removed by candidate truncation or matrix gating.",
        "- If candidate recall changes with K, accuracy changes cannot be attributed only to Banzhaf scoring.",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default="output/runs")
    parser.add_argument("--benchmark", nargs="*", default=None)
    parser.add_argument("--baseline", default="gabanzhaf")
    parser.add_argument("--trace-baseline", default="with_trace_full")
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["re2-ob", "re2-ss", "re2-tt"],
    )
    parser.add_argument(
        "--output",
        default="docs/benchmark/gabanzhaf-ablation-and-trace.md",
    )
    args = parser.parse_args()
    report = render_report(
        args.runs_root,
        args.benchmark,
        args.baseline,
        args.trace_baseline,
        args.variants,
        args.datasets,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
