"""Generate paper-ready case records, summaries, and paired statistics."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from RCAEval.benchmark.paper_evaluation import (
    aggregate_method,
    cluster_paired_permutation,
    discover_expected_cases,
    evaluate_method_dataset,
    exact_mcnemar,
    holm_adjust,
    pair_by_case,
    paired_bootstrap,
    paired_cluster_bootstrap,
    paired_seed_average_cluster_bootstrap,
    seed_average_cluster_permutation,
)


CONTROLLED_ABLATION_LABELS = {
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
}

ROBUSTNESS_LABELS = {
    "merit_graph_per_case_full",
    "merit_graph_first_case_pre_fault",
    "merit_graph_union_pre_fault",
    "merit_graph_first_case_order1",
    "merit_graph_first_case_order2",
    "merit_graph_first_case_order3",
}


def comparison_family(method: str, protocol: dict) -> str | None:
    base_method = re.sub(r"_seed\d+$", "", method)
    if base_method in protocol["baselines"]:
        return "rq1_baselines"
    if method in CONTROLLED_ABLATION_LABELS:
        return "rq2_ablations"
    if method in ROBUSTNESS_LABELS:
        return "rq4_robustness"
    if re.fullmatch(r"merit_trace_(dropout|noise|edge_noise)_\d+", method):
        return "rq4_robustness"
    return None


def _paired_record(
    reference: str,
    method: str,
    scope: str,
    datasets: list[str],
    paired_a: list,
    paired_b: list,
    family: str,
    protocol: dict,
) -> dict:
    return {
        "comparison_family": family,
        "reference": reference,
        "method": method,
        "dataset": scope,
        "scope": scope,
        "datasets": datasets,
        "case_ids": [outcome.case_id for outcome in paired_a],
        "mcnemar_ac1": exact_mcnemar(paired_a, paired_b, k=1),
        "cluster_permutation_ac1": cluster_paired_permutation(
            paired_a,
            paired_b,
            k=1,
            resamples=protocol["bootstrap_resamples"],
            seed=protocol["bootstrap_seed"],
        ),
        "bootstrap": [
            paired_bootstrap(
                paired_a,
                paired_b,
                metric,
                resamples=protocol["bootstrap_resamples"],
                seed=protocol["bootstrap_seed"],
            )
            for metric in ("AC@1", "AC@3", "AC@5", "Avg@5")
        ],
        "cluster_bootstrap": [
            paired_cluster_bootstrap(
                paired_a,
                paired_b,
                metric,
                resamples=protocol["bootstrap_resamples"],
                seed=protocol["bootstrap_seed"],
            )
            for metric in ("AC@1", "AC@3", "AC@5", "Avg@5")
        ],
    }


def _seed_average_record(
    reference: str,
    method: str,
    seed_labels: list[str],
    scope: str,
    datasets: list[str],
    paired_a: list,
    paired_seed_runs: list[list],
    family: str,
    protocol: dict,
) -> dict:
    return {
        "comparison_family": family,
        "reference": reference,
        "method": method,
        "seed_labels": seed_labels,
        "estimand": "frozen-seed mean",
        "dataset": scope,
        "scope": scope,
        "datasets": datasets,
        "case_ids": [outcome.case_id for outcome in paired_a],
        "mcnemar_ac1": {
            "status": "sensitivity_not_defined_for_fractional_seed_mean"
        },
        "cluster_permutation_ac1": seed_average_cluster_permutation(
            paired_a,
            paired_seed_runs,
            k=1,
            resamples=protocol["bootstrap_resamples"],
            seed=protocol["bootstrap_seed"],
        ),
        "bootstrap": [],
        "cluster_bootstrap": [
            paired_seed_average_cluster_bootstrap(
                paired_a,
                paired_seed_runs,
                metric,
                resamples=protocol["bootstrap_resamples"],
                seed=protocol["bootstrap_seed"],
            )
            for metric in ("AC@1", "AC@3", "AC@5", "Avg@5")
        ],
    }


def build_paired_statistics(
    all_outcomes: dict,
    methods: list[str],
    datasets: list[str],
    reference: str | None,
    protocol: dict,
) -> list[dict]:
    if reference is None:
        return []
    paired = []
    family_records = {
        "rq1_baselines": [],
        "rq2_ablations": [],
        "rq4_robustness": [],
    }
    stochastic_methods = set(protocol.get("stochastic_methods", []))
    seeded_groups = {}
    for method in methods:
        match = re.fullmatch(r"(.+)_seed(\d+)", method)
        if match and match.group(1) in stochastic_methods:
            seeded_groups.setdefault(match.group(1), []).append(method)

    # Deterministic methods and unseeded smoke-profile baselines.
    for method in methods:
        if method == reference:
            continue
        seed_match = re.fullmatch(r"(.+)_seed\d+", method)
        if seed_match and seed_match.group(1) in stochastic_methods:
            continue
        family = comparison_family(method, protocol)
        if family is None:
            continue
        pooled_a = []
        pooled_b = []
        pooled_datasets = []
        for dataset in datasets:
            a = all_outcomes.get((reference, dataset), [])
            b = all_outcomes.get((method, dataset), [])
            if not a or not b:
                continue
            paired_a, paired_b = pair_by_case(a, b)
            record = _paired_record(
                reference,
                method,
                dataset,
                [dataset],
                paired_a,
                paired_b,
                family,
                protocol,
            )
            paired.append(record)
            family_records[family].append(record)
            pooled_a.extend(paired_a)
            pooled_b.extend(paired_b)
            pooled_datasets.append(dataset)
        if pooled_a:
            record = _paired_record(
                reference,
                method,
                "pooled",
                pooled_datasets,
                pooled_a,
                pooled_b,
                family,
                protocol,
            )
            paired.append(record)
            family_records[family].append(record)

    # Stochastic baselines: aggregate the frozen seeds before inference.
    for base_method, seed_labels in seeded_groups.items():
        seed_labels = sorted(seed_labels)
        pooled_a = []
        pooled_seed_runs = [[] for _ in seed_labels]
        pooled_datasets = []
        for dataset in datasets:
            reference_outcomes = all_outcomes.get((reference, dataset), [])
            seed_outcomes = [
                all_outcomes.get((label, dataset), []) for label in seed_labels
            ]
            if not reference_outcomes or any(not outcomes for outcomes in seed_outcomes):
                continue
            aligned_a = None
            aligned_seed_runs = []
            for outcomes in seed_outcomes:
                current_a, current_b = pair_by_case(reference_outcomes, outcomes)
                if aligned_a is None:
                    aligned_a = current_a
                elif [item.case_id for item in current_a] != [
                    item.case_id for item in aligned_a
                ]:
                    raise ValueError("stochastic seed case sets do not match")
                aligned_seed_runs.append(current_b)
            record = _seed_average_record(
                reference,
                base_method,
                seed_labels,
                dataset,
                [dataset],
                aligned_a,
                aligned_seed_runs,
                "rq1_baselines",
                protocol,
            )
            paired.append(record)
            family_records["rq1_baselines"].append(record)
            pooled_a.extend(aligned_a)
            for target, run in zip(pooled_seed_runs, aligned_seed_runs):
                target.extend(run)
            pooled_datasets.append(dataset)
        if pooled_a:
            record = _seed_average_record(
                reference,
                base_method,
                seed_labels,
                "pooled",
                pooled_datasets,
                pooled_a,
                pooled_seed_runs,
                "rq1_baselines",
                protocol,
            )
            paired.append(record)
            family_records["rq1_baselines"].append(record)

    for records in family_records.values():
        adjusted = holm_adjust(
            record["cluster_permutation_ac1"]["p_value"]
            for record in records
        )
        mcnemar_records = [
            record for record in records if "p_value" in record["mcnemar_ac1"]
        ]
        mcnemar_adjusted = holm_adjust(
            record["mcnemar_ac1"]["p_value"] for record in mcnemar_records
        )
        for record, p_value in zip(records, adjusted):
            record["cluster_permutation_ac1"][
                "holm_adjusted_p_value"
            ] = p_value
            # McNemar is retained as an explicitly case-independent
            # sensitivity analysis, not the v14 primary binary test.
        for record, sensitivity_p in zip(mcnemar_records, mcnemar_adjusted):
            record["mcnemar_ac1"]["holm_adjusted_p_value"] = sensitivity_p
    return paired


def build_diagnostic_checks(all_outcomes: dict) -> dict:
    reference = all_outcomes.get(("merit_full", "re2-ss"), [])
    comparison = all_outcomes.get(("merit_no_trace", "re2-ss"), [])
    checks = {}
    if not reference or not comparison:
        checks["missing_trace"] = {
                "status": "not_applicable",
                "basis": "both merit_full and merit_no_trace on re2-ss are required",
        }
    else:
        paired_reference, paired_comparison = pair_by_case(reference, comparison)
        status_matches = [
            left.status == right.status
            for left, right in zip(paired_reference, paired_comparison)
        ]
        metric_matches = [
            left.ranks == right.ranks
            for left, right in zip(paired_reference, paired_comparison)
        ]
        service_matches = [
            left.ranked_services == right.ranked_services
            for left, right in zip(paired_reference, paired_comparison)
        ]
        checks["missing_trace"] = {
            "status": "completed",
            "dataset": "re2-ss",
            "reference": "merit_full",
            "comparison": "merit_no_trace",
            "n": len(paired_reference),
            "execution_status_matches": sum(status_matches),
            "exact_metric_rank_matches": sum(metric_matches),
            "exact_service_rank_matches": sum(service_matches),
            "all_execution_status_equal": all(status_matches),
            "all_metric_ranks_equal": all(metric_matches),
            "all_service_ranks_equal": all(service_matches),
            "metric_rank_mismatch_case_ids": [
                outcome.case_id
                for outcome, matches in zip(paired_reference, metric_matches)
                if not matches
            ],
            "service_rank_mismatch_case_ids": [
                outcome.case_id
                for outcome, matches in zip(paired_reference, service_matches)
                if not matches
            ],
        }

    graph_variants = (
        "merit_graph_per_case_full",
        "merit_graph_first_case_pre_fault",
        "merit_graph_union_pre_fault",
        "merit_graph_first_case_order1",
        "merit_graph_first_case_order2",
        "merit_graph_first_case_order3",
    )
    graph_checks = []
    for (method, dataset), outcomes in sorted(all_outcomes.items()):
        if method not in graph_variants or not outcomes:
            continue
        baseline = all_outcomes.get(("merit_full", dataset), [])
        if not baseline:
            continue
        paired_baseline, paired_variant = pair_by_case(baseline, outcomes)
        jaccards = []
        for left, right in zip(paired_baseline, paired_variant):
            left_edges = set(left.graph_edges)
            right_edges = set(right.graph_edges)
            union = left_edges | right_edges
            jaccards.append(
                len(left_edges & right_edges) / len(union) if union else 1.0
            )
        graph_checks.append(
            {
                "method": method,
                "dataset": dataset,
                "n": len(paired_baseline),
                "edge_jaccard_mean": sum(jaccards) / len(jaccards),
                "edge_jaccard_min": min(jaccards),
                "ac1_effect_pp_vs_primary": 100.0
                * sum(
                    right.hit(1) - left.hit(1)
                    for left, right in zip(paired_baseline, paired_variant)
                )
                / len(paired_baseline),
                "exact_service_rank_matches": sum(
                    left.ranked_services == right.ranked_services
                    for left, right in zip(paired_baseline, paired_variant)
                ),
            }
        )
    checks["graph_sensitivity"] = {
        "status": "completed" if graph_checks else "not_applicable",
        "comparisons": graph_checks,
    }

    # Explanation stability is deliberately descriptive: it asks whether the
    # largest auditable player->target contribution remains the same under
    # pre-registered graph and trace perturbations. It does not convert the
    # designed coverage strengths into human or structural-causal ground truth.
    stability_variants = {
        *graph_variants,
        "merit_no_trace",
        "merit_trace_dropout_50",
        "merit_trace_noise_30",
        "merit_trace_edge_noise_30",
    }
    stability_checks = []
    for (method, dataset), outcomes in sorted(all_outcomes.items()):
        if method not in stability_variants or not outcomes:
            continue
        baseline = all_outcomes.get(("merit_full", dataset), [])
        if not baseline:
            continue
        paired_baseline, paired_variant = pair_by_case(baseline, outcomes)
        comparable = [
            (left, right)
            for left, right in zip(paired_baseline, paired_variant)
            if left.success
            and right.success
            and left.top_explanation_player is not None
            and left.top_explanation_target is not None
            and right.top_explanation_player is not None
            and right.top_explanation_target is not None
            and left.top_explanation_contribution is not None
            and right.top_explanation_contribution is not None
        ]
        if not comparable:
            continue
        top_edge_matches = sum(
            (left.top_explanation_player, left.top_explanation_target)
            == (right.top_explanation_player, right.top_explanation_target)
            for left, right in comparable
        )
        top_player_matches = sum(
            left.top_explanation_player == right.top_explanation_player
            for left, right in comparable
        )
        stability_checks.append(
            {
                "method": method,
                "dataset": dataset,
                "n": len(paired_baseline),
                "n_explanations": len(comparable),
                "top_player_matches": top_player_matches,
                "top_edge_matches": top_edge_matches,
                "top_player_agreement": top_player_matches / len(comparable),
                "top_edge_agreement": top_edge_matches / len(comparable),
                "mean_absolute_top_contribution_change": sum(
                    abs(
                        float(left.top_explanation_contribution)
                        - float(right.top_explanation_contribution)
                    )
                    for left, right in comparable
                )
                / len(comparable),
            }
        )
    checks["explanation_stability"] = {
        "status": "completed" if stability_checks else "not_applicable",
        "comparisons": stability_checks,
    }
    return checks


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--runs-root", type=Path, default=Path("output/runs"))
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument(
        "--datasets", nargs="+", default=["re2-ob", "re2-ss", "re2-tt"]
    )
    parser.add_argument("--reference-method")
    parser.add_argument("--protocol", type=Path, default=Path("experiment/protocol.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args(argv)


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    runs_root = args.runs_root if args.runs_root.is_absolute() else repo_root / args.runs_root
    protocol_path = args.protocol if args.protocol.is_absolute() else repo_root / args.protocol
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)

    all_outcomes = {}
    dataset_summaries = []
    aggregate_summaries = []
    case_rows = []

    for method in args.methods:
        method_outcomes = []
        method_summaries = []
        for dataset in args.datasets:
            expected = discover_expected_cases(repo_root, dataset)
            key = f"{method}:{dataset}"
            infeasible = protocol.get("infeasible", {}).get(key)
            inapplicable = protocol.get("modality_inapplicable", {}).get(key)
            result_dir = runs_root / method / dataset / "results"
            outcomes, summary = evaluate_method_dataset(
                method,
                dataset,
                expected,
                result_dir,
                infeasible_reason=(
                    infeasible.get("basis")
                    if infeasible
                    else inapplicable.get("reason")
                    if inapplicable
                    else None
                ),
            )
            if inapplicable:
                summary["combination_status"] = "inapplicable"
                summary["declared_status"] = inapplicable.get("status", "N/A")
                summary["required_modality"] = inapplicable.get(
                    "required_modality"
                )
            if (
                summary["combination_status"] == "incomplete"
                and not args.allow_incomplete
            ):
                raise RuntimeError(
                    f"{method} x {dataset} is incomplete: "
                    f"{summary.get('n_missing')} missing artifacts"
                )
            all_outcomes[(method, dataset)] = outcomes
            method_outcomes.extend(outcomes)
            method_summaries.append(summary)
            dataset_summaries.append(summary)
            case_rows.extend(outcome.__dict__ for outcome in outcomes)
        aggregate_summaries.append(
            aggregate_method(method, method_summaries, method_outcomes)
        )

    paired = build_paired_statistics(
        all_outcomes,
        list(args.methods),
        list(args.datasets),
        args.reference_method,
        protocol,
    )

    _write_json(output_dir / "case_results.json", case_rows)
    _write_json(output_dir / "dataset_summaries.json", dataset_summaries)
    _write_json(output_dir / "aggregate_summaries.json", aggregate_summaries)
    _write_json(output_dir / "paired_statistics.json", paired)
    _write_json(
        output_dir / "diagnostic_checks.json",
        build_diagnostic_checks(all_outcomes),
    )

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "method", "dataset", "combination_status", "n_expected", "n_success",
            "execution_success_rate", "AC@1", "AC@3", "AC@5", "Avg@5",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in dataset_summaries:
            row = {key: summary.get(key) for key in fieldnames}
            if summary.get("overall"):
                row.update(summary["overall"])
            writer.writerow(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
