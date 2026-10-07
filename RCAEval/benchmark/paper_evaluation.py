"""Case-level evaluation and paired statistics for the MERIT paper.

Ground truth is intentionally derived here, never in an RCA method.  Missing,
crashed, and timed-out cases remain in the overall-accuracy denominator.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Sequence

import numpy as np


DATASET_PATHS = {
    "re2-ob": Path("data/RE2/RE2-OB"),
    "re2-ss": Path("data/RE2/RE2-SS"),
    "re2-tt": Path("data/RE2/RE2-TT"),
}


@dataclass(frozen=True)
class ExpectedCase:
    dataset: str
    case_id: str
    root_service: str
    fault_type: str
    repetition: str


@dataclass
class CaseOutcome:
    method: str
    dataset: str
    case_id: str
    root_service: str
    fault_type: str
    repetition: str
    status: str
    ranks: list[str]
    ranked_services: list[str]
    error: str | None
    candidate_recall: bool | None
    stage_timing: dict[str, float]
    base_candidate_recall: bool | None = None
    rescued_candidate_recall: bool | None = None
    io_seconds: float | None = None
    method_seconds: float | None = None
    peak_rss_bytes: int | None = None
    input_modality: str | None = None
    input_rows: int | None = None
    observed_case_seconds: float | None = None
    warmup_seconds: float | None = None
    warmup_method_call: bool | None = None
    graph_edges: list[tuple[str, str]] = field(default_factory=list)
    base_candidate_k: int | None = None
    coverage_rescued_k: int | None = None
    final_candidate_k: int | None = None
    target_count: int | None = None
    explanation_edge_count: int | None = None
    explanation_faithfulness_correlation: float | None = None
    explanation_faithfulness_mae: float | None = None
    explanation_sign_agreement: float | None = None
    top_explanation_player: str | None = None
    top_explanation_target: str | None = None
    top_explanation_contribution: float | None = None
    top_explanation_utility_drop: float | None = None

    @property
    def success(self) -> bool:
        return self.status == "success"

    def hit(self, k: int) -> float:
        if not self.success:
            return 0.0
        return float(self.root_service in self.ranked_services[:k])

    def avg_at_5(self) -> float:
        return sum(self.hit(k) for k in range(1, 6)) / 5.0


def parse_fault_directory(name: str) -> tuple[str, str]:
    try:
        service, fault = name.rsplit("_", 1)
    except ValueError as exc:
        raise ValueError(f"invalid fault directory name: {name}") from exc
    if not service or not fault:
        raise ValueError(f"invalid fault directory name: {name}")
    return service, fault.lower()


def discover_expected_cases(
    repo_root: Path,
    dataset: str,
    dataset_paths: dict[str, Path] | None = None,
) -> list[ExpectedCase]:
    paths = dataset_paths or DATASET_PATHS
    dataset_root = repo_root / paths[dataset]
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)

    expected = []
    for fault_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        root_service, fault = parse_fault_directory(fault_dir.name)
        for case_dir in sorted(p for p in fault_dir.iterdir() if p.is_dir()):
            if not (
                (case_dir / "data.csv").exists()
                or (case_dir / "simple_metrics.csv").exists()
            ):
                continue
            expected.append(
                ExpectedCase(
                    dataset=dataset,
                    case_id=f"{fault_dir.name}_{case_dir.name}",
                    root_service=root_service,
                    fault_type=fault,
                    repetition=case_dir.name,
                )
            )
    return expected


def metric_service(metric_name: str) -> str:
    text = str(metric_name)
    service = text.rsplit("_", 1)[0] if "_" in text else text
    return service.removesuffix("-db")


def unique_ranked_services(ranks: Sequence[str]) -> list[str]:
    services = []
    seen = set()
    for rank in ranks:
        service = metric_service(rank)
        if service not in seen:
            services.append(service)
            seen.add(service)
    return services


def _read_json_object(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("result must be a JSON object")
    return data


def _read_diagnostics(path: Path, root_service: str) -> tuple[bool | None, dict, dict]:
    if not path.exists():
        return None, {}, {}
    try:
        diagnostics = _read_json_object(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None, {}, {}
    candidates = diagnostics.get("game_candidates")
    recall = root_service in set(candidates) if isinstance(candidates, list) else None
    base_candidates = diagnostics.get("base_game_candidates")
    base_recall = (
        root_service in set(base_candidates)
        if isinstance(base_candidates, list)
        else None
    )
    rescued_candidates = set(base_candidates or ()) | set(
        diagnostics.get("coverage_rescued_candidates", ()) or ()
    )
    rescued_recall = (
        root_service in rescued_candidates
        if isinstance(base_candidates, list)
        else None
    )
    raw_timing = diagnostics.get("stage_timing", {})
    timing = {
        str(key): float(value)
        for key, value in raw_timing.items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    } if isinstance(raw_timing, dict) else {}
    raw_edges = diagnostics.get("call_graph_edges", [])
    graph_edges = [
        (str(edge[0]), str(edge[1]))
        for edge in raw_edges
        if isinstance(edge, list) and len(edge) == 2
    ] if isinstance(raw_edges, list) else []
    extra = {
        "graph_edges": graph_edges,
        "base_candidate_recall": base_recall,
        "rescued_candidate_recall": rescued_recall,
    }
    for key in ("base_candidate_k", "coverage_rescued_k", "final_candidate_k"):
        value = diagnostics.get(key)
        if isinstance(value, int) and value >= 0:
            extra[key] = value
    services = diagnostics.get("game_services")
    if isinstance(services, list):
        extra["target_count"] = len(services)

    # Reduce the auditable player->target decomposition to compact case-level
    # faithfulness diagnostics. The raw maps remain in the per-case diagnostic
    # artifact; case_results.json need not duplicate the full K x N matrices.
    contribution_map = diagnostics.get("player_target_contributions")
    removal_map = diagnostics.get("edge_removal_utility_drop")
    strengths = diagnostics.get("coverage_strength_matrix")
    if (
        isinstance(candidates, list)
        and isinstance(services, list)
        and isinstance(contribution_map, dict)
        and isinstance(removal_map, dict)
        and isinstance(strengths, list)
    ):
        pairs = []
        labels = []
        for i, player in enumerate(candidates):
            if i >= len(strengths) or not isinstance(strengths[i], list):
                continue
            player_contributions = contribution_map.get(player, {})
            player_removals = removal_map.get(player, {})
            if not isinstance(player_contributions, dict) or not isinstance(
                player_removals, dict
            ):
                continue
            for j, target in enumerate(services):
                if j >= len(strengths[i]):
                    continue
                strength = strengths[i][j]
                contribution = player_contributions.get(target)
                removal = player_removals.get(target)
                if not all(
                    isinstance(value, (int, float))
                    and math.isfinite(float(value))
                    for value in (strength, contribution, removal)
                ):
                    continue
                if float(strength) <= 0.0:
                    continue
                pairs.append((float(contribution), float(removal)))
                labels.append((str(player), str(target)))
        if pairs:
            contribution_values = np.asarray([pair[0] for pair in pairs])
            removal_values = np.asarray([pair[1] for pair in pairs])
            extra["explanation_edge_count"] = len(pairs)
            extra["explanation_faithfulness_mae"] = float(
                np.mean(np.abs(contribution_values - removal_values))
            )
            extra["explanation_sign_agreement"] = float(
                np.mean((contribution_values >= -1e-12) == (removal_values >= -1e-12))
            )
            if np.std(contribution_values) > 0 and np.std(removal_values) > 0:
                extra["explanation_faithfulness_correlation"] = float(
                    np.corrcoef(contribution_values, removal_values)[0, 1]
                )
            top = int(np.argmax(contribution_values))
            extra["top_explanation_player"] = labels[top][0]
            extra["top_explanation_target"] = labels[top][1]
            extra["top_explanation_contribution"] = float(
                contribution_values[top]
            )
            extra["top_explanation_utility_drop"] = float(removal_values[top])
    return recall, timing, extra


def _read_case_metadata(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        metadata = _read_json_object(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return {}

    clean = {}
    for key in (
        "io_seconds",
        "method_seconds",
        "observed_case_seconds",
        "warmup_seconds",
    ):
        value = metadata.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            clean[key] = float(value)
    peak = metadata.get("peak_rss_bytes")
    if isinstance(peak, int) and peak >= 0:
        clean["peak_rss_bytes"] = peak
    modality = metadata.get("input_modality")
    if isinstance(modality, str):
        clean["input_modality"] = modality
    rows = metadata.get("input_rows")
    if isinstance(rows, int) and rows >= 0:
        clean["input_rows"] = rows
    warmup = metadata.get("warmup_method_call")
    if isinstance(warmup, bool):
        clean["warmup_method_call"] = warmup
    return clean


def load_case_outcome(
    method: str,
    expected: ExpectedCase,
    result_dir: Path,
) -> CaseOutcome:
    result_path = result_dir / f"{expected.case_id}.json"
    diagnostics_path = result_dir / "diagnostics" / result_path.name
    metadata = _read_case_metadata(
        result_dir / "case_metadata" / result_path.name
    )
    candidate_recall, timing, diagnostic_metadata = _read_diagnostics(
        diagnostics_path, expected.root_service
    )
    if not result_path.exists():
        return CaseOutcome(
            method=method,
            **asdict(expected),
            status="missing",
            ranks=[],
            ranked_services=[],
            error="result artifact missing",
            candidate_recall=candidate_recall,
            stage_timing=timing,
            **diagnostic_metadata,
            **metadata,
        )

    try:
        data = _read_json_object(result_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return CaseOutcome(
            method=method,
            **asdict(expected),
            status="error",
            ranks=[],
            ranked_services=[],
            error=f"invalid result artifact: {exc}",
            candidate_recall=candidate_recall,
            stage_timing=timing,
            **diagnostic_metadata,
            **metadata,
        )

    if data.get("status") == "timeout":
        status = "timeout"
        ranks = []
        error = str(data.get("error", "per-case timeout"))
    elif "error" in data:
        status = "error"
        ranks = []
        error = str(data["error"])
    else:
        ranks = data.get("0", data.get(0))
        if not isinstance(ranks, list):
            status = "error"
            ranks = []
            error = "result does not contain a rank list"
        else:
            status = "success"
            ranks = [str(item) for item in ranks]
            error = None

    return CaseOutcome(
        method=method,
        **asdict(expected),
        status=status,
        ranks=ranks,
        ranked_services=unique_ranked_services(ranks),
        error=error,
        candidate_recall=candidate_recall,
        stage_timing=timing,
        **diagnostic_metadata,
        **metadata,
    )


def evaluate_method_dataset(
    method: str,
    dataset: str,
    expected_cases: Sequence[ExpectedCase],
    result_dir: Path,
    infeasible_reason: str | None = None,
) -> tuple[list[CaseOutcome], dict]:
    if infeasible_reason is not None:
        return [], {
            "method": method,
            "dataset": dataset,
            "combination_status": "infeasible",
            "infeasible_reason": infeasible_reason,
            "n_expected": len(expected_cases),
            "n_success": 0,
            "execution_success_rate": None,
            "overall": None,
            "successful_cases_only": None,
            "fault_breakdown": None,
        }

    outcomes = [
        load_case_outcome(method, expected, result_dir)
        for expected in expected_cases
    ]
    summary = summarize_outcomes(method, dataset, outcomes)
    task_path = result_dir.parent / "task.json"
    if task_path.is_file():
        try:
            task = json.loads(task_path.read_text(encoding="utf-8"))
            summary["task_observed_wall_seconds"] = task.get(
                "observed_task_wall_seconds"
            )
        except (OSError, json.JSONDecodeError):
            summary["task_observed_wall_seconds"] = None
    else:
        summary["task_observed_wall_seconds"] = None
    return outcomes, summary


def _metrics(outcomes: Sequence[CaseOutcome]) -> dict | None:
    if not outcomes:
        return None
    return {
        "AC@1": mean(outcome.hit(1) for outcome in outcomes),
        "AC@3": mean(outcome.hit(3) for outcome in outcomes),
        "AC@5": mean(outcome.hit(5) for outcome in outcomes),
        "Avg@5": mean(outcome.avg_at_5() for outcome in outcomes),
    }


def _distribution(values: Sequence[float]) -> dict | None:
    if not values:
        return None
    return {
        "n": len(values),
        "median": median(values),
        "iqr": [
            float(np.percentile(values, 25)),
            float(np.percentile(values, 75)),
        ],
        "p95": float(np.percentile(values, 95)),
    }


def summarize_outcomes(
    method: str, dataset: str, outcomes: Sequence[CaseOutcome]
) -> dict:
    successes = [outcome for outcome in outcomes if outcome.success]
    missing = [outcome for outcome in outcomes if outcome.status == "missing"]
    recalls = [
        outcome.candidate_recall
        for outcome in outcomes
        if outcome.candidate_recall is not None
    ]
    base_recalls = [
        outcome.base_candidate_recall
        for outcome in outcomes
        if outcome.base_candidate_recall is not None
    ]
    rescued_recalls = [
        outcome.rescued_candidate_recall
        for outcome in outcomes
        if outcome.rescued_candidate_recall is not None
    ]
    timing_totals = [
        sum(outcome.stage_timing.values())
        for outcome in outcomes
        if outcome.stage_timing
    ]
    stage_names = sorted(
        {stage for outcome in outcomes for stage in outcome.stage_timing}
    )
    stage_distributions = {
        stage: _distribution(
            [
                outcome.stage_timing[stage]
                for outcome in outcomes
                if stage in outcome.stage_timing
            ]
        )
        for stage in stage_names
    }
    io_times = [
        outcome.io_seconds
        for outcome in outcomes
        if outcome.io_seconds is not None
    ]
    method_times = [
        outcome.method_seconds
        for outcome in outcomes
        if outcome.method_seconds is not None
    ]
    warmup_times = [
        outcome.warmup_seconds
        for outcome in outcomes
        if outcome.warmup_seconds is not None
    ]
    def end_to_end(outcome: CaseOutcome) -> float | None:
        if outcome.observed_case_seconds is not None:
            return outcome.observed_case_seconds
        if outcome.io_seconds is not None and outcome.method_seconds is not None:
            return outcome.io_seconds + outcome.method_seconds
        return None

    end_to_end_times = [
        value for outcome in outcomes if (value := end_to_end(outcome)) is not None
    ]
    measured_end_to_end_times = [
        outcome.io_seconds + outcome.method_seconds
        for outcome in outcomes
        if outcome.io_seconds is not None and outcome.method_seconds is not None
    ]
    timeout_end_to_end_times = [
        value
        for outcome in outcomes
        if outcome.status == "timeout"
        and (value := end_to_end(outcome)) is not None
    ]
    peak_memory = [
        outcome.peak_rss_bytes
        for outcome in outcomes
        if outcome.peak_rss_bytes is not None
    ]
    candidate_stage_distributions = {
        key: _distribution(
            [
                float(value)
                for outcome in outcomes
                if (value := getattr(outcome, key)) is not None
            ]
        )
        for key in (
            "base_candidate_k",
            "coverage_rescued_k",
            "final_candidate_k",
            "target_count",
        )
    }
    graph_edge_counts = [len(outcome.graph_edges) for outcome in outcomes]
    explanation_edges = [
        float(outcome.explanation_edge_count)
        for outcome in outcomes
        if outcome.explanation_edge_count is not None
    ]
    faithfulness_correlations = [
        outcome.explanation_faithfulness_correlation
        for outcome in outcomes
        if outcome.explanation_faithfulness_correlation is not None
    ]
    faithfulness_mae = [
        outcome.explanation_faithfulness_mae
        for outcome in outcomes
        if outcome.explanation_faithfulness_mae is not None
    ]
    sign_agreement = [
        outcome.explanation_sign_agreement
        for outcome in outcomes
        if outcome.explanation_sign_agreement is not None
    ]
    first = outcomes[0] if outcomes else None
    warm_end_to_end = [
        value
        for outcome in outcomes[1:]
        if (value := end_to_end(outcome)) is not None
    ]
    faults = {}
    for fault in sorted({outcome.fault_type for outcome in outcomes}):
        fault_outcomes = [o for o in outcomes if o.fault_type == fault]
        faults[fault] = {
            "n": len(fault_outcomes),
            "successes": sum(o.success for o in fault_outcomes),
            **(_metrics(fault_outcomes) or {}),
        }
    return {
        "method": method,
        "dataset": dataset,
        "combination_status": "incomplete" if missing else "completed",
        "n_expected": len(outcomes),
        "n_success": len(successes),
        "n_error": sum(o.status == "error" for o in outcomes),
        "n_timeout": sum(o.status == "timeout" for o in outcomes),
        "n_missing": len(missing),
        "execution_success_rate": len(successes) / len(outcomes) if outcomes else None,
        "overall": _metrics(outcomes),
        "successful_cases_only": _metrics(successes),
        "candidate_recall": (
            sum(bool(value) for value in recalls) / len(recalls) if recalls else None
        ),
        "candidate_recall_n": len(recalls),
        "base_candidate_recall": (
            sum(bool(value) for value in base_recalls) / len(base_recalls)
            if base_recalls
            else None
        ),
        "base_candidate_recall_n": len(base_recalls),
        "rescued_candidate_recall": (
            sum(bool(value) for value in rescued_recalls) / len(rescued_recalls)
            if rescued_recalls
            else None
        ),
        "rescued_candidate_recall_n": len(rescued_recalls),
        "per_case_stage_time_median_seconds": (
            median(timing_totals) if timing_totals else None
        ),
        "per_case_stage_time_p95_seconds": (
            float(np.percentile(timing_totals, 95)) if timing_totals else None
        ),
        "stage_timing_seconds": stage_distributions,
        "dataset_observed_wall_seconds": (
            sum(end_to_end_times)
            if len(end_to_end_times) == len(outcomes) and outcomes
            else None
        ),
        "timing_seconds": {
            "io": _distribution(io_times),
            "method": _distribution(method_times),
            "warmup": _distribution(warmup_times),
            "cache_condition": (
                "warm_after_same_case_call"
                if any(outcome.warmup_method_call for outcome in outcomes)
                else "cold"
            ),
            "end_to_end": _distribution(end_to_end_times),
            # Excludes an explicit same-case warm-up call. This is the measured
            # invocation under either cache condition and is comparable across
            # the cold-independent and warm-same-case runs.
            "measured_end_to_end": _distribution(measured_end_to_end_times),
            "first_case_end_to_end": end_to_end(first) if first is not None else None,
            "remaining_cases_end_to_end": _distribution(warm_end_to_end),
            "timeouts_observed": _distribution(timeout_end_to_end_times),
        },
        "peak_rss_bytes": max(peak_memory) if peak_memory else None,
        "candidate_stage_k": candidate_stage_distributions,
        "graph_edge_count": _distribution(graph_edge_counts),
        "explanation_faithfulness": {
            "edge_count": _distribution(explanation_edges),
            "pearson_correlation": _distribution(faithfulness_correlations),
            "mean_absolute_difference": _distribution(faithfulness_mae),
            "sign_agreement": _distribution(sign_agreement),
        },
        "fault_breakdown": faults,
    }


def aggregate_method(method: str, summaries: Sequence[dict], outcomes: Sequence[CaseOutcome]) -> dict:
    completed = [s for s in summaries if s["combination_status"] == "completed"]
    all_completed = len(completed) == len(summaries) and bool(summaries)
    pooled = _metrics(outcomes) if all_completed else None
    macro = None
    if all_completed:
        macro = {
            metric: mean(s["overall"][metric] for s in summaries)
            for metric in ("AC@1", "AC@3", "AC@5", "Avg@5")
        }
    return {
        "method": method,
        "datasets": [s["dataset"] for s in summaries],
        "all_datasets_completed": all_completed,
        "macro_equal_system_weight": macro,
        "pooled": pooled,
        "pooled_n": len(outcomes) if all_completed else None,
    }


def exact_mcnemar(a: Sequence[CaseOutcome], b: Sequence[CaseOutcome], k: int = 1) -> dict:
    if len(a) != len(b):
        raise ValueError("paired outcomes must have the same length")
    if [o.case_id for o in a] != [o.case_id for o in b]:
        raise ValueError("paired outcomes must have identical case order")
    a_only = sum(x.hit(k) == 1.0 and y.hit(k) == 0.0 for x, y in zip(a, b))
    b_only = sum(x.hit(k) == 0.0 and y.hit(k) == 1.0 for x, y in zip(a, b))
    discordant = a_only + b_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = min(a_only, b_only)
        p_value = min(
            1.0,
            2.0
            * sum(math.comb(discordant, j) for j in range(tail + 1))
            / (2**discordant),
        )
    return {
        "n": len(a),
        "a_only": a_only,
        "b_only": b_only,
        "discordant": discordant,
        "p_value": p_value,
        "effect_pp_b_minus_a": 100.0 * mean(y.hit(k) - x.hit(k) for x, y in zip(a, b)),
    }


def _case_metric(outcome: CaseOutcome, metric: str) -> float:
    if metric == "Avg@5":
        return outcome.avg_at_5()
    if metric.startswith("AC@"):
        return outcome.hit(int(metric.split("@", 1)[1]))
    raise ValueError(f"unsupported paired metric: {metric}")


def paired_bootstrap(
    a: Sequence[CaseOutcome],
    b: Sequence[CaseOutcome],
    metric: str,
    resamples: int = 10000,
    seed: int = 20260713,
) -> dict:
    if len(a) != len(b) or [o.case_id for o in a] != [o.case_id for o in b]:
        raise ValueError("paired outcomes must have identical cases and order")
    if not a:
        raise ValueError("paired outcomes must not be empty")
    differences = np.asarray(
        [_case_metric(y, metric) - _case_metric(x, metric) for x, y in zip(a, b)],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(resamples, len(differences)))
    sampled = differences[indices].mean(axis=1)
    return {
        "metric": metric,
        "n": len(differences),
        "effect_pp_b_minus_a": float(100.0 * differences.mean()),
        "ci95_pp": [
            float(100.0 * np.percentile(sampled, 2.5)),
            float(100.0 * np.percentile(sampled, 97.5)),
        ],
        "resamples": resamples,
        "seed": seed,
    }


def _cluster_key(outcome: CaseOutcome) -> tuple[str, str, str]:
    return (outcome.dataset, outcome.root_service, outcome.fault_type)


def paired_cluster_bootstrap(
    a: Sequence[CaseOutcome],
    b: Sequence[CaseOutcome],
    metric: str,
    resamples: int = 10000,
    seed: int = 20260713,
) -> dict:
    """Paired bootstrap that resamples fault clusters, not independent cases."""

    if len(a) != len(b) or [o.case_id for o in a] != [o.case_id for o in b]:
        raise ValueError("paired outcomes must have identical cases and order")
    if not a:
        raise ValueError("paired outcomes must not be empty")
    differences = np.asarray(
        [_case_metric(y, metric) - _case_metric(x, metric) for x, y in zip(a, b)],
        dtype=float,
    )
    cluster_keys = [_cluster_key(outcome) for outcome in a]
    unique_clusters = sorted(set(cluster_keys))
    cluster_indices = [
        np.asarray(
            [index for index, key in enumerate(cluster_keys) if key == cluster],
            dtype=int,
        )
        for cluster in unique_clusters
    ]
    rng = np.random.default_rng(seed)
    sampled = np.empty(resamples, dtype=float)
    for iteration in range(resamples):
        selected = rng.integers(0, len(cluster_indices), size=len(cluster_indices))
        indices = np.concatenate([cluster_indices[index] for index in selected])
        sampled[iteration] = float(differences[indices].mean())
    return {
        "metric": metric,
        "n": len(differences),
        "n_clusters": len(unique_clusters),
        "cluster_definition": "dataset x root_service x fault_type",
        "effect_pp_b_minus_a": float(100.0 * differences.mean()),
        "ci95_pp": [
            float(100.0 * np.percentile(sampled, 2.5)),
            float(100.0 * np.percentile(sampled, 97.5)),
        ],
        "resamples": resamples,
        "seed": seed,
    }


def cluster_paired_permutation(
    a: Sequence[CaseOutcome],
    b: Sequence[CaseOutcome],
    k: int = 1,
    resamples: int = 10000,
    seed: int = 20260713,
) -> dict:
    """Two-sided paired sign-flip test with one sign per fault cluster."""

    if len(a) != len(b) or [o.case_id for o in a] != [o.case_id for o in b]:
        raise ValueError("paired outcomes must have identical cases and order")
    if not a:
        raise ValueError("paired outcomes must not be empty")
    differences = np.asarray(
        [y.hit(k) - x.hit(k) for x, y in zip(a, b)], dtype=float
    )
    cluster_keys = [_cluster_key(outcome) for outcome in a]
    unique_clusters = sorted(set(cluster_keys))
    cluster_indices = [
        np.asarray(
            [index for index, key in enumerate(cluster_keys) if key == cluster],
            dtype=int,
        )
        for cluster in unique_clusters
    ]
    observed = float(abs(differences.mean()))
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(resamples):
        permuted = differences.copy()
        signs = rng.choice((-1.0, 1.0), size=len(cluster_indices))
        for sign, indices in zip(signs, cluster_indices):
            permuted[indices] *= sign
        extreme += abs(float(permuted.mean())) >= observed - 1e-15
    return {
        "metric": f"AC@{k}",
        "n": len(differences),
        "n_clusters": len(unique_clusters),
        "cluster_definition": "dataset x root_service x fault_type",
        "effect_pp_b_minus_a": float(100.0 * differences.mean()),
        "p_value": float((extreme + 1) / (resamples + 1)),
        "resamples": resamples,
        "seed": seed,
    }


def paired_seed_average_cluster_bootstrap(
    reference: Sequence[CaseOutcome],
    seeded: Sequence[Sequence[CaseOutcome]],
    metric: str,
    resamples: int = 10000,
    seed: int = 20260713,
) -> dict:
    """Cluster bootstrap for the mean prediction metric across frozen seeds."""

    if not reference or not seeded:
        raise ValueError("reference and seeded outcomes must not be empty")
    case_ids = [outcome.case_id for outcome in reference]
    if any([outcome.case_id for outcome in run] != case_ids for run in seeded):
        raise ValueError("all seed runs must have identical cases and order")
    reference_values = np.asarray(
        [_case_metric(outcome, metric) for outcome in reference], dtype=float
    )
    seeded_values = np.asarray(
        [
            [_case_metric(outcome, metric) for outcome in run]
            for run in seeded
        ],
        dtype=float,
    )
    case_differences = seeded_values.mean(axis=0) - reference_values
    seed_effects = 100.0 * (
        seeded_values.mean(axis=1) - reference_values.mean()
    )
    cluster_keys = [_cluster_key(outcome) for outcome in reference]
    unique_clusters = sorted(set(cluster_keys))
    cluster_indices = [
        np.asarray(
            [index for index, key in enumerate(cluster_keys) if key == cluster],
            dtype=int,
        )
        for cluster in unique_clusters
    ]
    rng = np.random.default_rng(seed)
    sampled = np.empty(resamples, dtype=float)
    for iteration in range(resamples):
        selected = rng.integers(0, len(cluster_indices), size=len(cluster_indices))
        indices = np.concatenate([cluster_indices[index] for index in selected])
        sampled[iteration] = float(case_differences[indices].mean())
    return {
        "metric": metric,
        "n": len(reference),
        "n_clusters": len(unique_clusters),
        "n_seeds": len(seeded),
        "aggregation_order": (
            "case metric within seed -> seed-average case metric -> "
            "paired cluster resampling"
        ),
        "effect_pp_seed_mean_minus_reference": float(
            100.0 * case_differences.mean()
        ),
        "seed_effects_pp": [float(value) for value in seed_effects],
        "seed_effect_sd_pp": float(np.std(seed_effects, ddof=1))
        if len(seed_effects) > 1
        else 0.0,
        "ci95_pp": [
            float(100.0 * np.percentile(sampled, 2.5)),
            float(100.0 * np.percentile(sampled, 97.5)),
        ],
        "resamples": resamples,
        "seed": seed,
    }


def seed_average_cluster_permutation(
    reference: Sequence[CaseOutcome],
    seeded: Sequence[Sequence[CaseOutcome]],
    k: int = 1,
    resamples: int = 10000,
    seed: int = 20260713,
) -> dict:
    if not reference or not seeded:
        raise ValueError("reference and seeded outcomes must not be empty")
    case_ids = [outcome.case_id for outcome in reference]
    if any([outcome.case_id for outcome in run] != case_ids for run in seeded):
        raise ValueError("all seed runs must have identical cases and order")
    reference_values = np.asarray(
        [outcome.hit(k) for outcome in reference], dtype=float
    )
    seeded_values = np.asarray(
        [[outcome.hit(k) for outcome in run] for run in seeded], dtype=float
    )
    differences = seeded_values.mean(axis=0) - reference_values
    cluster_keys = [_cluster_key(outcome) for outcome in reference]
    unique_clusters = sorted(set(cluster_keys))
    cluster_indices = [
        np.asarray(
            [index for index, key in enumerate(cluster_keys) if key == cluster],
            dtype=int,
        )
        for cluster in unique_clusters
    ]
    observed = abs(float(differences.mean()))
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(resamples):
        permuted = differences.copy()
        signs = rng.choice((-1.0, 1.0), size=len(cluster_indices))
        for sign, indices in zip(signs, cluster_indices):
            permuted[indices] *= sign
        extreme += abs(float(permuted.mean())) >= observed - 1e-15
    return {
        "metric": f"AC@{k}",
        "n": len(reference),
        "n_clusters": len(unique_clusters),
        "n_seeds": len(seeded),
        "effect_pp_seed_mean_minus_reference": float(100.0 * differences.mean()),
        "p_value": float((extreme + 1) / (resamples + 1)),
        "resamples": resamples,
        "seed": seed,
    }


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    values = [float(value) for value in p_values]
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [0.0] * len(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def pair_by_case(
    a: Sequence[CaseOutcome], b: Sequence[CaseOutcome]
) -> tuple[list[CaseOutcome], list[CaseOutcome]]:
    a_index = {outcome.case_id: outcome for outcome in a}
    b_index = {outcome.case_id: outcome for outcome in b}
    common = sorted(set(a_index) & set(b_index))
    return [a_index[key] for key in common], [b_index[key] for key in common]
