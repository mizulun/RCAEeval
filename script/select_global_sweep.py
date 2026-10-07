"""Apply the frozen joint/LOSO rule to complete MERIT sweep artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median


PARAMETER_PREFIX = {
    "victim_floor": "merit_victim_",
    "trace_topology_conf": "merit_tconf_",
    "coverage_conf": "merit_cconf_",
    "exact_candidate_k": "merit_k",
}


def method_label(parameter: str, value: float | int) -> str:
    if parameter == "exact_candidate_k":
        return f"merit_k{int(value)}"
    return f"{PARAMETER_PREFIX[parameter]}{round(float(value) * 10):02d}"


def _tie_choice(parameter: str, values: list[float | int]) -> float | int:
    if parameter == "victim_floor":
        return max(values)
    return min(values)


def select_global_value(
    parameter: str,
    values: list[float | int],
    scores: dict[str, dict[float | int, float]],
    tolerance: float,
) -> dict:
    datasets = sorted(scores)
    if len(datasets) != 3:
        raise ValueError("global selection requires exactly three systems")
    for dataset in datasets:
        if set(scores[dataset]) != set(values):
            raise ValueError(f"incomplete grid for {dataset}")

    per_system_flat = {
        dataset: [
            value
            for value in values
            if max(scores[dataset].values()) - scores[dataset][value] <= tolerance
        ]
        for dataset in datasets
    }
    joint = sorted(set.intersection(*(set(v) for v in per_system_flat.values())))
    if joint:
        selected = _tie_choice(parameter, joint)
        return {
            "strategy": "joint_flat_region",
            "selected_value": selected,
            "per_system_flat_values": per_system_flat,
            "joint_values": joint,
            "folds": [],
        }

    folds = []
    selected_values = []
    for held_out in datasets:
        training = [dataset for dataset in datasets if dataset != held_out]
        training_scores = {
            value: sum(scores[dataset][value] for dataset in training) / 2.0
            for value in values
        }
        best = max(training_scores.values())
        eligible = [
            value
            for value in values
            if best - training_scores[value] <= tolerance
        ]
        selected = _tie_choice(parameter, eligible)
        selected_values.append(selected)
        folds.append(
            {
                "held_out": held_out,
                "training_systems": training,
                "eligible_values": eligible,
                "selected_value": selected,
                "training_macro_AC@1": training_scores[selected],
                "held_out_AC@1": scores[held_out][selected],
            }
        )
    return {
        "strategy": "leave_one_system_out",
        "selected_value": median(selected_values),
        "per_system_flat_values": per_system_flat,
        "joint_values": [],
        "folds": folds,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-summaries", type=Path, required=True)
    parser.add_argument(
        "--parameter", choices=tuple(PARAMETER_PREFIX), required=True
    )
    parser.add_argument(
        "--protocol", type=Path, default=Path("experiment/protocol.json")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    summaries = json.loads(args.dataset_summaries.read_text(encoding="utf-8"))
    values = protocol["sweeps"][args.parameter]
    labels = {method_label(args.parameter, value): value for value in values}
    datasets = protocol["datasets"]
    scores = {dataset: {} for dataset in datasets}
    rows = []
    for summary in summaries:
        value = labels.get(summary.get("method"))
        dataset = summary.get("dataset")
        if value is None or dataset not in scores:
            continue
        if summary.get("combination_status") != "completed":
            raise RuntimeError(
                f"incomplete sweep cell: {summary.get('method')} x {dataset}"
            )
        score = summary.get("overall", {}).get("AC@1")
        if not isinstance(score, (int, float)):
            raise RuntimeError(f"missing AC@1: {summary.get('method')} x {dataset}")
        if value in scores[dataset]:
            raise RuntimeError(f"duplicate sweep cell: {value} x {dataset}")
        scores[dataset][value] = float(score)
        rows.append(
            {
                "parameter": args.parameter,
                "value": value,
                "method": summary["method"],
                "dataset": dataset,
                "AC@1": float(score),
                "n_expected": summary["n_expected"],
                "n_success": summary["n_success"],
            }
        )

    selection = select_global_value(
        args.parameter,
        values,
        scores,
        protocol["flat_region_ac1_tolerance"],
    )
    payload = {
        "protocol_version": protocol["protocol_version"],
        "parameter": args.parameter,
        "tolerance": protocol["flat_region_ac1_tolerance"],
        "source": str(args.dataset_summaries),
        "generator": "script/select_global_sweep.py",
        **selection,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "selection.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "sweep.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
