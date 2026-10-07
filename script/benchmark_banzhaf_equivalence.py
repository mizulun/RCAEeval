"""Validate and benchmark MERIT's closed-form noisy-OR Banzhaf expression."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import tracemalloc
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from RCAEval.e2e.merit import (
    _bruteforce_probabilistic_banzhaf,
    _probabilistic_banzhaf,
)


def _measure(function, probability, weights):
    gc.collect()
    tracemalloc.start()
    started = perf_counter()
    value = function(probability, weights)
    seconds = perf_counter() - started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return value, seconds, peak_bytes


def _distribution(values):
    return {
        "median": median(values),
        "iqr": [
            float(np.percentile(values, 25)),
            float(np.percentile(values, 75)),
        ],
        "min": min(values),
        "max": max(values),
    }


def run_benchmark(
    k_values,
    target_count,
    repetitions,
    seed,
    absolute_tolerance,
):
    rng = np.random.default_rng(seed)
    raw = []
    summaries = []
    for k in k_values:
        probability = rng.uniform(0.0, 1.0, size=(k, target_count))
        weights = rng.uniform(0.0, 1.0, size=target_count)
        errors = []
        method_records = {"closed_form": [], "bruteforce": []}
        for repetition in range(repetitions):
            order = (
                ("closed_form", _probabilistic_banzhaf),
                ("bruteforce", _bruteforce_probabilistic_banzhaf),
            )
            if repetition % 2:
                order = tuple(reversed(order))
            values = {}
            for mode, function in order:
                result, seconds, peak_bytes = _measure(
                    function, probability, weights
                )
                values[mode] = result
                record = {
                    "k": k,
                    "target_count": target_count,
                    "repetition": repetition,
                    "mode": mode,
                    "seconds": seconds,
                    "tracemalloc_peak_bytes": peak_bytes,
                }
                raw.append(record)
                method_records[mode].append(record)
            errors.append(
                float(
                    np.max(
                        np.abs(values["closed_form"] - values["bruteforce"])
                    )
                )
            )

        max_error = max(errors)
        if max_error > absolute_tolerance:
            raise AssertionError(
                f"K={k}: max absolute error {max_error} exceeds "
                f"{absolute_tolerance}"
            )
        summary = {
            "k": k,
            "target_count": target_count,
            "repetitions": repetitions,
            "max_absolute_error": max_error,
            "tolerance": absolute_tolerance,
            "equivalent": True,
            "modes": {},
        }
        for mode, records in method_records.items():
            summary["modes"][mode] = {
                "seconds": _distribution([row["seconds"] for row in records]),
                "peak_bytes": _distribution(
                    [row["tracemalloc_peak_bytes"] for row in records]
                ),
            }
        closed = summary["modes"]["closed_form"]["seconds"]["median"]
        brute = summary["modes"]["bruteforce"]["seconds"]["median"]
        summary["median_speedup_bruteforce_over_closed_form"] = (
            brute / closed if closed > 0 else None
        )
        summaries.append(summary)
    return raw, summaries


def observed_game_shapes(case_records):
    counts = {}
    for record in case_records:
        if record.get("method") != "merit_full" or record.get("status") != "success":
            continue
        k = record.get("final_candidate_k")
        n = record.get("target_count")
        if isinstance(k, int) and k > 0 and isinstance(n, int) and n > 0:
            counts[(k, n)] = counts.get((k, n), 0) + 1
    return [
        {"k": k, "target_count": n, "observed_cases": count}
        for (k, n), count in sorted(counts.items())
    ]


def run_observed_shape_benchmark(
    shapes, repetitions, seed, absolute_tolerance, max_k
):
    raw = []
    summaries = []
    for index, shape in enumerate(shapes):
        k = int(shape["k"])
        target_count = int(shape["target_count"])
        if k > max_k:
            raise ValueError(f"observed K={k} exceeds frozen max_k={max_k}")
        shape_raw, shape_summaries = run_benchmark(
            [k],
            target_count,
            repetitions,
            seed + index,
            absolute_tolerance,
        )
        for record in shape_raw:
            record["observed_cases"] = int(shape["observed_cases"])
        for summary in shape_summaries:
            summary["observed_cases"] = int(shape["observed_cases"])
        raw.extend(shape_raw)
        summaries.extend(shape_summaries)
    return raw, summaries


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol", type=Path, default=Path("experiment/protocol.json")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-results", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    config = protocol["banzhaf_equivalence"]
    if max(config["k_values"]) > config["max_k"]:
        raise ValueError("equivalence grid exceeds frozen max_k")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    raw, summaries = run_benchmark(
        k_values=config["k_values"],
        target_count=config["target_count"],
        repetitions=config["repetitions"],
        seed=config["seed"],
        absolute_tolerance=config["absolute_tolerance"],
    )
    shapes = []
    observed_raw = []
    observed_summaries = []
    if args.case_results:
        case_records = json.loads(args.case_results.read_text(encoding="utf-8"))
        shapes = observed_game_shapes(case_records)
        observed_raw, observed_summaries = run_observed_shape_benchmark(
            shapes,
            config["repetitions"],
            config["seed"],
            config["absolute_tolerance"],
            config["max_k"],
        )
    payload = {
        "protocol_version": protocol["protocol_version"],
        "config": config,
        "raw": raw,
        "summaries": summaries,
        "observed_shapes": shapes,
        "observed_shape_raw": observed_raw,
        "observed_shape_summaries": observed_summaries,
    }
    (args.output_dir / "banzhaf_equivalence.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "banzhaf_equivalence.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw[0]))
        writer.writeheader()
        writer.writerows(raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
