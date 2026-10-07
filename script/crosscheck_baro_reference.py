"""Block paper tables unless self-run BARO reproduces RCAEval Table 6."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def crosscheck(summaries: list[dict], reference: dict) -> dict:
    matches = [
        summary
        for summary in summaries
        if summary.get("method") == reference["method"]
        and summary.get("dataset") == reference["dataset"]
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one {reference['method']} x {reference['dataset']} summary"
        )
    summary = matches[0]
    if summary.get("combination_status") != "completed":
        raise RuntimeError("BARO cross-check cell is incomplete")
    rows = []
    for fault, expected in reference["fault_values"].items():
        observed = summary.get("fault_breakdown", {}).get(fault, {}).get(
            reference["metric"]
        )
        if not isinstance(observed, (int, float)):
            raise RuntimeError(f"BARO cross-check lacks {reference['metric']}-{fault}")
        difference = abs(float(observed) - float(expected))
        rows.append(
            {
                "fault": fault,
                "expected": expected,
                "observed": observed,
                "absolute_difference": difference,
                "passed": difference <= reference["absolute_tolerance"],
            }
        )
    return {
        "method": reference["method"],
        "dataset": reference["dataset"],
        "metric": reference["metric"],
        "source": reference["source"],
        "source_url": reference["source_url"],
        "absolute_tolerance": reference["absolute_tolerance"],
        "rows": rows,
        "passed": all(row["passed"] for row in rows),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-summaries", type=Path, required=True)
    parser.add_argument(
        "--protocol", type=Path, default=Path("experiment/protocol.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    summaries = json.loads(args.dataset_summaries.read_text(encoding="utf-8"))
    reference = protocol["external_crosschecks"]["baro_re2_tt_fault_avg5"]
    result = {
        "protocol_version": protocol["protocol_version"],
        "generator": "script/crosscheck_baro_reference.py",
        **crosscheck(summaries, reference),
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
