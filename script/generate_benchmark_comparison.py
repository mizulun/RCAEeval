#!/usr/bin/env python3
"""Generate paper-style GA-Banzhaf versus BARO benchmark tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FAULTS = ("CPU", "MEM", "DISK", "SOCKET", "DELAY", "LOSS")
DATASETS = ("re2-ob", "re2-ss", "re2-tt")
METRICS = ("AC@1", "AC@3", "AC@5", "Avg@5")
METHOD_LABELS = {
    "baro": "BARO",
    "gabanzhaf": "GA-Banzhaf",
}


def format_delta_pp(delta_score: float) -> str:
    value = 100.0 * float(delta_score)
    if abs(value) < 0.05:
        return "0.0"
    return f"{value:+.1f}"


def _result_index(results: list[dict]) -> dict[tuple[str, str], dict]:
    return {
        (str(row["dataset"]), str(row["method"])): row
        for row in results
        if row.get("status", "OK") == "OK"
    }


def build_comparison_rows(
    results: list[dict],
    metric_prefix: str,
    faults: tuple[str, ...] = FAULTS,
) -> list[dict]:
    index = _result_index(results)
    datasets = [dataset for dataset in DATASETS if (dataset, "baro") in index]
    if not datasets:
        datasets = sorted({dataset for dataset, method in index if method == "baro"})

    rows: list[dict] = []
    for dataset in datasets:
        baro = index[(dataset, "baro")]
        gabanzhaf = index[(dataset, "gabanzhaf")]
        baro_values = [float(baro[f"{metric_prefix}-{fault}"]) for fault in faults]
        gabanzhaf_values = [
            float(gabanzhaf[f"{metric_prefix}-{fault}"]) for fault in faults
        ]

        baro_row = {"dataset": dataset, "method": METHOD_LABELS["baro"]}
        gabanzhaf_row = {
            "dataset": dataset,
            "method": METHOD_LABELS["gabanzhaf"],
        }
        delta_row = {"dataset": dataset, "method": "Delta (pp)"}
        for fault, baro_value, gabanzhaf_value in zip(
            faults, baro_values, gabanzhaf_values
        ):
            baro_row[fault] = baro_value
            gabanzhaf_row[fault] = gabanzhaf_value
            delta_row[fault] = 100.0 * (gabanzhaf_value - baro_value)

        baro_macro = sum(baro_values) / len(baro_values)
        gabanzhaf_macro = sum(gabanzhaf_values) / len(gabanzhaf_values)
        baro_row["Macro Avg."] = baro_macro
        gabanzhaf_row["Macro Avg."] = gabanzhaf_macro
        delta_row["Macro Avg."] = 100.0 * (gabanzhaf_macro - baro_macro)
        rows.extend((baro_row, gabanzhaf_row, delta_row))

    return rows


def _format_score(value: float, best: bool) -> str:
    text = f"{value:.2f}"
    return f"**{text}**" if best else text


def render_metric_table(
    results: list[dict],
    metric_prefix: str,
    title: str,
    faults: tuple[str, ...] = FAULTS,
) -> str:
    rows = build_comparison_rows(results, metric_prefix, faults)
    columns = (*faults, "Macro Avg.")
    lines = [
        f"### {title}",
        "",
        "| Dataset | Method | " + " | ".join(columns) + " |",
        "|---|---|" + "|".join("---:" for _ in columns) + "|",
    ]

    for offset in range(0, len(rows), 3):
        baro, gabanzhaf, delta = rows[offset : offset + 3]
        baro_cells = [
            _format_score(baro[column], baro[column] >= gabanzhaf[column])
            for column in columns
        ]
        gabanzhaf_cells = [
            _format_score(gabanzhaf[column], gabanzhaf[column] >= baro[column])
            for column in columns
        ]
        delta_cells = [
            format_delta_pp(
                (gabanzhaf[column] - baro[column])
                if column == "Macro Avg."
                else delta[column] / 100.0
            )
            for column in columns
        ]
        lines.append(
            f"| {baro['dataset']} | BARO | " + " | ".join(baro_cells) + " |"
        )
        lines.append(
            "|  | **GA-Banzhaf** | " + " | ".join(gabanzhaf_cells) + " |"
        )
        lines.append("|  | Delta (pp) | " + " | ".join(delta_cells) + " |")

    return "\n".join(lines)


def render_runtime_table(results: list[dict]) -> str:
    index = _result_index(results)
    lines = [
        "### Runtime comparison",
        "",
        "| Dataset | BARO (s/case) | GA-Banzhaf (s/case) | Ratio (GA/BARO) |",
        "|---|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        baro = float(index[(dataset, "baro")]["Avg speed"])
        gabanzhaf = float(index[(dataset, "gabanzhaf")]["Avg speed"])
        baro_text = f"{baro:.2f}"
        gabanzhaf_text = f"{gabanzhaf:.2f}"
        if baro <= gabanzhaf:
            baro_text = f"**{baro_text}**"
        if gabanzhaf <= baro:
            gabanzhaf_text = f"**{gabanzhaf_text}**"
        lines.append(
            f"| {dataset} | {baro_text} | {gabanzhaf_text} | "
            f"{gabanzhaf / baro:.2f}x |"
        )
    return "\n".join(lines)


def render_report(results: list[dict], source_name: str) -> str:
    sections = [
        "# GA-Banzhaf vs. BARO Benchmark Comparison",
        "",
        f"Source: `{source_name}`",
        "",
        "Accuracy differences use percentage points:",
        "",
        "`Delta (pp) = 100 x (GA-Banzhaf - BARO)`",
        "",
        "Higher accuracy is better; the best method value in each dataset/fault "
        "cell is bold. Macro Avg. is the unweighted mean across the six fault types.",
        "",
        "## Main Results",
        "",
        render_metric_table(results, "AC@1", "AC@1"),
        "",
        render_metric_table(results, "Avg@5", "Avg@5"),
        "",
        "## Supplementary Results",
        "",
        render_metric_table(results, "AC@3", "AC@3"),
        "",
        render_metric_table(results, "AC@5", "AC@5"),
        "",
        render_runtime_table(results),
        "",
    ]
    return "\n".join(sections)


def write_long_csv(results: list[dict], output_path: Path) -> None:
    index = _result_index(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "metric",
                "fault",
                "baro",
                "gabanzhaf",
                "delta_pp",
            ],
        )
        writer.writeheader()
        for dataset in DATASETS:
            for metric in METRICS:
                for fault in FAULTS:
                    baro = float(index[(dataset, "baro")][f"{metric}-{fault}"])
                    gabanzhaf = float(
                        index[(dataset, "gabanzhaf")][f"{metric}-{fault}"]
                    )
                    writer.writerow(
                        {
                            "dataset": dataset,
                            "metric": metric,
                            "fault": fault,
                            "baro": f"{baro:.2f}",
                            "gabanzhaf": f"{gabanzhaf:.2f}",
                            "delta_pp": format_delta_pp(gabanzhaf - baro),
                        }
                    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate GA-Banzhaf versus BARO paper-style tables."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("evidence/data/baro-re2.json"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("output/generated-docs/gabanzhaf-vs-baro.md"),
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("output/generated-docs/gabanzhaf-vs-baro.csv"),
    )
    args = parser.parse_args()

    with args.input.open(encoding="utf-8") as handle:
        results = json.load(handle)

    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(
        render_report(results, str(args.input)),
        encoding="utf-8",
    )
    write_long_csv(results, args.csv_output)


if __name__ == "__main__":
    main()
