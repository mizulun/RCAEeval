#!/usr/bin/env python3
"""Format per-fault MERIT ablation comparisons for paper tables."""

import argparse
import json
from pathlib import Path


DEFAULT_FAULTS = ["CPU", "MEM", "DISK", "SOCKET", "DELAY", "LOSS"]
DEFAULT_MATRIX_METRICS = ["AC@1", "Avg@5"]
METHOD_ALIASES = {
    "metric+trace": ["metric+trace", "no-banzhaf"],
    "no-banzhaf": ["no-banzhaf", "metric+trace"],
}


def _record_by_method(records):
    return {record["method"]: record for record in records}


def _require_float(record, key):
    if key not in record:
        raise KeyError(f"missing metric {key!r} for method {record.get('method')!r}")
    return float(record[key])


def _delta(variant_value, baseline_value):
    return round(variant_value - baseline_value, 10)


def build_rows(records, baseline, variants=None, faults=None):
    by_method = _record_by_method(records)
    if baseline not in by_method:
        raise KeyError(f"missing baseline method {baseline!r}")

    if variants is None:
        variants = [record["method"] for record in records if record["method"] != baseline]
    if faults is None:
        faults = DEFAULT_FAULTS

    base = by_method[baseline]
    rows = []
    for method in variants:
        if method not in by_method:
            raise KeyError(f"missing ablation method {method!r}")
        variant = by_method[method]
        for fault in faults:
            baseline_ac1 = _require_float(base, f"AC@1-{fault}")
            variant_ac1 = _require_float(variant, f"AC@1-{fault}")
            baseline_avg5 = _require_float(base, f"Avg@5-{fault}")
            variant_avg5 = _require_float(variant, f"Avg@5-{fault}")
            rows.append(
                {
                    "method": method,
                    "fault": fault,
                    "baseline_ac1": baseline_ac1,
                    "variant_ac1": variant_ac1,
                    "delta_ac1": _delta(variant_ac1, baseline_ac1),
                    "baseline_avg5": baseline_avg5,
                    "variant_avg5": variant_avg5,
                    "delta_avg5": _delta(variant_avg5, baseline_avg5),
                }
            )
    return rows


def _fmt(value):
    return f"{value:.3f}"


def _clean_cell(cell):
    return cell.strip().strip("`").replace("**", "").strip()


def _split_md_row(line):
    return [_clean_cell(cell) for cell in line.strip().strip("|").split("|")]


def _is_md_separator(line):
    cells = _split_md_row(line)
    return bool(cells) and all(set(cell.replace(":", "").strip()) <= {"-"} for cell in cells)


def parse_markdown_metric_tables(text, metrics=None):
    if metrics is None:
        metrics = DEFAULT_MATRIX_METRICS
    wanted = set(metrics)
    lines = text.splitlines()
    tables = {}
    i = 0
    current_metric = None
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith("## "):
            heading = stripped.lstrip("#").strip()
            current_metric = heading if heading in wanted else None
            i += 1
            continue

        if current_metric and stripped.startswith("|"):
            header = _split_md_row(stripped)
            if not header or header[0].lower() != "fault":
                i += 1
                continue
            if i + 1 >= len(lines) or not _is_md_separator(lines[i + 1]):
                i += 1
                continue
            method_names = header[1:]
            metric_table = {method: {} for method in method_names}
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                row = _split_md_row(lines[i])
                if len(row) >= len(method_names) + 1:
                    fault = row[0]
                    if fault.lower() != "mean":
                        for method, value in zip(method_names, row[1:]):
                            metric_table[method][fault] = float(value)
                i += 1
            tables[current_metric] = metric_table
            current_metric = None
            continue

        i += 1

    missing = [metric for metric in metrics if metric not in tables]
    if missing:
        raise KeyError(f"missing markdown metric sections: {missing}")
    return tables


def _resolve_matrix_method(tables, requested):
    available = set()
    for metric_table in tables.values():
        available.update(metric_table)
    candidates = METHOD_ALIASES.get(requested, [requested])
    for candidate in candidates:
        if candidate in available:
            return candidate
    raise KeyError(f"missing method {requested!r}; available methods: {sorted(available)}")


def build_matrix_rows(tables, methods, faults=None, metrics=None):
    if faults is None:
        faults = DEFAULT_FAULTS
    if metrics is None:
        metrics = DEFAULT_MATRIX_METRICS

    rows = []
    for requested_method in methods:
        source_method = _resolve_matrix_method(tables, requested_method)
        cells = {}
        for fault in faults:
            for metric in metrics:
                try:
                    cells[(fault, metric)] = float(tables[metric][source_method][fault])
                except KeyError as exc:
                    raise KeyError(
                        f"missing value for method={requested_method!r} "
                        f"source={source_method!r} fault={fault!r} metric={metric!r}"
                    ) from exc
        rows.append({"method": requested_method, "cells": cells})
    return rows


def _tex_escape(text):
    return str(text).replace("\\", "\\textbackslash{}").replace("_", "\\_")


def render_latex(rows, caption, label="tab:merit-re2tt-fault-ablation"):
    lines = [
        "\\begin{longtable}{llrrrrrr}",
        f"\\caption{{{_tex_escape(caption)}}}\\label{{{_tex_escape(label)}}}\\\\",
        "\\toprule",
        (
            "Ablation & Fault & Base AC@1 & Abl. AC@1 & $\\Delta$ AC@1 "
            "& Base Avg@5 & Abl. Avg@5 & $\\Delta$ Avg@5 \\\\"
        ),
        "\\midrule",
        "\\endfirsthead",
        "\\toprule",
        (
            "Ablation & Fault & Base AC@1 & Abl. AC@1 & $\\Delta$ AC@1 "
            "& Base Avg@5 & Abl. Avg@5 & $\\Delta$ Avg@5 \\\\"
        ),
        "\\midrule",
        "\\endhead",
    ]
    current_method = None
    for row in rows:
        method = _tex_escape(row["method"]) if row["method"] != current_method else ""
        current_method = row["method"]
        lines.append(
            " & ".join(
                [
                    method,
                    _tex_escape(row["fault"]),
                    _fmt(row["baseline_ac1"]),
                    _fmt(row["variant_ac1"]),
                    _fmt(row["delta_ac1"]),
                    _fmt(row["baseline_avg5"]),
                    _fmt(row["variant_avg5"]),
                    _fmt(row["delta_avg5"]),
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{longtable}", ""])
    return "\n".join(lines)


def render_markdown(rows):
    lines = [
        "| Ablation | Fault | Base AC@1 | Abl. AC@1 | Delta AC@1 | Base Avg@5 | Abl. Avg@5 | Delta Avg@5 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            (
                f"| `{row['method']}` | {row['fault']} | {_fmt(row['baseline_ac1'])} | "
                f"{_fmt(row['variant_ac1'])} | {_fmt(row['delta_ac1'])} | "
                f"{_fmt(row['baseline_avg5'])} | {_fmt(row['variant_avg5'])} | "
                f"{_fmt(row['delta_avg5'])} |"
            )
        )
    lines.append("")
    return "\n".join(lines)


def render_terminal_matrix(rows, faults, metrics=None):
    if metrics is None:
        metrics = DEFAULT_MATRIX_METRICS
    method_width = max(len("Method"), *(len(row["method"]) for row in rows)) + 2
    value_width = max([7, *(len(metric) for metric in metrics)])
    group_width = value_width * len(metrics) + (len(metrics) - 1)

    header_1 = f"{'Method':<{method_width}}"
    for fault in faults:
        header_1 += f"{fault:^{group_width}} "
    header_2 = " " * method_width
    for _fault in faults:
        header_2 += " ".join(f"{metric:>{value_width}}" for metric in metrics) + " "
    separator = "-" * len(header_2.rstrip())
    lines = [header_1.rstrip(), header_2.rstrip(), separator]
    for row in rows:
        line = f"{row['method']:<{method_width}}"
        for fault in faults:
            line += " ".join(
                f"{_fmt(row['cells'][(fault, metric)]):>{value_width}}"
                for metric in metrics
            )
            line += " "
        lines.append(line.rstrip())
    return "\n".join(lines)


def render_markdown_matrix(rows, faults, metrics=None):
    if metrics is None:
        metrics = DEFAULT_MATRIX_METRICS
    headers = ["Method"] + [f"{fault} {metric}" for fault in faults for metric in metrics]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|---" + "|---:" * (len(headers) - 1) + "|",
    ]
    for row in rows:
        values = [f"`{row['method']}`"]
        values.extend(_fmt(row["cells"][(fault, metric)]) for fault in faults for metric in metrics)
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    return "\n".join(lines)


def render_latex_matrix(
    rows,
    faults,
    metrics=None,
    caption="MERIT ablation results by fault type.",
    label="tab:merit-ablation-matrix",
):
    if metrics is None:
        metrics = DEFAULT_MATRIX_METRICS
    column_spec = "l" + "r" * (len(faults) * len(metrics))
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{{_tex_escape(caption)}}}",
        f"\\label{{{_tex_escape(label)}}}",
        f"\\begin{{tabular}}{{{column_spec}}}",
        "\\toprule",
        "Method"
        + "".join(f" & \\multicolumn{{{len(metrics)}}}{{c}}{{{_tex_escape(fault)}}}" for fault in faults)
        + " \\\\",
    ]
    cmidrules = []
    start = 2
    for fault in faults:
        end = start + len(metrics) - 1
        cmidrules.append(f"\\cmidrule(lr){{{start}-{end}}}")
        start = end + 1
    lines.append(" ".join(cmidrules))
    lines.append(" " + "".join(f" & {_tex_escape(metric)}" for _fault in faults for metric in metrics) + " \\\\")
    lines.append("\\midrule")
    for row in rows:
        values = [
            _fmt(row["cells"][(fault, metric)])
            for fault in faults
            for metric in metrics
        ]
        lines.append(_tex_escape(row["method"]) + " & " + " & ".join(values) + " \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Create paper-style per-fault ablation tables."
    )
    parser.add_argument("--benchmark", help="Input run123 benchmark JSON path")
    parser.add_argument("--source-md", help="Input Markdown ablation summary path")
    parser.add_argument("--baseline", default="merit_full", help="Baseline method name")
    parser.add_argument(
        "--variants",
        nargs="*",
        default=None,
        help="Ablation methods. Defaults to all methods except baseline.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        help="Matrix row method order for --source-md mode",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_MATRIX_METRICS,
        help="Metrics to include for --source-md matrix mode",
    )
    parser.add_argument(
        "--faults",
        nargs="+",
        default=DEFAULT_FAULTS,
        help="Fault types to include",
    )
    parser.add_argument(
        "--caption",
        default="Per-fault MERIT ablation on re2-tt.",
        help="LaTeX table caption",
    )
    parser.add_argument(
        "--label",
        default="tab:merit-re2tt-fault-ablation",
        help="LaTeX table label",
    )
    parser.add_argument("--tex-output", help="Output .tex path")
    parser.add_argument("--md-output", help="Optional Markdown preview output path")
    parser.add_argument(
        "--terminal",
        action="store_true",
        help="Print matrix table to terminal in --source-md mode",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    if args.source_md:
        if not args.methods:
            raise ValueError("--source-md mode requires --methods")
        text = Path(args.source_md).read_text(encoding="utf-8")
        tables = parse_markdown_metric_tables(text, metrics=args.metrics)
        matrix_rows = build_matrix_rows(
            tables,
            methods=args.methods,
            faults=args.faults,
            metrics=args.metrics,
        )
        if args.terminal:
            print(render_terminal_matrix(matrix_rows, faults=args.faults, metrics=args.metrics))
        if args.tex_output:
            tex_path = Path(args.tex_output)
            tex_path.parent.mkdir(parents=True, exist_ok=True)
            tex_path.write_text(
                render_latex_matrix(
                    matrix_rows,
                    faults=args.faults,
                    metrics=args.metrics,
                    caption=args.caption,
                    label=args.label,
                ),
                encoding="utf-8",
            )
            print(f"Wrote LaTeX matrix: {tex_path}")
        if args.md_output:
            md_path = Path(args.md_output)
            md_path.parent.mkdir(parents=True, exist_ok=True)
            md_path.write_text(
                render_markdown_matrix(matrix_rows, faults=args.faults, metrics=args.metrics),
                encoding="utf-8",
            )
            print(f"Wrote Markdown matrix: {md_path}")
        if not (args.terminal or args.tex_output or args.md_output):
            raise ValueError("--source-md mode requires --terminal, --tex-output, or --md-output")
        return

    if not args.benchmark:
        raise ValueError("either --benchmark or --source-md is required")
    if not args.tex_output:
        raise ValueError("--benchmark mode requires --tex-output")
    benchmark_path = Path(args.benchmark)
    records = json.loads(benchmark_path.read_text(encoding="utf-8"))
    rows = build_rows(
        records,
        baseline=args.baseline,
        variants=args.variants,
        faults=args.faults,
    )

    tex_path = Path(args.tex_output)
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(
        render_latex(rows, caption=args.caption, label=args.label),
        encoding="utf-8",
    )
    print(f"Wrote LaTeX table: {tex_path}")

    if args.md_output:
        md_path = Path(args.md_output)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(render_markdown(rows), encoding="utf-8")
        print(f"Wrote Markdown preview: {md_path}")


if __name__ == "__main__":
    main()
