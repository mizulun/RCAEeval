import importlib.util
import json
from pathlib import Path


_spec = importlib.util.spec_from_file_location(
    "format_fault_ablation_table",
    Path(__file__).resolve().parent.parent / "script" / "format_fault_ablation_table.py",
)
formatter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(formatter)


def test_build_rows_compare_baseline_and_ablation_by_fault_type():
    records = [
        {
            "method": "merit_full",
            "dataset": "re2-tt",
            "AC@1-CPU": 0.8,
            "Avg@5-CPU": 0.9,
            "AC@1-MEM": 1.0,
            "Avg@5-MEM": 1.0,
        },
        {
            "method": "merit_no_trace",
            "dataset": "re2-tt",
            "AC@1-CPU": 0.6,
            "Avg@5-CPU": 0.7,
            "AC@1-MEM": 1.0,
            "Avg@5-MEM": 0.95,
        },
    ]

    rows = formatter.build_rows(
        records,
        baseline="merit_full",
        variants=["merit_no_trace"],
        faults=["CPU", "MEM"],
    )

    assert rows == [
        {
            "method": "merit_no_trace",
            "fault": "CPU",
            "baseline_ac1": 0.8,
            "variant_ac1": 0.6,
            "delta_ac1": -0.2,
            "baseline_avg5": 0.9,
            "variant_avg5": 0.7,
            "delta_avg5": -0.2,
        },
        {
            "method": "merit_no_trace",
            "fault": "MEM",
            "baseline_ac1": 1.0,
            "variant_ac1": 1.0,
            "delta_ac1": 0.0,
            "baseline_avg5": 1.0,
            "variant_avg5": 0.95,
            "delta_avg5": -0.05,
        },
    ]


def test_render_latex_uses_paper_table_shape():
    rows = [
        {
            "method": "merit_no_trace",
            "fault": "CPU",
            "baseline_ac1": 0.8,
            "variant_ac1": 0.6,
            "delta_ac1": -0.2,
            "baseline_avg5": 0.9,
            "variant_avg5": 0.7,
            "delta_avg5": -0.2,
        },
    ]

    latex = formatter.render_latex(rows, caption="MERIT ablation on re2-tt")

    assert "\\begin{longtable}" in latex
    assert "\\toprule" in latex
    assert "MERIT ablation on re2-tt" in latex
    assert "merit\\_no\\_trace" in latex
    assert "CPU & 0.800 & 0.600 & -0.200 & 0.900 & 0.700 & -0.200" in latex


def test_cli_writes_latex_and_markdown(tmp_path):
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(
        json.dumps(
            [
                {
                    "method": "merit_full",
                    "dataset": "re2-tt",
                    "AC@1-CPU": 0.8,
                    "Avg@5-CPU": 0.9,
                },
                {
                    "method": "merit_no_trace",
                    "dataset": "re2-tt",
                    "AC@1-CPU": 0.6,
                    "Avg@5-CPU": 0.7,
                },
            ]
        ),
        encoding="utf-8",
    )
    tex = tmp_path / "table.tex"
    md = tmp_path / "table.md"

    formatter.main(
        [
            "--benchmark",
            str(benchmark),
            "--baseline",
            "merit_full",
            "--variants",
            "merit_no_trace",
            "--faults",
            "CPU",
            "--tex-output",
            str(tex),
            "--md-output",
            str(md),
        ]
    )

    assert "merit\\_no\\_trace" in tex.read_text(encoding="utf-8")
    assert "| `merit_no_trace` | CPU | 0.800 | 0.600 | -0.200 |" in md.read_text(
        encoding="utf-8"
    )


def test_parse_markdown_metric_tables_and_alias_metric_trace():
    text = """
## AC@1

| fault | full | no-metric | no-trace | no-banzhaf |
| --- | --- | --- | --- | --- |
| CPU | 0.80 | 0.00 | 0.80 | 0.75 |
| MEM | 1.00 | 0.00 | 1.00 | 0.95 |

## Avg@5

| fault | full | no-metric | no-trace | no-banzhaf |
| --- | --- | --- | --- | --- |
| CPU | 0.96 | 0.03 | 0.96 | 0.90 |
| MEM | 1.00 | 0.00 | 1.00 | 0.91 |
"""

    tables = formatter.parse_markdown_metric_tables(text, metrics=["AC@1", "Avg@5"])
    rows = formatter.build_matrix_rows(
        tables,
        methods=["full", "metric+trace"],
        faults=["CPU", "MEM"],
        metrics=["AC@1", "Avg@5"],
    )

    assert rows == [
        {
            "method": "full",
            "cells": {
                ("CPU", "AC@1"): 0.8,
                ("CPU", "Avg@5"): 0.96,
                ("MEM", "AC@1"): 1.0,
                ("MEM", "Avg@5"): 1.0,
            },
        },
        {
            "method": "metric+trace",
            "cells": {
                ("CPU", "AC@1"): 0.75,
                ("CPU", "Avg@5"): 0.9,
                ("MEM", "AC@1"): 0.95,
                ("MEM", "Avg@5"): 0.91,
            },
        },
    ]


def test_render_terminal_matrix_places_faults_on_top_and_methods_left():
    rows = [
        {
            "method": "full",
            "cells": {
                ("CPU", "AC@1"): 0.8,
                ("CPU", "Avg@5"): 0.96,
                ("MEM", "AC@1"): 1.0,
                ("MEM", "Avg@5"): 1.0,
            },
        },
        {
            "method": "no-metric",
            "cells": {
                ("CPU", "AC@1"): 0.0,
                ("CPU", "Avg@5"): 0.03,
                ("MEM", "AC@1"): 0.0,
                ("MEM", "Avg@5"): 0.0,
            },
        },
    ]

    table = formatter.render_terminal_matrix(
        rows,
        faults=["CPU", "MEM"],
        metrics=["AC@1", "Avg@5"],
    )

    lines = table.splitlines()
    assert lines[0].startswith("Method")
    assert "CPU" in lines[0]
    assert "MEM" in lines[0]
    assert lines[1].strip().startswith("AC@1")
    assert lines[3].startswith("full")
    assert lines[4].startswith("no-metric")


def test_cli_reads_markdown_and_writes_matrix_outputs(tmp_path, capsys):
    source = tmp_path / "ablation.md"
    source.write_text(
        """
## AC@1

| fault | full | no-metric |
| --- | --- | --- |
| CPU | 0.80 | 0.00 |

## Avg@5

| fault | full | no-metric |
| --- | --- | --- |
| CPU | 0.96 | 0.03 |
""",
        encoding="utf-8",
    )
    tex = tmp_path / "matrix.tex"
    md = tmp_path / "matrix.md"

    formatter.main(
        [
            "--source-md",
            str(source),
            "--methods",
            "full",
            "no-metric",
            "--faults",
            "CPU",
            "--terminal",
            "--tex-output",
            str(tex),
            "--md-output",
            str(md),
        ]
    )

    stdout = capsys.readouterr().out
    assert "Method" in stdout
    assert "no-metric" in stdout
    assert "\\multicolumn{2}{c}{CPU}" in tex.read_text(encoding="utf-8")
    assert "\\begin{table}" in tex.read_text(encoding="utf-8")
    assert "| Method | CPU AC@1 | CPU Avg@5 |" in md.read_text(encoding="utf-8")
