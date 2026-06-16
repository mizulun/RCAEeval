import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "script"
    / "generate_benchmark_comparison.py"
)
SPEC = importlib.util.spec_from_file_location(
    "generate_benchmark_comparison",
    MODULE_PATH,
)
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)

build_comparison_rows = comparison.build_comparison_rows
format_delta_pp = comparison.format_delta_pp
render_metric_table = comparison.render_metric_table


def _sample_results():
    return [
        {
            "method": "gabanzhaf",
            "dataset": "re2-ob",
            "AC@1-CPU": 0.40,
            "AC@1-MEM": 0.80,
        },
        {
            "method": "baro",
            "dataset": "re2-ob",
            "AC@1-CPU": 0.00,
            "AC@1-MEM": 0.60,
        },
    ]


def test_format_delta_pp_uses_percentage_points():
    assert format_delta_pp(0.40) == "+40.0"
    assert format_delta_pp(-0.05) == "-5.0"
    assert format_delta_pp(0.0) == "0.0"


def test_build_comparison_rows_calculates_macro_average_and_pp_delta():
    rows = build_comparison_rows(
        _sample_results(),
        metric_prefix="AC@1",
        faults=("CPU", "MEM"),
    )

    assert rows[0]["method"] == "BARO"
    assert rows[0]["Macro Avg."] == pytest.approx(0.30)
    assert rows[1]["method"] == "GA-Banzhaf"
    assert rows[1]["Macro Avg."] == pytest.approx(0.60)
    assert rows[2]["method"] == "Delta (pp)"
    assert rows[2]["CPU"] == pytest.approx(40.0)
    assert rows[2]["MEM"] == pytest.approx(20.0)
    assert rows[2]["Macro Avg."] == pytest.approx(30.0)


def test_render_metric_table_groups_methods_under_each_dataset():
    table = render_metric_table(
        _sample_results(),
        metric_prefix="AC@1",
        title="AC@1 comparison",
        faults=("CPU", "MEM"),
    )

    assert "### AC@1 comparison" in table
    assert "| re2-ob | BARO | 0.00 | 0.60 | 0.30 |" in table
    assert "|  | **GA-Banzhaf** | **0.40** | **0.80** | **0.60** |" in table
    assert "|  | Delta (pp) | +40.0 | +20.0 | +30.0 |" in table
