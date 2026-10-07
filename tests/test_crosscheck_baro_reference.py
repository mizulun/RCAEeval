import pytest

from script.crosscheck_baro_reference import crosscheck


def _reference():
    return {
        "source": "official",
        "source_url": "https://example.test",
        "method": "baro",
        "dataset": "re2-tt",
        "metric": "Avg@5",
        "absolute_tolerance": 0.0051,
        "fault_values": {"cpu": 0.72, "delay": 0.63},
    }


def test_baro_crosscheck_accepts_two_decimal_reference_rounding():
    summaries = [
        {
            "method": "baro",
            "dataset": "re2-tt",
            "combination_status": "completed",
            "fault_breakdown": {
                "cpu": {"Avg@5": 0.72},
                "delay": {"Avg@5": 0.6333333333},
            },
        }
    ]

    result = crosscheck(summaries, _reference())

    assert result["passed"] is True


def test_baro_crosscheck_reports_blocking_discrepancy():
    summaries = [
        {
            "method": "baro",
            "dataset": "re2-tt",
            "combination_status": "completed",
            "fault_breakdown": {
                "cpu": {"Avg@5": 0.70},
                "delay": {"Avg@5": 0.63},
            },
        }
    ]

    result = crosscheck(summaries, _reference())

    assert result["passed"] is False
    assert result["rows"][0]["absolute_difference"] == pytest.approx(0.02)
