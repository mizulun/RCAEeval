import json

import pandas as pd

from RCAEval.benchmark.input_boundaries import (
    BOUNDARY_SCHEMA_VERSION,
    ObservationBoundary,
    filter_span_frame,
    filter_timestamp_frame,
    full_record_boundary,
    inspect_case_exposure,
)
from script.run_paper_dataset import prepare_case, prepare_trace_case


def test_full_record_boundary_uses_metric_clock_and_half_open_cadence():
    boundary = full_record_boundary([10, 12, 14], inject_time=12)

    assert boundary.to_dict() == {
        "start": 10.0,
        "end": 16.0,
        "inject_time": 12.0,
        "mode": "full_record_retrospective",
        "version": BOUNDARY_SCHEMA_VERSION,
    }
    frame = pd.DataFrame({"time": [9, 10, 15.9, 16], "value": range(4)})
    assert filter_timestamp_frame(frame, boundary).time.tolist() == [10.0, 15.9]


def test_all_modalities_are_filtered_to_the_same_case_boundary(tmp_path):
    case_dir = tmp_path / "service_delay" / "1"
    case_dir.mkdir(parents=True)
    metric_path = case_dir / "simple_metrics.csv"
    pd.DataFrame(
        {"time": [100, 102, 104], "service_latency-90": [1, 2, 3]}
    ).to_csv(metric_path, index=False)
    (case_dir / "inject_time.txt").write_text("102\n", encoding="utf-8")
    pd.DataFrame(
        {"time": [98, 100, 104, 106], "service_op": [1, 2, 3, 4]}
    ).to_csv(case_dir / "tracets_lat.csv", index=False)
    pd.DataFrame(
        {
            "traceID": ["outside-left", "inside", "outside-right"],
            "spanID": ["1", "2", "3"],
            "serviceName": ["service"] * 3,
            "methodName": ["op"] * 3,
            "operationName": ["op"] * 3,
            "startTime": [98_000_000, 100_000_000, 106_000_000],
            "duration": [0, 0, 0],
            "parentSpanID": [""] * 3,
        }
    ).to_csv(case_dir / "traces.csv", index=False)

    exposure = inspect_case_exposure(metric_path)
    boundary = ObservationBoundary.from_dict(exposure["boundary"])
    metrics, inject_time = prepare_case(metric_path, boundary=boundary)
    spans, inject_us = prepare_trace_case(metric_path, boundary=boundary)

    assert metrics.time.tolist() == [100, 102, 104]
    assert spans.traceID.tolist() == ["inside"]
    assert inject_time == 102
    assert inject_us == 102_000_000
    assert exposure["modalities"]["metrics"]["sample_count"] == 3
    assert exposure["modalities"]["trace_latency"]["sample_count"] == 2
    assert exposure["modalities"]["trace_error"]["available"] is False
    assert exposure["modalities"]["raw_traces"]["sample_count"] == 1


def test_span_filter_uses_span_end_and_excludes_right_boundary():
    boundary = ObservationBoundary(
        start=10,
        end=20,
        inject_time=15,
        mode="full_record_retrospective",
    )
    frame = pd.DataFrame(
        {
            "startTime": [9_000_000, 19_000_000, 20_000_000],
            "duration": [1_000_000, 0, 0],
            "name": ["left-inclusive", "inside", "right-exclusive"],
        }
    )

    assert filter_span_frame(frame, boundary).name.tolist() == [
        "left-inclusive",
        "inside",
    ]
