"""Benchmark-defined telemetry exposure boundaries for paper experiments.

The paper protocol treats a case record as a half-open interval ``[start, end)``.
This module is the single place that derives that interval and applies it to
metrics, trace time series, and raw spans.  Method adapters may aggregate their
input differently, but they may not widen the externally supplied interval.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


BOUNDARY_SCHEMA_VERSION = "paper-full-record-v1"


@dataclass(frozen=True)
class ObservationBoundary:
    start: float
    end: float
    inject_time: float
    mode: str
    version: str = BOUNDARY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        values = (self.start, self.end, self.inject_time)
        if not all(np.isfinite(float(value)) for value in values):
            raise ValueError("observation boundary values must be finite")
        if not self.start < self.end:
            raise ValueError("observation boundary must satisfy start < end")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "ObservationBoundary":
        return cls(
            start=float(value["start"]),
            end=float(value["end"]),
            inject_time=float(value["inject_time"]),
            mode=str(value["mode"]),
            version=str(value.get("version", BOUNDARY_SCHEMA_VERSION)),
        )


def _numeric(values: Iterable) -> np.ndarray:
    array = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return array[np.isfinite(array)]


def full_record_boundary(metric_times: Iterable, inject_time: float) -> ObservationBoundary:
    """Derive the benchmark record from the metric clock.

    Metric telemetry is present for every RE2 case and therefore provides the
    common clock.  The right edge extends by one observed cadence so a sample at
    the final metric timestamp remains inside the half-open interval.
    """

    times = np.unique(_numeric(metric_times))
    if times.size == 0:
        raise ValueError("metric record has no finite timestamps")
    positive_steps = np.diff(times)
    positive_steps = positive_steps[positive_steps > 0]
    cadence = float(np.median(positive_steps)) if positive_steps.size else 1.0
    return ObservationBoundary(
        start=float(times[0]),
        end=float(times[-1] + cadence),
        inject_time=float(inject_time),
        mode="full_record_retrospective",
    )


def window_boundary(
    inject_time: float, length_minutes: int, tdelta_seconds: int = 0
) -> ObservationBoundary:
    if int(length_minutes) <= 0:
        raise ValueError("window length must be positive")
    shifted_inject = float(inject_time) + int(tdelta_seconds)
    half_window = float(length_minutes) * 60.0 / 2.0
    return ObservationBoundary(
        start=shifted_inject - half_window,
        end=shifted_inject + half_window,
        inject_time=shifted_inject,
        mode="window_sensitivity",
    )


def filter_timestamp_frame(
    frame: pd.DataFrame, boundary: ObservationBoundary, column: str = "time"
) -> pd.DataFrame:
    if column not in frame.columns:
        raise ValueError(f"input frame is missing timestamp column {column!r}")
    timestamps = pd.to_numeric(frame[column], errors="coerce")
    mask = timestamps.ge(boundary.start) & timestamps.lt(boundary.end)
    return frame.loc[mask].copy()


def filter_span_frame(
    frame: pd.DataFrame, boundary: ObservationBoundary
) -> pd.DataFrame:
    required = {"startTime", "duration"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"trace input is missing required columns: {missing}")
    start_us = pd.to_numeric(frame["startTime"], errors="coerce")
    duration_us = pd.to_numeric(frame["duration"], errors="coerce").fillna(0.0)
    end_us = start_us + duration_us
    lower = boundary.start * 1_000_000.0
    upper = boundary.end * 1_000_000.0
    return frame.loc[end_us.ge(lower) & end_us.lt(upper)].copy()


def _timestamp_summary(
    timestamps: Iterable, boundary: ObservationBoundary, scale: float = 1.0
) -> dict:
    values = _numeric(timestamps) / float(scale)
    values = values[(values >= boundary.start) & (values < boundary.end)]
    pre = values[values < boundary.inject_time]
    post = values[values >= boundary.inject_time]
    return {
        "available": True,
        "sample_count": int(values.size),
        "actual_start": float(values.min()) if values.size else None,
        "actual_end": float(values.max()) if values.size else None,
        "pre_sample_count": int(pre.size),
        "post_sample_count": int(post.size),
        "pre_duration_seconds": max(
            0.0, float(boundary.inject_time - boundary.start)
        ),
        "post_duration_seconds": max(
            0.0, float(boundary.end - boundary.inject_time)
        ),
    }


def _absent_summary(boundary: ObservationBoundary) -> dict:
    return {
        "available": False,
        "sample_count": 0,
        "actual_start": None,
        "actual_end": None,
        "pre_sample_count": 0,
        "post_sample_count": 0,
        "pre_duration_seconds": max(
            0.0, float(boundary.inject_time - boundary.start)
        ),
        "post_duration_seconds": max(
            0.0, float(boundary.end - boundary.inject_time)
        ),
    }


def inspect_case_exposure(
    data_path: Path,
    mode: str = "full_record_retrospective",
    length_minutes: int | None = None,
    tdelta_seconds: int = 0,
) -> dict:
    """Create one auditable modality record for a benchmark case."""

    metrics = pd.read_csv(data_path, usecols=lambda column: column == "time")
    inject_time = float(
        (data_path.parent / "inject_time.txt")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    if mode == "full_record_retrospective":
        boundary = full_record_boundary(metrics["time"], inject_time)
    elif mode == "window_sensitivity":
        if length_minutes is None:
            raise ValueError("window sensitivity requires length_minutes")
        boundary = window_boundary(inject_time, length_minutes, tdelta_seconds)
    else:
        raise ValueError(f"unsupported exposure mode: {mode}")

    modalities = {
        "metrics": _timestamp_summary(metrics["time"], boundary),
    }
    for name, filename in (
        ("trace_latency", "tracets_lat.csv"),
        ("trace_error", "tracets_err.csv"),
    ):
        path = data_path.parent / filename
        if path.is_file():
            frame = pd.read_csv(path, usecols=lambda column: column == "time")
            modalities[name] = _timestamp_summary(frame["time"], boundary)
        else:
            modalities[name] = _absent_summary(boundary)

    trace_path = data_path.parent / "traces.csv"
    if trace_path.is_file():
        spans = pd.read_csv(trace_path, usecols=["startTime", "duration"])
        span_end = (
            pd.to_numeric(spans["startTime"], errors="coerce")
            + pd.to_numeric(spans["duration"], errors="coerce").fillna(0.0)
        )
        modalities["raw_traces"] = _timestamp_summary(
            span_end, boundary, scale=1_000_000.0
        )
    else:
        modalities["raw_traces"] = _absent_summary(boundary)

    return {
        "schema_version": BOUNDARY_SCHEMA_VERSION,
        "case_id": f"{data_path.parent.parent.name}_{data_path.parent.name}",
        "fault_directory": data_path.parent.parent.name,
        "case_directory": data_path.parent.name,
        "metric_file": data_path.name,
        "boundary": boundary.to_dict(),
        "modalities": modalities,
    }
