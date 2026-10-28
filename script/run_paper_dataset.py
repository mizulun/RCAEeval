"""Run one method on one dataset with case-level failure accounting.

This is a paper-only harness.  It does not modify any baseline implementation,
does not choose an SLI from the injected service, and writes one artifact for
every attempted case so crashes and timeouts remain scoreable.
"""

from __future__ import annotations

import argparse
import importlib
import json
import multiprocessing
import os
import queue
import random
import threading
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter, sleep

import numpy as np
import pandas as pd
import psutil

from RCAEval.benchmark.input_boundaries import (
    ObservationBoundary,
    filter_span_frame,
    filter_timestamp_frame,
    full_record_boundary,
    window_boundary,
)


DATASET_PATHS = {
    "re2-ob": Path("data/RE2/RE2-OB"),
    "re2-ss": Path("data/RE2/RE2-SS"),
    "re2-tt": Path("data/RE2/RE2-TT"),
}


class CaseTimeout(TimeoutError):
    pass


@dataclass
class MemorySampler:
    interval_seconds: float = 0.01

    def __post_init__(self):
        self.peak_rss_bytes = psutil.Process().memory_info().rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        process = psutil.Process()
        while not self._stop.is_set():
            try:
                self.peak_rss_bytes = max(
                    self.peak_rss_bytes, process.memory_info().rss
                )
            except psutil.Error:
                return
            sleep(self.interval_seconds)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_args):
        self._stop.set()
        self._thread.join(timeout=1.0)
        try:
            self.peak_rss_bytes = max(
                self.peak_rss_bytes, psutil.Process().memory_info().rss
            )
        except psutil.Error:
            pass


def prefer_simple_data(paths: list[Path]) -> list[Path]:
    """Use the curated ``simple_data.csv`` next to a ``data.csv`` when the release ships both.

    RE1 sock-shop-2 and train-ticket ship ``data.csv`` as raw container metrics
    (438 / 1242 columns) and ``simple_data.csv`` as the standard service-level
    metrics (``<service>_cpu``, ``_mem``, ``_latency-90`` ...). Scoring the raw file
    is not comparable to published RCAEval results (issue #12). Opt-in only: the
    shared harness default stays ``data.csv``.
    """
    return [
        p.with_name("simple_data.csv") if p.name == "data.csv" and p.with_name("simple_data.csv").is_file() else p
        for p in paths
    ]


def discover_data_paths(
    repo_root: Path, dataset: str, prefer_simple: bool = False
) -> list[Path]:
    dataset_root = repo_root / DATASET_PATHS[dataset]
    paths = sorted(dataset_root.glob("*/*/data.csv"))
    paths = paths or sorted(dataset_root.glob("*/*/simple_metrics.csv"))
    return prefer_simple_data(paths) if prefer_simple else paths


def case_artifact_name(data_path: Path) -> str:
    # Evaluation-only naming. This value is never passed as a method argument.
    return f"{data_path.parent.parent.name}_{data_path.parent.name}.json"


def order_case_paths(
    paths: list[Path], case_order_seed: int | None, limit: int | None
) -> list[Path]:
    ordered = list(paths)
    if case_order_seed is not None:
        random.Random(case_order_seed).shuffle(ordered)
    return ordered if limit is None else ordered[:limit]


def prepare_case(
    data_path: Path,
    length: int | None = None,
    tdelta: int = 0,
    boundary: ObservationBoundary | None = None,
) -> tuple[pd.DataFrame, float]:
    data = pd.read_csv(data_path)
    data = data.loc[:, ~data.columns.str.endswith("_latency-50")]
    data = data.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    inject_time = float(
        (data_path.parent / "inject_time.txt").read_text(encoding="utf-8").splitlines()[0]
    )
    if boundary is None:
        boundary = (
            full_record_boundary(data["time"], inject_time)
            if length is None
            else window_boundary(inject_time, length, tdelta)
        )
    combined = filter_timestamp_frame(data, boundary).reset_index(drop=True)
    combined = combined.rename(
        columns={
            col: col.replace("_latency-90", "_latency")
            for col in combined.columns
            if col.endswith("_latency-90")
        }
    )
    return combined, boundary.inject_time


def prepare_trace_case(
    data_path: Path,
    inject_time: float | None = None,
    length: int | None = None,
    boundary: ObservationBoundary | None = None,
) -> tuple[pd.DataFrame, float]:
    """Load raw spans inside the same external boundary as metrics."""
    trace_path = data_path.parent / "traces.csv"
    if not trace_path.exists():
        raise FileNotFoundError(f"trace input is absent: {trace_path}")
    spans = pd.read_csv(trace_path)
    required = {
        "traceID",
        "spanID",
        "serviceName",
        "methodName",
        "operationName",
        "startTime",
        "duration",
        "parentSpanID",
    }
    missing = sorted(required - set(spans.columns))
    if missing:
        raise ValueError(f"trace input is missing required columns: {missing}")

    if boundary is None:
        if inject_time is None or length is None:
            raise ValueError("trace preparation requires a frozen boundary")
        boundary = window_boundary(inject_time, length)
    spans = filter_span_frame(spans, boundary)
    if spans.empty:
        raise ValueError("trace input has no spans in the frozen observation window")
    return spans, boundary.inject_time * 1_000_000.0


def _case_exposure(args, data_path: Path) -> tuple[dict, ObservationBoundary]:
    case_id = case_artifact_name(data_path).removesuffix(".json")
    exposure = args.exposure_manifest.get("cases", {}).get(case_id)
    if not isinstance(exposure, dict):
        raise ValueError(f"frozen input exposure is absent for case {case_id}")
    boundary = ObservationBoundary.from_dict(exposure.get("boundary", {}))
    if boundary.version != args.exposure_manifest.get("schema_version"):
        raise ValueError(f"input boundary schema mismatch for case {case_id}")
    return exposure, boundary


def load_method(method: str):
    if method == "rcd":
        return importlib.import_module("RCAEval.e2e.rcd").rcd
    package = importlib.import_module("RCAEval.e2e")
    if not hasattr(package, method):
        raise ValueError(f"method is not exported by RCAEval.e2e: {method}")
    return getattr(package, method)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except (ImportError, RuntimeError):
        pass


def _atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _execute_case(method, args, data_path: Path) -> dict:
    """Execute one case inside the persistent dataset worker."""
    seed_everything(args.seed)
    status = "success"
    error = None
    ranks = []
    diagnostics = None
    input_rows = 0
    method_started = None
    method_seconds = 0.0
    warmup_seconds = 0.0
    io_seconds = 0.0
    peak_rss_bytes = psutil.Process().memory_info().rss
    exposure = None
    boundary = None
    io_started = perf_counter()
    try:
        exposure, boundary = _case_exposure(args, data_path)
        metric_data, inject_time = prepare_case(data_path, boundary=boundary)
        if args.sli not in metric_data.columns:
            raise ValueError(
                f"frozen SLI {args.sli!r} is absent from telemetry columns"
            )
        if args.input_kind == "traces":
            method_data, method_inject_time = prepare_trace_case(
                data_path, boundary=boundary
            )
        else:
            method_data, method_inject_time = metric_data, inject_time
        input_rows = len(method_data)
        io_seconds = perf_counter() - io_started
        run_args = argparse.Namespace(
            root_path=str(args.repo_root.resolve()),
            data_path=str(data_path),
            input_boundary=boundary.to_dict(),
            case_exposure=exposure,
            dataset_exposure_manifest=args.exposure_manifest,
            graph_source=args.graph_source,
            graph_reference_case_id=getattr(args, "graph_reference_case_id", None),
        )
        method_kwargs = {
            "dataset": args.dataset,
            "anomalies": None,
            "dk_select_useful": False,
            "sli": args.sli,
            "verbose": False,
            "n_iter": len(method_data.columns) - 1,
            "args": run_args,
        }
        if args.warmup_method_call:
            warmup_started = perf_counter()
            warmup_output = method(
                method_data,
                method_inject_time,
                **method_kwargs,
            )
            warmup_seconds = perf_counter() - warmup_started
            if not isinstance(warmup_output, dict) or not isinstance(
                warmup_output.get("ranks"), list
            ):
                raise ValueError("warm-up method output must contain a rank list")
        method_started = perf_counter()
        with MemorySampler() as memory:
            output = method(
                method_data,
                method_inject_time,
                **method_kwargs,
            )
        method_seconds = perf_counter() - method_started
        if not isinstance(output, dict) or not isinstance(output.get("ranks"), list):
            raise ValueError("method output must contain a rank list")
        ranks = [str(item) for item in output["ranks"]]
        diagnostics = output.get("diagnostics")
        peak_rss_bytes = memory.peak_rss_bytes
    except Exception as exc:  # case-level integration/implementation failure
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
        method_seconds = (
            perf_counter() - method_started
            if isinstance(method_started, float)
            else 0.0
        )
        peak_rss_bytes = psutil.Process().memory_info().rss
        io_seconds = io_seconds or (perf_counter() - io_started)
    return {
        "status": status,
        "error": error,
        "ranks": ranks,
        "diagnostics": diagnostics,
        "io_seconds": io_seconds,
        "method_seconds": method_seconds,
        "warmup_seconds": warmup_seconds,
        "peak_rss_bytes": peak_rss_bytes,
        "input_rows": input_rows,
        "input_boundary": boundary.to_dict() if boundary is not None else None,
        "case_exposure": exposure,
    }


def _dataset_worker(args, input_queue, output_queue) -> None:
    """Keep method module state, including MERIT's dataset graph cache."""
    method = load_method(args.method)
    while True:
        raw_path = input_queue.get()
        if raw_path is None:
            return
        output_queue.put(_execute_case(method, args, Path(raw_path)))


def _rss_tree_bytes(pid: int) -> int:
    try:
        process = psutil.Process(pid)
        own_rss = process.memory_info().rss
    except (psutil.Error, OSError):
        return 0
    try:
        return own_rss + sum(
            child.memory_info().rss for child in process.children(recursive=True)
        )
    except (psutil.Error, OSError):
        # Managed sandboxes can deny KERN_PROC_ALL while still allowing the
        # worker's own RSS.  Preserve that valid lower-bound measurement.
        return own_rss


def _wait_for_worker_case(
    worker,
    input_queue,
    output_queue,
    data_path: Path,
    timeout_seconds: float,
) -> tuple[dict | None, int, float]:
    """Return a worker result, or ``None`` only after the hard deadline."""
    started = perf_counter()
    deadline = started + timeout_seconds
    peak_rss_bytes = _rss_tree_bytes(worker.pid)
    input_queue.put(str(data_path))
    while True:
        peak_rss_bytes = max(peak_rss_bytes, _rss_tree_bytes(worker.pid))
        remaining = deadline - perf_counter()
        if remaining <= 0:
            return None, peak_rss_bytes, perf_counter() - started
        try:
            result = output_queue.get(timeout=min(0.05, remaining))
            peak_rss_bytes = max(
                peak_rss_bytes,
                int(result.get("peak_rss_bytes") or 0),
                _rss_tree_bytes(worker.pid),
            )
            return result, peak_rss_bytes, perf_counter() - started
        except queue.Empty:
            if not worker.is_alive():
                raise RuntimeError(
                    f"dataset worker exited unexpectedly with code {worker.exitcode}"
                )


def _stop_worker(worker, input_queue=None) -> None:
    if not worker.is_alive():
        worker.join(timeout=1.0)
        return
    if input_queue is not None:
        input_queue.put(None)
        worker.join(timeout=5.0)
    if worker.is_alive():
        worker.terminate()
        worker.join(timeout=5.0)
    if worker.is_alive():
        worker.kill()
        worker.join(timeout=5.0)


def _start_worker(context, args):
    input_queue = context.Queue(maxsize=1)
    output_queue = context.Queue(maxsize=1)
    worker = context.Process(
        target=_dataset_worker, args=(args, input_queue, output_queue)
    )
    worker.start()
    return worker, input_queue, output_queue


def run_dataset(args) -> dict:
    repo_root = args.repo_root.resolve()
    result_dir = args.result_dir.resolve()
    if result_dir.exists() and any(result_dir.iterdir()):
        raise FileExistsError(f"result directory is not empty: {result_dir}")
    result_dir.mkdir(parents=True, exist_ok=True)
    cache_root = result_dir.parent / ".cache"
    cache_root.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    metadata_dir = result_dir / "case_metadata"
    diagnostics_dir = result_dir / "diagnostics"
    metadata_dir.mkdir()

    paths = order_case_paths(
        discover_data_paths(repo_root, args.dataset),
        args.case_order_seed,
        args.limit,
    )
    if not paths:
        raise FileNotFoundError(f"no cases found for {args.dataset}")
    # Resolve after ordering and limiting so every worker generation uses the
    # same experimental first case, including after a timeout/restart.
    args.graph_reference_case_id = case_artifact_name(paths[0]).removesuffix(".json")

    context = multiprocessing.get_context("spawn")
    worker, input_queue, output_queue = _start_worker(context, args)
    records = []
    try:
        for case_index, data_path in enumerate(paths):
            artifact_name = case_artifact_name(data_path)
            result_path = result_dir / artifact_name
            metadata_path = metadata_dir / artifact_name
            try:
                result, sampled_peak, observed_seconds = _wait_for_worker_case(
                    worker,
                    input_queue,
                    output_queue,
                    data_path,
                    args.per_case_timeout,
                )
            except Exception as exc:
                result = {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "ranks": [],
                    "diagnostics": None,
                    "io_seconds": 0.0,
                    "method_seconds": 0.0,
                    "peak_rss_bytes": _rss_tree_bytes(worker.pid),
                    "input_rows": 0,
                }
                sampled_peak = int(result["peak_rss_bytes"])
                observed_seconds = 0.0
                _stop_worker(worker)
                if case_index + 1 < len(paths):
                    worker, input_queue, output_queue = _start_worker(context, args)
            if result is None:
                _stop_worker(worker)
                result = {
                    "status": "timeout",
                    "error": f"per-case hard timeout: {args.per_case_timeout}s",
                    "ranks": [],
                    "diagnostics": None,
                    "io_seconds": None,
                    "method_seconds": None,
                    "peak_rss_bytes": sampled_peak,
                    "input_rows": 0,
                }
                if case_index + 1 < len(paths):
                    worker, input_queue, output_queue = _start_worker(context, args)

            status = result["status"]
            error = result["error"]
            result_payload = (
                {"0": result["ranks"]}
                if status == "success"
                else {"status": status, "error": error}
            )
            _atomic_json(result_path, result_payload)
            diagnostics = result.get("diagnostics")
            if isinstance(diagnostics, dict):
                diagnostics_dir.mkdir(exist_ok=True)
                _atomic_json(diagnostics_dir / artifact_name, diagnostics)
            metadata = {
                "status": status,
                "error": error,
                "io_seconds": result["io_seconds"],
                "method_seconds": result["method_seconds"],
                "warmup_seconds": result.get("warmup_seconds", 0.0),
                "warmup_method_call": args.warmup_method_call,
                "observed_case_seconds": observed_seconds,
                "peak_rss_bytes": max(
                    sampled_peak, int(result.get("peak_rss_bytes") or 0)
                ),
                "seed": args.seed,
                "per_case_timeout_seconds": args.per_case_timeout,
                "timeout_scope": "whole_case_hard_subprocess_deadline",
                "length_minutes": args.length,
                "tdelta_seconds": args.tdelta,
                "evaluation_mode": args.evaluation_mode,
                "sli": args.sli,
                "input_modality": args.input_kind,
                "input_rows": result["input_rows"],
                "input_boundary": result.get("input_boundary"),
                "case_exposure": result.get("case_exposure"),
                "graph_source": args.graph_source,
                "graph_reference_case_id": args.graph_reference_case_id,
                "case_order_seed": args.case_order_seed,
                "case_order_index": case_index,
            }
            _atomic_json(metadata_path, metadata)
            records.append(metadata)
    finally:
        _stop_worker(worker, input_queue)

    summary = {
        "method": args.method,
        "dataset": args.dataset,
        "cases_attempted": len(records),
        "successes": sum(record["status"] == "success" for record in records),
        "errors": sum(record["status"] == "error" for record in records),
        "timeouts": sum(record["status"] == "timeout" for record in records),
        "seed": args.seed,
        "fixed_sli": args.sli,
        "case_order_seed": args.case_order_seed,
        "graph_reference_case_id": args.graph_reference_case_id,
        "case_order": [case_artifact_name(path) for path in paths],
    }
    _atomic_json(result_dir.parent / "combination.json", summary)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--method", required=True)
    parser.add_argument("--dataset", choices=sorted(DATASET_PATHS), required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--sli", required=True)
    parser.add_argument("--input-kind", choices=["metrics", "traces"], default="metrics")
    parser.add_argument(
        "--evaluation-mode",
        choices=["full_record_retrospective", "window_sensitivity"],
        default="full_record_retrospective",
    )
    parser.add_argument("--case-order-seed", type=int)
    parser.add_argument("--warmup-method-call", action="store_true")
    parser.add_argument("--length", type=int)
    parser.add_argument("--tdelta", type=int, default=0)
    parser.add_argument("--exposure-manifest", type=Path, required=True)
    parser.add_argument(
        "--graph-source",
        choices=[
            "per_case_pre_fault",
            "per_case_full",
            "first_case_pre_fault",
            "union_pre_fault",
        ],
        default="per_case_pre_fault",
    )
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--per-case-timeout", type=int, default=900)
    parser.add_argument("--limit", type=int)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.evaluation_mode == "window_sensitivity" and args.length is None:
        raise ValueError("window sensitivity requires --length")
    args.exposure_manifest = json.loads(
        args.exposure_manifest.read_text(encoding="utf-8")
    )
    print(json.dumps(run_dataset(args), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
