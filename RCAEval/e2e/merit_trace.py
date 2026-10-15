"""Trace ingestion, graph-source resolution, and topology features for MERIT."""

from dataclasses import dataclass, replace
from hashlib import sha256
import json
import os
from time import perf_counter

import numpy as np
import pandas as pd

from RCAEval.benchmark.input_boundaries import (
    BOUNDARY_SCHEMA_VERSION,
    ObservationBoundary,
    filter_span_frame,
    filter_timestamp_frame,
)
from RCAEval.e2e.merit_ablation import (
    _add_trace_series_noise as _ablation_add_trace_series_noise,
    _degree_matched_random_edges,
    _deterministic_row_dropout,
    _inject_random_trace_edges as _ablation_inject_random_trace_edges,
    _time_shuffle_trace_series,
)
from RCAEval.e2e.merit_evidence import _finite_nonnegative, _norm_dict, _parse_svc


@dataclass(frozen=True)
class _TraceTopologyFeatures:
    edges: list
    service_evidence: dict
    latency_evidence: dict
    error_evidence: dict
    stage_timing: dict


@dataclass(frozen=True)
class GraphArtifact:
    graph: dict
    source: str
    source_scope: str
    source_case: str | None
    source_cases: tuple
    window: str
    boundary_version: str
    manifest_sha256: str | None = None
    cache_hit: bool = False
    empty_graph_cached: bool = False

    def provenance(self):
        return {
            "source": self.source,
            "source_scope": self.source_scope,
            "source_case": self.source_case,
            "source_cases": list(self.source_cases),
            "source_case_count": len(self.source_cases),
            "window": self.window,
            "boundary_version": self.boundary_version,
            "manifest_sha256": self.manifest_sha256,
            "cache_hit": self.cache_hit,
            "edge_count": int(sum(len(values) for values in self.graph.values())),
            "empty_graph_cached": self.empty_graph_cached,
        }


_CALL_GRAPH_CACHE = {}


def _add_trace_series_noise(trace_series, fraction, salt):
    return _ablation_add_trace_series_noise(
        trace_series, fraction, salt, _finite_nonnegative=_finite_nonnegative
    )


def _inject_random_trace_edges(edges, services, fraction):
    return _ablation_inject_random_trace_edges(
        edges, services, fraction, _finite_nonnegative=_finite_nonnegative
    )


def _match_service(trace_name, service_names):
    """Map a trace serviceName to a metric service name.

    Trace span service names occasionally differ from the metric column prefix
    (e.g. "frontendservice" vs "frontend"). With no exact match, pick the
    longest metric service name the trace name starts with; else keep it as-is.
    Purely data-driven — no per-dataset hardcoding.
    """
    if not service_names or trace_name in service_names:
        return trace_name
    best = None
    for svc in service_names:
        if trace_name.startswith(svc) and (best is None or len(svc) > len(best)):
            best = svc
    return best or trace_name


def _build_call_graph_from_traces(
    case_dir,
    service_names=None,
    cache_key=None,
    boundary=None,
    pre_fault=False,
    dropout_fraction=0.0,
):
    """Build {caller: [callees]} from a case's traces.csv span hierarchy.

    A span's parent service is the caller, the span's own service is the callee
    (the called downstream dependency). Same construction for every dataset; no
    precomputed files, no dataset-name branching. Returns {} when no traces.csv
    ships (e.g. sock-shop) so the pipeline degrades to pure metric ranking.

    When a cache key is supplied it must describe the graph source, source path,
    boundary version, and service schema.  Empty graphs are deliberately not
    cached, so a missing or malformed first record cannot poison later cases.
    """
    if not case_dir:
        return {}
    boundary_obj = (
        ObservationBoundary.from_dict(boundary)
        if isinstance(boundary, dict)
        else boundary
    )
    if pre_fault and boundary_obj is None:
        raise ValueError("pre-fault graph filtering requires an input boundary")
    boundary_signature = (
        (
            boundary_obj.version,
            float(boundary_obj.start),
            float(boundary_obj.end),
            float(boundary_obj.inject_time),
        )
        if boundary_obj is not None
        else (None, None, None, None)
    )
    boundary_signature += (
        bool(pre_fault),
        float(np.clip(dropout_fraction, 0.0, 1.0)),
    )
    key = (
        cache_key if cache_key is not None else os.path.realpath(case_dir),
        boundary_signature,
        tuple(sorted(service_names)) if service_names else None,
    )
    if key in _CALL_GRAPH_CACHE:
        return _CALL_GRAPH_CACHE[key]

    path = os.path.join(case_dir, "traces.csv")
    graph = {}
    if os.path.exists(path):
        try:
            usecols = ["spanID", "parentSpanID", "serviceName"]
            if boundary_obj is not None:
                usecols.extend(["startTime", "duration"])
            df = pd.read_csv(path, usecols=usecols, dtype={
                "spanID": str,
                "parentSpanID": str,
                "serviceName": str,
            }).dropna(subset=["spanID", "serviceName"])
            if boundary_obj is not None:
                graph_boundary = boundary_obj
                if pre_fault:
                    graph_boundary = ObservationBoundary(
                        start=boundary_obj.start,
                        end=min(boundary_obj.end, boundary_obj.inject_time),
                        inject_time=boundary_obj.inject_time,
                        mode=boundary_obj.mode,
                        version=boundary_obj.version,
                    )
                df = filter_span_frame(df, graph_boundary)
            df = _deterministic_row_dropout(
                df,
                dropout_fraction,
                f"raw-spans:{os.path.realpath(case_dir)}",
            )
            parent = df[["spanID", "serviceName"]].rename(
                columns={"spanID": "parentSpanID", "serviceName": "callerSvc"}
            )
            joined = df.merge(parent, on="parentSpanID", how="inner")
            # _match_service only depends on the raw name (and the fixed
            # service_names set), so resolve each unique raw name once and
            # broadcast via a vector map instead of calling it per row.
            unique_names = pd.unique(
                pd.concat([joined["callerSvc"], joined["serviceName"]], copy=False)
            )
            match_map = {
                name: _match_service(name, service_names) for name in unique_names
            }
            caller_matched = joined["callerSvc"].map(match_map)
            callee_matched = joined["serviceName"].map(match_map)
            edge_mask = caller_matched != callee_matched
            edge_df = pd.DataFrame(
                {"caller": caller_matched[edge_mask], "callee": callee_matched[edge_mask]}
            )
            graph = {
                caller: sorted(set(group))
                for caller, group in edge_df.groupby("caller")["callee"]
            }
        except (OSError, ValueError, KeyError):
            graph = {}

    if graph:
        _CALL_GRAPH_CACHE[key] = graph
    return graph


def _service_trace_series_from_frame(df, inject_time=None, residual=False):
    """Aggregate trace operation time series to service-level trace series.

    When ``residual`` and ``inject_time`` are set, each column is reduced to its
    fault-induced delta: subtract the pre-inject baseline (mean of rows before
    ``inject_time``) from the post-inject rows and clip negatives to zero. This
    drops services that merely co-activate during the fault window but are not on
    the propagation path, widening the signal-to-noise gap in edge scoring.
    """
    if df is None or df.empty:
        return {}

    use_residual = bool(residual) and inject_time is not None and "time" in df.columns
    if use_residual:
        times = pd.to_numeric(df["time"], errors="coerce").to_numpy(dtype=float)
        pre_mask = times < float(inject_time)
        post_mask = ~pre_mask

    service_values = {}
    for col in df.columns:
        if col == "time":
            continue
        svc = str(col).rsplit("_", 1)[0]
        values = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if use_residual:
            baseline = float(values[pre_mask].mean()) if pre_mask.any() else 0.0
            values = np.clip(values[post_mask] - baseline, 0.0, None)
        if svc in service_values:
            service_values[svc] = service_values[svc] + values
        else:
            service_values[svc] = values
    return service_values


def _norm_trace_arr(values):
    arr = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size == 0:
        return arr
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if hi <= lo:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


def _lagged_cosine_score(source, target, lag):
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    lag = max(0, int(lag))
    n = min(source.size, target.size)
    if n == 0 or lag >= n:
        return 0.0
    if lag == 0:
        a = source[:n]
        b = target[:n]
    else:
        a = source[: n - lag]
        b = target[lag:n]
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return max(0.0, min(float(np.dot(a, b) / denom), 1.0))


def _trace_multilag_edges(call_graph, trace_series, config, source_kind):
    """Build callee->caller propagation edges from trace time-series lag."""
    if not call_graph or not trace_series:
        return []

    max_lag = max(0, int(config.trace_topology_max_lag))
    base_conf = float(np.clip(config.trace_topology_conf, 0.0, 1.0))
    min_conf = _finite_nonnegative(config.trace_topology_min_conf)
    normalized = {
        svc: _norm_trace_arr(values)
        for svc, values in trace_series.items()
    }

    # Energy gate (residual mode only): drop services whose fault-induced bump is
    # negligible relative to the loudest service, so unperturbed-but-co-active
    # services never form edges. Measured on raw (pre-normalization) magnitude.
    low_energy = set()
    if config.trace_residual:
        peaks = {
            svc: float(np.max(values)) if np.asarray(values).size else 0.0
            for svc, values in trace_series.items()
        }
        max_peak = max(peaks.values(), default=0.0)
        min_energy = _finite_nonnegative(config.trace_min_energy)
        if max_peak > 0.0:
            low_energy = {
                svc for svc, peak in peaks.items()
                if peak / max_peak < min_energy
            }

    edges = []
    for caller, callees in call_graph.items():
        caller_values = normalized.get(caller)
        if caller_values is None or caller in low_energy:
            continue
        for callee in callees:
            callee_values = normalized.get(callee)
            if callee_values is None or callee in low_energy:
                continue
            best_score = 0.0
            best_lag = 0
            for lag in range(max_lag + 1):
                score = _lagged_cosine_score(callee_values, caller_values, lag)
                if score > best_score:
                    best_score = score
                    best_lag = lag
            confidence = base_conf * best_score
            if confidence >= min_conf:
                edges.append((callee, caller, float(confidence), source_kind, int(best_lag)))
    return sorted(edges, key=lambda edge: (edge[0], edge[1], edge[3], edge[4]))


def _merge_trace_topology_edges(edge_groups):
    merged = {}
    for edges in edge_groups:
        for source, target, confidence, source_kind, lag in edges:
            key = (source, target)
            confidence = float(np.clip(confidence, 0.0, 1.0))
            existing = merged.get(key)
            if existing is None or confidence > existing[2]:
                merged[key] = (source, target, confidence, source_kind, int(lag))
    return sorted(merged.values(), key=lambda edge: (edge[0], edge[1], edge[3], edge[4]))


def _trace_service_evidence(trace_series):
    """Score trace residual magnitude on a shared cross-service scale.

    The previous implementation min-max normalized each service separately and
    then took its maximum, assigning 1.0 to every non-constant series.  Here the
    within-service temporal magnitude is computed first, then normalized across
    services so amplitudes remain comparable.
    """
    raw_evidence = {}
    for svc, values in (trace_series or {}).items():
        arr = np.nan_to_num(
            np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0
        )
        arr = np.clip(arr, 0.0, None)
        raw_evidence[svc] = (
            float(0.7 * np.percentile(arr, 95) + 0.3 * np.max(arr))
            if arr.size
            else 0.0
        )
    return _norm_dict(raw_evidence) if raw_evidence else {}


def _trace_topology_from_case_dir(
    case_dir, call_graph, config, inject_time=None, boundary=None, need_evidence=True
):
    stage_timing = {}
    last_tick = perf_counter()

    def _mark_stage(name):
        nonlocal last_tick
        now = perf_counter()
        stage_timing[name] = float(now - last_tick)
        last_tick = now

    if not case_dir or not config.use_trace_topology:
        return _TraceTopologyFeatures([], {}, {}, {}, stage_timing)

    if config.trace_signal not in {"both", "latency", "error"}:
        raise ValueError(f"unsupported trace signal: {config.trace_signal}")
    if config.trace_control not in {
        "observed",
        "time_shuffle",
        "degree_matched_random_edges",
    }:
        raise ValueError(f"unsupported trace control: {config.trace_control}")

    residual = bool(config.trace_residual)
    boundary_obj = (
        ObservationBoundary.from_dict(boundary)
        if isinstance(boundary, dict)
        else boundary
    )
    latency_series = {}
    error_series = {}
    lat_path = os.path.join(case_dir, "tracets_lat.csv")
    err_path = os.path.join(case_dir, "tracets_err.csv")

    latency_evidence_series = {}
    error_evidence_series = {}
    if config.trace_signal in {"both", "latency"} and os.path.exists(lat_path):
        try:
            latency_frame = pd.read_csv(lat_path)
            if boundary_obj is not None:
                latency_frame = filter_timestamp_frame(
                    latency_frame, boundary_obj
                )
            latency_frame = _deterministic_row_dropout(
                latency_frame,
                config.trace_dropout_fraction,
                f"trace-latency:{os.path.realpath(case_dir)}",
            )
            latency_series = _service_trace_series_from_frame(
                latency_frame, inject_time=inject_time, residual=residual
            )
            if need_evidence:
                latency_evidence_series = _service_trace_series_from_frame(
                    latency_frame, inject_time=inject_time, residual=inject_time is not None
                )
        except Exception:
            latency_series = {}
            latency_evidence_series = {}
    _mark_stage("trace_latency")

    if config.trace_signal in {"both", "error"} and os.path.exists(err_path):
        try:
            error_frame = pd.read_csv(err_path)
            if boundary_obj is not None:
                error_frame = filter_timestamp_frame(error_frame, boundary_obj)
            error_frame = _deterministic_row_dropout(
                error_frame,
                config.trace_dropout_fraction,
                f"trace-error:{os.path.realpath(case_dir)}",
            )
            error_series = _service_trace_series_from_frame(
                error_frame, inject_time=inject_time, residual=residual
            )
            if need_evidence:
                error_evidence_series = _service_trace_series_from_frame(
                    error_frame, inject_time=inject_time, residual=inject_time is not None
                )
        except Exception:
            error_series = {}
            error_evidence_series = {}
    _mark_stage("trace_error")

    if config.trace_control == "time_shuffle":
        latency_series = _time_shuffle_trace_series(latency_series, "latency")
        error_series = _time_shuffle_trace_series(error_series, "error")

    latency_series = _add_trace_series_noise(
        latency_series, config.trace_noise_fraction, "latency"
    )
    error_series = _add_trace_series_noise(
        error_series, config.trace_noise_fraction, "error"
    )

    latency_edges = _trace_multilag_edges(
        call_graph,
        latency_series,
        config,
        source_kind="latency",
    )
    error_edges = _trace_multilag_edges(
        call_graph,
        error_series,
        config,
        source_kind="error",
    )
    edges = _merge_trace_topology_edges([latency_edges, error_edges])
    if config.trace_control == "degree_matched_random_edges":
        edges = _degree_matched_random_edges(edges)
    edges = _inject_random_trace_edges(
        edges,
        set(latency_series) | set(error_series),
        config.trace_edge_noise_fraction,
    )
    _mark_stage("trace_edges")

    if need_evidence:
        latency_evidence = _trace_service_evidence(latency_evidence_series)
        error_evidence = _trace_service_evidence(error_evidence_series)
        service_evidence = _norm_dict({
            svc: max(latency_evidence.get(svc, 0.0), error_evidence.get(svc, 0.0))
            for svc in set(latency_evidence) | set(error_evidence)
        }) if (latency_evidence or error_evidence) else {}
    else:
        latency_evidence = {}
        error_evidence = {}
        service_evidence = {}

    return _TraceTopologyFeatures(
        edges=edges,
        service_evidence=service_evidence,
        latency_evidence=latency_evidence,
        error_evidence=error_evidence,
        stage_timing=stage_timing,
    )




def _boundary_signature(boundary):
    obj = ObservationBoundary.from_dict(boundary) if isinstance(boundary, dict) else boundary
    if obj is None:
        return None
    return (obj.version, float(obj.start), float(obj.end), float(obj.inject_time))


def _case_id(case_dir):
    case_dir = os.path.realpath(case_dir)
    return f"{os.path.basename(os.path.dirname(case_dir))}_{os.path.basename(case_dir)}"


def _manifest_digest(exposure_manifest):
    payload = json.dumps(exposure_manifest or {}, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def _source_record(dataset_root, exposure_manifest, case_id):
    record = (exposure_manifest or {}).get("cases", {}).get(case_id)
    if not isinstance(record, dict):
        raise ValueError(f"graph reference case is absent from exposure manifest: {case_id}")
    fault = record.get("fault_directory")
    repetition = record.get("case_directory")
    boundary = record.get("boundary")
    if fault is None or repetition is None or not isinstance(boundary, dict):
        raise ValueError(f"graph reference case is incomplete: {case_id}")
    return os.path.join(dataset_root, str(fault), str(repetition)), boundary, record


def _source_service_names(source_dir, record):
    metric_file = record.get("metric_file")
    if not metric_file:
        raise ValueError("graph reference case is missing metric_file")
    metric_path = os.path.join(source_dir, str(metric_file))
    try:
        columns = pd.read_csv(metric_path, nrows=0).columns
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read graph reference metric schema: {metric_path}") from exc
    return {_parse_svc(column) for column in columns if str(column).lower() != "time"}


def _cached_artifact(key, factory):
    cached = _CALL_GRAPH_CACHE.get(key)
    if isinstance(cached, GraphArtifact):
        return replace(cached, cache_hit=True)
    artifact = factory()
    artifact = replace(artifact, empty_graph_cached=not bool(artifact.graph))
    _CALL_GRAPH_CACHE[key] = artifact
    return artifact


def _resolve_graph_artifact(
    *, case_dir, service_names, input_boundary, graph_source,
    exposure_manifest, graph_reference_case_id, dropout_fraction,
    build_graph=_build_call_graph_from_traces,
):
    dataset_root = os.path.dirname(os.path.dirname(case_dir))
    boundary_version = (
        input_boundary.get("version", BOUNDARY_SCHEMA_VERSION)
        if isinstance(input_boundary, dict) else BOUNDARY_SCHEMA_VERSION
    )
    services = tuple(sorted(service_names)) if service_names else None
    dropout = float(np.clip(dropout_fraction, 0.0, 1.0))

    if graph_source in {"per_case_pre_fault", "per_case_full"}:
        pre_fault = graph_source == "per_case_pre_fault"
        source_case = _case_id(case_dir)
        key = (
            "artifact", graph_source, os.path.realpath(case_dir),
            _boundary_signature(input_boundary), services, dropout,
        )
        return _cached_artifact(key, lambda: GraphArtifact(
            graph=build_graph(
                case_dir, service_names, boundary=input_boundary,
                pre_fault=pre_fault, dropout_fraction=dropout,
            ),
            source=graph_source,
            source_scope=os.path.realpath(case_dir),
            source_case=source_case,
            source_cases=(source_case,),
            window="pre_fault" if pre_fault else "full_record",
            boundary_version=boundary_version,
        ))

    if graph_source == "first_case_pre_fault":
        if not graph_reference_case_id:
            raise ValueError("first_case_pre_fault requires graph_reference_case_id")
        source_dir, source_boundary, source_record = _source_record(
            dataset_root, exposure_manifest, graph_reference_case_id
        )
        source_version = source_boundary.get("version", boundary_version)
        key = (
            "artifact", graph_source, os.path.realpath(dataset_root),
            graph_reference_case_id, os.path.realpath(source_dir),
            _boundary_signature(source_boundary),
            dropout,
        )
        def build_first():
            source_services = _source_service_names(source_dir, source_record)
            return GraphArtifact(
                graph=build_graph(
                    source_dir, source_services, boundary=source_boundary,
                    pre_fault=True, dropout_fraction=dropout,
                ),
                source=graph_source,
                source_scope=os.path.realpath(dataset_root),
                source_case=graph_reference_case_id,
                source_cases=(graph_reference_case_id,),
                window="pre_fault",
                boundary_version=source_version,
            )

        return _cached_artifact(key, build_first)

    if graph_source == "union_pre_fault":
        cases = (exposure_manifest or {}).get("cases", {})
        if not isinstance(cases, dict) or not cases:
            raise ValueError("union_pre_fault requires an exposure manifest")
        member_ids = tuple(sorted(cases))
        digest = _manifest_digest(exposure_manifest)
        key = (
            "artifact", graph_source, os.path.realpath(dataset_root), digest,
            dropout,
        )

        def build_union():
            merged = {}
            for member_id in member_ids:
                source_dir, source_boundary, source_record = _source_record(
                    dataset_root, exposure_manifest, member_id
                )
                source_services = _source_service_names(source_dir, source_record)
                graph = build_graph(
                    source_dir, source_services, boundary=source_boundary,
                    pre_fault=True, dropout_fraction=dropout,
                )
                for caller, callees in graph.items():
                    merged.setdefault(caller, set()).update(callees)
            union = {caller: sorted(callees) for caller, callees in merged.items()}
            return GraphArtifact(
                graph=union,
                source=graph_source,
                source_scope=os.path.realpath(dataset_root),
                source_case=None,
                source_cases=member_ids,
                window="pre_fault",
                boundary_version=boundary_version,
                manifest_sha256=digest,
            )

        return _cached_artifact(key, build_union)

    raise ValueError(f"unsupported graph source: {graph_source}")


def prepare_case_inputs(
    data, inject_time, args_ns, runtime_kwargs, config, graph_source,
    *, build_graph=_build_call_graph_from_traces,
    build_trace_features=_trace_topology_from_case_dir,
):
    """Load graph and trace inputs after the modality gate has been resolved."""
    case_dir = os.path.dirname(args_ns.data_path)
    service_names = {_parse_svc(c) for c in data.columns if c != "time"}
    input_boundary = getattr(args_ns, "input_boundary", None)
    exposure_manifest = getattr(args_ns, "dataset_exposure_manifest", {})
    reference_case = getattr(args_ns, "graph_reference_case_id", None)

    graph_started = perf_counter()
    artifact = _resolve_graph_artifact(
        case_dir=case_dir,
        service_names=service_names,
        input_boundary=input_boundary,
        graph_source=graph_source,
        exposure_manifest=exposure_manifest,
        graph_reference_case_id=reference_case,
        dropout_fraction=config.trace_dropout_fraction,
        build_graph=build_graph,
    )
    graph_seconds = perf_counter() - graph_started
    runtime_kwargs["call_graph"] = artifact.graph
    runtime_kwargs["graph_provenance"] = artifact.provenance()
    runtime_kwargs["input_boundary"] = input_boundary or {}

    need_trace_evidence = (
        not config.use_metrics
        or config.rank_mode in {"trace_evidence", "service_evidence_trace"}
        or bool(runtime_kwargs.get("return_diagnostics", False))
    )
    trace_features = build_trace_features(
        case_dir,
        artifact.graph,
        config,
        inject_time=inject_time,
        boundary=input_boundary,
        need_evidence=need_trace_evidence,
    )
    trace_stage_timing = dict(trace_features.stage_timing)
    trace_stage_timing["call_graph"] = float(graph_seconds)
    runtime_kwargs.setdefault("trace_topology_edges", trace_features.edges)
    runtime_kwargs.setdefault("trace_service_evidence", trace_features.service_evidence)
    runtime_kwargs.setdefault("trace_latency_evidence", trace_features.latency_evidence)
    runtime_kwargs.setdefault("trace_error_evidence", trace_features.error_evidence)
    runtime_kwargs.setdefault("trace_stage_timing", trace_stage_timing)
    return runtime_kwargs
