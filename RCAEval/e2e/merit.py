"""MERIT shared evidence, coverage-game and ranking pipeline.

Experiment switches and alternative implementations live in merit_ablation;
merit_config owns defaults. Legacy helper entry points below are thin adapters
so research monkeypatches continue to target this module at call time.
"""

import os
import numpy as np
import pandas as pd
import warnings
from dataclasses import dataclass
from time import perf_counter

warnings.filterwarnings("ignore")

from RCAEval.e2e.merit_config import _GabanzhafConfig
from RCAEval.e2e import merit_ablation as _ablation
from RCAEval.io.time_series import preprocess
from RCAEval.benchmark.input_boundaries import (
    BOUNDARY_SCHEMA_VERSION,
    ObservationBoundary,
    filter_span_frame,
    filter_timestamp_frame,
)


@dataclass(frozen=True)
class _TraceTopologyFeatures:
    edges: list
    service_evidence: dict
    latency_evidence: dict
    error_evidence: dict
    stage_timing: dict


# ==========================================
# Internal helpers
# ==========================================


def _parse_svc(col):
    """Parse service name from a metric column (last underscore separator)."""
    parts = col.rsplit("_", 1)
    return parts[0] if len(parts) == 2 else col


def _norm01_arr(a):
    a = np.asarray(a, dtype=float)
    lo = np.nanmin(a) if a.size else 0.
    hi = np.nanmax(a) if a.size else 1.
    return (a - lo) / (hi - lo + 1e-9)


def _norm_dict(d):
    lo, hi = min(d.values(), default=0.), max(d.values(), default=1.)
    span = hi - lo if hi > lo else 1.
    return {k: (v - lo) / span for k, v in d.items()}


def _metric_evidence(normal_mat, anomal_mat, cols, config=None):
    """Metric-level evidence from robust magnitude, persistence and peak.

    The score remains unsupervised: it only compares the post-injection window
    with the pre-injection baseline.  Percentile and persistence terms reduce
    the chance that a single noisy spike dominates the ranking.

    config.scorer selects the underlying anomaly scorer: "robust_z" (default)
    keeps this function's original magnitude/persistence/peak behaviour
    exactly as-is; "cusum" routes to _cusum_metric_evidence instead.
    """
    if config is None:
        config = _GabanzhafConfig.from_kwargs({})
    normal_mat = np.asarray(normal_mat, dtype=float)
    anomal_mat = np.asarray(anomal_mat, dtype=float)
    if getattr(config, "scorer", "robust_z") == "cusum":
        return _cusum_metric_evidence(normal_mat, anomal_mat, cols, config)
    med = np.nanmedian(normal_mat, axis=0)
    q75 = np.nanpercentile(normal_mat, 75, axis=0)
    q25 = np.nanpercentile(normal_mat, 25, axis=0)
    std = np.nanstd(normal_mat, axis=0)
    iqr = np.where((q75 - q25) > 0, q75 - q25, 1.)

    rz = np.abs((anomal_mat - med) / iqr)
    rz = np.nan_to_num(rz, nan=0., posinf=0., neginf=0.)
    # Selective Winsorization: cap metrics where IQR=0 AND std≈0 (truly constant
    # baseline). These produce inflated rz through the unit denominator even for
    # tiny shifts, masking real causes. The std<=1e-6 guard preserves columns
    # that have IQR=0 but non-trivial spread so their rz is still meaningful.
    near_zero_baseline = ((q75 - q25) <= 0) & (std <= 1e-6)   # shape (n_cols,)
    rz = np.where(near_zero_baseline, np.clip(rz, 0., 10.), rz)

    anomal_len = rz.shape[0]
    early_len = max(int(round(anomal_len * config.early_window_ratio)), 1)
    use_full_for_early = early_len < config.early_window_min_len
    early_rz = rz if use_full_for_early else rz[:early_len]

    mag = _norm01_arr(np.nanpercentile(rz, 95, axis=0))
    peak = _norm01_arr(np.nanmax(rz, axis=0))
    persistence = _norm01_arr(np.mean(rz > 3.0, axis=0))
    w_full = _norm_weights(
        (config.ev_w_mag, config.ev_w_persistence, config.ev_w_peak), (0.55, 0.30, 0.15)
    )
    full = _norm01_arr(w_full[0] * mag + w_full[1] * persistence + w_full[2] * peak)
    early_mag = _norm01_arr(np.nanpercentile(early_rz, 95, axis=0))
    early_peak = _norm01_arr(np.nanmax(early_rz, axis=0))
    w_early = _norm_weights((config.early_w_mag, config.early_w_peak), (0.70, 0.30))
    early = (
        full.copy()
        if use_full_for_early
        else _norm01_arr(w_early[0] * early_mag + w_early[1] * early_peak)
    )
    w_fuse = _norm_weights((config.fusion_w_early, config.fusion_w_full), (0.60, 0.40))
    score = (
        _norm01_arr(w_fuse[0] * early + w_fuse[1] * full)
        if config.use_early_full_fusion
        else full
    )

    return {col: float(score[i]) for i, col in enumerate(cols)}


def _norm_weights(values, fallback):
    """Renormalise a weight group to sum to 1 so any non-negative input is legal."""
    arr = np.asarray(
        [float(v) if np.isfinite(v) and v > 0 else 0.0 for v in values], dtype=float
    )
    total = arr.sum()
    return arr / total if total > 0 else np.asarray(fallback, dtype=float)


def _service_evidence(cols, metric_ev, use_metric_type_dominance=True, config=None):
    """Aggregate metric evidence into service evidence.

    Two complementary signals are blended:

    1. Aggregate (70 %): percentile-weighted blend of a service's own metric
       scores — the primary signal for clearly localised faults.

    2. Metric-type dominance (30 %): for each metric type (the suffix after
       the last underscore, e.g. "cpu", "mem", "latency") the service with the
       highest score earns a bonus proportional to its margin over the
       runner-up.  This is decisive for broadcast faults (e.g. CPU injection)
       where every service shows elevated metrics: the injected service still
       leads in the fault-relevant metric type even when absolute levels are
       similar across the system.

    No ground truth is used; no branching on system size.
    """
    svc_to_cols: dict = {}
    for col in cols:
        svc_to_cols.setdefault(_parse_svc(col), []).append(col)

    # ── 1. Per-service aggregate score ───────────────────────────────────────
    w_agg = (
        _norm_weights((config.agg_w_p95, config.agg_w_top3, config.agg_w_max),
                      (0.50, 0.30, 0.20))
        if config is not None
        else np.asarray((0.50, 0.30, 0.20))
    )
    svc_agg: dict = {}
    for svc, svc_cols in svc_to_cols.items():
        arr = np.asarray([metric_ev[c] for c in svc_cols], dtype=float)
        top = np.sort(arr)[-min(3, len(arr)):]
        svc_agg[svc] = float(
            w_agg[0] * np.percentile(arr, 95)
            + w_agg[1] * np.mean(top)
            + w_agg[2] * np.max(arr)
        )

    if not use_metric_type_dominance:
        return _norm_dict(svc_agg)

    # ── 2. Metric-type dominance ─────────────────────────────────────────────
    # Track the top-2 scores per metric type across services.
    # "metric type" = everything after the service prefix  (e.g. "cpu", "mem").
    type_best: dict   = {}   # mtype -> (winning_svc, score)
    type_second: dict = {}   # mtype -> best score from a *different* service

    for col in cols:
        svc   = _parse_svc(col)
        mtype = col[len(svc) + 1:] if col.startswith(svc + "_") else col
        sc    = metric_ev[col]

        if mtype not in type_best or sc > type_best[mtype][1]:
            prev = type_best.get(mtype)
            if prev is not None and prev[0] != svc:
                # Previous winner (different service) becomes second-best candidate
                if prev[1] > type_second.get(mtype, 0.):
                    type_second[mtype] = prev[1]
            type_best[mtype] = (svc, sc)
        elif svc != type_best[mtype][0] and sc > type_second.get(mtype, 0.):
            type_second[mtype] = sc

    dom: dict = {svc: 0. for svc in svc_agg}
    for mtype, (best_svc, best_sc) in type_best.items():
        margin = best_sc - type_second.get(mtype, 0.)
        dom[best_svc] = dom.get(best_svc, 0.) + margin

    dom_max  = max(dom.values(), default=1.)
    dom_norm = {s: v / (dom_max + 1e-9) for s, v in dom.items()}

    # ── 3. Blend ─────────────────────────────────────────────────────────────
    w_svc = (
        _norm_weights((config.svc_w_agg, config.svc_w_dominance), (0.70, 0.30))
        if config is not None
        else np.asarray((0.70, 0.30))
    )
    svc_scores = {
        svc: w_svc[0] * svc_agg[svc] + w_svc[1] * dom_norm.get(svc, 0.)
        for svc in svc_agg
    }
    return _norm_dict(svc_scores)


def _noisy_or_coverage(prob_matrix, coalition):
    prob = np.nan_to_num(
        np.asarray(prob_matrix, dtype=float),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    prob = np.clip(prob, 0.0, 1.0)
    if prob.ndim != 2:
        raise ValueError("prob_matrix must be a 2-D array")

    try:
        coalition_arr = np.asarray(list(coalition))
    except TypeError as exc:
        raise ValueError("coalition must be a 1-D integer index array") from exc
    if coalition_arr.ndim != 1:
        raise ValueError("coalition must be a 1-D integer index array")
    if coalition_arr.size == 0:
        return np.zeros(prob.shape[1], dtype=float)

    if not np.issubdtype(coalition_arr.dtype, np.integer):
        raise ValueError("coalition must contain integer indices")

    coalition_idx = coalition_arr.astype(int)
    if np.any((coalition_idx < 0) | (coalition_idx >= prob.shape[0])):
        raise ValueError("coalition indices must be in [0, n_rows)")

    rows = prob[np.unique(coalition_idx)]
    return 1.0 - np.prod(1.0 - rows, axis=0)


def _probabilistic_banzhaf(prob_matrix, anomaly_weights):
    prob = np.nan_to_num(
        np.asarray(prob_matrix, dtype=float),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    prob = np.clip(prob, 0.0, 1.0)
    weights = np.nan_to_num(
        np.asarray(anomaly_weights, dtype=float).reshape(-1),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    weights = np.maximum(weights, 0.0)
    if prob.ndim != 2:
        raise ValueError("prob_matrix must be a 2-D array")
    if prob.shape[1] != weights.shape[0]:
        raise ValueError("anomaly_weights length must match prob_matrix columns")
    if prob.shape[0] == 0:
        return np.zeros(0, dtype=float)

    # Delegate the half-complement/product-without-self computation to
    # _probabilistic_banzhaf_contributions (prob/weights are already cleaned
    # above, so its own nan_to_num/clip are no-ops here) and sum per-target
    # terms into per-candidate scores. Every term is already >= 0 (weights,
    # prob, and product_without are all non-negative), so this is numerically
    # identical to the previous inline computation.
    contributions = _probabilistic_banzhaf_contributions(prob, weights)
    return np.sum(contributions, axis=1)


def _probabilistic_banzhaf_contributions(prob_matrix, anomaly_weights):
    """Return the player-by-target terms whose row sums are Banzhaf scores."""

    prob = np.clip(
        np.nan_to_num(np.asarray(prob_matrix, dtype=float)), 0.0, 1.0
    )
    weights = np.maximum(
        np.nan_to_num(np.asarray(anomaly_weights, dtype=float).reshape(-1)), 0.0
    )
    if prob.ndim != 2 or prob.shape[1] != weights.shape[0]:
        raise ValueError("contribution dimensions do not match")
    if prob.shape[0] == 0:
        return np.zeros_like(prob)
    half_complements = 1.0 - 0.5 * prob
    product_all = np.prod(half_complements, axis=0)
    product_without = product_all / np.maximum(half_complements, 1e-12)
    return np.maximum(weights[None, :] * prob * product_without, 0.0)


def _edge_removal_faithfulness(prob_matrix, anomaly_weights):
    """Measure full-coalition utility loss after removing each explanation edge."""

    prob = np.clip(
        np.nan_to_num(np.asarray(prob_matrix, dtype=float)), 0.0, 1.0
    )
    weights = np.maximum(
        np.nan_to_num(np.asarray(anomaly_weights, dtype=float).reshape(-1)), 0.0
    )
    if prob.ndim != 2 or prob.shape[1] != weights.shape[0]:
        raise ValueError("faithfulness dimensions do not match")
    if prob.shape[0] == 0:
        return np.zeros_like(prob)
    complements = 1.0 - prob
    zero_mask = complements <= 1e-15
    zero_count = np.sum(zero_mask, axis=0)
    nonzero_product = np.prod(np.where(zero_mask, 1.0, complements), axis=0)
    product_without = np.zeros_like(prob)
    no_zero = zero_count == 0
    if np.any(no_zero):
        product_without[:, no_zero] = (
            nonzero_product[None, no_zero] / complements[:, no_zero]
        )
    one_zero = zero_count == 1
    if np.any(one_zero):
        product_without[:, one_zero] = np.where(
            zero_mask[:, one_zero],
            nonzero_product[None, one_zero],
            0.0,
        )
    return np.maximum(weights[None, :] * prob * product_without, 0.0)


_CALL_GRAPH_CACHE = {}


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
    boundary_signature = (
        (
            boundary_obj.version,
            float(boundary_obj.start),
            float(boundary_obj.end),
            float(boundary_obj.inject_time),
            bool(pre_fault),
            float(np.clip(dropout_fraction, 0.0, 1.0)),
        )
        if boundary_obj is not None
        else None
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


def _service_metric_family_scores(cols, metric_ev, matches):
    """Shared implementation: per-service max score over columns matching ``matches``."""
    out = {}
    for col in cols:
        metric_name = col.rsplit("_", 1)[-1]
        if not matches(metric_name):
            continue
        svc = _parse_svc(col)
        out[svc] = max(out.get(svc, 0.0), _finite_nonnegative(metric_ev.get(col, 0.0)))
    return out


def _service_latency_scores(cols, metric_ev):
    """Per-service latency-metric score (max over its *_latency columns).

    Used to tell propagation victims (loud because latency spiked) from
    services with their own resource fault (loud because of cpu/mem/disk).
    """
    return _service_metric_family_scores(cols, metric_ev, lambda name: name == "latency")


def _service_latency_family_scores(cols, metric_ev):
    """Per-service score for latency and latency percentile metrics."""
    return _service_metric_family_scores(
        cols, metric_ev, lambda name: name.startswith("latency")
    )


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


def _coverage_parents(call_graph, svc_anom_dict, svc_latency, config):
    """Parents that explain >=min_loud propagation victims (delay/loss).

    A child counts as a victim when it is loud overall (anomaly >= loud_thr)
    AND shows at least some latency signal (>= lat_floor). The latency floor
    excludes a child that is loud purely from its own resource fault
    (cpu/mem/disk, latency ~0) — that child is itself a root, so its parent
    must not steal its #1. Returns {parent: [victim_child, ...]}.
    """
    loud_thr = _finite_nonnegative(config.coverage_loud_thr)
    lat_floor = _finite_nonnegative(config.coverage_lat_floor)
    min_loud = max(1, int(config.coverage_min_loud))
    parents = {}
    for caller, callees in (call_graph or {}).items():
        loud = [
            c for c in callees
            if svc_anom_dict.get(c, 0.0) >= loud_thr
            and svc_latency.get(c, 0.0) >= lat_floor
        ]
        if len(loud) >= min_loud:
            parents[caller] = loud
    return parents


def _apply_coverage_edges(matrix, cand_list, service_list, coverage_parents, conf):
    """Overlay parent->loud-child coverage with fixed confidence.

    Bypasses the alignment damping in _causal_explanation_matrix (which zeroes
    edges from a quiet parent) so a low-anomaly root still gets credited for
    covering its loud victims.
    """
    cand_idx = {svc: i for i, svc in enumerate(cand_list)}
    svc_idx = {svc: j for j, svc in enumerate(service_list)}
    conf = float(np.clip(conf, 0.0, 1.0))
    for parent, children in coverage_parents.items():
        i = cand_idx.get(parent)
        if i is None:
            continue
        for child in children:
            j = svc_idx.get(child)
            if j is not None:
                matrix[i, j] = max(matrix[i, j], conf)
    return matrix


def _apply_trace_topology_edges(matrix, cand_list, service_list, trace_edges):
    cand_idx = {svc: i for i, svc in enumerate(cand_list)}
    svc_idx = {svc: j for j, svc in enumerate(service_list)}
    for source, target, confidence, _source_kind, _lag in trace_edges:
        i = cand_idx.get(source)
        j = svc_idx.get(target)
        if i is None or j is None:
            continue
        matrix[i, j] = max(matrix[i, j], float(np.clip(confidence, 0.0, 1.0)))
    return matrix


def _causal_edges(coverage_parents, trace_edges, coverage_conf):
    """Merge coverage + trace edges (cause->effect); consumed by victim demotion."""
    edges = []
    conf = float(np.clip(coverage_conf, 0.0, 1.0))
    for parent, children in (coverage_parents or {}).items():
        for child in children:
            edges.append((parent, child, conf, "coverage", 0))
    for source, target, confidence, source_kind, lag in (trace_edges or []):
        edges.append((
            source,
            target,
            float(np.clip(confidence, 0.0, 1.0)),
            source_kind,
            int(lag),
        ))
    return edges


def _victim_factors(causal_edges, svc_anom_dict, config, protected=None):
    """Per-service self-coverage multiplier ∈ [victim_floor, 1].

    Edges are cause->effect. A service is a propagation victim when one of its
    causal predecessors is itself abnormal — then its own anomaly is credited
    upstream, so its self-coverage (diagonal 1.0) is shrunk toward victim_floor.
    A service whose predecessors are all normal/absent is a root and keeps 1.0.

    `protected` services are immune to demotion: coverage already inferred them
    to be roots that uniquely explain a cluster of loud victims, so the (opposite
    -direction) trace evidence must not overrule that and demote them as victims.
    """
    protected = set(protected or ())
    floor = float(np.clip(config.victim_floor, 0.0, 1.0))
    abnormal_thr = _finite_nonnegative(config.frontier_abnormal_threshold)
    normal_thr = min(abnormal_thr, _finite_nonnegative(config.frontier_normal_threshold))
    edge_thr = _finite_nonnegative(config.frontier_edge_threshold)
    uncertain_factor = float(np.clip(config.frontier_uncertain_stop_factor, 0.0, 1.0))

    predecessors = {}
    for source, target, confidence, _kind, _lag in causal_edges or []:
        if float(np.clip(confidence, 0.0, 1.0)) < edge_thr:
            continue
        predecessors.setdefault(str(target), []).append(str(source))

    factors = {}
    for svc, upstreams in predecessors.items():
        if svc in protected:
            continue  # coverage-inferred root → keep self-coverage at 1.0
        up_scores = [_finite_nonnegative(svc_anom_dict.get(u, 0.0)) for u in upstreams]
        if any(s >= abnormal_thr for s in up_scores):
            factors[svc] = floor
        elif any(s > normal_thr for s in up_scores):
            factors[svc] = floor + (1.0 - floor) * uncertain_factor
        # all-normal predecessors → root → leave at default 1.0 (omit)
    return factors


def _split_raw_windows(data, inject_time, anomalies):
    if anomalies is not None:
        split_idx = int(anomalies[0])
        return data.head(split_idx), data.tail(len(data) - split_idx)
    if inject_time is not None and "time" in data.columns:
        return data[data["time"] < inject_time], data[data["time"] >= inject_time]
    split_idx = len(data) // 2
    return data.head(split_idx), data.tail(len(data) - split_idx)


def _preprocess_windows(normal_raw, anomal_raw, dataset, config):
    n_rows = len(normal_raw)
    full_proc = preprocess(
        pd.concat([normal_raw, anomal_raw], ignore_index=True),
        dataset,
        config.dk_select_useful,
    )

    normal_df = full_proc.iloc[:n_rows].reset_index(drop=True)
    anomal_df = full_proc.iloc[n_rows:].reset_index(drop=True)

    shared = [c for c in normal_df.columns if c in anomal_df.columns]
    normal_df = normal_df[shared]
    anomal_df = anomal_df[shared]

    # The v14 headline estimand uses the complete case record. Keep every
    # available pre- and post-injection row; metric evidence supports unequal
    # baseline and observation lengths. Fixed equal-window truncation belongs
    # only to the explicitly separate window-sensitivity jobs.
    normal_df = normal_df.reset_index(drop=True)
    anomal_df = anomal_df.reset_index(drop=True)
    no_time = pd.concat([normal_df, anomal_df], ignore_index=True).drop(
        columns=["time"], errors="ignore"
    )
    return normal_df, anomal_df, no_time


def _select_exact_candidates(candidates, svc_anom_dict, config):
    k = max(1, int(config.exact_candidate_k))
    candidates = list(dict.fromkeys(candidates))
    return sorted(
        candidates,
        key=lambda svc: (-_finite_nonnegative(svc_anom_dict.get(svc, 0.0)), svc),
    )[:k]


def _causal_explanation_matrix(cand_list, service_list):
    """Base explanation matrix: each candidate explains its own service (1.0).

    Off-diagonal coverage edges are overlaid separately by
    _apply_coverage_edges.
    """
    if len(cand_list) != len(set(cand_list)):
        raise ValueError("Duplicate candidate name")
    if len(service_list) != len(set(service_list)):
        raise ValueError("Duplicate service name")
    svc_idx = {svc: j for j, svc in enumerate(service_list)}
    matrix = np.zeros((len(cand_list), len(service_list)), dtype=float)
    for i, svc in enumerate(cand_list):
        if svc in svc_idx:
            matrix[i, svc_idx[svc]] = 1.0
    return matrix


def _finite_nonnegative(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(value) or value < 0.0:
        return 0.0
    return value


def _rank_metrics(all_cols, svc_banzhaf, metric_evidence=None):
    """Rank by responsibility, then evidence, with stable schema-order ties."""
    finite_banzhaf = {
        svc: _finite_nonnegative(score)
        for svc, score in (svc_banzhaf or {}).items()
    }
    max_banzhaf = max(finite_banzhaf.values(), default=0.0)
    if max_banzhaf > 0.0:
        nb = {svc: score / max_banzhaf for svc, score in finite_banzhaf.items()}
    else:
        nb = {svc: 1.0 for svc in {_parse_svc(col) for col in all_cols}}

    if metric_evidence is None:
        return sorted(
            all_cols,
            key=lambda col: (-nb.get(_parse_svc(col), 0.0), col),
        )
    evidence = metric_evidence
    input_order = {column: index for index, column in enumerate(all_cols)}
    return sorted(
        all_cols,
        key=lambda col: (
            -nb.get(_parse_svc(col), 0.0),
            -_finite_nonnegative(evidence.get(col, 0.0)),
            input_order[col],
        ),
    )


def _service_ranks_from_metric_ranks(metric_ranks):
    """Collapse RCAEval's metric-level output schema to a service ranking."""
    service_ranks = []
    seen = set()
    for metric in metric_ranks:
        service = _parse_svc(metric)
        if service not in seen:
            seen.add(service)
            service_ranks.append(service)
    return service_ranks


# ==========================================
# Core pipeline
# ==========================================


def _merit_core(data, inject_time, dataset, anomalies, kwargs):
    """
    Pipeline: window split → metric evidence → service evidence
              → call-graph coverage → Banzhaf → metric ranking
    """
    config = _GabanzhafConfig.from_kwargs(kwargs)
    trace_topology_edges = (
        list(kwargs.get("trace_topology_edges", []))
        if config.use_trace and config.use_trace_topology
        else []
    )
    trace_service_evidence = (
        kwargs.get("trace_service_evidence", {}) or {}
        if config.use_trace
        else {}
    )
    trace_latency_evidence = (
        kwargs.get("trace_latency_evidence", {}) or {}
        if config.use_trace
        else {}
    )
    trace_error_evidence = (
        kwargs.get("trace_error_evidence", {}) or {}
        if config.use_trace
        else {}
    )
    trace_stage_timing = (
        kwargs.get("trace_stage_timing", {}) or {}
        if config.use_trace
        else {}
    )
    return_stage_timing = bool(kwargs.get("return_stage_timing", False))
    stage_timing = {}
    last_tick = perf_counter()

    def _mark_stage(name):
        nonlocal last_tick
        if not return_stage_timing:
            return
        now = perf_counter()
        stage_timing[name] = float(now - last_tick)
        last_tick = now

    # ── 1. Window split ──────────────────────────────────────────────────────
    normal_raw, anomal_raw = _split_raw_windows(data, inject_time, anomalies)

    # ── 2. Preprocess jointly ────────────────────────────────────────────────
    if config.use_metrics:
        normal_df, anomal_df, no_time = _preprocess_windows(
            normal_raw, anomal_raw, dataset, config
        )
        cols = list(no_time.columns)
        normal_mat = normal_df.drop(
            columns=["time"], errors="ignore"
        ).values.astype(float)
        anomal_mat = anomal_df.drop(
            columns=["time"], errors="ignore"
        ).values.astype(float)
    else:
        cols, normal_mat, anomal_mat = _ablation.metric_free_windows(
            data, normal_raw, anomal_raw
        )
    _mark_stage("preprocess")

    # ── 3. Metric and service evidence ───────────────────────────────────────
    if config.use_metrics:
        metric_ev = _metric_evidence(normal_mat, anomal_mat, cols, config=config)
        svc_anom_dict = _service_evidence(
            cols,
            metric_ev,
            use_metric_type_dominance=config.use_metric_type_dominance,
            config=config,
        )
    else:
        metric_ev, svc_anom_dict = _ablation.metric_free_evidence(
            cols, trace_service_evidence,
            parse_service=_parse_svc, nonnegative=_finite_nonnegative,
        )
    svc_names = sorted(svc_anom_dict.keys())
    _mark_stage("metric_evidence")

    _ablation.validate_rank_mode(config.rank_mode)

    if config.rank_mode == "banzhaf":
        # ── 4. Candidates = top services by anomaly score ────────────────────
        svc_ranked = sorted(
            svc_names, key=lambda s: svc_anom_dict.get(s, 0.0), reverse=True
        )
        seed_svcs = set(svc_ranked[: config.top_k_svcs])
        cand_list = sorted(seed_svcs)
        _mark_stage("candidates")

        svc_latency = _service_latency_scores(cols, metric_ev)
        svc_latency_family = _service_latency_family_scores(cols, metric_ev)
        if config.use_metrics:
            trace_topology_edges = [
                edge
                for edge in trace_topology_edges
                if svc_latency_family.get(edge[1], 0.0)
                >= config.coverage_lat_floor
            ]

        # ── 4b. Call-graph coverage: rescue quiet roots of multi-victim faults
        if config.use_metrics and bool(kwargs.get("use_coverage", True)):
            call_graph = (
                kwargs.get("call_graph", {}) or {}
                if config.use_trace
                else {}
            )
            coverage_parents = _coverage_parents(
                call_graph, svc_anom_dict, svc_latency, config
            )
        else:
            coverage_parents = {}
        causal_edges = _causal_edges(
            coverage_parents,
            trace_topology_edges,
            config.coverage_conf,
        )

        # ── 5. Directed coverage-strength game ───────────────────────────────
        game_candidates = _select_exact_candidates(
            cand_list, svc_anom_dict, config
        )
        base_game_candidates = list(game_candidates)
        # Force qualified coverage parents into the game even when their own
        # anomaly is too low to make the anomaly-ranked top-k cut.
        for parent in coverage_parents:
            if parent in svc_anom_dict and parent not in game_candidates:
                game_candidates.append(parent)
        trace_added_candidates = 0
        game_services = sorted(svc_anom_dict.keys())
        prob_matrix = _causal_explanation_matrix(game_candidates, game_services)
        prob_matrix = _apply_coverage_edges(
            prob_matrix,
            game_candidates,
            game_services,
            coverage_parents,
            config.coverage_conf,
        )
        prob_matrix = _apply_trace_topology_edges(
            prob_matrix,
            game_candidates,
            game_services,
            trace_topology_edges,
        )
        if config.use_victim_demotion:
            victim_factor = _victim_factors(
                causal_edges, svc_anom_dict, config,
                protected=(
                    set(coverage_parents)
                    if config.protect_coverage_parents
                    else set()
                ),
            )
            svc_col = {svc: j for j, svc in enumerate(game_services)}
            for i, cand in enumerate(game_candidates):
                f = victim_factor.get(cand)
                if f is not None and cand in svc_col:
                    prob_matrix[i, svc_col[cand]] *= f
        anomaly_weights = np.asarray(
            [svc_anom_dict.get(svc, 0.0) for svc in game_services],
            dtype=float,
        )
    else:
        # Evidence-only modes terminate the architecture after Figure 1 step 2.
        # Keep the empty game fields for diagnostics/API compatibility.
        cand_list = []
        coverage_parents = {}
        causal_edges = []
        game_candidates = []
        base_game_candidates = []
        trace_added_candidates = 0
        game_services = []
        prob_matrix = np.zeros((0, 0), dtype=float)
        anomaly_weights = np.zeros(0, dtype=float)

    # ── 6. Banzhaf power index ────────────────────────────────────────────────
    return_diagnostics = bool(kwargs.get("return_diagnostics", False))
    svc_banzhaf = {}
    local_contributions = np.zeros_like(prob_matrix)
    edge_removal_utility_drop = np.zeros_like(prob_matrix)
    if config.rank_mode == "banzhaf":
        banzhaf_values = _banzhaf_power(prob_matrix, anomaly_weights, config)
        svc_banzhaf = {
            game_candidates[i]: float(banzhaf_values[i])
            for i in range(len(game_candidates))
        }
        # local_contributions / edge_removal_utility_drop are only surfaced in
        # the return_diagnostics payload below; skip the extra matrices on the
        # default (non-diagnostic) path.
        if return_diagnostics:
            local_contributions = _ablation.attribution_contributions(
                prob_matrix, anomaly_weights, config.banzhaf_mode,
                closed_form=_probabilistic_banzhaf_contributions,
            )
            edge_removal_utility_drop = _edge_removal_faithfulness(
                prob_matrix, anomaly_weights
            )
        _mark_stage("banzhaf")

    # ── 7. Final metric-level ranking by Banzhaf score ───────────────────────
    if config.rank_mode == "banzhaf":
        ranks = _rank_metrics(
            cols,
            svc_banzhaf,
            metric_ev if config.use_metrics else None,
        )
    else:
        ranks = _rank_evidence_metrics(
            cols,
            metric_ev,
            svc_anom_dict,
            config.rank_mode,
            trace_service_evidence=trace_service_evidence,
        )
    _mark_stage("ranking")

    direct_metric_evidence = {}
    direct_trace_evidence = {}
    direct_service_scores = {}
    if config.rank_mode == "service_evidence_trace":
        (
            direct_metric_evidence,
            direct_trace_evidence,
            direct_service_scores,
        ) = _direct_service_score_sum(svc_anom_dict, trace_service_evidence)
    service_ranks = _service_ranks_from_metric_ranks(ranks)

    result = {"ranks": ranks, "node_names": cols}
    if return_diagnostics:
        diagnostics = {
            "rank_mode": config.rank_mode,
            "configuration": {
                "banzhaf_mode": config.banzhaf_mode,
                "use_early_full_fusion": config.use_early_full_fusion,
                "use_metric_type_dominance": config.use_metric_type_dominance,
                "use_metrics": config.use_metrics,
                "use_trace": config.use_trace,
                "use_trace_topology": config.use_trace_topology,
                "trace_signal": config.trace_signal,
                "trace_control": config.trace_control,
                "trace_topology_max_lag": config.trace_topology_max_lag,
                "trace_dropout_fraction": config.trace_dropout_fraction,
                "trace_noise_fraction": config.trace_noise_fraction,
                "trace_edge_noise_fraction": config.trace_edge_noise_fraction,
                "use_victim_demotion": config.use_victim_demotion,
                "protect_coverage_parents": config.protect_coverage_parents,
                "use_coverage": bool(kwargs.get("use_coverage", True)),
            },
            "candidate_pool": list(cand_list),
            "game_candidates": list(game_candidates),
            "game_services": list(game_services),
            "candidate_pool_k": len(cand_list),
            "base_game_candidates": list(base_game_candidates),
            "base_candidate_k": len(base_game_candidates),
            "coverage_rescued_candidates": sorted(
                set(game_candidates) - set(base_game_candidates)
            ),
            "coverage_rescued_k": len(
                set(game_candidates) - set(base_game_candidates)
            ),
            "final_candidate_k": len(game_candidates),
            "coverage_strength_matrix": prob_matrix.tolist(),
            # Backward-compatible key for pre-v14 analysis code.  The v14
            # paper and generated artifacts use coverage_strength_matrix.
            "causal_probability_matrix": prob_matrix.tolist(),
            "metric_evidence": dict(metric_ev),
            "metric_service_evidence": dict(svc_anom_dict),
            "direct_metric_service_evidence": dict(direct_metric_evidence),
            "direct_trace_service_evidence": dict(direct_trace_evidence),
            "direct_service_scores": dict(direct_service_scores),
            "service_ranks": list(service_ranks),
            "service_banzhaf": dict(svc_banzhaf),
            "player_target_contributions": ({
                candidate: {
                    target: float(local_contributions[i, j])
                    for j, target in enumerate(game_services)
                }
                for i, candidate in enumerate(game_candidates)
            } if config.rank_mode == "banzhaf" else {}),
            "edge_removal_utility_drop": ({
                candidate: {
                    target: float(edge_removal_utility_drop[i, j])
                    for j, target in enumerate(game_services)
                }
                for i, candidate in enumerate(game_candidates)
            } if config.rank_mode == "banzhaf" else {}),
            "trace_topology_edges": list(trace_topology_edges),
            "trace_edge_count": len(trace_topology_edges),
            "trace_candidate_added_count": trace_added_candidates,
            "trace_service_evidence": dict(trace_service_evidence),
            "trace_latency_evidence": dict(trace_latency_evidence),
            "trace_error_evidence": dict(trace_error_evidence),
            "trace_stage_timing": dict(trace_stage_timing),
            "causal_edges": list(causal_edges),
            "causal_edge_count": len(causal_edges),
            "graph_provenance": dict(kwargs.get("graph_provenance", {}) or {}),
            "call_graph_edges": sorted(
                [
                    [caller, callee]
                    for caller, callees in (
                        kwargs.get("call_graph", {}) or {}
                    ).items()
                    for callee in callees
                ]
            ),
            "input_boundary": dict(kwargs.get("input_boundary", {}) or {}),
        }
        result["diagnostics"] = diagnostics
        result.update(diagnostics)
    if return_stage_timing:
        result["stage_timing"] = stage_timing
    return result


# ==========================================
# RCAEval entry point
# ==========================================

# Compatibility adapters: keep research imports and dynamic patch points stable.
# No alternate algorithm implementation belongs in this section.
def _cusum_metric_evidence(normal_mat, anomal_mat, cols, config):
    """Compatibility entry point; implementation lives in merit_ablation."""
    return _ablation._cusum_metric_evidence(
        normal_mat, anomal_mat, cols, config
    )


def _bruteforce_probabilistic_banzhaf(prob_matrix, anomaly_weights):
    """Compatibility entry point; implementation lives in merit_ablation."""
    return _ablation._bruteforce_probabilistic_banzhaf(
        prob_matrix, anomaly_weights,
        _noisy_or_coverage=_noisy_or_coverage
    )


def _banzhaf_power(prob_matrix, anomaly_weights, config):
    """Compatibility entry point; implementation lives in merit_ablation."""
    return _ablation._banzhaf_power(
        prob_matrix, anomaly_weights, config,
        _probabilistic_banzhaf=_probabilistic_banzhaf, _bruteforce_probabilistic_banzhaf=_bruteforce_probabilistic_banzhaf
    )


def _build_union_call_graph(dataset_root, exposure_manifest, service_names, boundary_version, dropout_fraction=0.0):
    """Compatibility entry point; implementation lives in merit_ablation."""
    return _ablation._build_union_call_graph(
        dataset_root, exposure_manifest, service_names, boundary_version, dropout_fraction,
        _CALL_GRAPH_CACHE=_CALL_GRAPH_CACHE, _build_call_graph_from_traces=_build_call_graph_from_traces
    )


def _time_shuffle_trace_series(trace_series, salt):
    """Compatibility entry point; implementation lives in merit_ablation."""
    return _ablation._time_shuffle_trace_series(
        trace_series, salt
    )


def _deterministic_row_dropout(frame, fraction, salt):
    """Compatibility entry point; implementation lives in merit_ablation."""
    return _ablation._deterministic_row_dropout(
        frame, fraction, salt
    )


def _add_trace_series_noise(trace_series, fraction, salt):
    """Compatibility entry point; implementation lives in merit_ablation."""
    return _ablation._add_trace_series_noise(
        trace_series, fraction, salt,
        _finite_nonnegative=_finite_nonnegative
    )


def _degree_matched_random_edges(edges):
    """Compatibility entry point; implementation lives in merit_ablation."""
    return _ablation._degree_matched_random_edges(
        edges
    )


def _inject_random_trace_edges(edges, services, fraction):
    """Compatibility entry point; implementation lives in merit_ablation."""
    return _ablation._inject_random_trace_edges(
        edges, services, fraction,
        _finite_nonnegative=_finite_nonnegative
    )


def _rank_evidence_metrics(all_cols, metric_ev, svc_anom_dict, rank_mode, trace_service_evidence=None):
    """Compatibility entry point; implementation lives in merit_ablation."""
    return _ablation._rank_evidence_metrics(
        all_cols, metric_ev, svc_anom_dict, rank_mode, trace_service_evidence,
        _norm_dict=_norm_dict, _finite_nonnegative=_finite_nonnegative, _parse_svc=_parse_svc, _direct_service_score_sum=_direct_service_score_sum
    )


def _direct_service_score_sum(metric_service_evidence, trace_service_evidence):
    """Compatibility entry point; implementation lives in merit_ablation."""
    return _ablation._direct_service_score_sum(
        metric_service_evidence, trace_service_evidence,
        _finite_nonnegative=_finite_nonnegative, _norm_dict=_norm_dict
    )



def _prepare_case_inputs(data, inject_time, kwargs, runtime_kwargs, disable_trace):
    """Load bounded per-case graph and trace inputs for the shared pipeline."""
    args_ns = kwargs.get("args")
    if args_ns and hasattr(args_ns, "data_path"):
        case_dir = os.path.dirname(args_ns.data_path)
        trace_config = _GabanzhafConfig.from_kwargs(runtime_kwargs)
        service_names = {_parse_svc(c) for c in data.columns if c != "time"}
        dataset_root = os.path.dirname(os.path.dirname(case_dir))
        input_boundary = getattr(args_ns, "input_boundary", None)
        (graph_source, pre_fault_graph, graph_cache_scope,
         boundary_version, graph_cache_key) = _ablation.graph_context(
            args_ns, case_dir, dataset_root, input_boundary, BOUNDARY_SCHEMA_VERSION
        )
        graph_started = perf_counter()
        if disable_trace:
            call_graph = {}
        elif graph_source == "union_pre_fault":
            call_graph = _build_union_call_graph(
                dataset_root,
                getattr(args_ns, "dataset_exposure_manifest", {}),
                service_names,
                boundary_version,
                trace_config.trace_dropout_fraction,
            )
        else:
            call_graph = _build_call_graph_from_traces(
                case_dir,
                service_names,
                cache_key=graph_cache_key,
                boundary=input_boundary,
                pre_fault=pre_fault_graph,
                dropout_fraction=trace_config.trace_dropout_fraction,
            )
        graph_seconds = perf_counter() - graph_started
        graph_provenance = (
            {
                "source": "disabled_no_trace",
                "edge_count": 0,
                "empty_graph_cached": False,
            }
            if disable_trace
            else {
                "source": graph_source,
                "source_scope": graph_cache_scope,
                "source_case": os.path.basename(case_dir),
                "window": "pre_fault" if pre_fault_graph else "full_record",
                "boundary_version": boundary_version,
                "edge_count": int(sum(len(values) for values in call_graph.values())),
                "empty_graph_cached": False,
            }
        )
        runtime_kwargs["call_graph"] = call_graph
        runtime_kwargs["graph_provenance"] = graph_provenance
        runtime_kwargs["input_boundary"] = input_boundary or {}
        # latency_evidence_series/error_evidence_series/service_evidence are
        # only ever read: (a) as svc_anom_dict's sole source when metrics are
        # disabled, (b) by rank_mode in {trace_evidence, service_evidence_trace},
        # or (c) in the return_diagnostics payload. Skip the extra series pass
        # on the default (metrics-on, rank_mode=banzhaf, no diagnostics) path.
        need_trace_evidence = (
            not trace_config.use_metrics
            or trace_config.rank_mode in {"trace_evidence", "service_evidence_trace"}
            or bool(runtime_kwargs.get("return_diagnostics", False))
        )
        trace_features = (
            _TraceTopologyFeatures([], {}, {}, {}, {})
            if disable_trace
            else _trace_topology_from_case_dir(
                case_dir,
                call_graph,
                trace_config,
                inject_time=inject_time,
                boundary=input_boundary,
                need_evidence=need_trace_evidence,
            )
        )
        trace_stage_timing = dict(trace_features.stage_timing)
        trace_stage_timing["call_graph"] = float(graph_seconds)
        runtime_kwargs.setdefault("trace_topology_edges", trace_features.edges)
        runtime_kwargs.setdefault("trace_service_evidence", trace_features.service_evidence)
        runtime_kwargs.setdefault("trace_latency_evidence", trace_features.latency_evidence)
        runtime_kwargs.setdefault("trace_error_evidence", trace_features.error_evidence)
        runtime_kwargs.setdefault("trace_stage_timing", trace_stage_timing)
    return runtime_kwargs


def merit(data, inject_time=None, dataset=None, num_loop=None,
              sli=None, anomalies=None, **kwargs):
    """Banzhaf attribution over a directed coverage-strength game.

    Pipeline: window split → metric evidence → service evidence →
    per-case pre-fault call-graph coverage (rescues quiet upstream roots of
    delay/loss propagation) → noisy-OR-inspired Banzhaf → rank by Banzhaf
    score. Coverage strengths are bounded heuristic evidence, not calibrated
    probabilities or structural-causal effects.
    Lag-1 correlation topology was removed after ablation showed it dead on
    re2-{tt,ob,ss} and re1-{ss,tt} and net-negative on re1-ob.

    Legacy experiment switches are resolved by merit_ablation; configuration
    defaults and keyword validation live in merit_config.
    """
    runtime_kwargs, _profile, disable_trace = _ablation.resolve_runtime_kwargs(kwargs)
    runtime_kwargs = _prepare_case_inputs(
        data, inject_time, kwargs, runtime_kwargs, disable_trace
    )
    if _profile:
        runtime_kwargs["return_stage_timing"] = True

    result = _merit_core(data, inject_time, dataset, anomalies, runtime_kwargs)
    return _ablation.finish_result(result, dataset, _profile)
