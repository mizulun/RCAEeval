"""Research controls for MERIT, independent of the core module.

Core-dependent operations receive callbacks explicitly. Environment parsing keeps
legacy precedence (including the force-disable-trace and diagnostics exceptions).
"""

import hashlib
import os

import numpy as np


def metric_free_windows(data, normal_raw, anomal_raw):
    """Preserve output names, never preprocess or consume metric values."""
    cols = [col for col in data.columns if col.lower() != "time"]
    return (cols, np.empty((len(normal_raw), len(cols)), dtype=float),
            np.empty((len(anomal_raw), len(cols)), dtype=float))


def metric_free_evidence(cols, trace_service_evidence, *, parse_service, nonnegative):
    metric_ev = {col: 0.0 for col in cols}
    services = {parse_service(col) for col in cols} | set(trace_service_evidence)
    return metric_ev, {
        svc: nonnegative(trace_service_evidence.get(svc, 0.0)) for svc in services
    }


def validate_rank_mode(mode):
    if mode not in {"banzhaf", "service_evidence", "metric_evidence",
                    "service_evidence_trace", "trace_evidence"}:
        raise ValueError(f"Unsupported rank_mode: {mode}")


def attribution_contributions(matrix, weights, mode, *, closed_form):
    if mode == "individual":
        return np.maximum(matrix * weights[None, :], 0.0)
    return closed_form(matrix, weights)


def _cusum_metric_evidence(normal_mat, anomal_mat, cols, config):
    """CUSUM-based anomaly scorer (alternative to robust z-score).

    Accumulates small, persistent deviations from the pre-injection median
    without a hard rz>3 threshold, so slow-onset gradual-degradation faults
    whose robust z-score never crosses 3 can still accumulate a
    detectable score. Vectorized standard CUSUM (floor at 0 each step);
    equivalence to the naive per-step loop is checked by a self-check in
    tests/test_merit.py.
    """
    med = np.nanmedian(normal_mat, axis=0)
    q75, q25 = np.nanpercentile(normal_mat, [75, 25], axis=0)
    sigma = np.where((q75 - q25) > 0, (q75 - q25) / 1.349, 1.0)
    k = config.cusum_slack * sigma
    dev = np.abs(anomal_mat - med) - k
    dev = np.nan_to_num(dev, nan=0.0)
    c = np.cumsum(dev, axis=0)
    c_ext = np.vstack([np.zeros((1, c.shape[1])), c])
    running_min = np.minimum.accumulate(c_ext, axis=0)
    peak = (c_ext[1:] - running_min[1:]).max(axis=0) / sigma
    # Log-compress before normalizing: peak's dynamic range spans 1000x+
    # across metrics/cases, so a fixed additive squash constant saturates
    # every score near 1.0 and destroys ranking discrimination. Normalize by
    # the 99th percentile (not max) so a single outlier metric can't rescale
    # everything else toward zero.
    s = np.log1p(peak)
    ref = np.nanpercentile(s, 99)
    if ref > 0:
        score = np.clip(s / ref, 0.0, 1.0)
    else:
        score = np.zeros_like(s)
    return {col: float(score[i]) for i, col in enumerate(cols)}


def _bruteforce_probabilistic_banzhaf(prob_matrix, anomaly_weights, *, _noisy_or_coverage):
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

    n_players = prob.shape[0]
    scores = np.zeros(n_players, dtype=float)
    players = list(range(n_players))
    for i in players:
        others = [p for p in players if p != i]
        total = 0.0
        count = 0
        for mask in range(1 << len(others)):
            coalition = [
                player
                for bit, player in enumerate(others)
                if mask & (1 << bit)
            ]
            with_i = coalition + [i]
            total += float(
                np.dot(weights, _noisy_or_coverage(prob, with_i))
                - np.dot(weights, _noisy_or_coverage(prob, coalition))
            )
            count += 1
        scores[i] = total / max(count, 1)
    return np.maximum(scores, 0.0)


def _banzhaf_power(prob_matrix, anomaly_weights, config, *, _probabilistic_banzhaf, _bruteforce_probabilistic_banzhaf):
    if config.banzhaf_mode == "closed_form":
        return _probabilistic_banzhaf(prob_matrix, anomaly_weights)
    if config.banzhaf_mode == "bruteforce":
        return _bruteforce_probabilistic_banzhaf(prob_matrix, anomaly_weights)
    if config.banzhaf_mode == "individual":
        # Controlled attribution comparator: keep players, targets, anomaly
        # weights, coverage strengths and victim factors unchanged,
        # but score each player's singleton weighted coverage instead of its
        # average marginal contribution over coalitions.
        prob = np.clip(
            np.nan_to_num(np.asarray(prob_matrix, dtype=float)), 0.0, 1.0
        )
        weights = np.maximum(
            np.nan_to_num(np.asarray(anomaly_weights, dtype=float).reshape(-1)),
            0.0,
        )
        if prob.ndim != 2 or prob.shape[1] != weights.shape[0]:
            raise ValueError("individual attribution dimensions do not match")
        return np.maximum(prob @ weights, 0.0)
    raise ValueError(f"Unsupported banzhaf_mode: {config.banzhaf_mode}")


def _build_union_call_graph(dataset_root, exposure_manifest, service_names, boundary_version, dropout_fraction=0.0, *, _CALL_GRAPH_CACHE, _build_call_graph_from_traces):
    """Union per-case pre-fault graphs for a transductive sensitivity control."""

    key = (
        "union_pre_fault",
        os.path.realpath(dataset_root),
        boundary_version,
        float(np.clip(dropout_fraction, 0.0, 1.0)),
        tuple(sorted(service_names)) if service_names else None,
    )
    if key in _CALL_GRAPH_CACHE:
        return _CALL_GRAPH_CACHE[key]
    merged = {}
    records = (exposure_manifest or {}).get("cases", {}).values()
    for record in records:
        fault_directory = record.get("fault_directory")
        repetition_directory = record.get("case_directory")
        if not fault_directory or not repetition_directory:
            continue
        source_dir = os.path.join(
            dataset_root, fault_directory, repetition_directory
        )
        graph = _build_call_graph_from_traces(
            source_dir,
            service_names,
            cache_key=(
                "union_member_pre_fault",
                os.path.realpath(source_dir),
                boundary_version,
            ),
            boundary=record.get("boundary"),
            pre_fault=True,
            dropout_fraction=dropout_fraction,
        )
        for caller, callees in graph.items():
            merged.setdefault(caller, set()).update(callees)
    union = {caller: sorted(callees) for caller, callees in merged.items()}
    if union:
        _CALL_GRAPH_CACHE[key] = union
    return union


def _time_shuffle_trace_series(trace_series, salt):
    """Deterministically shuffle each service timeline for a negative control."""

    shuffled = {}
    for service, values in (trace_series or {}).items():
        array = np.asarray(values, dtype=float).copy()
        digest = hashlib.sha256(
            f"{salt}:{service}:{array.size}".encode("utf-8")
        ).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        shuffled[service] = array[rng.permutation(array.size)]
    return shuffled


def _deterministic_row_dropout(frame, fraction, salt):
    fraction = float(np.clip(fraction, 0.0, 1.0))
    if fraction <= 0.0 or frame.empty:
        return frame.copy()
    if fraction >= 1.0:
        return frame.iloc[0:0].copy()
    digest = hashlib.sha256(
        f"{salt}:{len(frame)}:{fraction:.6f}".encode("utf-8")
    ).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    keep = rng.random(len(frame)) >= fraction
    return frame.loc[keep].copy()


def _add_trace_series_noise(trace_series, fraction, salt, *, _finite_nonnegative):
    fraction = _finite_nonnegative(fraction)
    if fraction <= 0.0:
        return {key: np.asarray(value, dtype=float) for key, value in trace_series.items()}
    noisy = {}
    for service, values in (trace_series or {}).items():
        array = np.asarray(values, dtype=float).copy()
        digest = hashlib.sha256(
            f"{salt}:{service}:{array.size}:{fraction:.6f}".encode("utf-8")
        ).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        scale = float(np.std(array))
        if scale <= 1e-12:
            scale = max(float(np.max(np.abs(array))), 1.0)
        noisy[service] = np.clip(
            array + rng.normal(0.0, fraction * scale, size=array.size),
            0.0,
            None,
        )
    return noisy


def _degree_matched_random_edges(edges):
    """Directed double-edge swaps preserve in/out degree for a null graph."""

    randomized = [tuple(edge) for edge in edges]
    if len(randomized) < 2:
        return randomized
    digest = hashlib.sha256(repr(sorted(randomized)).encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    for _ in range(max(20, 20 * len(randomized))):
        first, second = rng.choice(len(randomized), size=2, replace=False)
        edge_a = randomized[first]
        edge_b = randomized[second]
        source_a, target_a = edge_a[:2]
        source_b, target_b = edge_b[:2]
        proposed_a = (source_a, target_b, *edge_a[2:])
        proposed_b = (source_b, target_a, *edge_b[2:])
        if source_a == target_b or source_b == target_a:
            continue
        other_pairs = {
            (edge[0], edge[1])
            for index, edge in enumerate(randomized)
            if index not in {first, second}
        }
        if (source_a, target_b) in other_pairs or (source_b, target_a) in other_pairs:
            continue
        randomized[first] = proposed_a
        randomized[second] = proposed_b
    return sorted(
        randomized, key=lambda edge: (edge[0], edge[1], edge[3], edge[4])
    )


def _inject_random_trace_edges(edges, services, fraction, *, _finite_nonnegative):
    fraction = _finite_nonnegative(fraction)
    services = sorted(set(services))
    if fraction <= 0.0 or len(services) < 2:
        return list(edges)
    existing = {(edge[0], edge[1]) for edge in edges}
    candidates = [
        (source, target)
        for source in services
        for target in services
        if source != target and (source, target) not in existing
    ]
    if not candidates:
        return list(edges)
    digest = hashlib.sha256(
        f"edge-noise:{sorted(existing)}:{fraction:.6f}".encode("utf-8")
    ).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    count = max(1, int(round(max(len(edges), 1) * fraction)))
    selected = rng.choice(len(candidates), size=min(count, len(candidates)), replace=False)
    confidence = float(np.median([edge[2] for edge in edges])) if edges else 0.5
    augmented = list(edges)
    augmented.extend(
        (candidates[index][0], candidates[index][1], confidence, "noise", 0)
        for index in np.atleast_1d(selected)
    )
    return sorted(augmented, key=lambda edge: (edge[0], edge[1], edge[3], edge[4]))


def _rank_evidence_metrics(all_cols, metric_ev, svc_anom_dict, rank_mode, trace_service_evidence=None, *, _norm_dict, _finite_nonnegative, _parse_svc, _direct_service_score_sum):
    """Rank metrics without the causal game or Banzhaf power calculation."""
    if rank_mode == "trace_evidence":
        # No-metric ablation: rank purely by per-service trace evidence. Within
        # a service all columns tie (trace has no metric-type resolution), so
        # the column name breaks ties — only the service order is meaningful.
        norm_trace = _norm_dict(trace_service_evidence) if trace_service_evidence else {}
        return sorted(
            all_cols,
            key=lambda col: (
                -_finite_nonnegative(norm_trace.get(_parse_svc(col), 0.0)),
                col,
            ),
        )
    if rank_mode == "service_evidence":
        return sorted(
            all_cols,
            key=lambda col: (
                -_finite_nonnegative(svc_anom_dict.get(_parse_svc(col), 0.0)),
                -_finite_nonnegative(metric_ev.get(col, 0.0)),
                col,
            ),
        )
    if rank_mode == "service_evidence_trace":
        # Equal-weight normalized sum of metric and trace per-service evidence,
        # bypassing the Banzhaf game. metric column evidence breaks ties.
        _, _, combined = _direct_service_score_sum(
            svc_anom_dict, trace_service_evidence
        )
        return sorted(
            all_cols,
            key=lambda col: (
                -_finite_nonnegative(combined.get(_parse_svc(col), 0.0)),
                -_finite_nonnegative(metric_ev.get(col, 0.0)),
                col,
            ),
        )
    if rank_mode == "metric_evidence":
        return sorted(
            all_cols,
            key=lambda col: (
                -_finite_nonnegative(metric_ev.get(col, 0.0)),
                col,
            ),
        )
    raise ValueError(f"Unsupported evidence rank mode: {rank_mode}")


def _direct_service_score_sum(metric_service_evidence, trace_service_evidence, *, _finite_nonnegative, _norm_dict):
    """Return independently normalized metric, trace, and summed scores.

    This is the direct-evidence ablation from Figure 1: it stops after evidence
    extraction and assigns each service ``metric_score + trace_score``.  No
    candidate selection, coverage matrix, victim demotion, coverage game, or
    Banzhaf attribution is involved.
    """
    metric_scores = {
        service: _finite_nonnegative(score)
        for service, score in (metric_service_evidence or {}).items()
    }
    trace_scores = {
        service: _finite_nonnegative(score)
        for service, score in (trace_service_evidence or {}).items()
    }
    normalized_metric = _norm_dict(metric_scores) if metric_scores else {}
    normalized_trace = _norm_dict(trace_scores) if trace_scores else {}
    services = sorted(set(normalized_metric) | set(normalized_trace))
    combined = {
        service: (
            normalized_metric.get(service, 0.0)
            + normalized_trace.get(service, 0.0)
        )
        for service in services
    }
    return normalized_metric, normalized_trace, combined


def resolve_runtime_kwargs(kwargs, environ=None):
    """Resolve legacy switches without mutating kwargs or the environment."""
    if environ is None:
        environ = os.environ
    _profile = environ.get("GABANZHAF_PROFILE") == "1"

    runtime_kwargs = dict(kwargs)
    if environ.get("GABANZHAF_BANZHAF_MODE"):
        runtime_kwargs.setdefault("banzhaf_mode", environ["GABANZHAF_BANZHAF_MODE"])
    if environ.get("GABANZHAF_EXACT_CANDIDATE_K"):
        runtime_kwargs.setdefault(
            "exact_candidate_k",
            environ["GABANZHAF_EXACT_CANDIDATE_K"],
        )
    if environ.get("GABANZHAF_DISABLE_COVERAGE") == "1":
        runtime_kwargs.setdefault("use_coverage", False)
    if environ.get("GABANZHAF_DISABLE_TRACE_TOPOLOGY") == "1":
        runtime_kwargs.setdefault("use_trace_topology", False)
    if environ.get("GABANZHAF_DISABLE_METRIC_TYPE_DOMINANCE") == "1":
        runtime_kwargs.setdefault("use_metric_type_dominance", False)
    if environ.get("GABANZHAF_DISABLE_METRICS") == "1":
        runtime_kwargs.setdefault("use_metrics", False)
    for _field in (
        "ev_w_mag", "ev_w_persistence", "ev_w_peak",
        "early_w_mag", "early_w_peak",
        "fusion_w_early", "fusion_w_full",
        "agg_w_p95", "agg_w_top3", "agg_w_max",
        "svc_w_agg", "svc_w_dominance",
    ):
        _env = environ.get("GABANZHAF_" + _field.upper())
        if _env:
            runtime_kwargs.setdefault(_field, float(_env))
    if environ.get("GABANZHAF_DISABLE_EARLY_FULL_FUSION") == "1":
        runtime_kwargs.setdefault("use_early_full_fusion", False)
    if environ.get("GABANZHAF_TRACE_SIGNAL"):
        runtime_kwargs.setdefault("trace_signal", environ["GABANZHAF_TRACE_SIGNAL"])
    if environ.get("GABANZHAF_TRACE_MAX_LAG"):
        runtime_kwargs.setdefault(
            "trace_topology_max_lag", environ["GABANZHAF_TRACE_MAX_LAG"]
        )
    if environ.get("GABANZHAF_TRACE_CONTROL"):
        runtime_kwargs.setdefault("trace_control", environ["GABANZHAF_TRACE_CONTROL"])
    if environ.get("GABANZHAF_TRACE_DROPOUT"):
        runtime_kwargs.setdefault(
            "trace_dropout_fraction", environ["GABANZHAF_TRACE_DROPOUT"]
        )
    if environ.get("GABANZHAF_TRACE_NOISE"):
        runtime_kwargs.setdefault(
            "trace_noise_fraction", environ["GABANZHAF_TRACE_NOISE"]
        )
    if environ.get("GABANZHAF_TRACE_EDGE_NOISE"):
        runtime_kwargs.setdefault(
            "trace_edge_noise_fraction",
            environ["GABANZHAF_TRACE_EDGE_NOISE"],
        )
    if environ.get("GABANZHAF_TRACE_RESIDUAL") == "1":
        runtime_kwargs.setdefault("trace_residual", True)
    if environ.get("MERIT_SCORER"):
        runtime_kwargs.setdefault("scorer", environ["MERIT_SCORER"])
    if environ.get("MERIT_CUSUM_SLACK"):
        runtime_kwargs.setdefault("cusum_slack", float(environ["MERIT_CUSUM_SLACK"]))
    if environ.get("GABANZHAF_RANK_MODE"):
        runtime_kwargs.setdefault("rank_mode", environ["GABANZHAF_RANK_MODE"])
    if environ.get("GABANZHAF_TRACE_TOPOLOGY_CONF"):
        runtime_kwargs.setdefault("trace_topology_conf", float(environ["GABANZHAF_TRACE_TOPOLOGY_CONF"]))
    if environ.get("GABANZHAF_COVERAGE_CONF"):
        runtime_kwargs.setdefault("coverage_conf", float(environ["GABANZHAF_COVERAGE_CONF"]))
    if environ.get("GABANZHAF_COVERAGE_LOUD_THR"):
        runtime_kwargs.setdefault(
            "coverage_loud_thr", float(environ["GABANZHAF_COVERAGE_LOUD_THR"])
        )
    if environ.get("GABANZHAF_TOP_K_SVCS"):
        runtime_kwargs.setdefault("top_k_svcs", int(environ["GABANZHAF_TOP_K_SVCS"]))
    if environ.get("GABANZHAF_USE_VICTIM_DEMOTION") == "1":
        runtime_kwargs.setdefault("use_victim_demotion", True)
    if environ.get("GABANZHAF_DISABLE_VICTIM_DEMOTION") == "1":
        runtime_kwargs.setdefault("use_victim_demotion", False)
    if environ.get("GABANZHAF_DISABLE_COVERAGE_PARENT_PROTECTION") == "1":
        runtime_kwargs.setdefault("protect_coverage_parents", False)
    if environ.get("GABANZHAF_VICTIM_FLOOR"):
        runtime_kwargs.setdefault("victim_floor", float(environ["GABANZHAF_VICTIM_FLOOR"]))
    if environ.get("GABANZHAF_WRITE_DIAGNOSTICS") == "1":
        runtime_kwargs["return_diagnostics"] = True
        runtime_kwargs["return_stage_timing"] = True
    disable_trace = (
        environ.get("GABANZHAF_DISABLE_TRACE") == "1"
        or runtime_kwargs.get("use_trace") is False
    )
    if disable_trace:
        runtime_kwargs = disable_trace_inputs(runtime_kwargs)
    return runtime_kwargs, _profile, disable_trace


def disable_trace_inputs(kwargs):
    """Return a copy with every trace-derived input and provenance scrubbed."""
    runtime_kwargs = dict(kwargs)
    runtime_kwargs.update({
        "use_trace": False,
        "use_trace_topology": False,
        "call_graph": {},
        "trace_topology_edges": [],
        "trace_service_evidence": {},
        "trace_latency_evidence": {},
        "trace_error_evidence": {},
        "trace_stage_timing": {},
        "graph_provenance": {
            "source": "disabled_no_trace",
            "source_case": None,
            "source_cases": [],
            "source_case_count": 0,
            "edge_count": 0,
            "empty_graph_cached": False,
        },
    })
    return runtime_kwargs


def resolve_graph_source(args_ns, environ=None):
    """Resolve graph source without touching trace files.

    Protocol runs pass an input boundary and default to its pre-fault graph.
    Direct callers without a boundary use the full-record graph and receive
    matching provenance instead of being labeled pre-fault without a cutoff.
    """
    if environ is None:
        environ = os.environ
    graph_source = environ.get("GABANZHAF_GRAPH_SOURCE")
    if graph_source is None:
        graph_source = getattr(args_ns, "graph_source", None)
    if graph_source is None:
        graph_source = (
            "per_case_pre_fault"
            if getattr(args_ns, "input_boundary", None) is not None
            else "per_case_full"
        )
    if graph_source not in {
        "per_case_pre_fault",
        "per_case_full",
        "first_case_pre_fault",
        "union_pre_fault",
    }:
        raise ValueError(f"unsupported graph source: {graph_source}")
    return graph_source
