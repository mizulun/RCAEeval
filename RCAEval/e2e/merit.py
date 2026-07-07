import os
import numpy as np
import pandas as pd
import warnings
from dataclasses import dataclass
from time import perf_counter

warnings.filterwarnings("ignore")

from RCAEval.io.time_series import preprocess


@dataclass(frozen=True)
class _TraceTopologyFeatures:
    edges: list
    service_evidence: dict
    latency_evidence: dict
    error_evidence: dict
    stage_timing: dict


@dataclass(frozen=True)
class _GabanzhafConfig:
    """Centralized runtime configuration for the GA-Banzhaf pipeline."""

    top_k_svcs: int = 15
    dk_select_useful: bool = False
    exact_candidate_k: int = 12
    banzhaf_mode: str = "closed_form"
    rank_mode: str = "banzhaf"
    early_window_ratio: float = 0.25
    early_window_min_len: int = 5
    # Call-graph coverage: rescue a quiet upstream root that explains several
    # loud downstream victims (the delay/loss propagation signature). Gated on
    # >=min_loud children above loud_thr so single-strong-anomaly faults
    # (cpu/mem/disk/socket) are left untouched.
    coverage_conf: float = 0.7
    coverage_loud_thr: float = 0.4
    coverage_lat_floor: float = 0.1
    coverage_min_loud: int = 2
    use_trace_topology: bool = True
    trace_topology_max_lag: int = 3
    trace_topology_min_conf: float = 0.3
    trace_topology_conf: float = 0.85
    # Causal-edge thresholds used by victim demotion (historically named
    # frontier_*; names kept to preserve the kwargs API).
    frontier_abnormal_threshold: float = 0.45
    frontier_normal_threshold: float = 0.20
    frontier_edge_threshold: float = 0.30
    frontier_uncertain_stop_factor: float = 0.50
    # Victim demotion: when an abnormal service's causal upstream is also
    # abnormal, it is a propagation victim, so shrink its self-coverage (the
    # diagonal 1.0) toward victim_floor. Roots (normal/absent upstream) keep 1.0.
    use_victim_demotion: bool = True
    victim_floor: float = 0.5
    # Residual trace extraction: build edges from the inject pre/post delta
    # instead of the full-window normalized series, so unperturbed services
    # (which co-activate but are not the propagation path) drop out.
    trace_residual: bool = False
    trace_min_energy: float = 0.1

    @staticmethod
    def _positive_int_or_default(value, default):
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return int(default)
        if not np.isfinite(parsed) or parsed <= 0.0:
            return int(default)
        parsed_int = int(parsed)
        if parsed_int <= 0:
            return int(default)
        return parsed_int

    @staticmethod
    def _nonnegative_int_or_default(value, default):
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return int(default)
        if not np.isfinite(parsed) or parsed < 0.0:
            return int(default)
        return int(parsed)

    @classmethod
    def from_kwargs(cls, kwargs):
        return cls(
            top_k_svcs=int(kwargs.get("top_k_svcs", cls.top_k_svcs)),
            dk_select_useful=bool(kwargs.get("dk_select_useful", cls.dk_select_useful)),
            exact_candidate_k=cls._positive_int_or_default(
                kwargs.get("exact_candidate_k", cls.exact_candidate_k),
                cls.exact_candidate_k,
            ),
            banzhaf_mode=str(kwargs.get("banzhaf_mode", cls.banzhaf_mode)),
            rank_mode=str(kwargs.get("rank_mode", cls.rank_mode)),
            early_window_ratio=float(kwargs.get("early_window_ratio", cls.early_window_ratio)),
            early_window_min_len=cls._positive_int_or_default(
                kwargs.get("early_window_min_len", cls.early_window_min_len),
                cls.early_window_min_len,
            ),
            coverage_conf=float(kwargs.get("coverage_conf", cls.coverage_conf)),
            coverage_loud_thr=float(kwargs.get("coverage_loud_thr", cls.coverage_loud_thr)),
            coverage_lat_floor=float(kwargs.get("coverage_lat_floor", cls.coverage_lat_floor)),
            coverage_min_loud=cls._positive_int_or_default(
                kwargs.get("coverage_min_loud", cls.coverage_min_loud),
                cls.coverage_min_loud,
            ),
            use_trace_topology=bool(
                kwargs.get("use_trace_topology", cls.use_trace_topology)
            ),
            trace_topology_max_lag=cls._nonnegative_int_or_default(
                kwargs.get("trace_topology_max_lag", cls.trace_topology_max_lag),
                cls.trace_topology_max_lag,
            ),
            trace_topology_min_conf=float(
                kwargs.get("trace_topology_min_conf", cls.trace_topology_min_conf)
            ),
            trace_topology_conf=float(
                kwargs.get("trace_topology_conf", cls.trace_topology_conf)
            ),
            frontier_abnormal_threshold=float(
                kwargs.get("frontier_abnormal_threshold", cls.frontier_abnormal_threshold)
            ),
            frontier_normal_threshold=float(
                kwargs.get("frontier_normal_threshold", cls.frontier_normal_threshold)
            ),
            frontier_edge_threshold=float(
                kwargs.get("frontier_edge_threshold", cls.frontier_edge_threshold)
            ),
            frontier_uncertain_stop_factor=float(
                kwargs.get(
                    "frontier_uncertain_stop_factor",
                    cls.frontier_uncertain_stop_factor,
                )
            ),
            trace_residual=bool(kwargs.get("trace_residual", cls.trace_residual)),
            trace_min_energy=float(kwargs.get("trace_min_energy", cls.trace_min_energy)),
            use_victim_demotion=bool(
                kwargs.get("use_victim_demotion", cls.use_victim_demotion)
            ),
            victim_floor=float(kwargs.get("victim_floor", cls.victim_floor)),
        )


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
    """
    if config is None:
        config = _GabanzhafConfig.from_kwargs({})
    normal_mat = np.asarray(normal_mat, dtype=float)
    anomal_mat = np.asarray(anomal_mat, dtype=float)
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
    full = _norm01_arr(0.55 * mag + 0.30 * persistence + 0.15 * peak)
    early_mag = _norm01_arr(np.nanpercentile(early_rz, 95, axis=0))
    early_peak = _norm01_arr(np.nanmax(early_rz, axis=0))
    early = full.copy() if use_full_for_early else _norm01_arr(0.70 * early_mag + 0.30 * early_peak)
    score = _norm01_arr(0.60 * early + 0.40 * full)

    return {col: float(score[i]) for i, col in enumerate(cols)}


def _service_evidence(cols, metric_ev):
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
    svc_agg: dict = {}
    for svc, svc_cols in svc_to_cols.items():
        arr = np.asarray([metric_ev[c] for c in svc_cols], dtype=float)
        top = np.sort(arr)[-min(3, len(arr)):]
        svc_agg[svc] = float(
            0.50 * np.percentile(arr, 95)
            + 0.30 * np.mean(top)
            + 0.20 * np.max(arr)
        )

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
    svc_scores = {
        svc: 0.70 * svc_agg[svc] + 0.30 * dom_norm.get(svc, 0.)
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

    half_complements = 1.0 - 0.5 * prob                          # (K, N)
    product_all = np.prod(half_complements, axis=0)               # (N,)
    product_without = product_all / np.maximum(half_complements, 1e-12)  # (K, N)
    scores = np.sum(weights * prob * product_without, axis=1)     # (K,)
    return np.maximum(scores, 0.0)


def _bruteforce_probabilistic_banzhaf(prob_matrix, anomaly_weights):
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


def _banzhaf_power(prob_matrix, anomaly_weights, config):
    if config.banzhaf_mode == "closed_form":
        return _probabilistic_banzhaf(prob_matrix, anomaly_weights)
    if config.banzhaf_mode == "bruteforce":
        return _bruteforce_probabilistic_banzhaf(prob_matrix, anomaly_weights)
    raise ValueError(f"Unsupported banzhaf_mode: {config.banzhaf_mode}")


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


def _build_call_graph_from_traces(case_dir, service_names=None, cache_key=None):
    """Build {caller: [callees]} from a case's traces.csv span hierarchy.

    A span's parent service is the caller, the span's own service is the callee
    (the called downstream dependency). Same construction for every dataset; no
    precomputed files, no dataset-name branching. Returns {} when no traces.csv
    ships (e.g. sock-shop) so the pipeline degrades to pure metric ranking.

    The service call graph is system-static, so the result is cached under
    ``cache_key`` (the dataset) — only the first case of a run parses traces.csv;
    the rest reuse it. Defaults to per-case (``case_dir``) when no key is given.
    """
    if not case_dir:
        return {}
    key = (cache_key if cache_key is not None else case_dir,
           tuple(sorted(service_names)) if service_names else None)
    if key in _CALL_GRAPH_CACHE:
        return _CALL_GRAPH_CACHE[key]

    path = os.path.join(case_dir, "traces.csv")
    graph = {}
    if os.path.exists(path):
        try:
            df = pd.read_csv(
                path, usecols=["spanID", "parentSpanID", "serviceName"], dtype=str
            ).dropna(subset=["spanID", "serviceName"])
            parent = df[["spanID", "serviceName"]].rename(
                columns={"spanID": "parentSpanID", "serviceName": "callerSvc"}
            )
            joined = df.merge(parent, on="parentSpanID", how="inner")
            edges = {}
            for caller_raw, callee_raw in zip(joined["callerSvc"], joined["serviceName"]):
                caller = _match_service(caller_raw, service_names)
                callee = _match_service(callee_raw, service_names)
                if caller != callee:
                    edges.setdefault(caller, set()).add(callee)
            graph = {k: sorted(v) for k, v in edges.items()}
        except (OSError, ValueError, KeyError):
            graph = {}

    _CALL_GRAPH_CACHE[key] = graph
    return graph


def _service_latency_scores(cols, metric_ev):
    """Per-service latency-metric score (max over its *_latency columns).

    Used to tell propagation victims (loud because latency spiked) from
    services with their own resource fault (loud because of cpu/mem/disk).
    """
    out = {}
    for col in cols:
        if col.rsplit("_", 1)[-1] != "latency":
            continue
        svc = _parse_svc(col)
        out[svc] = max(out.get(svc, 0.0), _finite_nonnegative(metric_ev.get(col, 0.0)))
    return out


def _service_latency_family_scores(cols, metric_ev):
    """Per-service score for latency and latency percentile metrics."""
    out = {}
    for col in cols:
        metric_name = col.rsplit("_", 1)[-1]
        if not metric_name.startswith("latency"):
            continue
        svc = _parse_svc(col)
        out[svc] = max(out.get(svc, 0.0), _finite_nonnegative(metric_ev.get(col, 0.0)))
    return out


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
    evidence = {}
    for svc, values in (trace_series or {}).items():
        normed = _norm_trace_arr(values)
        evidence[svc] = float(np.max(normed)) if normed.size else 0.0
    return evidence


def _trace_topology_from_case_dir(case_dir, call_graph, config, inject_time=None):
    stage_timing = {}
    last_tick = perf_counter()

    def _mark_stage(name):
        nonlocal last_tick
        now = perf_counter()
        stage_timing[name] = float(now - last_tick)
        last_tick = now

    if not case_dir or not config.use_trace_topology:
        return _TraceTopologyFeatures([], {}, {}, {}, stage_timing)

    residual = bool(config.trace_residual)
    latency_series = {}
    error_series = {}
    lat_path = os.path.join(case_dir, "tracets_lat.csv")
    err_path = os.path.join(case_dir, "tracets_err.csv")

    if os.path.exists(lat_path):
        try:
            latency_series = _service_trace_series_from_frame(
                pd.read_csv(lat_path), inject_time=inject_time, residual=residual
            )
        except Exception:
            latency_series = {}
    _mark_stage("trace_latency")

    if os.path.exists(err_path):
        try:
            error_series = _service_trace_series_from_frame(
                pd.read_csv(err_path), inject_time=inject_time, residual=residual
            )
        except Exception:
            error_series = {}
    _mark_stage("trace_error")

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
    _mark_stage("trace_edges")

    latency_evidence = _trace_service_evidence(latency_series)
    error_evidence = _trace_service_evidence(error_series)
    service_evidence = _norm_dict({
        svc: max(latency_evidence.get(svc, 0.0), error_evidence.get(svc, 0.0))
        for svc in set(latency_evidence) | set(error_evidence)
    }) if (latency_evidence or error_evidence) else {}

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

    min_len = min(len(normal_df), len(anomal_df))
    normal_df = normal_df.tail(min_len).reset_index(drop=True)
    anomal_df = anomal_df.head(min_len).reset_index(drop=True)
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


def _rank_metrics(all_cols, svc_banzhaf):
    """Rank metric columns by their service's Banzhaf score (column name breaks ties)."""
    finite_banzhaf = {
        svc: _finite_nonnegative(score)
        for svc, score in (svc_banzhaf or {}).items()
    }
    max_banzhaf = max(finite_banzhaf.values(), default=0.0)
    if max_banzhaf > 0.0:
        nb = {svc: score / max_banzhaf for svc, score in finite_banzhaf.items()}
    else:
        nb = {svc: 1.0 for svc in {_parse_svc(col) for col in all_cols}}

    return sorted(all_cols, key=lambda col: (-nb.get(_parse_svc(col), 0.0), col))


def _rank_evidence_metrics(
    all_cols, metric_ev, svc_anom_dict, rank_mode, trace_service_evidence=None
):
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
        norm_metric = _norm_dict(svc_anom_dict) if svc_anom_dict else {}
        norm_trace = _norm_dict(trace_service_evidence) if trace_service_evidence else {}
        combined = {
            svc: norm_metric.get(svc, 0.0) + norm_trace.get(svc, 0.0)
            for svc in set(norm_metric) | set(norm_trace)
        }
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
        if config.use_trace_topology
        else []
    )
    trace_service_evidence = kwargs.get("trace_service_evidence", {}) or {}
    trace_latency_evidence = kwargs.get("trace_latency_evidence", {}) or {}
    trace_error_evidence = kwargs.get("trace_error_evidence", {}) or {}
    trace_stage_timing = kwargs.get("trace_stage_timing", {}) or {}
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
    normal_df, anomal_df, no_time = _preprocess_windows(
        normal_raw, anomal_raw, dataset, config
    )
    cols = list(no_time.columns)
    normal_mat = normal_df.drop(columns=["time"], errors="ignore").values.astype(float)
    anomal_mat = anomal_df.drop(columns=["time"], errors="ignore").values.astype(float)
    _mark_stage("preprocess")

    # ── 3. Metric and service evidence ───────────────────────────────────────
    metric_ev = _metric_evidence(normal_mat, anomal_mat, cols, config=config)
    svc_anom_dict = _service_evidence(cols, metric_ev)
    svc_names = sorted(svc_anom_dict.keys())
    _mark_stage("metric_evidence")

    if config.rank_mode not in {
        "banzhaf", "service_evidence", "metric_evidence",
        "service_evidence_trace", "trace_evidence",
    }:
        raise ValueError(f"Unsupported rank_mode: {config.rank_mode}")

    # ── 4. Candidates = top services by anomaly score ────────────────────────
    svc_ranked = sorted(svc_names, key=lambda s: svc_anom_dict.get(s, 0.), reverse=True)
    seed_svcs = set(svc_ranked[: config.top_k_svcs])
    cand_list = sorted(seed_svcs)
    _mark_stage("candidates")

    svc_latency = _service_latency_scores(cols, metric_ev)
    svc_latency_family = _service_latency_family_scores(cols, metric_ev)
    trace_topology_edges = [
        edge
        for edge in trace_topology_edges
        if svc_latency_family.get(edge[1], 0.0) >= config.coverage_lat_floor
    ]

    # ── 4b. Call-graph coverage: rescue quiet roots of multi-victim faults ────
    if bool(kwargs.get("use_coverage", True)):
        call_graph = kwargs.get("call_graph", {}) or {}
        coverage_parents = _coverage_parents(call_graph, svc_anom_dict, svc_latency, config)
    else:
        coverage_parents = {}
    causal_edges = _causal_edges(
        coverage_parents,
        trace_topology_edges,
        config.coverage_conf,
    )

    # ── 5. Probabilistic causal game ─────────────────────────────────────────
    game_candidates = _select_exact_candidates(cand_list, svc_anom_dict, config)
    # Force qualified coverage parents into the game even when their own
    # anomaly is too low to make the anomaly-ranked top-k cut.
    for parent in coverage_parents:
        if parent in svc_anom_dict and parent not in game_candidates:
            game_candidates.append(parent)
    trace_added_candidates = 0
    game_services = sorted(svc_anom_dict.keys())
    if config.rank_mode == "banzhaf":
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
                protected=set(coverage_parents),
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
        prob_matrix = np.zeros((0, 0), dtype=float)
        anomaly_weights = np.zeros(0, dtype=float)

    # ── 6. Banzhaf power index ────────────────────────────────────────────────
    svc_banzhaf = {}
    if config.rank_mode == "banzhaf":
        banzhaf_values = _banzhaf_power(prob_matrix, anomaly_weights, config)
        svc_banzhaf = {
            game_candidates[i]: float(banzhaf_values[i])
            for i in range(len(game_candidates))
        }
    _mark_stage("banzhaf")

    # ── 7. Final metric-level ranking by Banzhaf score ───────────────────────
    if config.rank_mode == "banzhaf":
        ranks = _rank_metrics(cols, svc_banzhaf)
    else:
        ranks = _rank_evidence_metrics(
            cols,
            metric_ev,
            svc_anom_dict,
            config.rank_mode,
            trace_service_evidence=trace_service_evidence,
        )
    _mark_stage("ranking")

    result = {"ranks": ranks, "node_names": cols}
    if bool(kwargs.get("return_diagnostics", False)):
        diagnostics = {
            "rank_mode": config.rank_mode,
            "candidate_pool": list(cand_list),
            "game_candidates": list(game_candidates),
            "game_services": list(game_services),
            "causal_probability_matrix": prob_matrix.tolist(),
            "metric_evidence": dict(metric_ev),
            "metric_service_evidence": dict(svc_anom_dict),
            "service_banzhaf": dict(svc_banzhaf),
            "trace_topology_edges": list(trace_topology_edges),
            "trace_edge_count": len(trace_topology_edges),
            "trace_candidate_added_count": trace_added_candidates,
            "trace_service_evidence": dict(trace_service_evidence),
            "trace_latency_evidence": dict(trace_latency_evidence),
            "trace_error_evidence": dict(trace_error_evidence),
            "trace_stage_timing": dict(trace_stage_timing),
            "causal_edges": list(causal_edges),
            "causal_edge_count": len(causal_edges),
            "root_service": kwargs.get("root_service"),
            "root_candidate_recall": (
                kwargs.get("root_service") in set(game_candidates)
                if kwargs.get("root_service") is not None
                else None
            ),
        }
        result["diagnostics"] = diagnostics
        result.update(diagnostics)
    if return_stage_timing:
        result["stage_timing"] = stage_timing
    return result


# ==========================================
# RCAEval entry point
# ==========================================

def merit(data, inject_time=None, dataset=None, num_loop=None,
              sli=None, anomalies=None, **kwargs):
    """Probabilistic causal Banzhaf RCA with call-graph coverage.

    Pipeline: window split → metric evidence → service evidence →
    static call-graph coverage (rescues quiet upstream roots of delay/loss
    propagation) → probabilistic noisy-OR Banzhaf → rank by Banzhaf score.
    Lag-1 correlation topology was removed after ablation showed it dead on
    re2-{tt,ob,ss} and re1-{ss,tt} and net-negative on re1-ob.

    Env knobs:
      GABANZHAF_BANZHAF_MODE   closed_form (default) | bruteforce
      GABANZHAF_EXACT_CANDIDATE_K   override the game candidate count
      GABANZHAF_DISABLE_COVERAGE=1  turn off call-graph coverage (ablation)
      GABANZHAF_DISABLE_TRACE=1  drop all trace info (empty call graph → metric)
      GABANZHAF_DISABLE_TRACE_TOPOLOGY=1  turn off trace-derived topology
      GABANZHAF_TRACE_RESIDUAL=1  build trace edges from inject pre/post residual
      GABANZHAF_PROFILE=1      print per-stage timing to stderr
                               (preprocess | metric_evidence | candidates |
                                banzhaf | ranking)
    """
    import sys
    _profile = os.environ.get("GABANZHAF_PROFILE") == "1"

    runtime_kwargs = dict(kwargs)
    if os.environ.get("GABANZHAF_BANZHAF_MODE"):
        runtime_kwargs.setdefault("banzhaf_mode", os.environ["GABANZHAF_BANZHAF_MODE"])
    if os.environ.get("GABANZHAF_EXACT_CANDIDATE_K"):
        runtime_kwargs.setdefault(
            "exact_candidate_k",
            os.environ["GABANZHAF_EXACT_CANDIDATE_K"],
        )
    if os.environ.get("GABANZHAF_DISABLE_COVERAGE") == "1":
        runtime_kwargs.setdefault("use_coverage", False)
    if os.environ.get("GABANZHAF_DISABLE_TRACE_TOPOLOGY") == "1":
        runtime_kwargs.setdefault("use_trace_topology", False)
    if os.environ.get("GABANZHAF_TRACE_RESIDUAL") == "1":
        runtime_kwargs.setdefault("trace_residual", True)
    if os.environ.get("GABANZHAF_RANK_MODE"):
        runtime_kwargs.setdefault("rank_mode", os.environ["GABANZHAF_RANK_MODE"])
    if os.environ.get("GABANZHAF_TRACE_TOPOLOGY_CONF"):
        runtime_kwargs.setdefault("trace_topology_conf", float(os.environ["GABANZHAF_TRACE_TOPOLOGY_CONF"]))
    if os.environ.get("GABANZHAF_USE_VICTIM_DEMOTION") == "1":
        runtime_kwargs.setdefault("use_victim_demotion", True)
    if os.environ.get("GABANZHAF_DISABLE_VICTIM_DEMOTION") == "1":
        runtime_kwargs.setdefault("use_victim_demotion", False)
    if os.environ.get("GABANZHAF_VICTIM_FLOOR"):
        runtime_kwargs.setdefault("victim_floor", float(os.environ["GABANZHAF_VICTIM_FLOOR"]))
    if os.environ.get("GABANZHAF_WRITE_DIAGNOSTICS") == "1":
        runtime_kwargs["return_diagnostics"] = True
        runtime_kwargs["return_stage_timing"] = True
    disable_trace = os.environ.get("GABANZHAF_DISABLE_TRACE") == "1"
    args_ns = kwargs.get("args")
    if args_ns and hasattr(args_ns, "data_path"):
        case_dir = os.path.dirname(args_ns.data_path)
        fault_dir = os.path.basename(os.path.dirname(case_dir))
        runtime_kwargs["root_service"] = fault_dir.split("_", 1)[0]
        trace_config = _GabanzhafConfig.from_kwargs(runtime_kwargs)
        # Build the service call graph from this case's own traces.csv — same
        # construction for every dataset, no precomputed files. Ablation knob
        # GABANZHAF_DISABLE_TRACE forces it empty (no trace information at all).
        service_names = {_parse_svc(c) for c in data.columns if c != "time"}
        # Cache per dataset root (parent of the fault dirs), so only the first
        # case of a run parses traces.csv — the call graph is system-static.
        dataset_root = os.path.dirname(os.path.dirname(case_dir))
        call_graph = (
            {} if disable_trace
            else _build_call_graph_from_traces(
                case_dir, service_names, cache_key=dataset_root
            )
        )
        runtime_kwargs["call_graph"] = call_graph
        trace_features = _trace_topology_from_case_dir(
            case_dir,
            call_graph,
            trace_config,
            inject_time=inject_time,
        )
        runtime_kwargs.setdefault("trace_topology_edges", trace_features.edges)
        runtime_kwargs.setdefault("trace_service_evidence", trace_features.service_evidence)
        runtime_kwargs.setdefault("trace_latency_evidence", trace_features.latency_evidence)
        runtime_kwargs.setdefault("trace_error_evidence", trace_features.error_evidence)
        runtime_kwargs.setdefault("trace_stage_timing", trace_features.stage_timing)
    if _profile:
        runtime_kwargs["return_stage_timing"] = True

    result = _merit_core(data, inject_time, dataset, anomalies, runtime_kwargs)
    if "diagnostics" in result:
        trace_stage_timing = result["diagnostics"].get("trace_stage_timing", {})
        merged_stage_timing = {}
        if trace_stage_timing:
            merged_stage_timing["trace"] = float(sum(trace_stage_timing.values()))
        merged_stage_timing.update(result.get("stage_timing", {}))
        result["diagnostics"]["stage_timing"] = dict(
            merged_stage_timing
        )

    if _profile and "stage_timing" in result:
        st = result["stage_timing"]
        total = sum(st.values())
        label = str(dataset) or "?"
        parts = [f"{k}={v:.3f}s" for k, v in st.items()] + [f"total={total:.3f}s"]
        print(f"[PROFILE] {label}  " + "  ".join(parts), file=sys.stderr)

    return result
