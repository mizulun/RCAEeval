"""local_banzhaf (Design B): Inference-only GAT + continuous Banzhaf RCA.

Pipeline overview:
  [1] Window split & preprocess → metric anomaly scores (RobustScaler)
  [2] Build service dependency graph:
      - traces.csv present → two-pass span-join (parent-then-child safe)
      - no traces         → lag-correlation fallback
  [3] Seed-based local subgraph expansion via Q(S) marginal-gain greedy search
  [4] Compute 5 service-level weight signals:
      w_local, w_sli, w_prop, w_onset, w_resource
  [5] Build 7-dim node feature vectors per service
  [6] Inference-only GAT on local subgraph:
      - Attention e(caller→callee) encodes causal plausibility of the edge
        (callee resource anomaly + early onset + SLI alignment + temporal)
      - 2-hop message passing: h_v' = blend(h_v, Σ α·h_neighbor)
      - Output: refined edge weights (attention) + updated node embeddings
  [7] GAT-weighted soft influence matrix (N×N, reverse-graph path products)
      influence[s, v] = max-path-product from s to v in reverse call graph
      Semantics: "how strongly does root-cause candidate s explain anomaly at v?"
  [8] Continuous cooperative game utility (replaces binary v(S)):
      u(S) = λ1·Coverage(S) + λ2·Align(S) + λ3·Temporal(S) - λ4·Parsimony(S)
      - Coverage: GAT-weighted reachability of anomalous nodes from S
      - Align:    best SLI correlation within S
      - Temporal: fraction of S-internal edges consistent with causal ordering
      - Parsimony: complexity penalty, discourages large coalitions
  [9] Block Monte Carlo Banzhaf: B_i = E_{S⊆N\\{i}}[u(S∪{i}) - u(S)]
      - Samples split into n_blocks independent blocks (same total budget)
      - Block estimates yield 95% t-based CIs per service (banzhaf_ci)
      - Split-half Kendall τ and top-5 overlap quantify ranking stability
  [10] Final score: η·norm(B_i) + (1-η)·norm(w_combined(i))
  [11] Map service ranks → representative metric strings; output explanations
      Return fields: ranks, banzhaf_values, banzhaf_ci, banzhaf_std,
                     stability, best_coalition, gat_edge_weights

Key design decisions:
  - GAT does NOT produce final root-cause scores; it only refines propagation
    strengths (edge attention weights) for the cooperative game utility.
  - Continuous u(S) gives richer gradient than binary v(S), reducing Banzhaf ties.
  - Inference-only GAT: no training, attention weights are heuristic linear
    combinations of node features encoding causal plausibility. Avoids heavy deps.
  - Block MC Banzhaf: same total samples, reorganised into blocks for free CIs.
  - RobustScaler throughout: prevents z-score explosion from near-zero normals.
  - Two-pass trace reading: parent spans always resolved before child spans.
"""

from __future__ import annotations

import os
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from RCAEval.io.time_series import preprocess


warnings.filterwarnings("ignore")


# ============================================================
# Utility: robust statistics
# ============================================================

def _to_1d_float(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size == 0:
        return x
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _safe_corr(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """Pearson correlation with guards for constant/short series."""
    a = _to_1d_float(a)
    b = _to_1d_float(b)
    n = min(a.size, b.size)
    if n < 3:
        return 0.0
    a, b = a[:n], b[:n]
    sa, sb = float(np.std(a)), float(np.std(b))
    if sa < eps or sb < eps:
        return 0.0
    c = float(np.corrcoef(a, b)[0, 1])
    return c if np.isfinite(c) else 0.0


def _lead_lag_corr(x: np.ndarray, y: np.ndarray, max_lag: int = 2) -> float:
    """Max absolute correlation across small lead/lag range."""
    x = _to_1d_float(x)
    y = _to_1d_float(y)
    n = min(x.size, y.size)
    if n < 5:
        return 0.0
    x, y = x[:n], y[:n]
    best = 0.0
    for lag in range(-max_lag, max_lag + 1):
        if lag == 0:
            c = _safe_corr(x, y)
        elif lag > 0:
            c = _safe_corr(x[:-lag], y[lag:])
        else:
            k = -lag
            c = _safe_corr(x[k:], y[:-k])
        best = max(best, abs(c))
    return float(best)


def _min_max_norm(values: Dict[str, float]) -> Dict[str, float]:
    if not values:
        return {}
    vs = np.array(list(values.values()), dtype=float)
    vs = np.nan_to_num(vs, nan=0.0, posinf=0.0, neginf=0.0)
    vmin, vmax = float(vs.min()), float(vs.max())
    if vmax - vmin < 1e-12:
        return {k: 0.0 for k in values}
    return {k: float((v - vmin) / (vmax - vmin)) for k, v in values.items()}


# ============================================================
# SLI auto-selection
# ============================================================

# Preferred SLI signals (golden signals: latency)
_SLI_PREFER_KEYWORDS: Tuple[str, ...] = (
    "lat50", "lat_50", "latency", "p50", "p95", "response_time",
)
# Secondary SLI signals (error / throughput)
_SLI_SECONDARY_KEYWORDS: Tuple[str, ...] = (
    "_error", "error_rate", "fail_rate", "_rps", "_throughput",
)
# Columns that are NOT SLI candidates
_NON_SLI_COLUMNS: frozenset = frozenset({"time", "timestamp", "Time"})


def _auto_select_sli(
    data: pd.DataFrame,
    sli_col_hint: Optional[str] = None,
    prefer_latency: bool = True,
) -> Optional[str]:
    """Heuristically select the SLI (golden signal) column from a DataFrame.

    Priority:
      1. ``sli_col_hint`` if it exists in ``data`` (explicit pass-through).
      2. Latency columns matching ``_SLI_PREFER_KEYWORDS``.
      3. Error-rate / throughput columns matching ``_SLI_SECONDARY_KEYWORDS``.
      4. Highest-std numeric column (fallback for datasets without keyword hints).

    Within each tier, the column with the highest standard deviation is
    preferred (more temporal variation → more signal for change-point detection).
    """
    cols = [c for c in data.columns if c not in _NON_SLI_COLUMNS]
    if not cols:
        return None

    if sli_col_hint is not None and sli_col_hint in data.columns:
        return sli_col_hint

    def _col_std(c: str) -> float:
        try:
            return float(pd.to_numeric(data[c], errors="coerce").std())
        except Exception:
            return 0.0

    if prefer_latency:
        t1 = [c for c in cols if any(kw in c.lower() for kw in _SLI_PREFER_KEYWORDS)]
        if t1:
            return max(t1, key=_col_std)

    t2 = [c for c in cols if any(kw in c.lower() for kw in _SLI_SECONDARY_KEYWORDS)]
    if t2:
        return max(t2, key=_col_std)

    return max(cols, key=_col_std) or None


# ============================================================
# Change-point detection (inject_time-free window split)
# ============================================================

def _cusum_change_point(
    series: np.ndarray,
    warmup_frac: float = 0.30,
    h: float = 5.0,
    k: float = 0.5,
) -> int:
    """One-sided CUSUM change-point detector.

    Estimates baseline (μ, σ) from the first ``warmup_frac`` fraction of the
    series, then returns the first index where the cumulative sum of
    standardised deviations exceeds alarm threshold ``h``.

    Args:
        series:      1-D float array.
        warmup_frac: fraction of series used to estimate baseline statistics.
        h:           CUSUM alarm threshold (σ units).
        k:           slack / reference value (typically ½).

    Returns:
        Index of the detected change-point, or ``-1`` if not detected.
    """
    x = _to_1d_float(series)
    T = len(x)
    if T < 6:
        return -1

    warmup = max(3, int(T * warmup_frac))
    mu = float(np.mean(x[:warmup]))
    sigma = float(np.std(x[:warmup]))
    if sigma < 1e-9:
        sigma = float(np.std(x))
    if sigma < 1e-9:
        return -1

    S = 0.0
    for t in range(warmup, T):
        z = (x[t] - mu) / sigma
        S = max(0.0, S + z - k)
        if S > h:
            return t
    return -1


def _zscore_change_point(
    series: np.ndarray,
    warmup_frac: float = 0.30,
    thresh: float = 3.0,
    min_streak: int = 3,
) -> int:
    """Rolling z-score change-point: first run of ≥ ``min_streak`` points above ``thresh``.

    Baseline (μ, σ) estimated from warmup window. Returns the first index of
    the qualifying streak, or ``-1`` if none found.
    """
    x = _to_1d_float(series)
    T = len(x)
    if T < 6:
        return -1

    warmup = max(3, int(T * warmup_frac))
    mu = float(np.mean(x[:warmup]))
    sigma = float(np.std(x[:warmup]))
    if sigma < 1e-9:
        sigma = float(np.std(x))
    if sigma < 1e-9:
        return -1

    streak = 0
    streak_start = -1
    for t in range(warmup, T):
        if abs((x[t] - mu) / sigma) > thresh:
            if streak == 0:
                streak_start = t
            streak += 1
            if streak >= min_streak:
                return streak_start
        else:
            streak = 0
            streak_start = -1
    return -1


def _detect_split_index(
    data: pd.DataFrame,
    sli_col: Optional[str],
    method: str = "cusum",
    min_normal_frac: float = 0.30,
    max_normal_frac: float = 0.80,
    cusum_h: float = 5.0,
    cusum_k: float = 0.5,
    cusum_warmup: float = 0.30,
    zscore_thresh: float = 3.0,
    zscore_warmup: float = 0.30,
    zscore_min_streak: int = 3,
) -> Tuple[int, str]:
    """Detect the row index separating the normal window from the anomalous window.

    Strategy:
      1. Run the selected ``method`` on the ``sli_col`` series.
      2. If that fails or the result is out of ``[min_normal_frac, max_normal_frac]``,
         retry on a composite signal (mean |z-score| across all numeric columns).
      3. If still out of bounds, clamp to the closest valid boundary.

    Args:
        data:             Full (unsplit) DataFrame including ``time`` column.
        sli_col:          SLI column name to prefer for detection.
        method:           ``"cusum"`` | ``"zscore"``.
        min_normal_frac:  Minimum fraction of rows that must be labelled normal.
        max_normal_frac:  Maximum fraction of rows that may be labelled normal.

    Returns:
        ``(split_row, detection_source)`` where ``detection_source`` is one of
        ``"cusum"``, ``"zscore"``, ``"cusum_composite"``, ``"zscore_composite"``,
        or ``"boundary_fallback"``.
    """
    T = len(data)
    lo = max(1, int(T * min_normal_frac))
    hi = min(T - 1, int(T * max_normal_frac))

    def _run(series: np.ndarray, src_suffix: str) -> Tuple[int, str]:
        if method == "cusum":
            cp = _cusum_change_point(series, warmup_frac=cusum_warmup,
                                     h=cusum_h, k=cusum_k)
            tag = "cusum" + src_suffix
        else:
            cp = _zscore_change_point(series, warmup_frac=zscore_warmup,
                                      thresh=zscore_thresh,
                                      min_streak=zscore_min_streak)
            tag = "zscore" + src_suffix
        return cp, tag

    # ── Primary: SLI column ──
    cp = -1
    tag = ""
    if sli_col is not None and sli_col in data.columns:
        series = _to_1d_float(
            pd.to_numeric(data[sli_col], errors="coerce").fillna(0).to_numpy()
        )
        cp, tag = _run(series, "")

    # ── Fallback: composite mean |z-score| across all numeric columns ──
    if cp < 0 or cp < lo or cp > hi:
        num_cols = [
            c for c in data.columns
            if c not in _NON_SLI_COLUMNS
            and pd.api.types.is_numeric_dtype(data[c])
        ]
        if num_cols:
            warmup = max(3, lo)
            composite = np.zeros(T, dtype=float)
            count = 0
            for col in num_cols:
                x = _to_1d_float(
                    pd.to_numeric(data[col], errors="coerce").fillna(0).to_numpy()
                )
                mu = float(np.mean(x[:warmup]))
                sigma = max(float(np.std(x[:warmup])), float(np.std(x)), 1e-9)
                composite += np.abs((x - mu) / sigma)
                count += 1
            composite /= max(count, 1)
            cp_comp, tag_comp = _run(composite, "_composite")
            if lo <= cp_comp <= hi:
                return cp_comp, tag_comp

    # ── Clamp to valid range ──
    if cp < lo or cp < 0:
        return lo, "boundary_fallback"
    if cp > hi:
        return hi, "boundary_fallback"
    return cp, tag


def _bit_iter(mask: int) -> Iterable[int]:
    """Iterate over set bit positions in mask."""
    while mask:
        lsb = mask & -mask
        yield lsb.bit_length() - 1
        mask ^= lsb


# ============================================================
# Service / metric parsing
# ============================================================

def _parse_service_metric(col: str) -> Tuple[str, str]:
    s = str(col)
    if "_" not in s:
        return s, "unknown"
    return s.split("_", 1)


def _group_cols_by_service(cols: Sequence[str]) -> Dict[str, List[str]]:
    by_service: Dict[str, List[str]] = {}
    for c in cols:
        svc, _ = _parse_service_metric(c)
        by_service.setdefault(svc, []).append(c)
    return by_service


def _is_host_like_service(service: str) -> bool:
    """Heuristic filter for host/node-exporter 'services'.

    In any Prometheus/k8s deployment, node-exporter metrics are keyed by
    hostnames or IP addresses (e.g. '10-192-15-225-9100_*').  These are not
    valid RCA targets (ground truth is always a microservice), but their
    anomaly signal may be crucial — especially for resource faults like disk
    or socket saturation.

    We treat any service token starting with a digit as host-like, which
    covers both IPv4 addresses and numeric hostnames.
    """
    s = str(service).strip()
    return bool(s) and s[0].isdigit()


def _attribute_host_metrics(
    host_series: Dict[str, np.ndarray],
    micro_series: Dict[str, np.ndarray],
    anom_start: int,
    corr_threshold: float = 0.40,
) -> Dict[str, float]:
    """Attribute host-level metric anomalies to the most co-anomalous microservice.

    Host metrics (disk I/O, socket counts, etc.) carry the raw fault signal for
    resource faults but are keyed by IP/hostname, not by microservice name.
    Without attribution they are silently dropped, causing the root-cause
    service to appear normal.

    For each host service whose anomaly window z-score exceeds 1.0 (i.e. a
    real anomaly, not noise), we find the microservice whose anomaly-window
    z-score series is most co-anomalous (highest absolute lead-lag correlation).
    The host's peak z-score is then offered as a bonus to that microservice's
    anomaly score, taken as a max so one host can boost at most one service.

    Args:
        host_series:      host-svc-name → full-window z-score series (N_f,)
        micro_series:     microservice-name → full-window z-score series (N_f,)
        anom_start:       row index where the anomaly window begins
        corr_threshold:   minimum absolute lead-lag correlation to accept attribution
    Returns:
        Dict microservice → anomaly bonus (to be merged via max into service_anom_score)
    """
    bonus: Dict[str, float] = {}
    if not host_series or not micro_series:
        return bonus

    for _host_svc, h_full in host_series.items():
        h_anom = _to_1d_float(h_full)[anom_start:]
        h_peak = float(h_anom.max()) if h_anom.size > 0 else 0.0
        if h_peak < 1.0:
            continue  # no meaningful anomaly in this host metric

        best_micro: Optional[str] = None
        best_corr = 0.0
        for micro_svc, m_full in micro_series.items():
            m_anom = _to_1d_float(m_full)[anom_start:]
            c = abs(float(_lead_lag_corr(h_anom, m_anom, max_lag=2)))
            if c > best_corr:
                best_corr, best_micro = c, micro_svc

        if best_micro is not None and best_corr >= corr_threshold:
            # max attribution: one host contributes to at most one microservice
            bonus[best_micro] = max(bonus.get(best_micro, 0.0), h_peak)

    return bonus


# ============================================================
# Resource vs. latency discrimination
# ============================================================

_RESOURCE_SUFFIXES = frozenset([
    '_cpu', '_mem', '_diskio', '_disk', '_socket', '_socketio', '_io'
])
_LATENCY_SUFFIXES = frozenset([
    '_latency', '_error', '_workload', '_rps', '_throughput'
])


def _compute_resource_type_weight(
    by_service: Dict[str, List[str]],
    metric_score: Dict[str, float],
) -> Dict[str, float]:
    """Resource-vs-latency ratio per service.

    High (→1): primarily resource anomaly → likely root cause.
    Low (→0):  primarily latency/effect anomaly → likely downstream.
    """
    w: Dict[str, float] = {}
    for svc, cols in by_service.items():
        res_max = lat_max = 0.0
        for c in cols:
            s = float(metric_score.get(c, 0.0))
            cl = c.lower()
            if any(cl.endswith(sf) for sf in _RESOURCE_SUFFIXES):
                res_max = max(res_max, s)
            elif any(cl.endswith(sf) for sf in _LATENCY_SUFFIXES):
                lat_max = max(lat_max, s)
        total = res_max + lat_max
        w[svc] = 0.5 if total < 1e-9 else res_max / total
    return w


# ============================================================
# Trace-based graph construction (two-pass)
# ============================================================

@dataclass(frozen=True)
class _TraceSchema:
    trace_id: str
    span_id: str
    parent_span_id: str
    service: str


def _detect_trace_schema(columns: Sequence[str]) -> Optional[_TraceSchema]:
    cols = {str(c) for c in columns}
    if {"traceID", "spanID", "parentSpanID", "serviceName"}.issubset(cols):
        return _TraceSchema("traceID", "spanID", "parentSpanID", "serviceName")
    for t, s, p, svc in [
        ("trace_id", "span_id", "parent_span_id", "service"),
        ("traceId", "spanId", "parentSpanId", "serviceName"),
        ("trace_id", "span_id", "parent_id", "serviceName"),
    ]:
        if {t, s, p, svc}.issubset(cols):
            return _TraceSchema(t, s, p, svc)
    return None


def _read_traces_direct(
    traces_path: str,
    colset: set,
    chunksize: int,
    max_rows: int,
    max_edges: int,
) -> Optional[List[Tuple[str, str]]]:
    """Handle traces with explicit caller/callee columns."""
    for a, b in [
        ("caller_service", "callee_service"),
        ("callerService", "calleeService"),
        ("parentServiceName", "serviceName"),
        ("source", "target"),
        ("src", "dst"),
    ]:
        if a not in colset or b not in colset:
            continue
        edges: set = set()
        seen = 0
        try:
            for chunk in pd.read_csv(
                traces_path, usecols=[a, b],
                dtype={a: "string", b: "string"}, chunksize=chunksize
            ):
                for x, y in zip(chunk[a].astype(str), chunk[b].astype(str)):
                    x, y = x.strip(), y.strip()
                    if x and y and x != y:
                        edges.add((x, y))
                        if len(edges) >= max_edges:
                            return list(edges)
                seen += len(chunk)
                if seen >= max_rows:
                    break
        except Exception:
            return None
        return list(edges) or None
    return None


def _read_traces_edges(case_dir: str) -> Optional[List[Tuple[str, str]]]:
    """Build caller→callee directed edges from traces.csv.

    Two-pass strategy (fixes single-pass parent-after-child miss):
      Pass 1: collect complete (traceID, spanID) → serviceName mapping.
      Pass 2: join parentSpanID to build directed edges.

    IMPORTANT: pd.read_csv with usecols returns columns in CSV file order,
    NOT the order of usecols. Always access by column name, not position.
    """
    traces_path = os.path.join(case_dir, "traces.csv")
    if not os.path.exists(traces_path):
        return None

    max_rows = int(os.getenv("LOCAL_BANZHAF_TRACE_MAX_ROWS", 200_000))
    max_edges = int(os.getenv("LOCAL_BANZHAF_TRACE_MAX_EDGES", 2_000))
    chunksize = int(os.getenv("LOCAL_BANZHAF_TRACE_CHUNKSIZE", 50_000))

    try:
        header = pd.read_csv(traces_path, nrows=0)
        cols = [str(c) for c in header.columns]
    except Exception:
        return None

    colset = set(cols)
    direct = _read_traces_direct(traces_path, colset, chunksize, max_rows, max_edges)
    if direct is not None:
        return direct

    schema = _detect_trace_schema(cols)
    if schema is None:
        return None

    usecols = [schema.trace_id, schema.span_id, schema.parent_span_id, schema.service]
    dtype = {c: "string" for c in usecols}
    tid, sid, pid, svc = schema.trace_id, schema.span_id, schema.parent_span_id, schema.service

    # Pass 1: build span→service map
    span_service: Dict[Tuple[str, str], str] = {}
    seen = 0
    try:
        for chunk in pd.read_csv(traces_path, usecols=usecols, dtype=dtype, chunksize=chunksize):
            for trace_id, span_id, service in zip(
                chunk[tid].astype(str), chunk[sid].astype(str), chunk[svc].astype(str)
            ):
                t, s, sv = trace_id.strip(), span_id.strip(), service.strip()
                if t and s and sv and sv.lower() not in {"nan", "none", ""}:
                    span_service[(t, s)] = sv
            seen += len(chunk)
            if seen >= max_rows:
                break
    except Exception:
        return None

    if not span_service:
        return None

    # Pass 2: find edges via parentSpanID join
    edges: set = set()
    seen = 0
    try:
        for chunk in pd.read_csv(traces_path, usecols=usecols, dtype=dtype, chunksize=chunksize):
            for trace_id, parent_id, service in zip(
                chunk[tid].astype(str), chunk[pid].astype(str), chunk[svc].astype(str)
            ):
                t = trace_id.strip()
                p = parent_id.strip()
                sv = service.strip()
                if sv.lower() in {"nan", "none", ""} or p.lower() in {"nan", "none", ""}:
                    continue
                parent_svc = span_service.get((t, p), "")
                if parent_svc and parent_svc != sv:
                    edges.add((parent_svc, sv))
                    if len(edges) >= max_edges:
                        return list(edges)
            seen += len(chunk)
            if seen >= max_rows:
                break
    except Exception:
        pass

    return list(edges) or None


# ============================================================
# Metrics-inferred graph (RE1 fallback)
# ============================================================

def _infer_edges_from_lag_corr(
    service_series: Dict[str, np.ndarray],
    corr_threshold: float = 0.35,
    max_in: int = 3,
    max_out: int = 3,
    dir_margin: float = 0.05,
) -> List[Tuple[str, str]]:
    """Infer directed edges i→j via asymmetric lag-correlation.

    An edge X→Y is added only when X *temporally leads* Y more than Y leads X:
        c_xy = corr(X[t-1], Y[t])   — X precedes Y by one step
        c_yx = corr(Y[t-1], X[t])   — Y precedes X by one step
        add X→Y iff c_xy >= corr_threshold AND c_xy > c_yx + dir_margin

    The directional advantage (c_xy − c_yx) is used as the edge score so that
    strongly asymmetric lead-lag relationships are preferred over symmetric ones
    that arise from shared common causes rather than direct causation.
    """
    services = sorted(service_series.keys())
    if len(services) < 2:
        return []
    pairs: List[Tuple[float, str, str]] = []
    for si in services:
        xi = _to_1d_float(service_series[si])
        if xi.size < 3:
            continue
        for sj in services:
            if si == sj:
                continue
            yj = _to_1d_float(service_series[sj])
            n = min(xi.size, yj.size)
            if n < 3:
                continue
            c_xy = _safe_corr(xi[:n - 1], yj[1:n])   # X leads Y
            c_yx = _safe_corr(yj[:n - 1], xi[1:n])   # Y leads X
            if c_xy >= corr_threshold and c_xy > c_yx + dir_margin:
                pairs.append((c_xy - c_yx, si, sj))   # score = directional advantage
    pairs.sort(reverse=True, key=lambda x: x[0])
    in_deg = {s: 0 for s in services}
    out_deg = {s: 0 for s in services}
    edges: List[Tuple[str, str]] = []
    for c, u, v in pairs:
        if out_deg[u] >= max_out or in_deg[v] >= max_in:
            continue
        edges.append((u, v))
        out_deg[u] += 1
        in_deg[v] += 1
    return edges


# ============================================================
# Local graph expansion
# ============================================================

def _build_neighbor_index(
    edges: Sequence[Tuple[str, str]]
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    out_n: Dict[str, List[str]] = {}
    in_n: Dict[str, List[str]] = {}
    for u, v in edges:
        out_n.setdefault(u, []).append(v)
        in_n.setdefault(v, []).append(u)
    return out_n, in_n


def _expand_local_nodes(
    seeds: Sequence[str],
    out_n: Dict[str, List[str]],
    in_n: Dict[str, List[str]],
    service_evidence: Dict[str, float],
    nmax: int,
    eps: float,
) -> List[str]:
    """BFS from seeds using evidence-weighted priority queue."""
    import heapq
    chosen: set = set(seeds)
    heap: List[Tuple[float, str]] = []

    def push_nbrs(x: str) -> None:
        for nb in (out_n.get(x, []) + in_n.get(x, [])):
            if nb not in chosen:
                heapq.heappush(heap, (-float(service_evidence.get(nb, 0.0)), nb))

    for s in seeds:
        push_nbrs(s)

    while heap and len(chosen) < nmax:
        neg_pr, v = heapq.heappop(heap)
        if v in chosen:
            continue
        if -neg_pr < eps:
            break
        chosen.add(v)
        push_nbrs(v)

    return list(chosen)


# ============================================================
# Greedy subgraph expansion via Q(S) marginal gain
# ============================================================

def _compute_expansion_influence(
    nodes: List[str],
    edges: List[Tuple[str, str]],
    max_hops: int = 3,
    hop_decay: float = 0.85,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Uniform-weight influence matrix for greedy subgraph expansion.

    Semantics: ``influence[i, j]`` = how strongly service *i* (potential
    root cause) explains the anomaly observed at service *j* through
    call-propagation.

    Uses 1/fan-out edge weights (no GAT — avoids circular dependency since
    this is called *before* node features are built in Step [7]).

    Returns:
        influence: (N, N) float32 array in [0, 1]
        idx:       service → row/col index
    """
    n = len(nodes)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64), {}

    idx: Dict[str, int] = {s: i for i, s in enumerate(nodes)}

    out_counts: Dict[str, int] = {}
    for (u, v) in edges:
        if u in idx and v in idx:
            out_counts[u] = out_counts.get(u, 0) + 1

    # Reverse adjacency: A_rev[callee_idx, caller_idx] = uniform weight
    # Semantics: callee is root cause → propagates anomaly to caller.
    A_rev = np.zeros((n, n), dtype=np.float64)
    for (caller, callee) in edges:
        ii = idx.get(caller, -1)
        jj = idx.get(callee, -1)
        if ii >= 0 and jj >= 0:
            w = 1.0 / max(1, out_counts.get(caller, 1))
            A_rev[jj, ii] = max(A_rev[jj, ii], w)

    # Max-product path (Bellman-Ford style).
    # next_hop[i, k] = max_j( A_rev[i, j] * current[j, k] )
    # Vectorised via (N, N, N) broadcasting — acceptable for N ≤ 150.
    influence = np.eye(n, dtype=np.float64)
    current = A_rev.copy()
    for hop in range(1, max_hops + 1):
        influence = np.maximum(influence, (hop_decay ** hop) * current)
        next_hop = np.max(
            A_rev[:, :, np.newaxis] * current[np.newaxis, :, :], axis=1
        )
        current = next_hop

    return np.clip(influence, 0.0, 1.0), idx


def _greedy_subgraph_expand(
    seeds: Sequence[str],
    out_n: Dict[str, List[str]],
    in_n: Dict[str, List[str]],
    service_evidence: Dict[str, float],
    sli_align: Dict[str, float],
    onset_idx_map: Dict[str, int],
    max_anom_len: int,
    nmax: int,
    rho: float = 0.01,
    lambda_cov: float = 0.50,
    lambda_align: float = 0.25,
    lambda_temp: float = 0.15,
    lambda_complex: float = 0.10,
    hop_decay: float = 0.85,
    eps_abs: float = 1e-6,
) -> List[str]:
    """Greedy subgraph expansion guided by marginal Q(S) gain.

    Replaces the hard-threshold BFS in ``_expand_local_nodes`` with a
    principled, self-calibrating stopping criterion.

    Q(S) = λ₁·Coverage(S) + λ₂·Align(S) + λ₃·Temporal(S) − λ₄·Complexity(S)

      Coverage(S)  = Σ_v e_v · max_{s∈S} I(s→v) / Σ_v e_v
      Align(S)     = max_{s∈S} a_s
      Temporal(S)  = (# causally-consistent edges in S) / (# edges in S)
      Complexity(S)= |S| / |V_L|

    Expansion stops when:
        ΔQ(v*|S) < max(ρ · Q(S), eps_abs)

    where v* is the best candidate in the 1-hop frontier of S.  The
    relative threshold ρ adapts to the overall signal scale — no manual
    tuning of an absolute eps value is needed.

    Incremental state per expansion step (avoids recomputing Q from scratch):
      max_inf[v]:  max_{s∈S} I(s→v)  — O(N) update via np.maximum
      cur_align:   max_{s∈S} a_s     — O(1) update
      edges_in_S, cons_in_S          — O(|S|) update

    Args:
        seeds:           Initial seed services (must be non-empty).
        out_n / in_n:    Neighbour index (caller→callee / callee←callers).
        service_evidence: e_v — per-service anomaly score (unnormalised).
        sli_align:       a_s — per-service SLI lag-correlation score.
        onset_idx_map:   Raw onset index (lower = earlier anomaly onset).
        max_anom_len:    Length of the anomaly window (fallback onset value).
        nmax:            Maximum number of nodes in the output subgraph.
        rho:             Relative stopping threshold (default 1%).
        lambda_*:        Q(S) component weights.
        hop_decay:       Decay per propagation hop in influence computation.
        eps_abs:         Absolute floor for the stopping threshold.

    Returns:
        List of service names in the selected subgraph (includes all seeds).
    """
    # ── 1. Collect candidate pool V_L via unconstrained BFS from seeds ──
    pool_limit = min(3 * nmax, 150)   # keep N ≤ 150 for O(N³) influence
    cand_set: set = set(seeds)
    bfs_queue: List[str] = list(seeds)
    head = 0
    while head < len(bfs_queue) and len(cand_set) < pool_limit:
        curr = bfs_queue[head]; head += 1
        for nb in (out_n.get(curr, []) + in_n.get(curr, [])):
            if nb not in cand_set:
                cand_set.add(nb)
                bfs_queue.append(nb)

    V_L = list(cand_set)
    n_VL = len(V_L)

    # ── 2. Build influence matrix over V_L ──
    vl_edges = [(u, v) for u in V_L for v in out_n.get(u, []) if v in cand_set]
    influence, vl_idx = _compute_expansion_influence(
        nodes=V_L, edges=vl_edges, max_hops=3, hop_decay=hop_decay,
    )

    # ── 3. Node-aligned arrays ──
    e_arr  = np.array([float(service_evidence.get(s, 0.0)) for s in V_L])
    a_arr  = np.array([float(sli_align.get(s, 0.0)) for s in V_L])
    on_arr = np.array([float(onset_idx_map.get(s, max_anom_len)) for s in V_L])
    denom_e = max(float(np.sum(e_arr)), 1e-12)

    # Forward adjacency within V_L (for temporal consistency)
    adj = np.zeros((n_VL, n_VL), dtype=np.float64)
    for (u, v) in vl_edges:
        ii, jj = vl_idx.get(u, -1), vl_idx.get(v, -1)
        if ii >= 0 and jj >= 0:
            adj[ii, jj] = 1.0

    # ── 4. Initialise S with seeds ──
    S_idx: List[int] = [vl_idx[s] for s in seeds if s in vl_idx]
    S_set: set = set(S_idx)
    if not S_idx:
        return list(seeds)

    max_inf = np.zeros(n_VL, dtype=np.float64)
    for si in S_idx:
        max_inf = np.maximum(max_inf, influence[si, :])

    cur_align = float(np.max(a_arr[np.array(S_idx, dtype=np.intp)]))

    # Edge counts within the initial seed set
    edges_in_S = 0
    cons_in_S  = 0
    for si in S_set:
        for sj in S_set:
            if si != sj and adj[si, sj] > 0.0:   # directed edge si → sj
                edges_in_S += 1
                if on_arr[sj] <= on_arr[si]:       # callee (sj) onset earlier ✓
                    cons_in_S += 1

    def _q_now() -> float:
        cov  = float(np.dot(max_inf, e_arr)) / denom_e
        temp = cons_in_S / max(1.0, float(edges_in_S))
        return (lambda_cov     * cov
                + lambda_align * cur_align
                + lambda_temp  * temp
                - lambda_complex * len(S_set) / max(1, n_VL))

    # ── 5. Greedy expansion over 1-hop frontier ──
    frontier: set = set()
    for si in S_set:
        svc = V_L[si]
        for nb in (out_n.get(svc, []) + in_n.get(svc, [])):
            vi = vl_idx.get(nb, -1)
            if vi >= 0 and vi not in S_set:
                frontier.add(vi)

    while len(S_set) < nmax and frontier:
        best_v  = -1
        best_dq = -np.inf

        for vi in frontier:
            # ΔCoverage — vectorised O(N)
            d_inf = np.maximum(0.0, influence[vi, :] - max_inf)
            dcov  = float(np.dot(d_inf, e_arr)) / denom_e

            # ΔAlign — O(1)
            dalign = max(0.0, float(a_arr[vi]) - cur_align)

            # ΔTemporal — O(|S|)
            new_e = 0; new_c = 0
            for si in S_set:
                if adj[si, vi] > 0.0:           # si → vi (vi is callee)
                    new_e += 1
                    if on_arr[vi] <= on_arr[si]:
                        new_c += 1
                if adj[vi, si] > 0.0:           # vi → si (si is callee)
                    new_e += 1
                    if on_arr[si] <= on_arr[vi]:
                        new_c += 1
            new_temp = (cons_in_S + new_c) / max(1.0, float(edges_in_S + new_e))
            old_temp = cons_in_S / max(1.0, float(edges_in_S))
            dtemp = new_temp - old_temp

            dq = (lambda_cov     * dcov
                  + lambda_align * dalign
                  + lambda_temp  * dtemp
                  - lambda_complex * (1.0 / max(1, n_VL)))
            if dq > best_dq:
                best_dq, best_v = dq, vi

        if best_v < 0:
            break

        stop_thr = max(rho * _q_now(), eps_abs)
        if best_dq < stop_thr:
            break

        # ── Add best_v and update incremental state ──
        S_set.add(best_v)
        max_inf   = np.maximum(max_inf, influence[best_v, :])
        cur_align = max(cur_align, float(a_arr[best_v]))

        for si in (S_set - {best_v}):
            if adj[si, best_v] > 0.0:
                edges_in_S += 1
                if on_arr[best_v] <= on_arr[si]:
                    cons_in_S += 1
            if adj[best_v, si] > 0.0:
                edges_in_S += 1
                if on_arr[si] <= on_arr[best_v]:
                    cons_in_S += 1

        frontier.discard(best_v)
        svc_v = V_L[best_v]
        for nb in (out_n.get(svc_v, []) + in_n.get(svc_v, [])):
            vi2 = vl_idx.get(nb, -1)
            if vi2 >= 0 and vi2 not in S_set:
                frontier.add(vi2)

    return [V_L[i] for i in S_set]


# ============================================================
# Node features (Design B Step 5)
# ============================================================

# Feature vector indices (7 dimensions)
_FEAT_ANOM = 0   # normalized anomaly score
_FEAT_RES  = 1   # resource-vs-latency ratio (high → resource root cause)
_FEAT_ONSET = 2  # onset score: high = early onset (more root-cause-like)
_FEAT_SLI  = 3   # SLI alignment
_FEAT_PROP = 4   # propagation consistency
_FEAT_IN   = 5   # in-degree (# callers), normalized
_FEAT_OUT  = 6   # out-degree (# callees), normalized
_FEAT_DIM  = 7


def _build_node_features(
    services: List[str],
    w_local_norm: Dict[str, float],
    w_resource: Dict[str, float],
    onset_score_norm: Dict[str, float],   # high = early onset
    w_sli_norm: Dict[str, float],
    w_prop_norm: Dict[str, float],
    out_n: Dict[str, List[str]],
    in_n: Dict[str, List[str]],
) -> Dict[str, np.ndarray]:
    """Build 7-dimensional feature vectors for each service."""
    all_svcs = set(services) | set(out_n) | set(in_n)
    max_in  = max(1, max(len(in_n.get(s, []))  for s in all_svcs))
    max_out = max(1, max(len(out_n.get(s, [])) for s in all_svcs))

    feats: Dict[str, np.ndarray] = {}
    for s in services:
        f = np.zeros(_FEAT_DIM, dtype=float)
        f[_FEAT_ANOM]  = float(w_local_norm.get(s, 0.0))
        f[_FEAT_RES]   = float(w_resource.get(s, 0.5))
        f[_FEAT_ONSET] = float(onset_score_norm.get(s, 0.0))
        f[_FEAT_SLI]   = float(w_sli_norm.get(s, 0.0))
        f[_FEAT_PROP]  = float(w_prop_norm.get(s, 0.0))
        f[_FEAT_IN]    = len(in_n.get(s, []))  / max_in
        f[_FEAT_OUT]   = len(out_n.get(s, [])) / max_out
        feats[s] = f
    return feats


# ============================================================
# Inference-only GAT (Design B Step 6)
# ============================================================

def _inference_gat(
    node_feats: Dict[str, np.ndarray],
    edges: List[Tuple[str, str]],
    num_hops: int = 2,
) -> Tuple[Dict[str, np.ndarray], Dict[Tuple[str, str], float]]:
    """Inference-only Graph Attention Network for causal edge refinement.

    No training is needed. Attention weights encode causal plausibility:

    For edge (caller → callee), we assign high attention when the callee:
      - Shows primarily resource anomaly (not just latency) [FEAT_RES]
      - Has early anomaly onset relative to its callers     [FEAT_ONSET]
      - Has high overall anomaly score                      [FEAT_ANOM]
      - Aligns with SLI anomaly                             [FEAT_SLI]
      - Has temporal ordering: callee onset < caller onset  (causal direction)

    Message passing (num_hops rounds):
      h_v' = 0.6·h_v + 0.4 · Σ_{u: u→v} α(u→v)·h_u
    where u is a caller of v. This propagates CALLER anomaly evidence into
    the callee embeddings, helping the utility function detect that callee v
    is implicated in callers' anomalies.

    Returns:
      embeddings:   updated feature vectors per service (after message passing)
      edge_weights: attention coefficient per edge ∈ [0,1] (softmax per source)
    """
    if not node_feats:
        return {}, {}

    services = sorted(node_feats.keys())
    n = len(services)
    idx = {s: i for i, s in enumerate(services)}

    H = np.array([node_feats[s] for s in services], dtype=float)   # (N, D)

    # ---- Compute attention logits per edge (caller → callee) ----
    raw_attn: Dict[Tuple[int, int], float] = {}
    for (src, dst) in edges:
        ii, jj = idx.get(src, -1), idx.get(dst, -1)
        if ii < 0 or jj < 0:
            continue
        f_src = H[ii]   # caller
        f_dst = H[jj]   # callee (potential root cause)

        # Causal plausibility of this edge: is dst a credible root cause?
        res_j    = float(f_dst[_FEAT_RES])    # resource anomaly ratio
        onset_j  = float(f_dst[_FEAT_ONSET])  # high = early onset
        anom_j   = float(f_dst[_FEAT_ANOM])   # anomaly strength
        sli_j    = float(f_dst[_FEAT_SLI])    # SLI alignment

        # Temporal: callee onset earlier than caller → consistent causal direction
        onset_i = float(f_src[_FEAT_ONSET])
        temporal = max(0.0, onset_j - onset_i)   # positive if callee is earlier

        e = (0.30 * res_j
             + 0.25 * onset_j
             + 0.20 * anom_j
             + 0.15 * temporal
             + 0.10 * sli_j)
        raw_attn[(ii, jj)] = float(e)

    # ---- Softmax normalization per source node (outgoing edges) ----
    out_nbrs_idx: Dict[int, List[int]] = defaultdict(list)
    for (ii, jj) in raw_attn:
        out_nbrs_idx[ii].append(jj)

    attn: Dict[Tuple[int, int], float] = {}
    for ii, nbrs in out_nbrs_idx.items():
        logits = np.array([raw_attn[(ii, jj)] for jj in nbrs])
        logits = logits - logits.max()
        exp_l = np.exp(logits)
        total = exp_l.sum() + 1e-12
        for jj, a in zip(nbrs, exp_l / total):
            attn[(ii, jj)] = float(a)

    # ---- Message passing ----
    # For edge caller(ii) → callee(jj): callee jj receives from caller ii.
    # After aggregation, each callee's embedding incorporates its callers'
    # anomaly patterns (weighted by causal attention).
    for _ in range(num_hops):
        in_msg = np.zeros_like(H)
        in_sum = np.zeros(n)
        for (ii, jj), a in attn.items():
            in_msg[jj] += a * H[ii]
            in_sum[jj] += a
        H_new = H.copy()
        for v in range(n):
            if in_sum[v] > 1e-12:
                H_new[v] = 0.6 * H[v] + 0.4 * (in_msg[v] / in_sum[v])
        H = np.clip(H_new, -5.0, 5.0)

    # Map back to service names
    embeddings = {s: H[idx[s]] for s in services}
    edge_weights_out = {
        (src, dst): attn.get((idx.get(src, -1), idx.get(dst, -1)), 0.0)
        for (src, dst) in edges
        if idx.get(src, -1) >= 0 and idx.get(dst, -1) >= 0
    }
    return embeddings, edge_weights_out


# ============================================================
# GAT-weighted soft influence matrix (Design B Step 7)
# ============================================================

def _compute_gat_influence(
    nodes: List[str],
    gat_edge_weights: Dict[Tuple[str, str], float],
    max_hops: int = 3,
    hop_decay: float = 0.85,
) -> np.ndarray:
    """Compute (N×N) soft influence matrix using GAT attention weights.

    Semantics: influence[i, j] ∈ [0,1] represents how strongly service i
    (acting as a root cause) can explain the anomaly observed at service j
    through the call-propagation structure.

    Uses the REVERSE call graph: if A calls B (edge A→B with attention α),
    then B being slow causes A's latency to increase. In the reverse graph,
    B→A has weight α. So influence[B, A] = α (B explains A's latency).

    Multi-hop: influence[B, C] via A = α(A→B) · α(C→A) · hop_decay.
    Approximated via max-product path algorithm (Bellman-Ford style).

    hop_decay < 1 discounts longer propagation paths (signal attenuates).
    """
    n = len(nodes)
    if n == 0:
        return np.zeros((0, 0))

    idx = {s: i for i, s in enumerate(nodes)}

    # Reverse adjacency: callee → caller, weight = attention(caller→callee)
    # Semantics: callee being root cause propagates to caller via this weight
    A_rev = np.zeros((n, n))
    for (caller, callee), alpha in gat_edge_weights.items():
        ii = idx.get(caller, -1)
        jj = idx.get(callee, -1)
        if ii >= 0 and jj >= 0 and alpha > 0:
            # callee (jj) influences caller (ii): A_rev[jj, ii]
            A_rev[jj, ii] = max(A_rev[jj, ii], float(alpha))

    # Max-product path via Bellman-Ford:
    # influence[i,j] = max over all paths P (i→j in reverse graph):
    #                  hop_decay^|P| * Π_{e in P} w_e
    influence = np.eye(n)       # self-influence = 1
    current = A_rev.copy()      # 1-hop paths

    for hop in range(1, max_hops + 1):
        influence = np.maximum(influence, (hop_decay ** hop) * current)
        # Extend by one more hop (max-product semiring)
        next_current = np.zeros((n, n))
        for i in range(n):
            for k in range(n):
                # max over intermediate j: A_rev[i,j] * current[j,k]
                vals = A_rev[i, :] * current[:, k]
                next_current[i, k] = float(np.max(vals)) if vals.size else 0.0
        current = next_current

    return np.clip(influence, 0.0, 1.0)


# ============================================================
# Continuous utility function u(S) (Design B Step 8)
# ============================================================

def _compute_utility(
    S_mask: int,
    n_nodes: int,
    w_anom_arr: np.ndarray,          # (N,) anomaly score per node
    w_sli_arr: np.ndarray,           # (N,) SLI direct correlation per node (fallback)
    influence: np.ndarray,           # (N,N) GAT influence matrix
    edge_exists_mat: np.ndarray,     # (N,N) 1 if edge i→j exists in candidate set
    temporal_consistent: np.ndarray, # (N,N) 1 if edge i→j is causally consistent
    complexity_lambda: float = 0.08,
    lambda_coverage: float = 0.45,
    lambda_align: float = 0.25,
    lambda_temporal: float = 0.15,
    lambda_parsimony: float = 0.15,
    denom_cov: float = 1.0,
    sli_reach_arr: Optional[np.ndarray] = None,  # (N,) propagation reach to SLI service
) -> float:
    """Continuous cooperative game utility for coalition S (encoded as bitmask).

    u(S) = λ1·Coverage(S) + λ2·Align(S) + λ3·Temporal(S) - λ4·Parsimony(S)

    Coverage(S):
      Σ_v w_anom(v) · max_{s∈S} influence[s,v] / Σ_v w_anom(v)
      Fully vectorized via numpy fancy indexing.

    Align(S):
      If sli_reach_arr is provided (recommended):
        max_{s∈S} sli_reach_arr[s]   — max causal reach from S to SLI service
        through the call graph.  Root causes score high because they are
        upstream of SLI; pure downstream nodes score low because anomaly
        must propagate *to* them, not *from* them to SLI.
      Else: max_{s∈S} w_sli_arr[s]  — direct lag-correlation fallback.

    Temporal(S):
      (# edges within S that are causally consistent) / (# edges within S)
      Causal consistency: callee v has earlier onset than caller u.
      Uses precomputed matrices for O(N²) vectorized computation.

    Parsimony(S): λ·|S|/N  (penalizes large coalitions)
    """
    if S_mask == 0:
        return 0.0

    members = np.array([i for i in range(n_nodes) if (S_mask >> i) & 1], dtype=np.intp)
    K = len(members)

    # Coverage (vectorized): max influence from any coalition member to each node
    max_inf = influence[np.ix_(members, np.arange(n_nodes))].max(axis=0)  # (N,)
    coverage = min(1.0, float(np.dot(max_inf, w_anom_arr)) / denom_cov)

    # Alignment: propagation reach to SLI (preferred) or direct correlation (fallback)
    _align_src = sli_reach_arr if sli_reach_arr is not None else w_sli_arr
    align = float(_align_src[members].max())

    # Temporal consistency (vectorized via membership outer product)
    S_vec = np.zeros(n_nodes, dtype=np.float64)
    S_vec[members] = 1.0
    outer = np.outer(S_vec, S_vec)              # (N,N) S×S membership mask
    edges_in_S      = float(np.sum(outer * edge_exists_mat))
    consistent_in_S = float(np.sum(outer * temporal_consistent))
    temporal = consistent_in_S / max(1.0, edges_in_S)

    # Parsimony
    parsimony = complexity_lambda * K / max(1, n_nodes)

    return float(
        lambda_coverage * coverage
        + lambda_align * align
        + lambda_temporal * temporal
        - lambda_parsimony * parsimony
    )


# ============================================================
# Vectorised Monte Carlo Banzhaf estimator
# ============================================================

def _mc_banzhaf_vectorized(
    n_nodes: int,
    influence: np.ndarray,            # (K, K)
    w_anom_arr: np.ndarray,           # (K,)
    w_sli_arr: np.ndarray,            # (K,) direct SLI correlation (fallback)
    edge_exists_mat: np.ndarray,      # (K, K)
    temporal_consistent: np.ndarray,  # (K, K)
    mc_T: int,
    n_blocks: int,
    denom_cov: float,
    rng: np.random.Generator,
    complexity_lambda: float = 0.08,
    lambda_coverage: float = 0.45,
    lambda_align: float = 0.25,
    lambda_temporal: float = 0.15,
    lambda_parsimony: float = 0.15,
    sli_reach_arr: Optional[np.ndarray] = None,  # (K,) propagation reach to SLI
) -> np.ndarray:
    """Fully vectorised block Monte Carlo Banzhaf estimator.

    Computes ``B_i = E[u(S∪{i}) − u(S)]`` for **all** K nodes simultaneously
    using (T, K, K) numpy tensors — no Python loops over T or K.

    Coverage marginal preserves the original *max-influence* formula::

        Δcov_i(S) = Σ_v max(0, inf[i,v] − max_{j∈S\\{i}} inf[j,v]) · w_anom[v] / denom

    The "max with one element excluded" is computed via a prefix/suffix-max
    trick along the j-dimension, giving O(T·K·K) work vs O(T·K²·K_inner)
    for naïve Python loops.

    Returns:
        ``block_pivs`` — shape ``(n_blocks, K)``, raw (un-normalised) block
        Banzhaf estimates.  Post-normalisation and CI computation remain in
        the caller to keep this function pure.
    """
    K = n_nodes
    block_size = max(10, mc_T // n_blocks)
    T = n_blocks * block_size

    masks = (rng.random((T, K)) < 0.5).astype(np.float64)  # (T, K)

    _NEG = -1e9   # sentinel for "not in coalition"

    # ------------------------------------------------------------------
    # [A] Coverage marginal — max-influence formula
    #
    #   inf_mm[t, j, v] = inf[j,v]  if j ∈ S[t]  else  -∞
    #   maxinf_S_i[t, i, v] = max_{j≠i, j∈S[t]} inf[j,v]
    #                        computed via prefix/suffix max along j-axis
    # ------------------------------------------------------------------
    inf_mm = np.where(
        masks[:, :, np.newaxis] > 0.5,
        influence[np.newaxis, :, :],
        _NEG,
    )  # (T, K, K)

    pmax = np.maximum.accumulate(inf_mm, axis=1)                              # (T, K, K)
    smax = np.maximum.accumulate(inf_mm[:, ::-1, :], axis=1)[:, ::-1, :]     # (T, K, K)

    left_v  = np.concatenate([np.full((T, 1, K), _NEG), pmax[:, :-1, :]], axis=1)   # (T,K,K)
    right_v = np.concatenate([smax[:, 1:, :], np.full((T, 1, K), _NEG)], axis=1)    # (T,K,K)

    maxinf_S_i = np.maximum(left_v, right_v)  # (T, K, K)

    delta_inf = np.maximum(0.0, influence[np.newaxis, :, :] - maxinf_S_i)  # (T, K, K)
    delta_cov = np.clip(
        np.einsum('tiv,v->ti', delta_inf, w_anom_arr) / denom_cov,
        0.0, 1.0,
    )  # (T, K)

    # ------------------------------------------------------------------
    # [B] Alignment marginal — prefix/suffix max along j
    #
    # Use sli_reach_arr (causal propagation reach to SLI service) when
    # available; fall back to w_sli_arr (direct lag-correlation) otherwise.
    # The prefix/suffix-max structure is identical in both cases.
    # ------------------------------------------------------------------
    _align_arr = sli_reach_arr if sli_reach_arr is not None else w_sli_arr
    contrib_a = masks * _align_arr[np.newaxis, :]                              # (T, K)
    pmax_a    = np.maximum.accumulate(contrib_a, axis=1)                       # (T, K)
    smax_a    = np.maximum.accumulate(contrib_a[:, ::-1], axis=1)[:, ::-1]    # (T, K)

    left_a  = np.concatenate([np.full((T, 1), _NEG), pmax_a[:, :-1]], axis=1)   # (T, K)
    right_a = np.concatenate([smax_a[:, 1:], np.full((T, 1), _NEG)], axis=1)    # (T, K)

    align_excl  = np.maximum(np.maximum(left_a, right_a), 0.0)               # (T, K)
    delta_align = np.maximum(0.0, _align_arr[np.newaxis, :] - align_excl)    # (T, K)

    # ------------------------------------------------------------------
    # [C] Temporal consistency marginal
    #
    #   edges_S_i[t,i]   = directed edges in S\{i}  (remove node i)
    #   edges_Si_plus[t,i] = directed edges in S\{i}∪{i}  (add node i back)
    #
    #   Correction term: corr[t,i] = masks[t,i] · Σ_j masks[t,j]·D[i,j]
    #   where D = edge_exists_mat + edge_exists_mat.T  (symmetric combined).
    #   D[i,i]=0 (no self-loops) ensures the formula is exact.
    # ------------------------------------------------------------------
    D_edge = edge_exists_mat + edge_exists_mat.T  # (K, K)
    D_cons  = temporal_consistent + temporal_consistent.T

    edges_full = np.einsum('ti,ij,tj->t', masks, edge_exists_mat, masks)  # (T,)
    cons_full  = np.einsum('ti,ij,tj->t', masks, temporal_consistent,  masks)  # (T,)

    MD_edge = masks @ D_edge   # (T, K): Σ_j masks[t,j]*D[i,j] per node i
    MD_cons  = masks @ D_cons

    corr_edge = masks * MD_edge   # (T, K)
    corr_cons = masks * MD_cons

    edges_S_i    = np.clip(edges_full[:, np.newaxis] - corr_edge, 0.0, None)   # (T, K)
    cons_S_i     = np.clip(cons_full[:, np.newaxis]  - corr_cons,  0.0, None)
    edges_Si_plus = edges_S_i + MD_edge   # (T, K)
    cons_Si_plus  = cons_S_i  + MD_cons

    temporal_S_i     = cons_S_i     / np.maximum(edges_S_i,     1.0)
    temporal_Si_plus = cons_Si_plus / np.maximum(edges_Si_plus,  1.0)
    delta_temporal   = temporal_Si_plus - temporal_S_i  # (T, K)

    # ------------------------------------------------------------------
    # [D] Parsimony (constant: always adds exactly one node)
    # ------------------------------------------------------------------
    delta_parsimony = complexity_lambda / max(1, K)

    # ------------------------------------------------------------------
    # [E] Aggregate marginal utility + block estimates
    # ------------------------------------------------------------------
    delta_u = (
        lambda_coverage  * delta_cov
        + lambda_align   * delta_align
        + lambda_temporal * delta_temporal
        - lambda_parsimony * delta_parsimony
    )  # (T, K)

    block_pivs = delta_u.reshape(n_blocks, block_size, K).mean(axis=1)  # (n_blocks, K)
    return block_pivs


# ============================================================
# Downstream-only node pruning
# ============================================================

def _prune_downstream_nodes(
    services: List[str],
    local_in_n: Dict[str, List[str]],
    w_resource: Dict[str, float],
    onset_score_norm: Dict[str, float],
    w_local_norm: Dict[str, float],
    w_combined: Dict[str, float],
    seed_services: List[str],
    resource_thresh: float = 0.25,
    onset_thresh: float = 0.20,
    anom_thresh: float = 0.10,
    min_keep: int = 5,
) -> Tuple[List[str], List[str]]:
    """Remove clearly downstream-only nodes from the Banzhaf player set.

    A node v is pruned only when **all four** of the following hold:

      (A) ``w_resource[v] < resource_thresh``
          Anomaly is effect-type (latency / error), not cause-type
          (CPU / MEM / disk).  High resource ratio → plausible root cause.

      (B) ``onset_score_norm[v] < onset_thresh``
          Late anomaly onset.  Root causes tend to anomalise first.

      (C) ``w_local_norm[v] < anom_thresh``
          Weak overall anomaly magnitude — not a strong signal either way.

      (D) ``v`` has at least one caller in the local subgraph.
          Source nodes (no callers) are never pruned — they cannot be
          explained by an upstream fault and must remain candidates.

    Seed services (top-m by anomaly score) are unconditionally kept,
    since they are the primary evidence regardless of structural features.

    Safety: if pruning leaves fewer than ``min_keep`` services, the
    highest-``w_combined`` pruned nodes are restored until ``min_keep`` is
    reached.

    Returns:
        ``(kept_services, pruned_services)``
    """
    seed_set = set(seed_services)
    kept: List[str] = []
    pruned: List[str] = []

    for v in services:
        if v in seed_set:
            kept.append(v)
            continue

        is_effect_type = float(w_resource.get(v, 0.5)) < resource_thresh
        is_late_onset  = float(onset_score_norm.get(v, 0.0)) < onset_thresh
        is_weak_anom   = float(w_local_norm.get(v, 0.0)) < anom_thresh
        has_callers    = bool(local_in_n.get(v, []))

        if is_effect_type and is_late_onset and is_weak_anom and has_callers:
            pruned.append(v)
        else:
            kept.append(v)

    # Safety: restore best pruned nodes if too few survive
    while len(kept) < min_keep and pruned:
        best = max(pruned, key=lambda x: float(w_combined.get(x, 0.0)))
        pruned.remove(best)
        kept.append(best)

    return kept, pruned


# ============================================================
# Per-case automatic hyperparameter calibration
# ============================================================

def _calibrate_onset_threshold(
    service_series: Dict[str, np.ndarray],
    anom_start: int,
    target_pct: float = 75.0,
) -> float:
    """Auto-calibrate ``onset_thr`` from this case's anomaly-window z-scores.

    We want the threshold to sit at the ``target_pct``-th percentile of all
    anomaly-window z-scores across services.  This means roughly the top
    ``(100 - target_pct)``% of readings are treated as "anomalous" — adaptive
    to the overall signal magnitude in each case.

    Falls back to 2.0 if no data is available.
    """
    all_vals: List[float] = []
    for x in service_series.values():
        xa = _to_1d_float(x)[anom_start:]
        all_vals.extend(xa[xa > 0].tolist())
    if not all_vals:
        return 2.0
    return float(max(0.5, np.percentile(all_vals, target_pct)))


def _calibrate_corr_threshold(
    service_series: Dict[str, np.ndarray],
    sample_pairs: int = 200,
    target_pct: float = 80.0,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Auto-calibrate ``corr_thr`` so only genuine correlations pass.

    Computes lead-lag correlations on a random subsample of service pairs and
    returns the ``target_pct``-th percentile.  Only correlations above this
    value are used to build the lag-corr fallback graph — i.e., we keep only
    the top ``(100 - target_pct)``% most correlated pairs.

    Falls back to 0.35 for small graphs (< 2 services).
    """
    svcs = sorted(service_series.keys())
    if len(svcs) < 2:
        return 0.35
    if rng is None:
        rng = np.random.default_rng(7)

    corrs: List[float] = []
    all_pairs = [(i, j) for i in range(len(svcs)) for j in range(len(svcs)) if i != j]
    chosen = (
        all_pairs if len(all_pairs) <= sample_pairs
        else [all_pairs[k] for k in rng.choice(len(all_pairs), sample_pairs, replace=False)]
    )
    for ii, jj in chosen:
        xi = _to_1d_float(service_series[svcs[ii]])
        xj = _to_1d_float(service_series[svcs[jj]])
        n = min(xi.size, xj.size)
        if n < 3:
            continue
        c = _safe_corr(xi[: n - 1], xj[1:n])
        corrs.append(c)

    if not corrs:
        return 0.35
    return float(np.clip(np.percentile(corrs, target_pct), 0.20, 0.80))


def _calibrate_prune_thresholds(
    local_services: List[str],
    w_resource: Dict[str, float],
    onset_score_norm: Dict[str, float],
    w_local_norm: Dict[str, float],
) -> Tuple[float, float, float]:
    """Auto-calibrate prune thresholds from this case's score distributions.

    Returns ``(resource_thresh, onset_thresh, anom_thresh)``.

    Strategy: set each threshold at the 25th percentile of the corresponding
    score distribution across ``local_services``.  This means only the bottom
    quarter on *all three axes simultaneously* will be pruned — a conservative
    policy that avoids pruning ambiguous nodes.
    """
    def _p25(vals: List[float]) -> float:
        return float(np.percentile(vals, 25)) if vals else 0.25

    res_vals   = [float(w_resource.get(s, 0.5))       for s in local_services]
    onset_vals = [float(onset_score_norm.get(s, 0.0)) for s in local_services]
    anom_vals  = [float(w_local_norm.get(s, 0.0))     for s in local_services]

    return _p25(res_vals), _p25(onset_vals), _p25(anom_vals)


def _calibrate_cusum_h(
    service_series: Dict[str, np.ndarray],
    anom_start: int,
    sigma_factor: float = 4.0,
) -> float:
    """Estimate a reasonable CUSUM alarm threshold for *this case's* data.

    Note: ``_cusum_change_point`` already normalises by the warmup std, so
    ``h`` is dimensionless (σ-units) and 5.0 works across most datasets.
    This function is provided for completeness and diagnostic output only;
    the value is stored in the ``calibrated`` return-dict field but is **not**
    used to alter the split that already happened in [1].

    Computes: ``sigma_factor × median(anomaly-window / normal-window std ratio)``
    to give a sense of how large the actual anomaly signal is vs. the baseline
    noise — helpful when deciding whether to tighten or relax ``cusum_h`` for
    a future run via the ``cusum_h`` kwarg.
    """
    ratios: List[float] = []
    for x in service_series.values():
        xn = _to_1d_float(x)[:anom_start]
        xa = _to_1d_float(x)[anom_start:]
        sn = float(np.std(xn)) if xn.size >= 2 else 0.0
        sa = float(np.std(xa)) if xa.size >= 2 else 0.0
        if sn > 1e-9:
            ratios.append(sa / sn)
    if not ratios:
        return 5.0
    # Suggest h ≈ sigma_factor × sqrt(median_ratio) so a stronger signal
    # doesn't require a lower threshold.
    return float(np.clip(sigma_factor * float(np.sqrt(np.median(ratios))), 2.0, 20.0))


def _calibrate_expand_params(
    service_series: Dict[str, np.ndarray],
    anom_start: int,
    sli_col: Optional[str],
    col_idx: Dict[str, int],
    z_full: np.ndarray,
    onset_idx_map: Dict[str, int],
    max_anom_len: int,
    seed_services: List[str],
    sli_align: Dict[str, float],
    service_evidence: Dict[str, float],
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float, float, float]:
    """Auto-calibrate Q(S) expansion parameters from this case's data.

    **λ weights** are proportional to the empirical *signal-to-noise ratio*
    of each Q component:

      lam_cov  ∝ mean(|anom z-score|) / max(std(normal z-score), 1)
                 High → anomaly scores are loud → Coverage matters more.

      lam_aln  ∝ std(SLI in anomaly window) / std(SLI in normal window)
                 High → SLI is clearly disrupted → Align matters more.
                 Falls back to a safe 0.3 when SLI is unavailable.

      lam_tmp  ∝ std of service onset-time fractions across services
                 High → services anomalise at very different times →
                 Temporal ordering is a meaningful discriminator.

      lam_cmp  = fixed at 0.10 (complexity penalty is scale-independent).

    Each raw SNR is clipped to [0.1, 3.0] before normalisation, and a
    per-component floor of 0.15 is applied so no term is completely silenced.
    The three variable weights are renormalized to sum to 0.90.

    **ρ (expand_rho)** is kept fixed at 0.01 (1% relative threshold).
    Estimating ρ from the per-case noise floor is unreliable: the distribution
    of ΔQ for random non-seed additions overlaps too heavily with genuine
    root-cause additions, causing ρ to be overestimated and the expansion to
    stop before key services are included.  The 1% relative threshold is
    well-motivated theoretically (stops when marginal gain < 1% of current Q)
    and empirically robust across RE2/RE3 datasets.

    Returns:
        (rho, lam_cov, lam_aln, lam_tmp, lam_cmp)
    """
    if rng is None:
        rng = np.random.default_rng(7)

    svcs = list(service_series.keys())

    # ── 1. lam_cov: anomaly SNR ──────────────────────────────────────────────
    normal_z_flat: List[float] = []
    anom_z_flat: List[float] = []
    for x in service_series.values():
        xn = _to_1d_float(x)[:anom_start]
        xa = _to_1d_float(x)[anom_start:]
        normal_z_flat.extend(xn.tolist())
        anom_z_flat.extend(xa.tolist())

    cov_snr = 1.0
    if normal_z_flat and anom_z_flat:
        mu_anom = float(np.mean(np.abs(anom_z_flat)))
        sd_norm = max(float(np.std(normal_z_flat)), 1e-9)
        cov_snr = max(mu_anom / sd_norm, 0.1)

    # ── 2. lam_aln: SLI disruption ratio ─────────────────────────────────────
    aln_snr = 0.3   # safe default when SLI unavailable
    if sli_col is not None:
        sli_j = col_idx.get(str(sli_col), -1)
        if sli_j >= 0:
            sli_n = z_full[:anom_start, sli_j]
            sli_a = z_full[anom_start:, sli_j]
            sd_n = max(float(np.std(sli_n)), 1e-9)
            aln_snr = max(float(np.std(sli_a)) / sd_n, 0.1)

    # ── 3. lam_tmp: onset-time spread ────────────────────────────────────────
    onset_vals = [
        float(onset_idx_map.get(s, max_anom_len)) / max(1, max_anom_len)
        for s in svcs
        if onset_idx_map.get(s, max_anom_len) < max_anom_len
    ]
    # tmp_snr: use raw std without amplification.
    # If few services have clear onsets, std is naturally small → temporal matters less.
    tmp_snr = max(float(np.std(onset_vals)), 0.1) if len(onset_vals) >= 2 else 0.3

    # ── Normalize λ₁, λ₂, λ₃ to sum = 0.90; fix λ₄ = 0.10 ─────────────────
    # Clip each SNR to [0.1, 3.0] before normalizing so no single component
    # can dominate (prevents lam_tmp from reaching 0.45+ on noisy onset signals).
    # Floor at 0.15 per component after normalization ensures each term keeps
    # a minimal voice — useful when one signal is genuinely absent.
    raw = np.clip([cov_snr, aln_snr, tmp_snr], 0.1, 3.0)
    raw = raw / raw.sum()
    # Soft floor: each weight ≥ 0.15 (avoids completely ignoring any component)
    raw = np.maximum(raw, 0.15)
    raw = raw / raw.sum()          # renormalize after floor
    lam_cov = float(raw[0] * 0.90)
    lam_aln = float(raw[1] * 0.90)
    lam_tmp = float(raw[2] * 0.90)
    lam_cmp = 0.10

    # ── 4. ρ: fixed at 0.01 ──────────────────────────────────────────────────
    # Noise-floor estimation from random non-seed ΔQ is unreliable: in many
    # cases the non-seed pool overlaps heavily with genuine expansion targets,
    # causing ρ to be estimated too high and the expansion to stop prematurely
    # (see: CPU fault case where ts-travel-service is excluded when ρ≈0.065).
    # A fixed ρ=0.01 (1% relative threshold) is theoretically well-motivated
    # and empirically robust.  Only λ weights are data-driven.
    rho = 0.01

    return rho, lam_cov, lam_aln, lam_tmp, lam_cmp


# ============================================================
# Main method
# ============================================================

def local_banzhaf(
    data: pd.DataFrame,
    inject_time: int | None = None,
    dataset: str | None = None,
    num_loop=None,
    sli: str | None = None,
    anomalies=None,
    **kwargs,
) -> dict:
    """RCAEval-compatible entrypoint: Design B (Inference-only GAT + continuous Banzhaf).

    Window split priority (from highest to lowest):
      1. ``anomalies`` (row-index based) — passed explicitly.
      2. ``inject_time`` (time-column based) — passed explicitly.
      3. Auto change-point detection on SLI series — used when neither above is given.

    SLI selection (``sli_mode`` kwarg):
      ``"provided"``   — use only the ``sli`` argument; no auto-detection.
      ``"auto"``       — always auto-detect; ignore ``sli`` argument.
      ``"auto_first"`` — (default) use ``sli`` if given, else auto-detect.
    """
    from sklearn.preprocessing import RobustScaler

    # ---- Hyper-parameters (kwargs > env > defaults) ----
    seed_m   = int(kwargs.get("seed_m",   os.getenv("LOCAL_BANZHAF_SEED_M", 7)))
    nmax     = int(kwargs.get("nmax",     os.getenv("LOCAL_BANZHAF_NMAX",   30)))
    eps      = float(kwargs.get("eps",    os.getenv("LOCAL_BANZHAF_EPS",    1e-3)))

    topk     = int(kwargs.get("topk",     os.getenv("LOCAL_BANZHAF_TOPK",   15)))
    mc_T     = int(kwargs.get("mc_samples", os.getenv("LOCAL_BANZHAF_T",    2000)))

    # Node weight blend coefficients (sum ≈ 1)
    alpha    = float(kwargs.get("alpha",     os.getenv("LOCAL_BANZHAF_ALPHA",     0.30)))
    beta     = float(kwargs.get("beta",      os.getenv("LOCAL_BANZHAF_BETA",      0.15)))
    gamma    = float(kwargs.get("gamma",     os.getenv("LOCAL_BANZHAF_GAMMA",     0.10)))
    delta    = float(kwargs.get("delta",     os.getenv("LOCAL_BANZHAF_DELTA",     0.25)))
    eps_r    = float(kwargs.get("epsilon_r", os.getenv("LOCAL_BANZHAF_EPSILON_R", 0.20)))

    eta      = float(kwargs.get("eta",       os.getenv("LOCAL_BANZHAF_ETA",       0.65)))

    # "auto" → calibrate from data; any numeric string → use as fixed value
    corr_thr_cfg  = kwargs.get("corr_threshold",    os.getenv("LOCAL_BANZHAF_CORR_T",    "auto"))
    onset_thr_cfg = kwargs.get("onset_threshold",   os.getenv("LOCAL_BANZHAF_ONSET_T",   "auto"))
    cusum_h_cfg   = kwargs.get("cusum_h",           os.getenv("LOCAL_BANZHAF_CUSUM_H",   "auto"))
    # Prune thresholds — "auto" or fixed numeric
    prune_res_cfg   = kwargs.get("prune_resource_thresh", os.getenv("LOCAL_BANZHAF_PRUNE_RES",   "auto"))
    prune_onset_cfg = kwargs.get("prune_onset_thresh",    os.getenv("LOCAL_BANZHAF_PRUNE_ONSET", "auto"))
    prune_anom_cfg  = kwargs.get("prune_anom_thresh",     os.getenv("LOCAL_BANZHAF_PRUNE_ANOM",  "auto"))
    prune_min_keep  = int(kwargs.get("prune_min_keep",    os.getenv("LOCAL_BANZHAF_PRUNE_MIN",   5)))
    prune_enable    = str(kwargs.get("prune_enable",      os.getenv("LOCAL_BANZHAF_PRUNE",       "true"))).lower() != "false"

    # Q(S)-guided greedy expansion parameters
    # Default "auto" → calibrated from data in [3c]; pass a numeric string to fix.
    expand_rho_cfg     = kwargs.get("expand_rho",     os.getenv("LOCAL_BANZHAF_EXPAND_RHO",  "auto"))
    expand_lam_cov_cfg = kwargs.get("expand_lam_cov", os.getenv("LOCAL_BANZHAF_EXPAND_COV",  "auto"))
    expand_lam_aln_cfg = kwargs.get("expand_lam_aln", os.getenv("LOCAL_BANZHAF_EXPAND_ALN",  "auto"))
    expand_lam_tmp_cfg = kwargs.get("expand_lam_tmp", os.getenv("LOCAL_BANZHAF_EXPAND_TMP",  "auto"))
    expand_lam_cmp_cfg = kwargs.get("expand_lam_cmp", os.getenv("LOCAL_BANZHAF_EXPAND_CMP",  "auto"))

    max_lag  = int(kwargs.get("max_lag",   os.getenv("LOCAL_BANZHAF_MAX_LAG",     2)))
    gat_hops  = int(kwargs.get("gat_hops", os.getenv("LOCAL_BANZHAF_GAT_HOPS",   2)))
    inf_hops  = int(kwargs.get("inf_hops", os.getenv("LOCAL_BANZHAF_INF_HOPS",   3)))
    hop_decay = float(kwargs.get("hop_decay", os.getenv("LOCAL_BANZHAF_HOP_DECAY", 0.85)))
    cx_lambda = float(kwargs.get("complexity_lambda", os.getenv("LOCAL_BANZHAF_COMPLEXITY", 0.08)))

    # ---- SLI selection mode ----
    # "provided"   : (default) use only the `sli` argument as-is (no auto-detection).
    #                This matches RCAEval's runner (`main.py`) which already picks
    #                a dataset-specific SLI column and passes it in.
    # "auto"       : always auto-detect; ignore the `sli` argument entirely.
    # "auto_first" : use `sli` if given and exists in data; else auto-detect.
    sli_mode = str(kwargs.get("sli_mode", os.getenv("LOCAL_BANZHAF_SLI_MODE", "provided")))

    # ---- Change-point detection parameters ----
    cp_method     = str(kwargs.get("cp_method",        os.getenv("LOCAL_BANZHAF_CP_METHOD",     "cusum")))
    cp_min_normal = float(kwargs.get("cp_min_normal_frac", os.getenv("LOCAL_BANZHAF_CP_MIN",    0.30)))
    cp_max_normal = float(kwargs.get("cp_max_normal_frac", os.getenv("LOCAL_BANZHAF_CP_MAX",    0.80)))
    cusum_h       = float(kwargs.get("cusum_h",        os.getenv("LOCAL_BANZHAF_CUSUM_H",       5.0)))
    cusum_k       = float(kwargs.get("cusum_k",        os.getenv("LOCAL_BANZHAF_CUSUM_K",       0.5)))
    cp_zscore_thresh     = float(kwargs.get("cp_zscore_thresh",     os.getenv("LOCAL_BANZHAF_CP_ZSCORE_T",  3.0)))
    cp_zscore_min_streak = int(kwargs.get("cp_zscore_min_streak",   os.getenv("LOCAL_BANZHAF_CP_STREAK",    3)))

    # GAT is optional (default off) — skipping it saves ~40% of wall time.
    # When off, graph structure is still used via uniform-attention influence.
    use_gat = str(kwargs.get("use_gat", os.getenv("LOCAL_BANZHAF_USE_GAT", "false"))).lower() != "false"

    rng = np.random.default_rng(int(os.getenv("LOCAL_BANZHAF_SEED", 7)))

    # ============================================================
    # [0] Resolve effective SLI column
    #     Done before window split so change-point detection can use it.
    # ============================================================
    if sli_mode == "auto":
        effective_sli: Optional[str] = _auto_select_sli(data)
    elif sli_mode == "auto_first":
        # Use provided SLI if it exists; otherwise fall back to auto selection.
        if sli is not None and str(sli) in data.columns:
            effective_sli = str(sli)
        else:
            effective_sli = _auto_select_sli(data, sli_col_hint=sli)
    else:  # "provided"
        effective_sli = sli

    # ============================================================
    # [1] Window split
    #     Priority: anomalies > inject_time > change-point detection
    # ============================================================
    split_source = "unknown"
    try:
        if anomalies is not None:
            # Explicit row-index split (takes highest priority)
            split_row = int(anomalies[0])
            normal_raw = data.head(split_row)
            anomal_raw = data.tail(len(data) - split_row)
            split_source = "anomalies"

        elif inject_time is not None and "time" in data.columns:
            # Explicit time-based split (backward compatible)
            normal_raw = data[data["time"] < inject_time]
            anomal_raw = data[data["time"] >= inject_time]
            split_source = "inject_time"

        else:
            # Auto change-point detection on the SLI (or composite) series
            if cp_method not in ("cusum", "zscore"):
                raise ValueError(f"Unknown cp_method: {cp_method!r}")

            split_row, split_source = _detect_split_index(
                data=data,
                sli_col=effective_sli,
                method=cp_method,
                min_normal_frac=cp_min_normal,
                max_normal_frac=cp_max_normal,
                cusum_h=cusum_h,
                cusum_k=cusum_k,
                cusum_warmup=cp_min_normal,
                zscore_thresh=cp_zscore_thresh,
                zscore_warmup=cp_min_normal,
                zscore_min_streak=cp_zscore_min_streak,
            )
            normal_raw = data.head(split_row)
            anomal_raw = data.tail(len(data) - split_row)

        if len(normal_raw) == 0 or len(anomal_raw) == 0:
            raise ValueError("Empty window after split")

    except Exception:
        return _fallback_baro_like(data=data, inject_time=inject_time, dataset=dataset,
                                   anomalies=anomalies, **kwargs)

    # Propagate auto-detected SLI back for downstream weight computation.
    # This replaces the original sli variable so section [6] works unchanged.
    sli = effective_sli

    # ============================================================
    # [2] Preprocess + metric anomaly scores (RobustScaler)
    # ============================================================
    try:
        normal_df = preprocess(data=normal_raw, dataset=dataset,
                               dk_select_useful=kwargs.get("dk_select_useful", False))
        anomal_df = preprocess(data=anomal_raw, dataset=dataset,
                               dk_select_useful=kwargs.get("dk_select_useful", False))

        intersects = [c for c in normal_df.columns if c in anomal_df.columns]
        normal_df = normal_df[intersects]
        anomal_df = anomal_df[intersects]

        if normal_df.shape[1] == 0:
            return _fallback_baro_like(data=data, inject_time=inject_time, dataset=dataset,
                                       anomalies=anomalies, **kwargs)

        min_len = min(len(normal_df), len(anomal_df))
        if min_len >= 3:
            normal_df = normal_df.tail(min_len)
            anomal_df = anomal_df.head(min_len)

        full_df = pd.concat([normal_df, anomal_df], ignore_index=True)
    except Exception:
        return _fallback_baro_like(data=data, inject_time=inject_time, dataset=dataset,
                                   anomalies=anomalies, **kwargs)

    # ---- Vectorised metric scoring (all columns, including host metrics) ----
    # Compute z-scores on the full (unfiltered) column set so that host-level
    # metrics (disk I/O, socket counters) can be attributed to microservices
    # before being dropped from the main pipeline.  We restrict the working
    # column set to non-host metrics afterwards.
    _all_cols  = list(normal_df.columns)
    _by_svc_all = _group_cols_by_service(_all_cols)

    _nn_all = np.nan_to_num(normal_df.to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    _an_all = np.nan_to_num(anomal_df.to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    _fn_all = np.nan_to_num(full_df.to_numpy(dtype=np.float64),   nan=0.0, posinf=0.0, neginf=0.0)

    _med_all = np.median(_nn_all, axis=0)
    _iqr_all = np.percentile(_nn_all, 75, axis=0) - np.percentile(_nn_all, 25, axis=0)
    _sc_all  = np.where(_iqr_all > 0, _iqr_all, 1.0)

    _z_anomal_all = np.clip(np.abs((_an_all - _med_all) / _sc_all), 0.0, 100.0)
    _z_full_all   = np.nan_to_num(
        np.clip(np.abs((_fn_all - _med_all) / _sc_all), 0.0, 100.0), nan=0.0
    )
    _col_idx_all: Dict[str, int] = {col: j for j, col in enumerate(_all_cols)}

    # Split: host-like services vs microservices
    _host_svcs  = [svc for svc in _by_svc_all if _is_host_like_service(svc)]
    _micro_svcs = [svc for svc in _by_svc_all if not _is_host_like_service(svc)]

    # Restrict working matrices to microservice columns only
    metric_cols  = [c for c in _all_cols if not _is_host_like_service(_parse_service_metric(c)[0])]
    _kept_pos    = np.array([_col_idx_all[c] for c in metric_cols], dtype=np.intp)
    _z_anomal    = _z_anomal_all[:, _kept_pos]   # (N_a, C_micro)
    _z_full      = _z_full_all[:, _kept_pos]     # (N_f, C_micro)
    _col_idx: Dict[str, int] = {col: j for j, col in enumerate(metric_cols)}

    by_service   = {svc: cols for svc, cols in _by_svc_all.items() if not _is_host_like_service(svc)}

    metric_score: Dict[str, float] = {
        col: float(_z_anomal[:, j].max()) for j, col in enumerate(metric_cols)
    }

    # ============================================================
    # [3] Service-level aggregation
    # ============================================================
    service_anom_score: Dict[str, float] = {}
    service_rep_metric: Dict[str, str]   = {}
    for svc, cols in by_service.items():
        best_c, best_s = None, -1.0
        for c in cols:
            s = float(metric_score.get(c, 0.0))
            if s > best_s:
                best_s, best_c = s, c
        service_anom_score[svc] = max(0.0, best_s)
        if best_c is not None:
            service_rep_metric[svc] = best_c

    # Build service-level anomaly time series (max z-score envelope over full window)
    anom_start = len(normal_df)
    service_series: Dict[str, np.ndarray] = {}
    for svc, cols in by_service.items():
        idxs = [_col_idx[c] for c in cols if c in _col_idx]
        if idxs:
            service_series[svc] = _z_full[:, idxs].max(axis=1)

    # ---- [3a] Host metric attribution ----
    # Build host z-score series and attribute anomalies to microservices.
    # A host service's peak anomaly is credited to the microservice whose
    # anomaly-window series shows the highest co-anomaly (lead-lag correlation).
    if _host_svcs:
        _host_series: Dict[str, np.ndarray] = {}
        for _h_svc in _host_svcs:
            _h_cols = _by_svc_all[_h_svc]
            _h_idxs = [_col_idx_all[c] for c in _h_cols if c in _col_idx_all]
            if _h_idxs:
                _host_series[_h_svc] = _z_full_all[:, _h_idxs].max(axis=1)
        _host_bonus = _attribute_host_metrics(
            host_series=_host_series,
            micro_series=service_series,
            anom_start=anom_start,
        )
        for _svc, _bonus in _host_bonus.items():
            if _svc in service_anom_score:
                service_anom_score[_svc] = max(service_anom_score[_svc], _bonus)

    # ============================================================
    # [3b] Per-case automatic hyperparameter calibration
    #      Runs after service_series is available so we have the
    #      actual anomaly-window distribution for this case.
    #      Any parameter set to "auto" (or left at default) is
    #      derived from data; explicit numeric values are kept as-is.
    # ============================================================
    def _resolve(cfg, calibrated_val: float) -> float:
        """Return calibrated_val when cfg is "auto", else parse cfg as float."""
        if str(cfg).lower() == "auto":
            return calibrated_val
        try:
            return float(cfg)
        except (TypeError, ValueError):
            return calibrated_val

    # onset_thr: 75th-pct of anomaly-window z-scores (adapt to signal magnitude)
    onset_thr = _resolve(
        onset_thr_cfg,
        _calibrate_onset_threshold(service_series, anom_start, target_pct=75.0),
    )

    # corr_thr: 80th-pct of pairwise lag-correlations (only keep top 20%)
    corr_thr = _resolve(
        corr_thr_cfg,
        _calibrate_corr_threshold(service_series, rng=rng),
    )

    # cusum_h: diagnostic only — the split in [1] already used the pre-calibration
    # value (default 5.0 in σ-units, which is scale-independent since CUSUM
    # normalises internally).  We compute the case-specific suggestion here so
    # it appears in the "calibrated" return field for the user's reference.
    _cusum_h_suggested = _calibrate_cusum_h(service_series, anom_start)
    # cusum_h stays as the pre-calibration value used in [1]; don't overwrite it.

    # ============================================================
    # [3c] Pre-expansion signals + Q(S) parameter calibration
    #
    #  Compute onset_idx and SLI alignment for ALL services so that
    #  _greedy_subgraph_expand can use them in Step [5] without circular
    #  dependency on local_services (which doesn't exist yet).
    #
    #  Also auto-calibrate expand_rho and lambda weights from data:
    #   lam_cov  ∝ anomaly SNR  (loud anomalies → Coverage matters)
    #   lam_aln  ∝ SLI disruption ratio (clear SLI → Align matters)
    #   lam_tmp  ∝ spread of service onset times (varied timing → Temporal matters)
    #   rho      ← 75th-pct of |ΔQ| from random non-seed additions (noise floor)
    # ============================================================
    _max_anom_len_pre = max(1, len(anomal_df))

    _onset_idx_pre: Dict[str, int] = {}
    for _s, _x in service_series.items():
        _xa = _to_1d_float(_x)[anom_start:]
        _idxs = np.where(_xa >= onset_thr)[0]
        _onset_idx_pre[_s] = int(_idxs[0]) if _idxs.size else _max_anom_len_pre

    _sli_align_pre: Dict[str, float] = {}
    if sli is not None:
        _sli_j_pre = _col_idx.get(str(sli), -1)
        if _sli_j_pre >= 0:
            _sli_anom_pre = _z_full[anom_start:, _sli_j_pre]
            for _s in service_series:
                _xs = _to_1d_float(service_series[_s])[anom_start:]
                _sli_align_pre[_s] = float(_lead_lag_corr(_xs, _sli_anom_pre, max_lag=max_lag))

    # Seed services (top-m by anomaly) needed for noise-floor ρ estimate.
    # The full [5] expansion runs later; here we only need the seed list.
    _seed_pre = [
        s for s, _ in sorted(service_anom_score.items(), key=lambda x: x[1], reverse=True)[:seed_m]
        if s
    ]

    # Calibrate from data (any "auto" cfg → use calibrated value; numeric → keep)
    _auto_rho, _auto_cov, _auto_aln, _auto_tmp, _auto_cmp = _calibrate_expand_params(
        service_series=service_series,
        anom_start=anom_start,
        sli_col=sli,
        col_idx=_col_idx,
        z_full=_z_full,
        onset_idx_map=_onset_idx_pre,
        max_anom_len=_max_anom_len_pre,
        seed_services=_seed_pre,
        sli_align=_sli_align_pre,
        service_evidence=service_anom_score,
        rng=rng,
    )
    # Note: _auto_rho is always 0.01 (fixed); expand_rho_cfg override is still
    # respected so users can pass "expand_rho=0.02" to tighten the stopping rule.

    expand_rho     = _resolve(expand_rho_cfg,     _auto_rho)
    expand_lam_cov = _resolve(expand_lam_cov_cfg, _auto_cov)
    expand_lam_aln = _resolve(expand_lam_aln_cfg, _auto_aln)
    expand_lam_tmp = _resolve(expand_lam_tmp_cfg, _auto_tmp)
    expand_lam_cmp = _resolve(expand_lam_cmp_cfg, _auto_cmp)

    # ============================================================
    # [4] Build directed call graph
    # ============================================================
    args_obj = kwargs.get("args", None)
    case_dir = None
    if args_obj is not None and getattr(args_obj, "data_path", None) is not None:
        try:
            case_dir = os.path.dirname(str(args_obj.data_path))
        except Exception:
            pass

    traces_edges = _read_traces_edges(case_dir) if case_dir else None
    if traces_edges is not None:
        all_edges = traces_edges
    else:
        all_edges = _infer_edges_from_lag_corr(
            service_series=service_series,
            corr_threshold=corr_thr,
            max_in=int(kwargs.get("max_in", os.getenv("LOCAL_BANZHAF_MAX_IN", 3))),
            max_out=int(kwargs.get("max_out", os.getenv("LOCAL_BANZHAF_MAX_OUT", 3))),
        )

    out_n, in_n = _build_neighbor_index(all_edges)

    # ============================================================
    # [5] Local subgraph expansion (seed → local_services)
    # ============================================================
    seed_services = [
        s for s, _ in sorted(service_anom_score.items(), key=lambda x: x[1], reverse=True)[:seed_m]
        if s
    ]
    local_services = _greedy_subgraph_expand(
        seeds=seed_services,
        out_n=out_n,
        in_n=in_n,
        service_evidence=service_anom_score,
        sli_align=_sli_align_pre,
        onset_idx_map=_onset_idx_pre,
        max_anom_len=_max_anom_len_pre,
        nmax=nmax,
        rho=expand_rho,
        lambda_cov=expand_lam_cov,
        lambda_align=expand_lam_aln,
        lambda_temp=expand_lam_tmp,
        lambda_complex=expand_lam_cmp,
        hop_decay=hop_decay,
    )
    if not local_services:
        return _fallback_baro_like(data=data, inject_time=inject_time, dataset=dataset,
                                   anomalies=anomalies, **kwargs)

    local_set = set(local_services)
    local_edges = [(u, v) for (u, v) in all_edges if u in local_set and v in local_set]
    local_out_n, local_in_n = _build_neighbor_index(local_edges)

    # ============================================================
    # [6] Service weight signals
    # ============================================================

    # w_local: normalized anomaly score
    w_local_raw = {s: float(service_anom_score.get(s, 0.0)) for s in local_services}
    w_local_norm = _min_max_norm(w_local_raw)

    # w_sli: lead-lag correlation with SLI metric
    # z-scores are pre-computed in _z_full; just index by column.
    w_sli_raw: Dict[str, float] = {s: 0.0 for s in local_services}
    if sli is not None:
        sli_col = str(sli)
        sli_j = _col_idx.get(sli_col, -1)
        if sli_j >= 0:
            sli_anom = _z_full[anom_start:, sli_j]
            for s in local_services:
                x = service_series.get(s)
                if x is not None:
                    x_anom = _to_1d_float(x)[anom_start:]
                    w_sli_raw[s] = float(_lead_lag_corr(x_anom, sli_anom, max_lag=max_lag))
    w_sli_norm = _min_max_norm(w_sli_raw)

    # w_prop: propagation consistency (upstream onset < downstream onset)
    max_anom_len = max(1, len(anomal_df))
    onset_idx: Dict[str, int] = {}
    for s in local_services:
        x = service_series.get(s)
        if x is None:
            onset_idx[s] = max_anom_len
            continue
        xa = _to_1d_float(x)[anom_start:]
        idxs = np.where(xa >= onset_thr)[0]
        onset_idx[s] = int(idxs[0]) if idxs.size else max_anom_len

    w_prop_raw: Dict[str, float] = {}
    for s in local_services:
        outs = local_out_n.get(s, [])
        if not outs:
            w_prop_raw[s] = 0.0
            continue
        good = sum(1 for u in outs if onset_idx.get(s, max_anom_len) <= onset_idx.get(u, max_anom_len))
        w_prop_raw[s] = float(good) / float(len(outs))
    w_prop_norm = _min_max_norm(w_prop_raw)

    # w_onset: early-onset services are more likely root causes
    onset_score_raw = {
        s: (0.0 if onset_idx.get(s, max_anom_len) >= max_anom_len
            else 1.0 - float(onset_idx[s]) / float(max_anom_len))
        for s in local_services
    }
    onset_score_norm = _min_max_norm(onset_score_raw)

    # w_resource: resource-vs-latency anomaly ratio
    resource_type_w = _compute_resource_type_weight(by_service, metric_score)
    w_resource: Dict[str, float] = {s: float(resource_type_w.get(s, 0.5)) for s in local_services}

    # Combined weight for candidate selection and final blend
    w_combined: Dict[str, float] = {}
    for s in local_services:
        w_combined[s] = (
            alpha * float(w_local_norm.get(s, 0.0))
            + beta  * float(w_sli_norm.get(s, 0.0))
            + gamma * float(w_prop_norm.get(s, 0.0))
            + delta * float(onset_score_norm.get(s, 0.0))
            + eps_r * float(w_resource.get(s, 0.5))
        )

    # ============================================================
    # [6b] Downstream-only node pruning
    #
    #  Nodes that are "obviously downstream" are removed from the
    #  Banzhaf player set BEFORE candidate selection.  This reduces K,
    #  shrinks the MC budget needed for stable estimates, and prevents
    #  pure-effect nodes from splitting Banzhaf mass away from the real
    #  root causes.
    #
    #  GAT ([7]) still runs on the full local subgraph (local_services +
    #  local_edges) for richer propagation context; only the candidate
    #  pool fed into the Banzhaf game ([8]) is restricted.
    #
    #  Prune thresholds are auto-calibrated from this case's score
    #  distributions (25th percentile of each axis) unless overridden.
    # ============================================================
    if prune_enable:
        _p_res, _p_onset, _p_anom = _calibrate_prune_thresholds(
            local_services, w_resource, onset_score_norm, w_local_norm
        )
        prune_res   = _resolve(prune_res_cfg,   _p_res)
        prune_onset = _resolve(prune_onset_cfg, _p_onset)
        prune_anom  = _resolve(prune_anom_cfg,  _p_anom)

        banzhaf_pool, pruned_services = _prune_downstream_nodes(
            services=local_services,
            local_in_n=local_in_n,
            w_resource=w_resource,
            onset_score_norm=onset_score_norm,
            w_local_norm=w_local_norm,
            w_combined=w_combined,
            seed_services=seed_services,
            resource_thresh=prune_res,
            onset_thresh=prune_onset,
            anom_thresh=prune_anom,
            min_keep=prune_min_keep,
        )
    else:
        banzhaf_pool = local_services[:]
        pruned_services: List[str] = []

    # ============================================================
    # [7] Build node features + (optional) Inference GAT
    #
    #  GAT is disabled by default (use_gat=False) to save ~40% wall time.
    #  When off, uniform 1/fan-out attention weights are used; the graph
    #  structure is still reflected in the influence matrix via [8].
    #  Enable with use_gat=True or env LOCAL_BANZHAF_USE_GAT=true.
    # ============================================================
    node_feats = _build_node_features(
        services=local_services,
        w_local_norm=w_local_norm,
        w_resource=w_resource,
        onset_score_norm=onset_score_norm,
        w_sli_norm=w_sli_norm,
        w_prop_norm=w_prop_norm,
        out_n=local_out_n,
        in_n=local_in_n,
    )

    if use_gat:
        _embeddings, gat_edge_weights = _inference_gat(
            node_feats=node_feats,
            edges=local_edges,
            num_hops=gat_hops,
        )
    else:
        # Uniform attention: 1/fan-out per outgoing edge
        gat_edge_weights: Dict[Tuple[str, str], float] = {}
        for src in set(u for u, _ in local_edges):
            out_edges = [(u, v) for (u, v) in local_edges if u == src]
            fan_out = max(1, len(out_edges))
            for e in out_edges:
                gat_edge_weights[e] = 1.0 / fan_out

    # ============================================================
    # [8] Select top-K candidates + compute GAT influence matrix
    #
    #  Candidate pool is banzhaf_pool (already pruned of downstream-only
    #  nodes in [6b]).  topk further limits to the highest-w_combined
    #  subset so K stays small for MC.
    # ============================================================
    banzhaf_pool_set = set(banzhaf_pool)
    candidates = [
        s for s, _ in sorted(w_combined.items(), key=lambda x: x[1], reverse=True)
        if s in banzhaf_pool_set
    ][:topk]
    if not candidates:
        candidates = banzhaf_pool[:] or local_services[:]

    nodes = list(dict.fromkeys(candidates))   # deduplicated, order preserved
    n_nodes = len(nodes)
    node_idx = {n: i for i, n in enumerate(nodes)}

    # GAT influence matrix restricted to candidates
    cand_edge_weights = {
        (u, v): w for (u, v), w in gat_edge_weights.items()
        if u in node_idx and v in node_idx
    }
    influence = _compute_gat_influence(
        nodes=nodes,
        gat_edge_weights=cand_edge_weights,
        max_hops=inf_hops,
        hop_decay=hop_decay,
    )

    # Arrays aligned to nodes index
    w_anom_arr   = np.array([float(w_local_raw.get(n, 0.0)) for n in nodes])
    w_sli_arr    = np.array([float(w_sli_raw.get(n, 0.0))   for n in nodes])
    onset_arr    = np.array([float(onset_score_norm.get(n, 0.0)) for n in nodes])
    denom_cov    = float(np.sum(w_anom_arr)) + 1e-12

    # Precompute two matrices for fast temporal consistency in u(S):
    # edge_exists[i,j]       = 1 if directed edge i→j in candidate set, else 0
    # temporal_consistent[i,j] = 1 if edge i→j exists AND causally consistent, else 0
    #   Causal consistency: callee(j) onset_score >= caller(i) onset_score
    #   (onset_score is high when early → callee must onset earlier than caller)
    edge_exists_mat = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    temporal_consistent = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for (u_s, v_s) in local_edges:
        ii, jj = node_idx.get(u_s, -1), node_idx.get(v_s, -1)
        if ii >= 0 and jj >= 0:
            edge_exists_mat[ii, jj] = 1.0
            if float(onset_arr[jj]) >= float(onset_arr[ii]):
                temporal_consistent[ii, jj] = 1.0
    n_internal_edges = max(1, int(np.sum(edge_exists_mat)))

    # ---- SLI propagation reach (for Align term in utility) ----
    # Compute how strongly each candidate can propagate anomaly to the SLI
    # service through the full local call graph.  This is used in place of
    # direct lag-correlation for the Align term: root causes are upstream and
    # can reach SLI via the call chain, while pure downstream services cannot.
    #
    # We use the full gat_edge_weights (over local_services, not just
    # candidates) so the SLI service is reachable even if it was not selected
    # as a Banzhaf candidate.
    sli_svc_reach: Optional[np.ndarray] = None
    if sli is not None:
        _sli_svc, _ = _parse_service_metric(str(sli))
        _local_svc_set = set(local_services)
        if _sli_svc in _local_svc_set:
            _inf_local = _compute_gat_influence(
                nodes=local_services,
                gat_edge_weights=gat_edge_weights,
                max_hops=inf_hops,
                hop_decay=hop_decay,
            )
            _local_idx = {s: i for i, s in enumerate(local_services)}
            _sli_j = _local_idx.get(_sli_svc, -1)
            if _sli_j >= 0:
                sli_svc_reach = np.array([
                    _inf_local[_local_idx[c], _sli_j] if c in _local_idx else float(w_sli_raw.get(c, 0.0))
                    for c in nodes
                ], dtype=np.float64)

    # ============================================================
    # [9] Monte Carlo Banzhaf — fully vectorised
    # ============================================================
    # B_i = E_{S⊆N\{i}}[u(S∪{i}) - u(S)]
    # All T×K marginals are computed simultaneously via (T,K,K) numpy ops.
    # Scalar u() closure is kept only for the best-coalition search in [12].
    def u(mask: int) -> float:
        return _compute_utility(
            S_mask=mask, n_nodes=n_nodes,
            w_anom_arr=w_anom_arr, w_sli_arr=w_sli_arr,
            influence=influence, edge_exists_mat=edge_exists_mat,
            temporal_consistent=temporal_consistent,
            complexity_lambda=cx_lambda, denom_cov=denom_cov,
            sli_reach_arr=sli_svc_reach,
        )

    K = n_nodes

    if K == 1:
        banzhaf = {nodes[0]: 1.0}
        banzhaf_ci: Dict[str, Tuple[float, float]] = {nodes[0]: (0.0, 1.0)}
        banzhaf_std: Dict[str, float] = {nodes[0]: 0.0}
        stability_metrics: dict = {"kendall_tau": 1.0, "top_k_overlap": 1.0,
                                   "n_blocks": 1, "block_size": mc_T}
    else:
        n_blocks   = max(4, min(10, mc_T // 200))
        block_size = max(10, mc_T // n_blocks)

        block_pivs = _mc_banzhaf_vectorized(
            n_nodes=K,
            influence=influence,
            w_anom_arr=w_anom_arr,
            w_sli_arr=w_sli_arr,
            edge_exists_mat=edge_exists_mat,
            temporal_consistent=temporal_consistent,
            mc_T=mc_T,
            n_blocks=n_blocks,
            denom_cov=denom_cov,
            rng=rng,
            complexity_lambda=cx_lambda,
            sli_reach_arr=sli_svc_reach,
        )

        # Mean Banzhaf estimate across blocks
        piv = block_pivs.mean(axis=0)

        # Normalize raw estimates to [0, 1]
        piv_min, piv_max = float(piv.min()), float(piv.max())
        scale = max(1e-12, piv_max - piv_min)
        piv_norm = (piv - piv_min) / scale if piv_max - piv_min > 1e-12 else np.zeros_like(piv)

        # ---- Block-bootstrap 95% confidence intervals ----
        # SE = std(block estimates) / sqrt(n_blocks), scaled to normalised range.
        # t-quantile from t(n_blocks - 1) table for 95% two-tailed CI.
        _T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78,
                5: 2.57,  6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26}
        t_factor = _T95.get(n_blocks - 1, 2.0)
        piv_std_raw = block_pivs.std(axis=0, ddof=1)      # raw-unit std
        se_norm = (piv_std_raw / scale) / max(1.0, float(n_blocks) ** 0.5)

        banzhaf_ci = {
            nodes[i]: (
                float(max(0.0, piv_norm[i] - t_factor * float(se_norm[i]))),
                float(min(1.0, piv_norm[i] + t_factor * float(se_norm[i]))),
            )
            for i in range(K)
        }
        banzhaf_std = {nodes[i]: float(se_norm[i]) for i in range(K)}
        banzhaf = {nodes[i]: float(piv_norm[i]) for i in range(K)}

        # ---- Stability: split-half Kendall τ + top-k overlap ----
        # Compare rankings derived from the first vs. second half of blocks.
        # High τ → the Banzhaf ranking is stable across different random seeds.
        half = max(1, n_blocks // 2)
        rank_a = block_pivs[:half].mean(axis=0)
        rank_b = block_pivs[half:].mean(axis=0)

        # Pure-NumPy Kendall τ (O(K²), fine for K ≤ 30)
        concordant = discordant = 0
        for _ii in range(K):
            for _jj in range(_ii + 1, K):
                d = (float(rank_a[_ii]) - float(rank_a[_jj])) * \
                    (float(rank_b[_ii]) - float(rank_b[_jj]))
                if d > 1e-15:
                    concordant += 1
                elif d < -1e-15:
                    discordant += 1
        _denom_kt = concordant + discordant
        kendall_tau = float(concordant - discordant) / _denom_kt if _denom_kt > 0 else 1.0

        top_k_n = min(K, 5)
        r_a = sorted(range(K), key=lambda _i: float(rank_a[_i]), reverse=True)[:top_k_n]
        r_b = sorted(range(K), key=lambda _i: float(rank_b[_i]), reverse=True)[:top_k_n]
        topk_overlap = float(len(set(r_a) & set(r_b))) / float(max(1, top_k_n))

        stability_metrics = {
            "kendall_tau": float(kendall_tau),
            "top_k_overlap": float(topk_overlap),
            "n_blocks": n_blocks,
            "block_size": block_size,
        }

    # ============================================================
    # [10] Final score: blend Banzhaf + combined weight
    # ============================================================
    w_cand_norm = _min_max_norm({n: float(w_combined.get(n, 0.0)) for n in nodes})
    final_score = {
        n: eta * float(banzhaf.get(n, 0.0)) + (1.0 - eta) * float(w_cand_norm.get(n, 0.0))
        for n in nodes
    }
    service_rank = sorted(final_score, key=lambda x: final_score[x], reverse=True)

    # ============================================================
    # [11] Map service ranking → metric-level strings
    # ============================================================
    ranks: List[str] = []
    seen: set = set()
    for svc in service_rank:
        rep = service_rep_metric.get(svc)
        if rep and rep not in seen:
            ranks.append(rep)
            seen.add(rep)

    # Append remaining metrics by anomaly score
    for c, _ in sorted(metric_score.items(), key=lambda x: x[1], reverse=True):
        if c not in seen:
            ranks.append(c)
            seen.add(c)

    # ============================================================
    # [12] Build explanations
    # ============================================================
    # Find representative coalition (highest-u single-member or pair)
    best_coalition: List[str] = []
    best_u = -1.0
    for i in range(K):
        score_i = u(1 << i)
        if score_i > best_u:
            best_u = score_i
            best_coalition = [nodes[i]]
    for i in range(K):
        for j in range(i + 1, K):
            score_ij = u((1 << i) | (1 << j))
            if score_ij > best_u:
                best_u = score_ij
                best_coalition = [nodes[i], nodes[j]]

    # Top GAT edges by attention weight (for visualization)
    top_gat_edges = sorted(
        [(u_s, v_s, w) for (u_s, v_s), w in gat_edge_weights.items()],
        key=lambda x: x[2], reverse=True
    )[:10]

    return {
        # Required by RCAEval evaluator
        "node_names": metric_cols,
        "ranks": ranks,
        # Banzhaf explanations
        "banzhaf_values": {n: float(banzhaf.get(n, 0.0)) for n in nodes},
        # 95% block-bootstrap CI: {service: (lower, upper)} in normalised [0,1] space
        "banzhaf_ci": banzhaf_ci,
        # Block-bootstrap standard error per service
        "banzhaf_std": banzhaf_std,
        # Stability diagnostics (split-half Kendall τ + top-5 overlap)
        "stability": stability_metrics,
        # Representative coalition (highest-utility 1- or 2-member set)
        "best_coalition": best_coalition,
        "best_coalition_utility": float(best_u),
        # GAT propagation structure (top-10 edges by attention weight)
        "gat_edge_weights": {f"{u_s}→{v_s}": round(w, 4) for u_s, v_s, w in top_gat_edges},
        # Provenance
        "local_services": local_services,
        "seed_services": seed_services,
        "used_traces": traces_edges is not None,
        # Window split provenance
        "split_source": split_source,
        "effective_sli": effective_sli,
        # Downstream pruning provenance
        "pruned_services": pruned_services,
        "banzhaf_pool_size": len(banzhaf_pool),
        # Auto-calibrated hyperparameter values (for reproducibility / debugging).
        # "cusum_h_suggested" is diagnostic: the split in [1] used the
        # pre-calibration default; pass it as kwarg `cusum_h=<value>` next run
        # if you want to apply it.
        "calibrated": {
            "onset_thr":         round(float(onset_thr), 4),
            "corr_thr":          round(float(corr_thr), 4),
            "cusum_h_suggested": round(float(_cusum_h_suggested), 4),
            "expand_rho":        round(float(expand_rho), 4),
            "expand_lam_cov":    round(float(expand_lam_cov), 4),
            "expand_lam_aln":    round(float(expand_lam_aln), 4),
            "expand_lam_tmp":    round(float(expand_lam_tmp), 4),
        },
    }


# ============================================================
# Fallback: BARO-like robust ranking
# ============================================================

def _fallback_baro_like(
    data: pd.DataFrame,
    inject_time: int | None = None,
    dataset: str | None = None,
    anomalies=None,
    **kwargs,
) -> dict:
    """Robust fallback: BARO-like RobustScaler ranking.

    Invoked when the main algorithm encounters any unexpected error.
    Ensures the case never fails completely.

    Window split priority mirrors the main function:
      anomalies > inject_time > change-point (CUSUM) > 50/50 split.
    """
    try:
        from sklearn.preprocessing import RobustScaler as _RS
        if anomalies is not None:
            normal_df = data.head(anomalies[0])
            anomal_df = data.tail(len(data) - anomalies[0])
        elif inject_time is not None and "time" in data.columns:
            normal_df = data[data["time"] < inject_time]
            anomal_df = data[data["time"] >= inject_time]
        else:
            # Attempt lightweight change-point detection on the best SLI column
            _sli_col = _auto_select_sli(data)
            _split, _ = _detect_split_index(
                data=data, sli_col=_sli_col, method="cusum",
                min_normal_frac=0.30, max_normal_frac=0.80,
            )
            normal_df = data.head(_split)
            anomal_df = data.tail(len(data) - _split)

        normal_df = preprocess(data=normal_df, dataset=dataset,
                               dk_select_useful=kwargs.get("dk_select_useful", False))
        anomal_df = preprocess(data=anomal_df, dataset=dataset,
                               dk_select_useful=kwargs.get("dk_select_useful", False))
        intersects = [c for c in normal_df.columns if c in anomal_df.columns]
        normal_df = normal_df[intersects]
        anomal_df = anomal_df[intersects]

        scores = []
        for col in normal_df.columns:
            a = _to_1d_float(normal_df[col].to_numpy())
            b = _to_1d_float(anomal_df[col].to_numpy())
            if a.size < 2 or b.size == 0:
                scores.append((col, 0.0))
                continue
            sc = _RS().fit(a.reshape(-1, 1))
            z = np.abs(np.clip(sc.transform(b.reshape(-1, 1))[:, 0], 0.0, 100.0))
            scores.append((col, float(np.max(z))))

        scores.sort(key=lambda x: x[1], reverse=True)
        ranks = [c for c, _ in scores]
        return {"node_names": list(normal_df.columns), "ranks": ranks}

    except Exception:
        try:
            d = preprocess(data=data, dataset=dataset,
                           dk_select_useful=kwargs.get("dk_select_useful", False))
            cols = list(d.columns)
            return {"node_names": cols, "ranks": cols}
        except Exception:
            return {"node_names": [], "ranks": []}
