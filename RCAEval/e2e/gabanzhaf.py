import os
import numpy as np
import pandas as pd
import warnings
from dataclasses import dataclass
from time import perf_counter

warnings.filterwarnings("ignore")

from RCAEval.io.time_series import preprocess


@dataclass(frozen=True)
class _TraceFeatures:
    trace_scores: dict | None
    call_graph: dict


@dataclass(frozen=True)
class _GabanzhafConfig:
    """Centralized runtime configuration for the GA-Banzhaf pipeline."""

    top_k_svcs: int = 15
    max_candidate_svcs: int = 20
    banzhaf_samples: int = 1000
    banzhaf_mode: str = "deterministic" # "deterministic" or "mc"
    seed: int = 42
    dk_select_useful: bool = False
    topology_pool_multiplier: int = 2
    topology_pool_extra: int = 5
    call_graph_confidence: float = 0.5
    call_graph_callee_min_score: float = 0.2
    call_graph_caller_min_score: float = 0.1
    residual_call_graph_ratio: float = 0.8
    residual_confidence_scale: float = 2.0
    residual_confidence_cap: float = 0.95
    topology_reliability_scale: float = 2.0
    topology_reliability_cap: float = 0.65
    residual_weight_base: float = 0.30
    residual_weight_topology_gain: float = 0.50
    topology_onset_tolerance: float = 0.05
    topology_onset_boost: float = 0.50
    fitness_temporal_weight: float = 0.12
    fitness_impact_weight: float = 0.10

    @classmethod
    def from_kwargs(cls, kwargs):
        top_k_svcs = int(kwargs.get("top_k_svcs", cls.top_k_svcs))
        return cls(
            top_k_svcs=top_k_svcs,
            max_candidate_svcs=int(
                kwargs.get("max_candidate_svcs", top_k_svcs + cls.topology_pool_extra)
            ),
            banzhaf_samples=int(kwargs.get("banzhaf_samples", cls.banzhaf_samples)),
            banzhaf_mode=str(kwargs.get("banzhaf_mode", cls.banzhaf_mode)),
            seed=int(kwargs.get("seed", cls.seed)),
            dk_select_useful=bool(kwargs.get("dk_select_useful", cls.dk_select_useful)),
            topology_onset_tolerance=float(
                kwargs.get("topology_onset_tolerance", cls.topology_onset_tolerance)
            ),
            topology_onset_boost=float(
                kwargs.get("topology_onset_boost", cls.topology_onset_boost)
            ),
            fitness_temporal_weight=float(
                kwargs.get("fitness_temporal_weight", cls.fitness_temporal_weight)
            ),
            fitness_impact_weight=float(
                kwargs.get("fitness_impact_weight", cls.fitness_impact_weight)
            ),
        )


def _trace_scores_from_chunks(chunks_normal, chunks_anomal):
    if not chunks_anomal:
        return None

    anomal_df = pd.concat(chunks_anomal, ignore_index=True)
    anomal_df["is_err"] = (
        pd.to_numeric(anomal_df["statusCode"], errors="coerce").fillna(0.) != 0.
    )
    anomal_df["dur"] = pd.to_numeric(anomal_df["duration"], errors="coerce").fillna(0.)

    a_stats = anomal_df.groupby("serviceName", sort=False).agg(
        cnt=("serviceName", "size"),
        err_cnt=("is_err", "sum"),
        dur_sum=("dur", "sum"),
    )

    n_dur_avg: dict = {}
    if chunks_normal:
        normal_df = pd.concat(chunks_normal, ignore_index=True)
        normal_df["dur"] = pd.to_numeric(normal_df["duration"], errors="coerce").fillna(0.)
        n_stats = normal_df.groupby("serviceName", sort=False).agg(
            cnt=("serviceName", "size"),
            dur_sum=("dur", "sum"),
        )
        for svc, row in n_stats.iterrows():
            n_dur_avg[str(svc)] = float(row["dur_sum"]) / max(int(row["cnt"]), 1)

    scores: dict = {}
    for svc, row in a_stats.iterrows():
        svc = str(svc)
        a_cnt = max(int(row["cnt"]), 1)
        err_rate = float(row["err_cnt"]) / a_cnt

        a_avg_dur = float(row["dur_sum"]) / a_cnt
        n_avg_dur = n_dur_avg.get(svc, 1.)
        dur_ratio = (a_avg_dur / max(n_avg_dur, 1.)) - 1.
        dur_signal = max(0., min(dur_ratio, 10.)) / 10.

        scores[svc] = max(err_rate, dur_signal)

    if not scores:
        return None

    lo, hi = min(scores.values()), max(scores.values())
    span = hi - lo if hi > lo else 1.
    return {svc: (v - lo) / span for svc, v in scores.items()}


def _trace_service_from_timeseries_col(col):
    if col == "time":
        return None
    return str(col).split("_", 1)[0]


def _preaggregated_trace_scores(case_dir, inject_time):
    scores: dict = {}

    def _add_score(svc, value):
        if svc and np.isfinite(value):
            scores[svc] = max(scores.get(svc, 0.0), float(value))

    err_path = os.path.join(case_dir, "tracets_err.csv")
    if os.path.exists(err_path):
        try:
            err_df = pd.read_csv(err_path)
            if "time" in err_df:
                anomal = err_df[err_df["time"] >= inject_time]
            else:
                anomal = err_df.tail(len(err_df) // 2)
            for col in anomal.columns:
                svc = _trace_service_from_timeseries_col(col)
                if svc is None:
                    continue
                values = pd.to_numeric(anomal[col], errors="coerce").fillna(0.0)
                _add_score(svc, float(values.mean()))
        except Exception:
            pass

    lat_path = os.path.join(case_dir, "tracets_lat.csv")
    if os.path.exists(lat_path):
        try:
            lat_df = pd.read_csv(lat_path)
            if "time" in lat_df:
                normal = lat_df[lat_df["time"] < inject_time]
                anomal = lat_df[lat_df["time"] >= inject_time]
            else:
                split = len(lat_df) // 2
                normal = lat_df.head(split)
                anomal = lat_df.tail(len(lat_df) - split)
            for col in anomal.columns:
                svc = _trace_service_from_timeseries_col(col)
                if svc is None:
                    continue
                a_values = pd.to_numeric(anomal[col], errors="coerce").fillna(0.0)
                n_values = pd.to_numeric(normal[col], errors="coerce").fillna(0.0)
                a_avg = float(a_values.mean())
                n_avg = float(n_values.mean()) if len(n_values) else 0.0
                dur_ratio = (a_avg / max(n_avg, 1.0)) - 1.0
                dur_signal = max(0.0, min(dur_ratio, 10.0)) / 10.0
                _add_score(svc, dur_signal)
        except Exception:
            pass

    if not scores:
        return None
    return _norm_dict(scores)


def _extract_trace_features(case_dir, inject_time):
    """Extract trace onset scores and service call graph with one traces.csv scan."""
    if not case_dir:
        return _TraceFeatures(trace_scores=None, call_graph={})

    traces_path = os.path.join(case_dir, "traces.csv")
    fast_trace_scores = _preaggregated_trace_scores(case_dir, inject_time)
    if fast_trace_scores is not None:
        return _TraceFeatures(
            trace_scores=fast_trace_scores,
            call_graph=_extract_call_graph(traces_path),
        )

    if not os.path.exists(traces_path):
        return _TraceFeatures(trace_scores=None, call_graph={})

    try:
        inject_ms = inject_time * 1000
        score_cols = {"serviceName", "startTimeMillis", "statusCode", "duration"}
        span_cols = {"serviceName", "spanID", "parentSpanID"}
        needed = score_cols | span_cols
        max_anomal = 30_000
        max_graph_rows = 50_000

        chunks_normal: list = []
        chunks_anomal: list = []
        graph_chunks: list = []
        anomal_rows = 0
        graph_rows = 0
        saw_score_cols = False
        saw_span_cols = False

        for chunk in pd.read_csv(
            traces_path,
            usecols=lambda col: col in needed,
            chunksize=10_000,
            on_bad_lines="skip",
        ):
            cols = set(chunk.columns)
            has_score_cols = score_cols.issubset(cols)
            has_span_cols = span_cols.issubset(cols)
            saw_score_cols = saw_score_cols or has_score_cols
            saw_span_cols = saw_span_cols or has_span_cols

            if has_span_cols and graph_rows < max_graph_rows:
                remaining = max_graph_rows - graph_rows
                graph_chunk = chunk.loc[:, ["serviceName", "spanID", "parentSpanID"]].head(
                    remaining
                )
                if not graph_chunk.empty:
                    graph_chunks.append(graph_chunk)
                    graph_rows += len(graph_chunk)

            if has_score_cols and anomal_rows < max_anomal:
                score_chunk = chunk.loc[
                    :, ["serviceName", "startTimeMillis", "statusCode", "duration"]
                ]
                ts = pd.to_numeric(score_chunk["startTimeMillis"], errors="coerce")
                is_norm = ts < inject_ms
                is_anom = ts >= inject_ms

                if is_norm.any():
                    chunks_normal.append(score_chunk[is_norm])
                if is_anom.any():
                    remaining = max_anomal - anomal_rows
                    ac = score_chunk[is_anom].head(remaining)
                    if not ac.empty:
                        chunks_anomal.append(ac)
                        anomal_rows += len(ac)

            score_done = (not saw_score_cols) or anomal_rows >= max_anomal
            graph_done = (not saw_span_cols) or graph_rows >= max_graph_rows
            if score_done and graph_done:
                break

        call_graph: dict = {}
        if graph_chunks:
            graph_df = pd.concat(graph_chunks, ignore_index=True)
            span_to_svc = graph_df.set_index("spanID")["serviceName"].to_dict()
            graph_df["parentSvc"] = graph_df["parentSpanID"].map(span_to_svc)
            edges = graph_df[
                graph_df["parentSvc"].notna()
                & (graph_df["parentSvc"] != graph_df["serviceName"])
            ]
            for caller, callee in zip(edges["parentSvc"], edges["serviceName"]):
                call_graph.setdefault(caller, set()).add(callee)

        trace_scores = (
            _trace_scores_from_chunks(chunks_normal, chunks_anomal)
            if saw_score_cols
            else None
        )
        return _TraceFeatures(trace_scores=trace_scores, call_graph=call_graph)
    except Exception:
        return _TraceFeatures(trace_scores=None, call_graph={})


def _extract_call_graph(traces_path):
    """Extract service call graph from the first 50 K trace spans.

    Reads only spanID / parentSpanID / serviceName, so I/O is ~0.15 s even
    for the largest TT files (114 MB).  50 K rows covers ≥ 98 % of distinct
    call pairs in practice.

    Returns
    -------
    dict  {caller_svc: set(callee_svc, ...)}
        "caller calls callee" — i.e. if callee is anomalous it *explains*
        caller's symptoms (anomaly propagates caller ← callee).
    """
    if not traces_path or not os.path.exists(traces_path):
        return {}
    try:
        df = pd.read_csv(
            traces_path,
            usecols=["serviceName", "spanID", "parentSpanID"],
            nrows=50_000,
            on_bad_lines="skip",
        )
        span_to_svc = df.set_index("spanID")["serviceName"].to_dict()
        df["parentSvc"] = df["parentSpanID"].map(span_to_svc)
        edges = df[
            df["parentSvc"].notna() & (df["parentSvc"] != df["serviceName"])
        ]
        result: dict = {}
        for caller, callee in zip(edges["parentSvc"], edges["serviceName"]):
            result.setdefault(caller, set()).add(callee)
        return result
    except Exception:
        return {}


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


def _metric_evidence(normal_mat, anomal_mat, cols):
    """Metric-level evidence from robust magnitude, persistence and peak.

    The score remains unsupervised: it only compares the post-injection window
    with the pre-injection baseline.  Percentile and persistence terms reduce
    the chance that a single noisy spike dominates the ranking.
    """
    med = np.nanmedian(normal_mat, axis=0)
    q75 = np.nanpercentile(normal_mat, 75, axis=0)
    q25 = np.nanpercentile(normal_mat, 25, axis=0)
    std = np.nanstd(normal_mat, axis=0)
    iqr = np.where((q75 - q25) > 0, q75 - q25, 1.)

    rz = np.abs((anomal_mat - med) / iqr)
    rz = np.nan_to_num(rz, nan=0., posinf=0., neginf=0.)
    # Selective Winsorization: cap metrics where IQR=0 AND std≈0 (truly constant
    # baseline).  These produce inflated rz through the fallback denominator of
    # 1.0 even for tiny shifts, masking real root causes.  The std<=1e-6 guard
    # preserves columns that have IQR=0 but non-trivial spread (step-function
    # or bursty behaviour) so their rz is still meaningful.
    near_zero_baseline = ((q75 - q25) <= 0) & (std <= 1e-6)   # shape (n_cols,)
    rz = np.where(near_zero_baseline, np.clip(rz, 0., 10.), rz)

    mag = _norm01_arr(np.nanpercentile(rz, 95, axis=0))
    peak = _norm01_arr(np.nanmax(rz, axis=0))
    persistence = _norm01_arr(np.mean(rz > 3.0, axis=0))

    n_steps = anomal_mat.shape[0]
    if n_steps > 0:
        onset_thr = np.maximum(3.0, 0.5 * np.nanpercentile(rz, 95, axis=0))
        exceed = rz >= onset_thr.reshape(1, -1)
        first_idx = np.argmax(exceed, axis=0)
        seen = exceed.any(axis=0)
        denom = max(n_steps - 1, 1)
        earliness = np.where(seen, 1.0 - (first_idx / denom), 0.0)

        time_idx = np.arange(n_steps, dtype=float).reshape(-1, 1)
        rz_mass = np.sum(rz, axis=0)
        centroid = np.divide(
            np.sum(rz * time_idx, axis=0),
            rz_mass + 1e-9,
        )
        frontload = np.where(rz_mass > 0.0, 1.0 - (centroid / denom), 0.0)
        onset = _norm01_arr(0.65 * earliness + 0.35 * frontload)
    else:
        onset = np.zeros(len(cols), dtype=float)

    raw = 0.52 * mag + 0.28 * persistence + 0.15 * peak + 0.05 * onset
    score = _norm01_arr(raw)

    return {
        col: {
            "score": float(score[i]),
            "mag": float(mag[i]),
            "persistence": float(persistence[i]),
            "peak": float(peak[i]),
            "onset": float(onset[i]),
        }
        for i, col in enumerate(cols)
    }


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
        arr = np.asarray([metric_ev[c]["score"] for c in svc_cols], dtype=float)
        onset_arr = np.asarray([metric_ev[c].get("onset", 0.0) for c in svc_cols], dtype=float)
        top = np.sort(arr)[-min(3, len(arr)):]
        svc_agg[svc] = float(
            0.48 * np.percentile(arr, 95)
            + 0.27 * np.mean(top)
            + 0.15 * np.max(arr)
            + 0.10 * np.percentile(onset_arr, 90)
        )

    # ── 2. Metric-type dominance ─────────────────────────────────────────────
    # Track the top-2 scores per metric type across services.
    # "metric type" = everything after the service prefix  (e.g. "cpu", "mem").
    type_best: dict   = {}   # mtype -> (winning_svc, score)
    type_second: dict = {}   # mtype -> best score from a *different* service

    for col in cols:
        svc   = _parse_svc(col)
        mtype = col[len(svc) + 1:] if col.startswith(svc + "_") else col
        sc    = metric_ev[col]["score"]

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
        svc: 0.72 * svc_agg[svc] + 0.28 * dom_norm.get(svc, 0.)
        for svc in svc_agg
    }
    return _norm_dict(svc_scores)


def _service_temporal_prior(cols, metric_ev):
    svc_to_cols: dict = {}
    for col in cols:
        svc_to_cols.setdefault(_parse_svc(col), []).append(col)

    priors = {}
    for svc, svc_cols in svc_to_cols.items():
        onset_arr = np.asarray([metric_ev[c].get("onset", 0.0) for c in svc_cols], dtype=float)
        persistence_arr = np.asarray(
            [metric_ev[c].get("persistence", 0.0) for c in svc_cols], dtype=float
        )
        combined = 0.7 * onset_arr + 0.3 * onset_arr * persistence_arr
        priors[svc] = float(np.max(combined)) if combined.size else 0.0
    return _norm_dict(priors)


def _deterministic_banzhaf_coverage(fitness_signal, topo_matrix):
    """Deterministic Banzhaf values for causal coverage games."""
    fitness = np.asarray(fitness_signal, dtype=float).reshape(-1)
    n = len(fitness)
    if n == 0:
        return np.zeros(0, dtype=float)

    topo = np.nan_to_num(
        np.asarray(topo_matrix, dtype=float),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    explains = topo > 0
    np.fill_diagonal(explains, True)

    num_explainers = explains.sum(axis=0)
    marginal_weight = fitness * np.power(0.5, num_explainers - 1)
    return np.maximum(0.0, explains @ marginal_weight)


def _banzhaf_power(fitness_signal, topo_matrix, config):
    if config.banzhaf_mode == "mc":
        return _mc_banzhaf(
            fitness_signal,
            topo_matrix,
            n_samples=config.banzhaf_samples,
            seed=config.seed,
        )
    if config.banzhaf_mode == "deterministic":
        return _deterministic_banzhaf_coverage(fitness_signal, topo_matrix)
    raise ValueError(f"Unsupported banzhaf_mode: {config.banzhaf_mode}")


def _mc_banzhaf(fitness_signal, topo_matrix, n_samples, seed=42):
    """Monte Carlo Banzhaf estimation with causal-coverage v(S)."""
    n = len(fitness_signal)
    rng = np.random.default_rng(seed)
    banzhaf = np.zeros(n, dtype=float)
    counts = np.zeros(n, dtype=float)

    downstream_masks = [0] * n
    for i in range(n):
        for j in range(n):
            if topo_matrix[i, j] > 0:
                downstream_masks[i] |= (1 << j)

    def _v(m):
        explained = m
        for i in range(n):
            if (m >> i) & 1:
                explained |= downstream_masks[i]
        total = 0.
        for j in range(n):
            if (explained >> j) & 1:
                total += fitness_signal[j]
        return total

    for _ in range(n_samples):
        k = int(rng.integers(0, n + 1))
        bits = rng.choice(n, size=k, replace=False) if k > 0 else []
        m = sum(1 << int(b) for b in bits)

        v_m = _v(m)
        for i in range(n):
            mask_i = 1 << i
            if m & mask_i:
                v_without = _v(m ^ mask_i)
                banzhaf[i] += v_m - v_without
            else:
                v_with = _v(m | mask_i)
                banzhaf[i] += v_with - v_m
            counts[i] += 1

    safe_counts = np.where(counts > 0, counts, 1.)
    return np.maximum(0., banzhaf / safe_counts)


def _service_root_score(
    banzhaf_score,
    residual_score,
    anomaly_score,
    trace_score=0.0,
    temporal_score=0.0,
    corroboration_score=0.0,
):
    """Fixed service-level blend; missing trace is represented by trace_score=0."""
    return (
        0.35 * banzhaf_score
        + 0.23 * residual_score
        + 0.17 * anomaly_score
        + 0.10 * trace_score
        + 0.15 * corroboration_score
    )


def _build_topology(
    svc_names,
    svc_series,
    corr_thr=0.08,
    max_in=3,
    svc_temporal=None,
    onset_tol=0.05,
    onset_boost=0.50,
):
    """Build weighted directed topology from asymmetric lag-1 correlations.

    Vectorised: stacks all available service series into a matrix and
    computes the full n_svc × n_svc lag-1 correlation matrix in one
    numpy matmul.  Edge confidence is asymmetric: u→v is strong only when
    u leads v more than v leads u.
    """
    svcs = [s for s in svc_names if svc_series.get(s) is not None]
    if len(svcs) < 2:
        return []

    T = min(len(svc_series[s]) for s in svcs) - 1
    if T < 3:
        return []

    # Stack: (n_svcs, T)
    X = np.array([svc_series[s][:T]   for s in svcs], dtype=float)  # current
    Y = np.array([svc_series[s][1:T+1] for s in svcs], dtype=float)  # lag-1

    # Z-score each row
    X -= X.mean(axis=1, keepdims=True)
    Y -= Y.mean(axis=1, keepdims=True)
    xn = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    yn = np.linalg.norm(Y, axis=1, keepdims=True) + 1e-9
    X /= xn;  Y /= yn

    # Lag-1 correlation matrix: C[i,j] = corr(X[i], Y[j]) = X_i leads Y_j.
    C = X @ Y.T  # (n_svcs, n_svcs)
    np.fill_diagonal(C, 0.)

    confidence = np.maximum(0., C - C.T)
    if svc_temporal:
        onset = np.array([float(svc_temporal.get(s, 0.0)) for s in svcs], dtype=float)
        onset_gap = onset[:, None] - onset[None, :]
        onset_gate = np.where(
            onset_gap < -onset_tol,
            0.0,
            1.0 + onset_boost * np.clip(onset_gap, 0.0, 1.0),
        )
        confidence *= onset_gate
    np.fill_diagonal(confidence, 0.)

    edges = []
    for j, v in enumerate(svcs):
        parents = np.where(confidence[:, j] > corr_thr)[0]
        if len(parents) > max_in:
            parents = parents[np.argsort(confidence[parents, j])[-max_in:]]
        for i in parents:
            edges.append((svcs[i], v, float(confidence[i, j])))
    return edges


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


def _service_series_for_pool(cols, no_time, pool_svcs):
    values = no_time.values
    svc_series = {}
    for svc in pool_svcs:
        idxs = [i for i, col in enumerate(cols) if _parse_svc(col) == svc]
        if idxs:
            svc_series[svc] = values[:, idxs].max(axis=1).astype(float)
    return svc_series


def _initial_service_pool(svc_ranked, seed_svcs, config):
    pool_k = min(
        len(svc_ranked),
        max(
            config.top_k_svcs * config.topology_pool_multiplier,
            config.top_k_svcs + config.topology_pool_extra,
        ),
    )
    return set(svc_ranked[:pool_k]) | set(seed_svcs)


def _call_graph_topology_edges(lag_corr_edges, call_graph, pool_svcs, svc_anom_dict, config):
    if not call_graph:
        return []

    existing_edges = {(u, v) for u, v, _ in lag_corr_edges}
    call_graph_edges = []
    for caller, callees in call_graph.items():
        for callee in callees:
            if callee not in pool_svcs or caller not in pool_svcs:
                continue
            callee_score = svc_anom_dict.get(callee, 0.)
            caller_score = svc_anom_dict.get(caller, 0.)
            if (
                callee_score < config.call_graph_callee_min_score
                or caller_score < config.call_graph_caller_min_score
            ):
                continue
            if (callee, caller) not in existing_edges:
                call_graph_edges.append((callee, caller, config.call_graph_confidence))
                existing_edges.add((callee, caller))
    return call_graph_edges


def _upstream_maps(svc_names, lag_corr_edges, call_graph_edges, svc_anom_dict, config):
    upstream_map_full = {s: set() for s in svc_names}
    upstream_map_residual = {s: set() for s in svc_names}
    edge_conf = {}

    for u, v, w in lag_corr_edges + call_graph_edges:
        upstream_map_full.get(v, set()).add(u)
        edge_conf[(u, v)] = max(edge_conf.get((u, v), 0.), w)
    for u, v, _ in lag_corr_edges:
        upstream_map_residual.get(v, set()).add(u)
    for u, v, _ in call_graph_edges:
        if svc_anom_dict.get(u, 0.) >= svc_anom_dict.get(v, 0.) * config.residual_call_graph_ratio:
            upstream_map_residual.get(v, set()).add(u)

    return upstream_map_full, upstream_map_residual, edge_conf


def _candidate_services(seed_svcs, upstream_map_full, svc_anom_dict, config):
    candidates = set(seed_svcs)
    for svc in list(candidates):
        candidates |= upstream_map_full.get(svc, set())
    if len(candidates) <= config.max_candidate_svcs:
        return sorted(candidates)
    return sorted(
        sorted(candidates, key=lambda s: svc_anom_dict.get(s, 0.), reverse=True)[
            : config.max_candidate_svcs
        ]
    )


def _residual_scores(svc_anom_dict, upstream_map_residual, edge_conf, config):
    residual = {}
    for svc, score in svc_anom_dict.items():
        explain_fail = 1.0
        for u in upstream_map_residual.get(svc, set()):
            eff = svc_anom_dict.get(u, 0.0) * min(
                edge_conf.get((u, svc), 0.0) * config.residual_confidence_scale,
                config.residual_confidence_cap,
            )
            explain_fail *= max(0.0, 1.0 - eff)
        explained = 1.0 - explain_fail
        residual[svc] = max(0.0, score * (1.0 - explained))
    return residual


def _fitness_signal(
    cand_list,
    svc_anom_dict,
    svc_residual,
    svc_temporal,
    topology_edges,
    config,
):
    svc_anom_arr = np.array([svc_anom_dict.get(s, 0.) for s in cand_list])
    svc_residual_arr = np.array([svc_residual.get(s, 0.) for s in cand_list])
    svc_temporal_arr = np.array([svc_temporal.get(s, 0.) for s in cand_list])
    conf_vals = np.array([w for _, _, w in topology_edges], dtype=float)
    topo_rel = (
        float(
            np.clip(
                np.mean(conf_vals) * config.topology_reliability_scale,
                0.0,
                config.topology_reliability_cap,
            )
        )
        if conf_vals.size
        else 0.0
    )
    res_w = config.residual_weight_base + config.residual_weight_topology_gain * topo_rel
    intrinsic = (
        res_w * svc_residual_arr
        + (1.0 - res_w) * (
            (1.0 - config.fitness_temporal_weight) * svc_anom_arr
            + config.fitness_temporal_weight * svc_temporal_arr
        )
    )

    impact = np.zeros(len(cand_list), dtype=float)
    svc_idx = {s: i for i, s in enumerate(cand_list)}
    for u, v, w in topology_edges:
        if u in svc_idx and v in svc_idx:
            impact[svc_idx[u]] += float(w) * svc_anom_dict.get(v, 0.0)
    impact = _norm01_arr(impact) if impact.size else impact

    return intrinsic + config.fitness_impact_weight * topo_rel * impact


def _topology_matrix(cand_list, lag_corr_edges):
    svc_idx = {s: i for i, s in enumerate(cand_list)}
    matrix = np.zeros((len(cand_list), len(cand_list)), dtype=np.float32)
    for u, v, w in lag_corr_edges:
        if u in svc_idx and v in svc_idx:
            matrix[svc_idx[u], svc_idx[v]] = w
    return matrix


def _rank_metrics(
    all_cols,
    svc_banzhaf,
    svc_residual,
    svc_anom_dict,
    svc_temporal,
    metric_score,
    trace_scores,
):
    nb = _norm_dict(svc_banzhaf)
    nr = _norm_dict(svc_residual)
    na = _norm_dict(svc_anom_dict)
    nm = _norm_dict(metric_score)
    trace_source = trace_scores or {}
    nt = _norm_dict({s: trace_source.get(s, 0.) for s in svc_anom_dict})

    def _col_score(col):
        svc = _parse_svc(col)
        corroboration = float(np.sqrt(max(na.get(svc, 0.0), 0.0) * max(nt.get(svc, 0.0), 0.0)))
        service_root = _service_root_score(
            nb.get(svc, 0.),
            nr.get(svc, 0.),
            na.get(svc, 0.),
            nt.get(svc, 0.),
            corroboration_score=corroboration,
        )
        metric_local = nm.get(col, 0.)
        return 0.70 * service_root + 0.30 * metric_local + 0.20 * service_root * metric_local

    return sorted(all_cols, key=_col_score, reverse=True)


# ==========================================
# Core pipeline
# ==========================================


def _gabanzhaf_core(data, inject_time, dataset, anomalies, kwargs):
    """
    Pipeline: window split → metric evidence → service evidence
              → bounded weighted topology → Banzhaf → metric ranking
    """
    config = _GabanzhafConfig.from_kwargs(kwargs)
    trace_scores = kwargs.get("trace_scores", None)
    sli = kwargs.get("sli", None)
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
    metric_ev = _metric_evidence(normal_mat, anomal_mat, cols)
    metric_score = {c: metric_ev[c]["score"] for c in cols}
    svc_anom_dict = _service_evidence(cols, metric_ev)
    svc_temporal = _service_temporal_prior(cols, metric_ev)
    svc_names = sorted(svc_anom_dict.keys())
    _mark_stage("metric_evidence")

    # ── 4. Candidate-first bounded topology ──────────────────────────────────
    svc_ranked = sorted(svc_names, key=lambda s: svc_anom_dict.get(s, 0.), reverse=True)
    seed_svcs = set(svc_ranked[: config.top_k_svcs])
    if sli:
        seed_svcs.add(_parse_svc(str(sli)))

    pool_svcs = _initial_service_pool(svc_ranked, seed_svcs, config)
    svc_series_full = _service_series_for_pool(cols, no_time, pool_svcs)
    lag_corr_edges = _build_topology(
        sorted(pool_svcs),
        svc_series_full,
        svc_temporal=svc_temporal,
        onset_tol=config.topology_onset_tolerance,
        onset_boost=config.topology_onset_boost,
    )

    # ── 4b. Augment topology with trace-derived call graph ────────────────────
    # Call graph: {caller: {callee}}.  When callee is anomalous it *explains*
    # caller's symptoms → topology edge: callee → caller (callee upstream of caller).
    # Confidence is moderate; both callee AND caller must show meaningful anomaly
    # to avoid adding spurious edges in non-propagating faults.
    call_graph_edges = _call_graph_topology_edges(
        lag_corr_edges,
        kwargs.get("call_graph", {}),
        pool_svcs,
        svc_anom_dict,
        config,
    )

    topology_edges = lag_corr_edges + call_graph_edges

    # upstream_map_full: for candidate expansion (all topology edges)
    upstream_map_full, upstream_map_residual, edge_conf = _upstream_maps(
        svc_names, lag_corr_edges, call_graph_edges, svc_anom_dict, config
    )

    # ── 5. Pre-filter with upstream rescue, then bound runtime ───────────────
    cand_list = _candidate_services(seed_svcs, upstream_map_full, svc_anom_dict, config)

    # ── 7. Weighted residual + topology matrix ───────────────────────────────
    # Scale effective confidence by 2× (cap 0.95) so call_graph (0.5→0.95) fully
    # suppresses victims while weak lag-corr edges (0.1→0.2) still matter.
    svc_residual = _residual_scores(
        svc_anom_dict, upstream_map_residual, edge_conf, config
    )

    # topo_rel uses all topology edges (lag-corr + call-graph) so that systems
    # with rich call graphs get higher res_w → stronger residual weighting.
    fitness_signal = _fitness_signal(
        cand_list, svc_anom_dict, svc_residual, svc_temporal, topology_edges, config
    )
    topo_matrix = _topology_matrix(cand_list, topology_edges)
    _mark_stage("topology")

    # ── 8. Banzhaf power index ────────────────────────────────────────────────
    banzhaf_values = _banzhaf_power(
        fitness_signal,
        topo_matrix,
        config,
    )
    svc_banzhaf = {cand_list[i]: float(banzhaf_values[i]) for i in range(len(cand_list))}
    _mark_stage("banzhaf")

    # ── 9. Final metric-level ranking ────────────────────────────────────────
    ranks = _rank_metrics(
        cols,
        svc_banzhaf,
        svc_residual,
        svc_anom_dict,
        svc_temporal,
        metric_score,
        trace_scores,
    )
    _mark_stage("ranking")

    result = {"ranks": ranks, "node_names": cols}
    if return_stage_timing:
        result["stage_timing"] = stage_timing
    return result


# ==========================================
# RCAEval entry point
# ==========================================

def gabanzhaf(data, inject_time=None, dataset=None, num_loop=None,
              sli=None, anomalies=None, **kwargs):
    """Approximate GA-Banzhaf with metric evidence and weighted topology.

    The default deterministic Banzhaf backend is used for small and large
    systems. MC is available only by explicitly setting banzhaf_mode="mc".
    Trace signals are folded into the fixed service score formula as zero-valued
    evidence when unavailable, instead of switching ranking formulas.

    Final ranking keeps metric-local evidence so metric-level AC@k is not
    collapsed to a service-only score.
    """
    args_ns  = kwargs.get("args")
    case_dir = (os.path.dirname(args_ns.data_path)
                if args_ns and hasattr(args_ns, "data_path") else None)
    _t_inject    = inject_time if inject_time is not None else 0
    trace_features = _extract_trace_features(case_dir, _t_inject)

    runtime_kwargs = dict(kwargs)
    runtime_kwargs["trace_scores"] = trace_features.trace_scores
    runtime_kwargs["call_graph"] = trace_features.call_graph
    runtime_kwargs["sli"] = sli
    algorithm_start = perf_counter()
    result = _gabanzhaf_core(data, inject_time, dataset, anomalies, runtime_kwargs)
    result["algorithm_time_sec"] = float(perf_counter() - algorithm_start)
    return result
