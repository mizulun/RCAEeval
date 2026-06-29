import os
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
import warnings
from dataclasses import dataclass
from time import perf_counter

warnings.filterwarnings("ignore")

from RCAEval.io.time_series import preprocess


@dataclass(frozen=True)
class _TraceFeatures:
    trace_scores: dict | None
    trace_error_scores: dict | None
    trace_latency_scores: dict | None
    call_graph: dict


@dataclass(frozen=True)
class _GabanzhafConfig:
    """Centralized runtime configuration for the GA-Banzhaf pipeline."""

    top_k_svcs: int = 15
    dk_select_useful: bool = False
    topology_pool_multiplier: int = 2
    topology_pool_extra: int = 5
    call_graph_confidence: float = 0.5
    call_graph_callee_min_score: float = 0.2
    call_graph_caller_min_score: float = 0.1
    residual_call_graph_ratio: float = 0.8
    residual_confidence_scale: float = 2.0
    residual_confidence_cap: float = 0.95
    exact_candidate_k: int = 12
    causal_probability_lambda: float = 1.5
    metric_rank_epsilon: float = 0.15
    early_window_ratio: float = 0.25
    early_window_min_len: int = 5
    onset_threshold: float = 1.5
    onset_consecutive: int = 2
    use_trace_scores: bool = True
    use_call_graph: bool = True
    use_residual: bool = True

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

    @classmethod
    def from_kwargs(cls, kwargs):
        top_k_svcs = int(kwargs.get("top_k_svcs", cls.top_k_svcs))
        return cls(
            top_k_svcs=top_k_svcs,
            dk_select_useful=bool(kwargs.get("dk_select_useful", cls.dk_select_useful)),
            exact_candidate_k=cls._positive_int_or_default(
                kwargs.get("exact_candidate_k", cls.exact_candidate_k),
                cls.exact_candidate_k,
            ),
            causal_probability_lambda=float(
                kwargs.get("causal_probability_lambda", cls.causal_probability_lambda)
            ),
            metric_rank_epsilon=float(kwargs.get("metric_rank_epsilon", cls.metric_rank_epsilon)),
            early_window_ratio=float(kwargs.get("early_window_ratio", cls.early_window_ratio)),
            early_window_min_len=cls._positive_int_or_default(
                kwargs.get("early_window_min_len", cls.early_window_min_len),
                cls.early_window_min_len,
            ),
            onset_threshold=float(kwargs.get("onset_threshold", cls.onset_threshold)),
            onset_consecutive=cls._positive_int_or_default(
                kwargs.get("onset_consecutive", cls.onset_consecutive),
                cls.onset_consecutive,
            ),
            use_trace_scores=bool(
                kwargs.get("use_trace_scores", cls.use_trace_scores)
            ),
            use_call_graph=bool(kwargs.get("use_call_graph", cls.use_call_graph)),
            use_residual=bool(kwargs.get("use_residual", cls.use_residual)),
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

    combined: dict = {}
    error_scores: dict = {}
    latency_scores: dict = {}
    for svc, row in a_stats.iterrows():
        svc = str(svc)
        a_cnt = max(int(row["cnt"]), 1)
        err_rate = float(row["err_cnt"]) / a_cnt
        a_avg_dur = float(row["dur_sum"]) / a_cnt
        n_avg_dur = n_dur_avg.get(svc, 1.0)
        dur_ratio = (a_avg_dur / max(n_avg_dur, 1.0)) - 1.0
        dur_signal = max(0.0, min(dur_ratio, 10.0)) / 10.0
        combined[svc] = max(err_rate, dur_signal)
        error_scores[svc] = err_rate
        latency_scores[svc] = dur_signal

    if not combined:
        return None

    return (
        _norm_dict(combined),
        _norm_trace_component_dict(error_scores),
        _norm_trace_component_dict(latency_scores),
    )


def _trace_service_from_timeseries_col(col):
    if col == "time":
        return None
    return str(col).rsplit("_", 1)[0]


def _call_graph_from_span_frame(graph_df):
    if graph_df.empty:
        return {}

    if "traceID" in graph_df.columns:
        span_to_svc = {
            (str(trace_id), span_id): service
            for trace_id, span_id, service in zip(
                graph_df["traceID"], graph_df["spanID"], graph_df["serviceName"]
            )
        }
        parent_svc = [
            span_to_svc.get((str(trace_id), parent_id))
            for trace_id, parent_id in zip(graph_df["traceID"], graph_df["parentSpanID"])
        ]
    else:
        span_to_svc = graph_df.set_index("spanID")["serviceName"].to_dict()
        parent_svc = graph_df["parentSpanID"].map(span_to_svc)

    edges = graph_df.assign(parentSvc=parent_svc)
    edges = edges[
        edges["parentSvc"].notna() & (edges["parentSvc"] != edges["serviceName"])
    ]
    result: dict = {}
    for caller, callee in zip(edges["parentSvc"], edges["serviceName"]):
        result.setdefault(caller, set()).add(callee)
    return result


def _preaggregated_trace_scores(case_dir, inject_time):
    combined: dict = {}
    error_scores: dict = {}
    latency_scores: dict = {}

    def _add_score(scores, svc, value):
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
                score = float(values.mean())
                _add_score(error_scores, svc, score)
                _add_score(combined, svc, score)
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
                _add_score(latency_scores, svc, dur_signal)
                _add_score(combined, svc, dur_signal)
        except Exception:
            pass

    if not combined:
        return None
    return (
        _norm_dict(combined),
        _norm_trace_component_dict(error_scores) if error_scores else None,
        _norm_trace_component_dict(latency_scores) if latency_scores else None,
    )


def _extract_trace_features(case_dir, inject_time):
    """Extract trace onset scores and service call graph with one traces.csv scan."""
    if not case_dir:
        return _TraceFeatures(
            trace_scores=None,
            trace_error_scores=None,
            trace_latency_scores=None,
            call_graph={},
        )

    traces_path = os.path.join(case_dir, "traces.csv")
    fast_trace_scores = _preaggregated_trace_scores(case_dir, inject_time)
    if fast_trace_scores is not None:
        trace_scores, trace_error_scores, trace_latency_scores = fast_trace_scores
        return _TraceFeatures(
            trace_scores=trace_scores,
            trace_error_scores=trace_error_scores,
            trace_latency_scores=trace_latency_scores,
            call_graph=_extract_call_graph(traces_path),
        )

    if not os.path.exists(traces_path):
        return _TraceFeatures(
            trace_scores=None,
            trace_error_scores=None,
            trace_latency_scores=None,
            call_graph={},
        )

    try:
        inject_ms = inject_time * 1000
        score_cols = {"serviceName", "startTimeMillis", "statusCode", "duration"}
        span_cols = {"serviceName", "spanID", "parentSpanID"}
        optional_span_cols = {"traceID"}
        needed = score_cols | span_cols | optional_span_cols
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
                graph_cols = ["serviceName", "spanID", "parentSpanID"]
                if "traceID" in chunk.columns:
                    graph_cols.append("traceID")
                graph_chunk = chunk.loc[:, graph_cols].head(remaining)
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
            call_graph = _call_graph_from_span_frame(graph_df)

        trace_tuple = (
            _trace_scores_from_chunks(chunks_normal, chunks_anomal)
            if saw_score_cols
            else None
        )
        if trace_tuple is None:
            trace_scores = trace_error_scores = trace_latency_scores = None
        else:
            trace_scores, trace_error_scores, trace_latency_scores = trace_tuple
        return _TraceFeatures(
            trace_scores=trace_scores,
            trace_error_scores=trace_error_scores,
            trace_latency_scores=trace_latency_scores,
            call_graph=call_graph,
        )
    except Exception:
        return _TraceFeatures(
            trace_scores=None,
            trace_error_scores=None,
            trace_latency_scores=None,
            call_graph={},
        )



def _extract_call_graph(traces_path):
    """Extract service call graph from the first ``GABANZHAF_CG_NROWS`` spans.

    Reads only spanID / parentSpanID / serviceName.  The row cap defaults to
    10 K (override via the GABANZHAF_CG_NROWS env var).  10 K was verified to
    yield byte-identical AC@1/AC@3/AC@5/Avg@5 to the previous 50 K cap across
    all benchmarked datasets and fault types, while roughly halving the
    per-case trace cost — the call pairs beyond 10 K spans do not change the
    ranking.

    Returns
    -------
    dict  {caller_svc: set(callee_svc, ...)}
        "caller calls callee" — i.e. if callee is anomalous it *explains*
        caller's symptoms (anomaly propagates caller ← callee).
    """
    if not traces_path or not os.path.exists(traces_path):
        return {}
    try:
        _cg_nrows = int(os.environ.get("GABANZHAF_CG_NROWS", "10000"))
        df = pd.read_csv(
            traces_path,
            usecols=lambda col: col
            in {"serviceName", "spanID", "parentSpanID", "traceID"},
            nrows=_cg_nrows,
            on_bad_lines="skip",
        )
        if not {"serviceName", "spanID", "parentSpanID"}.issubset(df.columns):
            return {}
        return _call_graph_from_span_frame(df)
    except Exception:
        return {}


# ==========================================
# Internal helpers
# ==========================================


def _cheap_service_filter(cols, normal_mat, anomal_mat, top_k):
    """Fast service-level pre-filter resistant to constant-baseline inflation.

    Uses mean-shift normalized by max(col_IQR, col_median * 0.1), so a
    perfectly-flat high-value metric (e.g. redis_diskio always at 5000)
    needs a proportionally large shift to score high, instead of being
    inflated by the IQR=0 → denominator=1 fallback in _metric_evidence.

    Returns list of up to top_k service names, sorted by pre-filter score.
    """
    normal_mat = np.asarray(normal_mat, dtype=float)
    anomal_mat = np.asarray(anomal_mat, dtype=float)

    q75 = np.nanpercentile(normal_mat, 75, axis=0)
    q25 = np.nanpercentile(normal_mat, 25, axis=0)
    col_iqr = q75 - q25                                    # (M,)

    normal_mean = np.nanmean(normal_mat, axis=0)           # (M,)
    anomal_mean = np.nanmean(anomal_mat, axis=0)           # (M,)

    # Two-level denominator prevents constant-baseline inflation:
    #   1. baseline_floor: 10% of the metric's own level — stops high-value
    #      flat metrics (e.g. diskio always at 5000) from inflating on tiny bumps.
    #   2. global_iqr_floor: 25th-percentile of positive IQRs across all
    #      metrics — stops near-zero flat metrics (e.g. idle bytes at ~0)
    #      from inflating via the 1e-6 absolute minimum.
    baseline_floor = np.abs(normal_mean) * 0.10
    positive_iqrs = col_iqr[col_iqr > 0]
    global_iqr_floor = float(np.percentile(positive_iqrs, 25)) if positive_iqrs.size > 0 else 1.0
    denom = np.maximum(col_iqr, baseline_floor)
    denom = np.maximum(denom, global_iqr_floor)

    full_score = np.abs(anomal_mean - normal_mean) / denom  # (M,)
    early_len = max(int(round(anomal_mat.shape[0] * _GabanzhafConfig.early_window_ratio)), 1)
    if early_len < _GabanzhafConfig.early_window_min_len:
        early_score = full_score
    else:
        early_mean = np.nanmean(anomal_mat[:early_len], axis=0)
        early_score = np.abs(early_mean - normal_mean) / denom
    col_score = 0.60 * early_score + 0.40 * full_score

    svc_scores: dict = {}
    for i, col in enumerate(cols):
        svc = _parse_svc(col)
        svc_scores[svc] = max(svc_scores.get(svc, 0.0), float(col_score[i]))

    ranked = sorted(svc_scores, key=lambda s: svc_scores[s], reverse=True)
    return ranked[:top_k]


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


def _norm_trace_component_dict(d):
    lo = min(d.values(), default=0.0)
    hi = max(d.values(), default=1.0)
    if hi > 0.0 and lo == hi:
        return {k: max(0.0, min(float(v) / hi, 1.0)) for k, v in d.items()}
    span = hi - lo if hi > lo else 1.0
    return {k: max(0.0, min((float(v) - lo) / span, 1.0)) for k, v in d.items()}


def _first_sustained_onset(values, threshold=1.5, consecutive=2):
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.size == 0:
        return np.full(arr.shape[1] if arr.ndim == 2 else 0, 0, dtype=int)
    n_rows, n_cols = arr.shape
    run_len = max(1, int(consecutive))
    sentinel = n_rows + 1
    if n_rows < run_len:
        return np.full(n_cols, sentinel, dtype=int)

    active = arr > float(threshold)
    # window_all[s, j] is True when active[s : s + run_len, j] are all True.
    windows = sliding_window_view(active, run_len, axis=0)  # (n_starts, n_cols, run_len)
    window_all = windows.all(axis=2)                        # (n_starts, n_cols)
    has_run = window_all.any(axis=0)                        # (n_cols,)
    first_start = window_all.argmax(axis=0)                 # smallest qualifying start
    return np.where(has_run, first_start, sentinel).astype(int)


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
    full_raw = 0.55 * mag + 0.30 * persistence + 0.15 * peak
    full = _norm01_arr(full_raw)
    early_mag = _norm01_arr(np.nanpercentile(early_rz, 95, axis=0))
    early_peak = _norm01_arr(np.nanmax(early_rz, axis=0))
    early_raw = 0.70 * early_mag + 0.30 * early_peak
    early = full.copy() if use_full_for_early else _norm01_arr(early_raw)
    onset_idx = _first_sustained_onset(
        rz,
        threshold=config.onset_threshold,
        consecutive=config.onset_consecutive,
    )
    raw = 0.60 * early + 0.40 * full
    score = _norm01_arr(raw)

    return {
        col: {
            "score": float(score[i]),
            "full": float(full[i]),
            "early": float(early[i]),
            "onset_idx": int(onset_idx[i]),
            "mag": float(mag[i]),
            "persistence": float(persistence[i]),
            "peak": float(peak[i]),
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
        svc: 0.70 * svc_agg[svc] + 0.30 * dom_norm.get(svc, 0.)
        for svc in svc_agg
    }
    return _norm_dict(svc_scores)


def _service_metric_component(cols, metric_ev, key, reducer="top"):
    svc_to_values: dict = {}
    for col in cols:
        svc_to_values.setdefault(_parse_svc(col), []).append(
            _finite_nonnegative(metric_ev.get(col, {}).get(key, 0.0))
        )
    result = {}
    for svc, values in svc_to_values.items():
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            result[svc] = 0.0
        elif reducer == "min":
            result[svc] = float(np.min(arr))
        else:
            top = np.sort(arr)[-min(3, len(arr)):]
            result[svc] = float(0.60 * np.max(arr) + 0.40 * np.mean(top))
    return result



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


def _banzhaf_power(prob_matrix, anomaly_weights, config):
    return _probabilistic_banzhaf(prob_matrix, anomaly_weights)



def _build_topology(svc_names, svc_series, corr_thr=0.08, max_in=3):
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


def _trace_seed_services(svc_names, trace_scores, top_k):
    trace_source = trace_scores or {}
    observed = [svc for svc in svc_names if svc in trace_source]
    return sorted(
        observed,
        key=lambda svc: (-_finite_nonnegative(trace_source.get(svc, 0.0)), svc),
    )[:max(0, int(top_k))]


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


def _call_graph_residual_allowed(u, v, svc_anom_dict, config, svc_onset=None, svc_early=None):
    if svc_anom_dict.get(u, 0.) < svc_anom_dict.get(v, 0.) * config.residual_call_graph_ratio:
        return False
    if svc_onset is None or svc_early is None:
        return True

    u_onset = svc_onset.get(u)
    v_onset = svc_onset.get(v)
    if u_onset is None or v_onset is None:
        return True
    if u_onset < v_onset:
        return True
    if u_onset == v_onset:
        return svc_early.get(u, 0.0) > svc_early.get(v, 0.0)
    return False


def _upstream_maps(
    svc_names,
    lag_corr_edges,
    call_graph_edges,
    svc_anom_dict,
    config,
    svc_onset=None,
    svc_early=None,
):
    upstream_map_full = {s: set() for s in svc_names}
    upstream_map_residual = {s: set() for s in svc_names}
    edge_conf = {}
    call_edge_set = {(u, v) for u, v, _ in call_graph_edges}

    for u, v, w in lag_corr_edges + call_graph_edges:
        upstream_map_full.get(v, set()).add(u)
        edge_conf[(u, v)] = max(edge_conf.get((u, v), 0.), w)
    for u, v, _ in lag_corr_edges:
        if (v, u) not in call_edge_set:
            upstream_map_residual.get(v, set()).add(u)
    for u, v, _ in call_graph_edges:
        if _call_graph_residual_allowed(
            u,
            v,
            svc_anom_dict,
            config,
            svc_onset=svc_onset,
            svc_early=svc_early,
        ):
            upstream_map_residual.get(v, set()).add(u)

    return upstream_map_full, upstream_map_residual, edge_conf


def _candidate_services(seed_svcs, upstream_map_full):
    candidates = set(seed_svcs)
    for svc in list(candidates):
        candidates |= upstream_map_full.get(svc, set())
    return sorted(candidates)


def _select_exact_candidates(
    candidates,
    svc_anom_dict,
    svc_residual,
    trace_scores,
    config,
):
    k = max(1, int(config.exact_candidate_k))
    candidates = list(dict.fromkeys(candidates))
    sources = (svc_anom_dict or {}, svc_residual or {}, trace_scores or {})
    source_ranks = []
    for source in sources:
        present = [svc for svc in candidates if svc in source]
        ranked = sorted(
            present,
            key=lambda svc: (-_finite_nonnegative(source.get(svc, 0.0)), svc),
        )
        ranks = {svc: rank for rank, svc in enumerate(ranked, start=1)}
        missing_rank = len(ranked) + 1
        source_ranks.append((ranks, missing_rank))

    def score(svc):
        return sum(
            1.0 / (10.0 + ranks.get(svc, missing_rank))
            for ranks, missing_rank in source_ranks
        )

    return sorted(candidates, key=lambda svc: (-score(svc), svc))[:k]


def _causal_explanation_matrix(
    cand_list,
    service_list,
    svc_anom_dict,
    topology_edges,
    config,
):
    if len(cand_list) != len(set(cand_list)):
        raise ValueError("Duplicate candidate name")
    if len(service_list) != len(set(service_list)):
        raise ValueError("Duplicate service name")
    cand_idx = {svc: i for i, svc in enumerate(cand_list)}
    svc_idx = {svc: j for j, svc in enumerate(service_list)}
    matrix = np.zeros((len(cand_list), len(service_list)), dtype=float)
    lam = _finite_nonnegative(config.causal_probability_lambda)
    if lam <= 0.0:
        lam = 1.0

    for svc, i in cand_idx.items():
        if svc in svc_idx:
            matrix[i, svc_idx[svc]] = 1.0

    for u, v, w in topology_edges:
        if u not in cand_idx or v not in svc_idx:
            continue
        alignment = np.sqrt(
            _finite_nonnegative(svc_anom_dict.get(u, 0.0))
            * _finite_nonnegative(svc_anom_dict.get(v, 0.0))
        )
        raw = _finite_nonnegative(w) * alignment
        matrix[cand_idx[u], svc_idx[v]] = max(
            matrix[cand_idx[u], svc_idx[v]],
            1.0 - np.exp(-lam * raw),
        )

    return np.clip(matrix, 0.0, 1.0)


def _residual_scores(svc_anom_dict, upstream_map_residual, edge_conf, config):
    if not config.use_residual:
        return dict(svc_anom_dict)
    residual = {}
    for svc, score in svc_anom_dict.items():
        explained = max(
            (
                svc_anom_dict.get(u, 0.)
                * min(
                    edge_conf.get((u, svc), 0.) * config.residual_confidence_scale,
                    config.residual_confidence_cap,
                )
                for u in upstream_map_residual.get(svc, set())
            ),
            default=0.,
        )
        residual[svc] = max(0., score - explained)
    return residual


def _finite_nonnegative(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(value) or value < 0.0:
        return 0.0
    return value


def _rank_metrics(all_cols, svc_banzhaf, metric_score, config):
    finite_banzhaf = {
        svc: _finite_nonnegative(score)
        for svc, score in (svc_banzhaf or {}).items()
    }
    if not finite_banzhaf or len(set(finite_banzhaf.values())) == 1:
        nb = {svc: 1.0 for svc in {_parse_svc(col) for col in all_cols}}
    else:
        max_banzhaf = max(finite_banzhaf.values())
        nb = {
            svc: score / max_banzhaf
            for svc, score in finite_banzhaf.items()
        }
    nm = _norm_dict(metric_score)
    epsilon = _finite_nonnegative(config.metric_rank_epsilon)

    def _col_score(col):
        svc = _parse_svc(col)
        return nb.get(svc, 0.0) * (epsilon + nm.get(col, 0.0))

    return sorted(all_cols, key=_col_score, reverse=True)


def _trace_metric_overlap_diagnostics(
    svc_anom_dict,
    trace_scores,
    top_k,
):
    trace_scores = trace_scores or {}
    shared = sorted(set(svc_anom_dict) & set(trace_scores))
    if len(shared) < 2:
        spearman = 0.0
    else:
        metric_ranks = pd.Series(
            [svc_anom_dict[svc] for svc in shared],
            index=shared,
            dtype=float,
        ).rank(method="average", ascending=False)
        trace_ranks = pd.Series(
            [trace_scores[svc] for svc in shared],
            index=shared,
            dtype=float,
        ).rank(method="average", ascending=False)
        corr = metric_ranks.corr(trace_ranks, method="pearson")
        spearman = float(corr) if np.isfinite(corr) else 0.0

    k = min(max(1, int(top_k)), len(shared)) if shared else 0
    if k == 0:
        overlap = 0.0
    else:
        metric_top = set(sorted(
            shared,
            key=lambda svc: (-_finite_nonnegative(svc_anom_dict[svc]), svc),
        )[:k])
        trace_top = set(sorted(
            shared,
            key=lambda svc: (-_finite_nonnegative(trace_scores[svc]), svc),
        )[:k])
        overlap = len(metric_top & trace_top) / k
    return spearman, float(overlap)


def _raw_call_graph_edge_count(call_graph):
    return sum(len(callees) for callees in (call_graph or {}).values())


# ==========================================
# Core pipeline
# ==========================================


def _gabanzhaf_core(data, inject_time, dataset, anomalies, kwargs):
    """
    Pipeline: window split → metric evidence → service evidence
              → bounded weighted topology → Banzhaf → metric ranking
    """
    config = _GabanzhafConfig.from_kwargs(kwargs)
    supplied_trace_scores = kwargs.get("trace_scores", None)
    trace_scores = supplied_trace_scores if config.use_trace_scores else None
    supplied_call_graph = kwargs.get(
        "raw_call_graph",
        kwargs.get("call_graph", {}),
    )
    active_call_graph = kwargs.get("call_graph", supplied_call_graph)
    call_graph = active_call_graph if config.use_call_graph else {}
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
    # Optional pre-filter: run cheap_service_filter first to restrict
    # metric_evidence to candidate services only, preventing constant-baseline
    # metrics (e.g. always-flat diskio) from inflating IQR-based anomaly scores.
    use_prefilter = bool(kwargs.get("use_prefilter", False))
    if use_prefilter:
        pre_svcs = set(_cheap_service_filter(cols, normal_mat, anomal_mat, top_k=config.top_k_svcs))
        active_cols = [c for c in cols if _parse_svc(c) in pre_svcs]
        active_idxs = [i for i, c in enumerate(cols) if c in set(active_cols)]
        active_normal = normal_mat[:, active_idxs]
        active_anomal = anomal_mat[:, active_idxs]
    else:
        active_cols, active_normal, active_anomal = cols, normal_mat, anomal_mat

    metric_ev = _metric_evidence(active_normal, active_anomal, active_cols, config=config)
    metric_score = {c: metric_ev[c]["score"] for c in active_cols}
    svc_anom_dict = _service_evidence(active_cols, metric_ev)
    svc_early = _norm_dict(_service_metric_component(active_cols, metric_ev, "early"))
    svc_onset = _service_metric_component(active_cols, metric_ev, "onset_idx", reducer="min")
    svc_names = sorted(svc_anom_dict.keys())
    _mark_stage("metric_evidence")

    # ── 4. Candidate-first bounded topology ──────────────────────────────────
    svc_ranked = sorted(svc_names, key=lambda s: svc_anom_dict.get(s, 0.), reverse=True)
    metric_seed_svcs = set(svc_ranked[: config.top_k_svcs])
    trace_seed_svcs = set(
        _trace_seed_services(svc_names, trace_scores, config.top_k_svcs)
    )
    seed_svcs = metric_seed_svcs | trace_seed_svcs

    pool_svcs = _initial_service_pool(svc_ranked, seed_svcs, config)
    svc_series_full = _service_series_for_pool(cols, no_time, pool_svcs)
    lag_corr_edges = _build_topology(sorted(pool_svcs), svc_series_full)

    # ── 4b. Augment topology with trace-derived call graph ────────────────────
    # Call graph: {caller: {callee}}.  When callee is anomalous it *explains*
    # caller's symptoms → topology edge: callee → caller (callee upstream of caller).
    # Confidence is moderate; both callee AND caller must show meaningful anomaly
    # to avoid adding spurious edges in non-propagating faults.
    call_graph_edges = _call_graph_topology_edges(
        lag_corr_edges,
        call_graph,
        pool_svcs,
        svc_anom_dict,
        config,
    )

    topology_edges = lag_corr_edges + call_graph_edges

    # upstream_map_full: for candidate expansion (all topology edges)
    upstream_map_full, upstream_map_residual, edge_conf = _upstream_maps(
        svc_names,
        lag_corr_edges,
        call_graph_edges,
        svc_anom_dict,
        config,
        svc_onset=svc_onset,
        svc_early=svc_early,
    )

    # ── 5. Pre-filter with upstream rescue, then bound runtime ───────────────
    cand_list = _candidate_services(seed_svcs, upstream_map_full)

    # ── 7. Weighted residual + probabilistic causal game ─────────────────────
    # Scale effective confidence by 2× (cap 0.95) so call_graph (0.5→0.95) fully
    # suppresses victims while weak lag-corr edges (0.1→0.2) still matter.
    svc_residual = _residual_scores(
        svc_anom_dict, upstream_map_residual, edge_conf, config
    )

    game_candidates = _select_exact_candidates(
        cand_list,
        svc_anom_dict,
        svc_residual,
        trace_scores,
        config,
    )
    game_services = sorted(svc_anom_dict.keys())
    prob_matrix = _causal_explanation_matrix(
        game_candidates,
        game_services,
        svc_anom_dict,
        topology_edges,
        config,
    )
    anomaly_weights = np.asarray(
        [svc_anom_dict.get(svc, 0.0) for svc in game_services],
        dtype=float,
    )
    _mark_stage("topology")

    # ── 8. Banzhaf power index ────────────────────────────────────────────────
    banzhaf_values = _banzhaf_power(prob_matrix, anomaly_weights, config)
    svc_banzhaf = {
        game_candidates[i]: float(banzhaf_values[i])
        for i in range(len(game_candidates))
    }
    _mark_stage("banzhaf")

    # ── 9. Final metric-level ranking ────────────────────────────────────────
    ranks = _rank_metrics(cols, svc_banzhaf, metric_score, config)
    _mark_stage("ranking")

    result = {"ranks": ranks, "node_names": cols}
    if bool(kwargs.get("return_diagnostics", False)):
        trace_metric_spearman, trace_metric_top_k_overlap = (
            _trace_metric_overlap_diagnostics(
                svc_anom_dict,
                trace_scores,
                config.top_k_svcs,
            )
        )
        game_candidate_set = set(game_candidates)
        game_service_idx = {svc: i for i, svc in enumerate(game_services)}
        effective_call_edges = [
            (u, v, w)
            for u, v, w in call_graph_edges
            if (
                u in game_candidate_set
                and v in game_service_idx
                and prob_matrix[game_candidates.index(u), game_service_idx[v]] > 0.0
            )
        ]
        root_service = kwargs.get("root_service")
        diagnostics = {
            "trace_metric_spearman": trace_metric_spearman,
            "trace_metric_top_k_overlap": trace_metric_top_k_overlap,
            "metric_seed_services": sorted(metric_seed_svcs),
            "trace_seed_services": sorted(trace_seed_svcs),
            "trace_added_seed_count": len(trace_seed_svcs - metric_seed_svcs),
            "raw_call_graph_edge_count": _raw_call_graph_edge_count(
                supplied_call_graph
            ),
            "accepted_call_graph_edge_count": len(call_graph_edges),
            "nonduplicate_call_graph_edge_count": len(call_graph_edges),
            "effective_call_graph_edge_count": len(effective_call_edges),
            "lag_edge_count": len(lag_corr_edges),
            "candidate_pool": list(cand_list),
            "game_candidates": list(game_candidates),
            "game_services": list(game_services),
            "root_service": root_service,
            "root_candidate_recall": (
                root_service in game_candidate_set
                if root_service is not None
                else None
            ),
            "metric_service_evidence": dict(svc_anom_dict),
            "trace_service_evidence": {
                svc: _finite_nonnegative((trace_scores or {}).get(svc, 0.0))
                for svc in svc_names
            },
            "residual_service_evidence": dict(svc_residual),
            "service_banzhaf": dict(svc_banzhaf),
            "lag_edges": [list(edge) for edge in lag_corr_edges],
            "call_graph_edges": [list(edge) for edge in call_graph_edges],
            "causal_probability_matrix": prob_matrix.tolist(),
        }
        result["diagnostics"] = diagnostics
        result["game_candidates"] = game_candidates
        result["metric_service_evidence"] = dict(svc_anom_dict)
        result["trace_service_evidence"] = {
            svc: _finite_nonnegative((trace_scores or {}).get(svc, 0.0))
            for svc in svc_names
        }
        result["residual_service_evidence"] = dict(svc_residual)
    if return_stage_timing:
        result["stage_timing"] = stage_timing
    return result


# ==========================================
# RCAEval entry point
# ==========================================

def gabanzhaf(data, inject_time=None, dataset=None, num_loop=None,
              sli=None, anomalies=None, **kwargs):
    """Approximate GA-Banzhaf with metric evidence and weighted topology.

    The service game uses a single probabilistic noisy-OR Banzhaf flow.
    Trace signals are folded into the fixed service score formula as zero-valued
    evidence when unavailable, instead of switching ranking formulas.

    Final ranking keeps metric-local evidence so metric-level AC@k is not
    collapsed to a service-only score.

    Set env var GABANZHAF_PROFILE=1 to print per-case stage timing to stderr.
    Stages: trace | preprocess | metric_evidence | topology | banzhaf | ranking
    """
    import sys
    _profile = os.environ.get("GABANZHAF_PROFILE") == "1"
    disable_trace = os.environ.get("GABANZHAF_TRACE_DISABLE_TRACE") == "1"
    use_trace_scores = not (
        disable_trace
        or os.environ.get("GABANZHAF_TRACE_DISABLE_SCORES") == "1"
    )
    use_call_graph = not (
        disable_trace
        or os.environ.get("GABANZHAF_TRACE_DISABLE_CALL_GRAPH") == "1"
    )
    use_residual = os.environ.get("GABANZHAF_TRACE_DISABLE_RESIDUAL") != "1"

    args_ns  = kwargs.get("args")
    case_dir = (os.path.dirname(args_ns.data_path)
                if args_ns and hasattr(args_ns, "data_path") else None)
    _t_inject = inject_time if inject_time is not None else 0

    _t0 = perf_counter()
    trace_features = (
        _extract_trace_features(case_dir, _t_inject)
        if use_trace_scores or use_call_graph
        else _TraceFeatures(None, None, None, {})
    )
    _trace_sec = perf_counter() - _t0

    runtime_kwargs = dict(kwargs)
    runtime_kwargs.setdefault("use_trace_scores", use_trace_scores)
    runtime_kwargs.setdefault("use_call_graph", use_call_graph)
    runtime_kwargs.setdefault("use_residual", use_residual)
    # Experiment flag: GABANZHAF_PREFILTER=1 enables early candidate selection.
    if os.environ.get("GABANZHAF_PREFILTER") == "1":
        runtime_kwargs.setdefault("use_prefilter", True)
    if _profile:
        runtime_kwargs["return_stage_timing"] = True
    if os.environ.get("GABANZHAF_WRITE_DIAGNOSTICS") == "1":
        runtime_kwargs["return_diagnostics"] = True
        runtime_kwargs["return_stage_timing"] = True
    runtime_kwargs["trace_scores"] = (
        trace_features.trace_scores if use_trace_scores else None
    )
    runtime_kwargs["trace_error_scores"] = (
        trace_features.trace_error_scores if use_trace_scores else None
    )
    runtime_kwargs["trace_latency_scores"] = (
        trace_features.trace_latency_scores if use_trace_scores else None
    )
    runtime_kwargs["call_graph"] = (
        trace_features.call_graph if use_call_graph else {}
    )
    runtime_kwargs["raw_call_graph"] = trace_features.call_graph
    runtime_kwargs["sli"] = sli
    if case_dir:
        fault_dir = os.path.basename(os.path.dirname(case_dir))
        runtime_kwargs["root_service"] = fault_dir.split("_", 1)[0]

    result = _gabanzhaf_core(data, inject_time, dataset, anomalies, runtime_kwargs)
    if "diagnostics" in result:
        result["diagnostics"]["trace_extraction_sec"] = float(_trace_sec)
        result["diagnostics"]["stage_timing"] = dict(
            result.get("stage_timing", {})
        )

    if _profile and "stage_timing" in result:
        st = result["stage_timing"]
        core_total = sum(st.values())
        grand_total = _trace_sec + core_total
        label = case_dir or str(dataset) or "?"
        parts = (
            [f"trace={_trace_sec:.3f}s"]
            + [f"{k}={v:.3f}s" for k, v in st.items()]
            + [f"core={core_total:.3f}s", f"total={grand_total:.3f}s"]
        )
        print(f"[PROFILE] {label}  " + "  ".join(parts), file=sys.stderr)

    return result
