from __future__ import annotations

import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import grangercausalitytests

from RCAEval.io.time_series import preprocess


_EPS = 1e-9
_TRACE_ROW_LIMIT = 50000


def _to_1d_float(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _norm01_arr(x) -> np.ndarray:
    arr = _to_1d_float(x)
    if arr.size == 0:
        return arr
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if hi - lo < _EPS:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


def _norm_dict(values: Dict[str, float]) -> Dict[str, float]:
    if not values:
        return {}
    keys = list(values.keys())
    arr = np.array([float(values[k]) for k in keys], dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < _EPS:
        return {k: 0.0 for k in keys}
    return {k: float((arr[i] - lo) / (hi - lo)) for i, k in enumerate(keys)}


def _clean_trace_id(value) -> str | None:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    return text


def _parse_service_metric(col: str) -> Tuple[str, str]:
    name = str(col)
    if name == "time":
        return "time", "unknown"
    if "_" not in name:
        return name, "unknown"
    svc, metric = name.rsplit("_", 1)
    return svc, metric or "unknown"


def _service_groups(cols: Iterable[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for col in cols:
        if str(col) == "time":
            continue
        svc, _ = _parse_service_metric(str(col))
        groups.setdefault(svc, []).append(str(col))
    return groups


def _build_service_health_scores(
    df: pd.DataFrame,
    groups: Dict[str, List[str]],
    split_idx: int,
) -> Dict[str, np.ndarray]:
    scores: Dict[str, np.ndarray] = {}
    for svc, cols in groups.items():
        if not cols:
            scores[svc] = np.zeros(len(df), dtype=float)
            continue

        X = (
            df[cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        try:
            X_scaled = StandardScaler().fit_transform(X)
            comp = PCA(n_components=1, random_state=42).fit_transform(X_scaled)[:, 0]
        except Exception:
            comp = np.zeros(X.shape[0], dtype=float)

        split = int(np.clip(split_idx, 1, max(len(comp) - 1, 1)))
        if comp.size and float(np.mean(comp[split:])) < float(np.mean(comp[:split])):
            comp = -comp
        scores[svc] = _norm01_arr(comp)
    return scores


def _stl_decompose_health(
    health: np.ndarray,
    split_idx: int,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    y = _to_1d_float(health)
    if y.size < 6:
        return np.zeros_like(y), np.zeros_like(y), False

    normal_len = max(2, min(int(split_idx), len(y)))
    period = int(min(24, max(2, normal_len // 4)))
    if len(y) < period * 2:
        return np.zeros_like(y), np.zeros_like(y), False

    try:
        res = STL(y, period=period, robust=True).fit()
        base = np.nan_to_num(res.trend + res.seasonal, nan=0.0)
        residual = np.nan_to_num(res.resid, nan=0.0)
        return base, residual, True
    except Exception:
        return np.zeros_like(y), np.zeros_like(y), False


def _lag_features(series: np.ndarray, max_lag: int) -> Tuple[np.ndarray, np.ndarray]:
    y = _to_1d_float(series)
    lag = int(max(1, max_lag))
    if y.size <= lag:
        return np.empty((0, lag), dtype=float), np.empty((0,), dtype=float)
    X = np.column_stack([y[lag - i - 1 : y.size - i - 1] for i in range(lag)])
    target = y[lag:]
    return X, target


def _elasticnet_residual_prediction(
    residual: np.ndarray,
    split_idx: int,
    max_lag: int,
    random_state: int,
) -> np.ndarray:
    r = _to_1d_float(residual)
    pred = np.zeros_like(r, dtype=float)
    lag = int(max(1, min(max_lag, max(split_idx - 2, 1))))
    X, y = _lag_features(r[:split_idx], lag)
    if len(y) < 4 or X.shape[0] < 4:
        return pred
    try:
        model = ElasticNet(alpha=0.01, l1_ratio=0.2, max_iter=1000, random_state=random_state)
        model.fit(X, y)
        X_all, _ = _lag_features(r, lag)
        if X_all.size:
            pred[lag:] = model.predict(X_all)
    except Exception:
        return np.zeros_like(r, dtype=float)
    return np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)


def _forecast_service(
    health: np.ndarray,
    split_idx: int,
    max_lag: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    y = _to_1d_float(health)
    base, residual, stl_ok = _stl_decompose_health(y, split_idx)
    if not stl_ok:
        return np.zeros_like(y), np.zeros_like(y), 0.0
    residual_pred = _elasticnet_residual_prediction(residual, split_idx, max_lag, random_state)
    combined = base + residual_pred

    normal_err = np.abs(y[:split_idx] - combined[:split_idx])
    anomal_err = np.abs(y[split_idx:] - combined[split_idx:])
    if anomal_err.size == 0:
        return combined, residual, 0.0

    med = float(np.median(normal_err)) if normal_err.size else 0.0
    q75 = float(np.percentile(normal_err, 75)) if normal_err.size else 1.0
    q25 = float(np.percentile(normal_err, 25)) if normal_err.size else 0.0
    denom = max(q75 - q25, float(np.std(normal_err)) if normal_err.size else 0.0, _EPS)
    rz = np.nan_to_num(
        np.maximum(0.0, (anomal_err - med) / denom),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    p95 = float(np.percentile(rz, 95)) if rz.size else 0.0
    persistence = float(np.mean(rz > 3.0)) if rz.size else 0.0
    peak = float(np.max(rz)) if rz.size else 0.0
    score = 0.55 * p95 + 0.30 * persistence + 0.15 * peak
    return combined, residual, float(score)


def _granger_edge_weight(x: np.ndarray, y: np.ndarray, max_lag: int, alpha: float) -> float:
    src = _to_1d_float(x)
    dst = _to_1d_float(y)
    n = min(src.size, dst.size)
    if n < max(8, max_lag * 4) or np.std(src[:n]) < _EPS or np.std(dst[:n]) < _EPS:
        return 0.0
    lag = int(max(1, min(max_lag, (n - 2) // 3)))
    if lag < 1:
        return 0.0
    try:
        arr = np.column_stack([dst[:n], src[:n]])
        result = grangercausalitytests(arr, maxlag=lag, verbose=False)
        pvals = [float(result[l][0]["ssr_ftest"][1]) for l in result]
        best_p = min(pvals)
        if not np.isfinite(best_p):
            return 0.0
        return float(np.clip(1.0 - best_p / max(alpha, _EPS), 0.0, 1.0))
    except Exception:
        return 0.0


def _mc_banzhaf(
    anomaly: np.ndarray,
    adj: np.ndarray,
    n_samples: int = 1000,
    random_state: int = 42,
) -> np.ndarray:
    score = _to_1d_float(anomaly)
    A = np.nan_to_num(np.asarray(adj, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    n = len(score)
    if n == 0:
        return np.zeros(0, dtype=float)
    rng = np.random.default_rng(random_state)
    values = np.zeros(n, dtype=float)

    def utility(mask: np.ndarray) -> float:
        if not mask.any():
            return 0.0
        explained = mask.astype(float).copy()
        upstream = np.where(mask)[0]
        if upstream.size:
            explained = np.maximum(explained, np.max(A[upstream, :], axis=0))
        return float(np.dot(score, explained))

    for _ in range(int(max(1, n_samples))):
        mask = rng.random(n) < 0.5
        for i in range(n):
            with_i = mask.copy()
            without_i = mask.copy()
            with_i[i] = True
            without_i[i] = False
            values[i] += utility(with_i) - utility(without_i)

    return np.maximum(0.0, values / float(max(1, n_samples)))


def _split_windows(
    data: pd.DataFrame,
    inject_time,
    anomalies,
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    if anomalies is not None:
        split = int(anomalies[0])
        normal = data.head(split)
        anomal = data.tail(len(data) - split)
    elif inject_time is not None and "time" in data.columns:
        normal = data[data["time"] < inject_time]
        anomal = data[data["time"] >= inject_time]
        split = len(normal)
    else:
        split = len(data) // 2
        normal = data.head(split)
        anomal = data.tail(len(data) - split)
    return normal.reset_index(drop=True), anomal.reset_index(drop=True), int(split)


def _preprocess_joint(
    normal: pd.DataFrame,
    anomal: pd.DataFrame,
    dataset,
    dk_select_useful: bool,
) -> Tuple[pd.DataFrame, int]:
    n_normal = len(normal)
    full = pd.concat([normal, anomal], ignore_index=True)
    full_proc = preprocess(full, dataset, dk_select_useful)
    normal_proc = full_proc.iloc[:n_normal].reset_index(drop=True)
    anomal_proc = full_proc.iloc[n_normal:].reset_index(drop=True)
    shared = [c for c in normal_proc.columns if c in anomal_proc.columns]
    normal_proc = normal_proc[shared]
    anomal_proc = anomal_proc[shared]
    full_proc = pd.concat([normal_proc, anomal_proc], ignore_index=True)
    return full_proc, len(normal_proc)


def _metric_evidence(
    normal_df: pd.DataFrame,
    anomal_df: pd.DataFrame,
    cols: List[str],
) -> Dict[str, float]:
    if not cols:
        return {}
    normal = (
        normal_df[cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    anomal = (
        anomal_df[cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    if normal.size == 0 or anomal.size == 0:
        return {c: 0.0 for c in cols}

    med = np.nanmedian(normal, axis=0)
    q75 = np.nanpercentile(normal, 75, axis=0)
    q25 = np.nanpercentile(normal, 25, axis=0)
    std = np.nanstd(normal, axis=0)
    denom = np.maximum.reduce([q75 - q25, std, np.full_like(std, _EPS)])
    rz = np.abs((anomal - med) / denom)
    rz = np.nan_to_num(rz, nan=0.0, posinf=0.0, neginf=0.0)
    p95 = _norm01_arr(np.nanpercentile(rz, 95, axis=0))
    persistence = _norm01_arr(np.mean(rz > 3.0, axis=0))
    peak = _norm01_arr(np.nanmax(rz, axis=0))
    raw = 0.55 * p95 + 0.30 * persistence + 0.15 * peak
    norm = _norm01_arr(raw)
    return {c: float(norm[i]) for i, c in enumerate(cols)}


def _read_trace_edges(case_dir: str | None) -> List[Tuple[str, str, float]]:
    if not case_dir:
        return []
    path = os.path.join(case_dir, "traces.csv")
    if not os.path.exists(path):
        return []
    try:
        cols = set(pd.read_csv(path, nrows=0).columns)
        if {"parentServiceName", "serviceName"}.issubset(cols):
            edges = set()
            for chunk in pd.read_csv(
                path,
                usecols=["parentServiceName", "serviceName"],
                chunksize=10000,
                nrows=_TRACE_ROW_LIMIT,
                dtype=str,
            ):
                parents = chunk["parentServiceName"].map(_clean_trace_id)
                services = chunk["serviceName"].map(_clean_trace_id)
                for caller, callee in zip(parents, services):
                    if caller and callee and caller != callee:
                        edges.add((callee, caller, 1.0))
            return sorted(edges)

        for caller, callee in [
            ("caller_service", "callee_service"),
            ("callerService", "calleeService"),
            ("source", "target"),
            ("src", "dst"),
        ]:
            if {caller, callee}.issubset(cols):
                edges = set()
                for chunk in pd.read_csv(
                    path,
                    usecols=[caller, callee],
                    chunksize=10000,
                    nrows=_TRACE_ROW_LIMIT,
                    dtype=str,
                ):
                    callers = chunk[caller].map(_clean_trace_id)
                    callees = chunk[callee].map(_clean_trace_id)
                    for c, d in zip(callers, callees):
                        if c and d and c != d:
                            edges.add((d, c, 1.0))
                return sorted(edges)

        if {"serviceName", "spanID", "parentSpanID"}.issubset(cols):
            trace_col = next(
                (c for c in ["traceID", "trace_id", "traceId"] if c in cols),
                None,
            )
            usecols = ["serviceName", "spanID", "parentSpanID"]
            if trace_col is not None:
                usecols.append(trace_col)
            df = pd.read_csv(
                path,
                usecols=usecols,
                nrows=_TRACE_ROW_LIMIT,
                dtype=str,
                on_bad_lines="skip",
            )
            for col in usecols:
                df[col] = df[col].map(_clean_trace_id)
            if trace_col is not None:
                span_to_svc = {
                    (trace_id, span_id): svc
                    for trace_id, span_id, svc in zip(
                        df[trace_col],
                        df["spanID"],
                        df["serviceName"],
                    )
                    if trace_id and span_id and svc
                }
                df["parentSvc"] = [
                    span_to_svc.get((trace_id, parent_span_id))
                    for trace_id, parent_span_id in zip(
                        df[trace_col],
                        df["parentSpanID"],
                    )
                ]
            else:
                span_to_svc = {
                    span_id: svc
                    for span_id, svc in zip(df["spanID"], df["serviceName"])
                    if span_id and svc
                }
                df["parentSvc"] = df["parentSpanID"].map(span_to_svc)
            edges = set()
            for caller, callee in zip(df["parentSvc"], df["serviceName"]):
                if caller and callee and caller != callee:
                    edges.add((callee, caller, 1.0))
            return sorted(edges)
    except Exception:
        return []
    return []


def _candidate_services(
    services: List[str],
    anomaly_scores: Dict[str, float],
    sli: str | None,
    trace_edges: List[Tuple[str, str, float]],
    top_k: int,
    max_candidate: int,
) -> List[str]:
    ranked = sorted(services, key=lambda s: (-anomaly_scores.get(s, 0.0), s))
    service_set = set(services)
    cap = max(1, int(max_candidate))
    sli_svc = None
    must_keep = set()
    if sli:
        svc, _ = _parse_service_metric(sli)
        if svc in service_set:
            sli_svc = svc
            must_keep.add(svc)
    for u, v, _ in trace_edges:
        if u in service_set:
            must_keep.add(u)
        if v in service_set:
            must_keep.add(v)

    selected: List[str] = []
    if sli_svc is not None:
        selected.append(sli_svc)

    remaining_must = sorted(
        must_keep - set(selected),
        key=lambda s: (-anomaly_scores.get(s, 0.0), s),
    )
    selected.extend(remaining_must[: max(0, cap - len(selected))])
    if len(selected) < cap:
        fill_ranked = [
            svc
            for svc in ranked[: max(1, int(top_k))]
            if svc not in set(selected)
        ]
        selected.extend(fill_ranked[: cap - len(selected)])
    if len(selected) < cap:
        fallback = [svc for svc in ranked if svc not in set(selected)]
        selected.extend(fallback[: cap - len(selected)])
    return sorted(selected[:cap])


def _build_adjacency(
    candidates: List[str],
    residuals: Dict[str, np.ndarray],
    trace_edges: List[Tuple[str, str, float]],
    max_lag: int,
    alpha: float,
    max_in_edges: int,
    trace_prior: float,
) -> np.ndarray:
    n = len(candidates)
    idx = {s: i for i, s in enumerate(candidates)}
    G = np.zeros((n, n), dtype=float)
    for u in candidates:
        for v in candidates:
            if u == v:
                continue
            G[idx[u], idx[v]] = _granger_edge_weight(
                residuals.get(u, []),
                residuals.get(v, []),
                max_lag,
                alpha,
            )

    for j in range(n):
        parents = np.where(G[:, j] > 0.0)[0]
        if len(parents) > max_in_edges:
            keep = parents[np.argsort(G[parents, j])[-max_in_edges:]]
            drop = set(parents) - set(keep)
            for i in drop:
                G[i, j] = 0.0

    T = np.zeros((n, n), dtype=float)
    for u, v, w in trace_edges:
        if u in idx and v in idx and u != v:
            T[idx[u], idx[v]] = max(T[idx[u], idx[v]], float(w))

    A = np.maximum(G, float(trace_prior) * T)
    np.fill_diagonal(A, 0.0)
    if max_in_edges <= 0:
        return np.zeros_like(A)
    for j in range(n):
        parents = np.where(A[:, j] > 0.0)[0]
        if len(parents) > max_in_edges:
            keep = parents[np.argsort(A[parents, j])[-max_in_edges:]]
            drop = set(parents) - set(keep)
            for i in drop:
                A[i, j] = 0.0
    np.fill_diagonal(A, 0.0)
    return A


def _rank_metrics(
    cols: List[str],
    service_banzhaf: Dict[str, float],
    metric_scores: Dict[str, float],
) -> List[str]:
    nb = _norm_dict(service_banzhaf)
    nm = _norm_dict(metric_scores)

    def score(col: str) -> float:
        svc, _ = _parse_service_metric(col)
        sb = nb.get(svc, 0.0)
        ms = nm.get(col, 0.0)
        return 0.70 * sb + 0.30 * ms + 0.15 * sb * ms

    return sorted(cols, key=lambda c: (-score(c), c))


def predict_banzhaf(
    data,
    inject_time=None,
    dataset=None,
    num_loop=None,
    sli=None,
    anomalies=None,
    **kwargs,
):
    normal_raw, anomal_raw, _ = _split_windows(data, inject_time, anomalies)
    full_proc, split_idx = _preprocess_joint(
        normal_raw,
        anomal_raw,
        dataset,
        bool(kwargs.get("dk_select_useful", False)),
    )

    cols = [c for c in full_proc.columns if c != "time"]
    normal_df = full_proc.iloc[:split_idx].reset_index(drop=True)
    anomal_df = full_proc.iloc[split_idx:].reset_index(drop=True)
    groups = _service_groups(cols)
    services = sorted(groups)

    max_lag = int(kwargs.get("max_lag", 3))
    random_state = int(kwargs.get("random_state", 42))
    health_scores = _build_service_health_scores(full_proc[cols], groups, split_idx)

    residuals: Dict[str, np.ndarray] = {}
    raw_anomaly: Dict[str, float] = {}
    for svc in services:
        health = health_scores.get(svc, np.zeros(len(full_proc)))
        _, residual, score = _forecast_service(
            health,
            split_idx,
            max_lag,
            random_state,
        )
        residuals[svc] = residual
        raw_anomaly[svc] = score
    anomaly_scores = _norm_dict(raw_anomaly)

    args_ns = kwargs.get("args")
    case_dir = None
    if args_ns is not None and hasattr(args_ns, "data_path"):
        case_dir = os.path.dirname(args_ns.data_path)
    trace_edges = _read_trace_edges(case_dir)

    candidates = _candidate_services(
        services,
        anomaly_scores,
        sli,
        trace_edges,
        int(kwargs.get("top_k_svcs", 30)),
        int(kwargs.get("max_candidate_svcs", 40)),
    )
    adj = _build_adjacency(
        candidates,
        residuals,
        trace_edges,
        max_lag,
        float(kwargs.get("granger_alpha", 0.05)),
        int(kwargs.get("max_in_edges", 3)),
        float(kwargs.get("trace_prior", 0.6)),
    )

    anomaly_arr = np.array([anomaly_scores.get(s, 0.0) for s in candidates], dtype=float)
    banzhaf_arr = _mc_banzhaf(
        anomaly_arr,
        adj,
        n_samples=int(kwargs.get("banzhaf_samples", 1000)),
        random_state=random_state,
    )
    service_banzhaf = {svc: float(banzhaf_arr[i]) for i, svc in enumerate(candidates)}
    for svc in services:
        service_banzhaf.setdefault(svc, 0.0)

    metric_scores = _metric_evidence(normal_df, anomal_df, cols)
    ranks = _rank_metrics(cols, service_banzhaf, metric_scores)
    service_ranks = sorted(services, key=lambda s: (-service_banzhaf.get(s, 0.0), s))

    return {
        "ranks": ranks,
        "node_names": cols,
        "service_ranks": service_ranks,
        "service_node_names": candidates,
        "adj": adj,
    }
