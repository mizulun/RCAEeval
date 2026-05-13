from __future__ import annotations

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


def _stl_decompose_health(health: np.ndarray, split_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    y = _to_1d_float(health)
    if y.size < 6:
        return np.zeros_like(y), np.zeros_like(y)

    normal_len = max(2, min(int(split_idx), len(y)))
    period = int(min(24, max(2, normal_len // 4)))
    if len(y) < period * 2:
        return np.zeros_like(y), np.zeros_like(y)

    try:
        res = STL(y, period=period, robust=True).fit()
        base = np.nan_to_num(res.trend + res.seasonal, nan=0.0)
        residual = np.nan_to_num(res.resid, nan=0.0)
        return base, residual
    except Exception:
        return np.zeros_like(y), np.zeros_like(y)


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
    base, residual = _stl_decompose_health(y, split_idx)
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
    rz = np.nan_to_num(np.abs((anomal_err - med) / denom), nan=0.0, posinf=0.0, neginf=0.0)
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
