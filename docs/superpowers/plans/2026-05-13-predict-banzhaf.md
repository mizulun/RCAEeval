# predict_banzhaf Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new RCAEval e2e method named `predict_banzhaf` that uses PCA health scores, STL/ElasticNet residual prediction, residual Granger topology, trace-prior fusion, and Monte Carlo Banzhaf ranking while returning RCAEval-compatible metric-level ranks.

**Architecture:** Implement a new independent module `RCAEval/e2e/predict_banzhaf.py` and keep `gabanzhaf` unchanged. The method always runs one fixed pipeline; missing trace or invalid statistical evidence becomes zero-confidence evidence in the same matrices rather than switching to another RCA method. Export the method through `RCAEval/e2e/__init__.py` and import it in `main.py`.

**Tech Stack:** Python 3.10, numpy, pandas, scikit-learn (`StandardScaler`, `PCA`, `ElasticNet`), statsmodels (`STL`, `grangercausalitytests`), networkx, pytest, RCAEval preprocessing utilities.

---

## File Structure

- Create: `RCAEval/e2e/predict_banzhaf.py`
  - Owns all helper functions and the public `predict_banzhaf(...)` entry point.
  - Keep helpers module-private with leading underscore.
- Create: `tests/test_predict_banzhaf.py`
  - Focused unit tests for parser/grouping, PCA health score, residual Granger, MC Banzhaf, and end-to-end output format.
- Modify: `RCAEval/e2e/__init__.py`
  - Import `predict_banzhaf` in the Python 3.10/3.12-compatible import block next to `gabanzhaf`.
- Modify: `main.py`
  - Import `predict_banzhaf` in the Python 3.10 branch so `--method predict_banzhaf` works.

Do not modify `RCAEval/e2e/gabanzhaf.py`.

## Task 1: Add Failing Helper Tests

**Files:**
- Create: `tests/test_predict_banzhaf.py`
- Create later: `RCAEval/e2e/predict_banzhaf.py`

- [ ] **Step 1: Write failing tests for parsing, normalization, PCA, Granger, and Banzhaf**

Create `tests/test_predict_banzhaf.py` with:

```python
import numpy as np
import pandas as pd

from RCAEval.e2e.predict_banzhaf import (
    _build_service_health_scores,
    _granger_edge_weight,
    _mc_banzhaf,
    _norm_dict,
    _parse_service_metric,
    _service_groups,
)


def test_parse_service_metric_uses_last_underscore():
    assert _parse_service_metric("ts_ui_dashboard_latency") == (
        "ts_ui_dashboard",
        "latency",
    )
    assert _parse_service_metric("frontend") == ("frontend", "unknown")


def test_norm_dict_is_deterministic_for_constant_values():
    assert _norm_dict({"a": 3.0, "b": 3.0}) == {"a": 0.0, "b": 0.0}


def test_service_groups_excludes_time_and_groups_by_service():
    cols = ["time", "svc_a_cpu", "svc_a_mem", "svc_b_latency"]
    assert _service_groups(cols) == {
        "svc_a": ["svc_a_cpu", "svc_a_mem"],
        "svc_b": ["svc_b_latency"],
    }


def test_build_service_health_scores_returns_one_series_per_service():
    rng = np.random.default_rng(0)
    n = 80
    df = pd.DataFrame(
        {
            "svc_a_cpu": np.r_[rng.normal(0, 0.1, 40), rng.normal(3, 0.1, 40)],
            "svc_a_mem": np.r_[rng.normal(0, 0.1, 40), rng.normal(2, 0.1, 40)],
            "svc_b_cpu": rng.normal(0, 0.1, n),
        }
    )
    groups = _service_groups(df.columns)

    scores = _build_service_health_scores(df, groups, split_idx=40)

    assert set(scores) == {"svc_a", "svc_b"}
    assert scores["svc_a"].shape == (n,)
    assert float(np.mean(scores["svc_a"][40:])) > float(np.mean(scores["svc_a"][:40]))


def test_granger_edge_weight_detects_synthetic_direction():
    rng = np.random.default_rng(1)
    n = 120
    a = rng.normal(0, 1, n)
    b = np.zeros(n)
    noise = rng.normal(0, 0.05, n)
    for t in range(2, n):
        b[t] = 0.8 * a[t - 1] + noise[t]

    ab = _granger_edge_weight(a, b, max_lag=2, alpha=0.05)
    ba = _granger_edge_weight(b, a, max_lag=2, alpha=0.05)

    assert ab > 0.5
    assert ab > ba


def test_mc_banzhaf_promotes_root_over_downstream_victim():
    anomaly = np.array([0.8, 1.0], dtype=float)
    adj = np.array(
        [
            [0.0, 1.0],
            [0.0, 0.0],
        ],
        dtype=float,
    )

    values = _mc_banzhaf(anomaly, adj, n_samples=2000, random_state=7)

    assert values.shape == (2,)
    assert values[0] > values[1]
```

- [ ] **Step 2: Run tests and verify they fail because the module does not exist**

Run:

```bash
python -m pytest tests/test_predict_banzhaf.py -v
```

Expected:

```text
ModuleNotFoundError: No module named 'RCAEval.e2e.predict_banzhaf'
```

- [ ] **Step 3: Commit failing tests**

Run:

```bash
git add tests/test_predict_banzhaf.py
git commit -m "test: add predict_banzhaf helper tests"
```

## Task 2: Implement Core Helpers

**Files:**
- Create: `RCAEval/e2e/predict_banzhaf.py`
- Test: `tests/test_predict_banzhaf.py`

- [ ] **Step 1: Add the module skeleton and helper implementations**

Create `RCAEval/e2e/predict_banzhaf.py`:

```python
from __future__ import annotations

import os
from typing import Dict, Iterable, List, Tuple

import networkx as nx
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

        X = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
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
```

- [ ] **Step 2: Run helper tests**

Run:

```bash
python -m pytest tests/test_predict_banzhaf.py -v
```

Expected:

```text
6 passed
```

- [ ] **Step 3: Commit helper implementation**

Run:

```bash
git add RCAEval/e2e/predict_banzhaf.py tests/test_predict_banzhaf.py
git commit -m "feat: add predict_banzhaf core helpers"
```

## Task 3: Add End-to-End Pipeline and Tests

**Files:**
- Modify: `RCAEval/e2e/predict_banzhaf.py`
- Modify: `tests/test_predict_banzhaf.py`

- [ ] **Step 1: Add failing end-to-end tests**

Append to `tests/test_predict_banzhaf.py`:

```python
from RCAEval.e2e.predict_banzhaf import predict_banzhaf


def _make_predict_banzhaf_frame(n=100, split=50):
    rng = np.random.default_rng(11)
    return pd.DataFrame(
        {
            "time": np.arange(n),
            "root_cpu": np.r_[rng.normal(0, 0.1, split), rng.normal(4, 0.2, n - split)],
            "root_mem": np.r_[rng.normal(0, 0.1, split), rng.normal(3, 0.2, n - split)],
            "victim_latency": np.r_[rng.normal(0, 0.1, split), rng.normal(2, 0.2, n - split)],
            "quiet_cpu": rng.normal(0, 0.1, n),
        }
    )


def test_predict_banzhaf_returns_rcaeval_format_trace_less():
    df = _make_predict_banzhaf_frame()

    result = predict_banzhaf(
        df,
        inject_time=50,
        dataset=None,
        banzhaf_samples=200,
        top_k_svcs=10,
        max_candidate_svcs=10,
        random_state=3,
    )

    assert "ranks" in result
    assert "node_names" in result
    assert "service_ranks" in result
    assert "adj" in result
    assert set(result["ranks"]) == {"root_cpu", "root_mem", "victim_latency", "quiet_cpu"}
    assert result["node_names"] == ["root_cpu", "root_mem", "victim_latency", "quiet_cpu"]
    assert result["ranks"][0].startswith("root_")


def test_predict_banzhaf_is_deterministic_for_fixed_seed():
    df = _make_predict_banzhaf_frame()
    kwargs = {
        "inject_time": 50,
        "dataset": None,
        "banzhaf_samples": 200,
        "random_state": 123,
    }

    first = predict_banzhaf(df, **kwargs)
    second = predict_banzhaf(df, **kwargs)

    assert first["ranks"] == second["ranks"]
    assert first["service_ranks"] == second["service_ranks"]
    assert np.allclose(first["adj"], second["adj"])
```

- [ ] **Step 2: Run the new test and verify failure**

Run:

```bash
python -m pytest tests/test_predict_banzhaf.py::test_predict_banzhaf_returns_rcaeval_format_trace_less -v
```

Expected:

```text
ImportError: cannot import name 'predict_banzhaf'
```

or:

```text
AttributeError: module 'RCAEval.e2e.predict_banzhaf' has no attribute 'predict_banzhaf'
```

- [ ] **Step 3: Implement trace parsing, topology, metric evidence, and public function**

Append these functions to `RCAEval/e2e/predict_banzhaf.py`:

```python
def _split_windows(data: pd.DataFrame, inject_time, anomalies) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
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


def _preprocess_joint(normal: pd.DataFrame, anomal: pd.DataFrame, dataset, dk_select_useful: bool):
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


def _metric_evidence(normal_df: pd.DataFrame, anomal_df: pd.DataFrame, cols: List[str]) -> Dict[str, float]:
    if not cols:
        return {}
    normal = normal_df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    anomal = anomal_df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
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
        for caller, callee in [
            ("caller_service", "callee_service"),
            ("callerService", "calleeService"),
            ("source", "target"),
            ("src", "dst"),
        ]:
            if {caller, callee}.issubset(cols):
                edges = set()
                for chunk in pd.read_csv(path, usecols=[caller, callee], chunksize=10000):
                    for c, d in zip(chunk[caller].astype(str), chunk[callee].astype(str)):
                        if c and d and c.lower() != "nan" and d.lower() != "nan" and c != d:
                            edges.add((d, c, 1.0))
                    if len(edges) >= 1000:
                        break
                return sorted(edges)

        if {"serviceName", "spanID", "parentSpanID"}.issubset(cols):
            df = pd.read_csv(
                path,
                usecols=["serviceName", "spanID", "parentSpanID"],
                nrows=50000,
                on_bad_lines="skip",
            )
            span_to_svc = df.set_index("spanID")["serviceName"].astype(str).to_dict()
            df["parentSvc"] = df["parentSpanID"].map(span_to_svc)
            edges = set()
            for caller, callee in zip(df["parentSvc"], df["serviceName"].astype(str)):
                if pd.notna(caller) and str(caller) != str(callee):
                    edges.add((str(callee), str(caller), 1.0))
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
    candidates = set(ranked[: max(1, int(top_k))])
    if sli:
        svc, _ = _parse_service_metric(sli)
        if svc in services:
            candidates.add(svc)
    for u, v, _ in trace_edges:
        if u in services:
            candidates.add(u)
        if v in services:
            candidates.add(v)
    capped = sorted(candidates, key=lambda s: (-anomaly_scores.get(s, 0.0), s))
    return sorted(capped[: max(1, int(max_candidate))])


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
            G[idx[u], idx[v]] = _granger_edge_weight(residuals.get(u, []), residuals.get(v, []), max_lag, alpha)

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
        _, residual, score = _forecast_service(health_scores.get(svc, np.zeros(len(full_proc))), split_idx, max_lag, random_state)
        residuals[svc] = residual
        raw_anomaly[svc] = score
    anomaly_scores = _norm_dict(raw_anomaly)

    args_ns = kwargs.get("args")
    case_dir = os.path.dirname(args_ns.data_path) if args_ns is not None and hasattr(args_ns, "data_path") else None
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
        "adj": adj,
    }
```

- [ ] **Step 4: Run all predict_banzhaf tests**

Run:

```bash
python -m pytest tests/test_predict_banzhaf.py -v
```

Expected:

```text
8 passed
```

- [ ] **Step 5: Commit pipeline implementation**

Run:

```bash
git add RCAEval/e2e/predict_banzhaf.py tests/test_predict_banzhaf.py
git commit -m "feat: implement predict_banzhaf pipeline"
```

## Task 4: Register Method with RCAEval

**Files:**
- Modify: `RCAEval/e2e/__init__.py`
- Modify: `main.py`
- Modify: `tests/test_predict_banzhaf.py`

- [ ] **Step 1: Add failing import registration test**

Append to `tests/test_predict_banzhaf.py`:

```python
def test_predict_banzhaf_exported_from_e2e_package():
    from RCAEval.e2e import predict_banzhaf as exported

    assert exported is predict_banzhaf
```

- [ ] **Step 2: Run registration test and verify failure**

Run:

```bash
python -m pytest tests/test_predict_banzhaf.py::test_predict_banzhaf_exported_from_e2e_package -v
```

Expected:

```text
ImportError
```

or:

```text
AssertionError
```

- [ ] **Step 3: Export `predict_banzhaf` from `RCAEval/e2e/__init__.py`**

In `RCAEval/e2e/__init__.py`, inside the `if is_py310() or is_py312():` block, add this import near `gabanzhaf`:

```python
    from .predict_banzhaf import predict_banzhaf
```

- [ ] **Step 4: Import `predict_banzhaf` in `main.py`**

In `main.py`, inside the `if is_py310(): from RCAEval.e2e import (...)` list, add:

```python
        predict_banzhaf,
```

Place it near `gabanzhaf` or other e2e methods. The `parse_args()` method checks `globals()`, so no argparse choices need to be changed.

- [ ] **Step 5: Run registration test**

Run:

```bash
python -m pytest tests/test_predict_banzhaf.py::test_predict_banzhaf_exported_from_e2e_package -v
```

Expected:

```text
1 passed
```

- [ ] **Step 6: Run full predict_banzhaf tests**

Run:

```bash
python -m pytest tests/test_predict_banzhaf.py -v
```

Expected:

```text
9 passed
```

- [ ] **Step 7: Commit registration**

Run:

```bash
git add RCAEval/e2e/__init__.py main.py tests/test_predict_banzhaf.py
git commit -m "feat: register predict_banzhaf method"
```

## Task 5: Verification and Smoke Tests

**Files:**
- No new source files expected.
- May update `RCAEval/e2e/predict_banzhaf.py` only for fixes found by verification.

- [ ] **Step 1: Run focused tests**

Run:

```bash
python -m pytest tests/test_predict_banzhaf.py -v
```

Expected:

```text
9 passed
```

- [ ] **Step 2: Run existing gabanzhaf tests to verify the baseline was not broken**

Run:

```bash
python -m pytest tests/test_gabanzhaf_improvements.py -v
```

Expected:

```text
passed
```

If these fail because of pre-existing dirty worktree changes, inspect failures and do not revert unrelated user changes. Report the exact failure before deciding on a fix.

- [ ] **Step 3: Run import smoke test**

Run:

```bash
python - <<'PY'
from RCAEval.e2e import predict_banzhaf
print(predict_banzhaf.__name__)
PY
```

Expected:

```text
predict_banzhaf
```

- [ ] **Step 4: Run main.py test-mode smoke path when dataset access is available**

Run:

```bash
python main.py --method predict_banzhaf --dataset re2-ss --test
```

Expected:

```text
--- Evaluation results ---
```

If this attempts to download data and fails because network access is blocked, rerun with the proper sandbox escalation or record that the command could not be completed due to dataset download/network restrictions.

- [ ] **Step 5: Commit any verification fixes**

If Step 1-4 required source changes, run:

```bash
git add RCAEval/e2e/predict_banzhaf.py tests/test_predict_banzhaf.py RCAEval/e2e/__init__.py main.py
git commit -m "fix: stabilize predict_banzhaf verification"
```

If no source changes were required, do not create an empty commit.

## Self-Review

Spec coverage:

- New independent method: covered by Tasks 2-4.
- PCA health score: covered by Task 2 helper tests and implementation.
- STL + ElasticNet forecasting: covered by Task 3 pipeline implementation.
- Residual Granger topology: covered by Task 2 helper tests and Task 3 adjacency builder.
- Trace prior fusion: covered by Task 3 `_read_trace_edges` and `_build_adjacency`.
- Same MC Banzhaf for all graph sizes: covered by Task 2 `_mc_banzhaf`; no exact Banzhaf task exists.
- Metric-level RCAEval ranks: covered by Task 3 end-to-end tests.
- RCAEval/main.py integration: covered by Task 4.

Placeholder scan:

- No TBD/TODO placeholders.
- No task says "add tests" without concrete test code.
- Code snippets define every helper later referenced by tests.

Type consistency:

- Public function name is consistently `predict_banzhaf`.
- Diagnostic outputs are consistently `service_ranks` and `adj`.
- Helper names in tests match helper names in implementation snippets.
