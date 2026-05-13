# predict_banzhaf Design

**Date:** 2026-05-13
**Status:** Approved for implementation planning
**Target:** RCAEval Python 3.10 e2e method

## Goal

Add a new RCAEval method named `predict_banzhaf` for high-performance root cause
analysis across trace and trace-less microservice scenarios. The method must keep
a single academic pipeline across system sizes and telemetry modalities:

```text
PCA health score
  -> STL decomposition
  -> ElasticNet residual prediction
  -> residual Granger topology + trace prior fusion
  -> Monte Carlo Banzhaf ranking
  -> metric-level RCAEval ranks
```

The implementation must not replace the RCA methodology based on whether traces
exist or whether the graph is small or large. Missing or invalid evidence is
represented as zero confidence inside the same matrices and equations.

## Scope

Create a new independent method:

- Add `RCAEval/e2e/predict_banzhaf.py`
- Export `predict_banzhaf` from `RCAEval/e2e/__init__.py`
- Import it in the Python 3.10 branch of `main.py`
- Add focused tests in `tests/test_predict_banzhaf.py`

Do not modify `gabanzhaf` behavior. Existing `gabanzhaf` remains a baseline.

## Public Interface

```python
def predict_banzhaf(
    data,
    inject_time=None,
    dataset=None,
    num_loop=None,
    sli=None,
    anomalies=None,
    **kwargs,
):
    return {
        "ranks": ranks,
        "node_names": all_cols,
        "service_ranks": service_ranks,
        "adj": adj,
    }
```

`ranks` is the compatibility surface used by RCAEval and must contain original
metric column names, sorted from most likely root-cause metric to least likely.
`service_ranks` and `adj` are diagnostic outputs and are not required by
`main.py`.

## Fixed Pipeline

### 1. Window Split and Preprocessing

Use the same RCAEval conventions as existing e2e methods:

- If `anomalies` is provided, split at `anomalies[0]`.
- Else if `inject_time` and `time` column exist, split by time.
- Else split at the midpoint.

Concatenate normal and anomalous windows before calling
`RCAEval.io.time_series.preprocess`, then split the processed frame back into
normal and anomalous windows. This keeps column selection and scaling consistent
across both windows.

### 2. Service Grouping

Group metric columns by service using the existing RCAEval naming convention:

```text
service_metric
```

The service name is parsed with the same practical convention used by the
existing metric-ranking methods: split on the last underscore, so
`service_metric` becomes service `service` and metric `metric`. Columns without
an underscore are treated as service-only columns with metric `unknown`. The
helper must be deterministic for ambiguous names.

### 3. PCA Health Score

For each service:

1. Build a service metric matrix from its metric columns.
2. Apply `StandardScaler`.
3. Apply `PCA(n_components=1)`.
4. Orient the first component so larger values correspond to more anomalous
   behavior. If the anomalous-window mean is lower than the normal-window mean,
   multiply by `-1`.
5. Robust-normalize the resulting service health score.

The result is one health score time series per service. Single-metric services
still pass through the same PCA function with a two-dimensional input matrix.

### 4. Two-Stage Forecasting

Stage 1 uses `statsmodels.tsa.seasonal.STL` on the service health score:

```text
health = trend + seasonal + residual
base_prediction = trend + seasonal
stage1_residual = residual
```

The STL period is chosen by a deterministic rule bounded by the available
sequence length, for example:

```text
period = min(24, max(2, len(normal_window) // 4))
```

Stage 2 uses `ElasticNet` on lagged residual features:

```text
residual_t ~ residual_{t-1}, residual_{t-2}, ..., residual_{t-L}
```

`L` defaults to `max_lag=3` and is capped by the available sample count. The
combined prediction is:

```text
combined_prediction = STL_base_prediction + ElasticNet_residual_prediction
```

The service anomaly score is computed from the anomalous-window absolute
forecast error using a robust blend:

```text
score = 0.55 * p95_robust_z + 0.30 * persistence + 0.15 * peak_robust_z
```

Scores are normalized to `[0, 1]` across services.

### 5. Residual Granger Topology

Topology inference uses residuals from Stage 1, not raw health scores. This
reduces spurious causality caused by shared trend and seasonality.

For candidate services `u` and `v`, test whether residual `u` Granger-causes
residual `v` using `statsmodels.tsa.stattools.grangercausalitytests`.

Convert the best p-value across lags to a confidence:

```text
G[u, v] = max(0, 1 - p_value / granger_alpha)
```

Default `granger_alpha` is `0.05`. The direction `u -> v` means `u` can explain
the symptoms of `v`.

To keep latency bounded, all cases use the same candidate-bounded graph
construction:

- rank services by anomaly score
- keep up to `top_k_svcs`
- include SLI service when available
- include services that appear in trace evidence when available
- cap candidate count at `max_candidate_svcs`
- keep at most `max_in_edges` incoming edges per target

This is not a different RCA method for large systems; it is the fixed graph
construction step used by the method.

### 6. Trace Prior Fusion

Trace data is represented as a prior matrix `T` over the same service set. The
method always computes:

```text
A = max(G, trace_prior * T)
```

where:

- `G` is the residual Granger confidence matrix
- `T` is the trace prior matrix
- `trace_prior` defaults to `0.6`
- `A` is the final service adjacency matrix

If traces are absent or unsupported, `T` is the all-zero matrix. The formula and
pipeline are unchanged.

Trace direction follows causal explanation direction. If a trace says caller
`c` calls callee `d`, the topology prior is:

```text
d -> c
```

because a callee fault can explain caller symptoms.

### 7. Monte Carlo Banzhaf Ranking

Use one Banzhaf computation strategy for every graph size: Monte Carlo
estimation with a fixed random seed.

Utility:

```text
v(S) = sum_j anomaly_score[j] * explained(j, S)
```

Weighted explanation:

```text
explained(j, S) = 1                         if j in S
explained(j, S) = max_i A[i, j] for i in S  otherwise
```

Banzhaf value for service `i` is estimated by sampling coalitions `S` and
averaging:

```text
v(S union {i}) - v(S without {i})
```

Defaults:

- `banzhaf_samples=1000`
- `random_state=42`

This supports the academic claim that all system sizes use the same low-cost
stochastic approximation of the Banzhaf power index.

### 8. Metric-Level Projection

RCAEval evaluates metric-level ranks, so service-level root-cause scores are
projected back to metric columns.

For each original metric column, compute local metric anomaly evidence from the
normal/anomalous split using robust z-score magnitude, persistence, and peak.

Final metric score:

```text
final_metric_score =
    0.70 * normalized_service_banzhaf[service]
  + 0.30 * normalized_metric_evidence[column]
  + 0.15 * normalized_service_banzhaf[service] * normalized_metric_evidence[column]
```

Sort all metric columns by this score and return them as `ranks`.

## Parameters

Default kwargs:

```python
top_k_svcs = 30
max_candidate_svcs = 40
max_lag = 3
granger_alpha = 0.05
max_in_edges = 3
banzhaf_samples = 1000
trace_prior = 0.6
random_state = 42
dk_select_useful = False
```

## Evidence Validity Policy

The method uses fixed operators and fixed equations. Data validity checks do not
switch to a different RCA algorithm:

- Invalid PCA evidence produces a zero health confidence for that service.
- Invalid STL residual evidence produces zero residual confidence for that
  service.
- Invalid ElasticNet residual prediction contributes zero refinement
  confidence.
- Invalid Granger evidence produces a zero edge confidence.
- Missing trace data produces an all-zero trace prior matrix.

These are evidence-confidence outcomes inside the same pipeline, not alternate
RCA paths.

## Testing

Add `tests/test_predict_banzhaf.py` with focused coverage:

1. `predict_banzhaf` returns RCAEval-compatible `ranks` and `node_names`.
2. PCA health score converts multi-metric service data to one series per
   service.
3. Trace-less data still follows the same path and returns deterministic ranks.
4. Synthetic residual `A -> B` produces a stronger `A -> B` Granger edge than
   `B -> A`.
5. Monte Carlo Banzhaf ranks a root service above a downstream victim when
   `root -> victim`.
6. `main.py --method predict_banzhaf --dataset re2-ss --test` smoke path can
   import and execute the method.

## Non-Goals

- Do not replace or refactor `gabanzhaf`.
- Do not add deep learning dependencies.
- Do not introduce trace-only or no-trace-only ranking paths.
- Do not use exact Banzhaf for small graphs; use MC Banzhaf for all sizes.

## Implementation Notes

Use only the requested libraries and existing RCAEval utilities:

- `numpy`
- `pandas`
- `scikit-learn`
- `statsmodels`
- `networkx`
- `RCAEval.io.time_series.preprocess`

The implementation should be deterministic for a fixed `random_state` and should
avoid global warning suppression beyond the local method module.
