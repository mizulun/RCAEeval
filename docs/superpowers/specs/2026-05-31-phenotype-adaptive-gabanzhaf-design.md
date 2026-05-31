# Phenotype-Adaptive Gabanzhaf Design

## Goal

Improve `RCAEval/e2e/gabanzhaf.py` so service and metric rankings are more precise across all fault types, with a target of pushing both `AC@1` and `Avg@5` toward `0.9+` on `re2-ob`, `re2-ss`, and `re2-tt`.

The improvement must remain fully unsupervised. It must not use ground-truth root causes, fault labels, dataset names, case IDs, service-name exceptions, or evaluation feedback. Runtime should stay close to the current Baro-compatible budget by adding only linear-time incident characterization overhead.

## Current Baseline

The current `gabanzhaf.py` pipeline already includes:

- robust metric evidence from pre/post injection windows;
- service evidence aggregation;
- bounded candidate pools;
- lag-correlation topology;
- trace-derived call graph fallback;
- residual victim suppression;
- deterministic causal-coverage Banzhaf;
- metric-level final ranking.

The current benchmark profile shows good top-3/top-5 behavior, but weaker top-1 precision. DELAY and LOSS remain difficult because propagation makes downstream services look anomalous, while CPU and SOCKET in some small datasets have top-1 failures because local metric evidence is not always enough to identify the injected service.

The current trace score combines duration and error evidence into one value. The new design should preserve that combined score for backward-compatible ranking, but also expose separate trace duration and trace error components for phenotype inference.

## Design Principle

Do not add fault-type rules such as "if DELAY then boost latency" or "if LOSS then boost errors." Instead, infer a continuous incident phenotype vector from observed telemetry:

```text
pi = [pi_resource, pi_latency, pi_error, pi_propagation]
```

The phenotype vector is not a classifier and has no access to labels. It is a normalized summary of which evidence families dominate the current incident. This vector conditions the causal game payoff and final ranking, so the same algorithm adapts to resource, latency, error, and propagation-heavy incidents.

## Mathematical Model

Let:

- `G = (V, E)` be the bounded service topology.
- `i, j in V` be services.
- `S subset V` be a candidate root-cause coalition.
- `A_j in [0, 1]` be service anomaly evidence from metric shifts.
- `R_j in [0, 1]` be residual evidence after upstream explanation.
- `T_j in [0, 1]` be trace evidence from error rate and duration ratio.
- `L_j in [0, 1]` be latency-local evidence aggregated from latency-like metrics.
- `Q_j in [0, 1]` be resource-local evidence aggregated from CPU, memory, disk, and socket-like metrics.
- `H_j in [0, 1]` be topology propagation pressure, high when service `j` is anomalous and has anomalous downstream or upstream neighbors.
- `C_j(S) in {0, 1}` indicate whether `j` is explained by coalition `S`: `j in S` or some upstream explainer of `j` is in `S`.

The incident phenotype is:

```text
z_resource    = mean_top(Q_j)
z_latency     = mean_top(L_j) + mean_top(trace_latency_j)
z_error       = mean_top(trace_error_j)
z_propagation = normalized_edge_anomaly_alignment(G, A)

pi_k = exp(z_k / tau) / sum_l exp(z_l / tau)
```

`mean_top` uses the mean of the top few service scores, not the maximum alone, to avoid one noisy metric determining the incident type. `tau` is a fixed smoothing temperature. No component is tied to a ground-truth fault label.

The phenotype-conditioned payoff is:

```text
v(S | pi) = sum_j W_j(pi) * C_j(S)
```

where:

```text
W_j(pi) =
    A_j
  * (1 + gamma_r * pi_resource    * Q_j)
  * (1 + gamma_l * pi_latency     * L_j)
  * (1 + gamma_e * pi_error       * T_j)
  * (1 + gamma_p * pi_propagation * H_j)
  * (lambda_a * A_j + lambda_r * R_j + lambda_t * T_j)
```

This remains an additive coverage game. Therefore deterministic Banzhaf is still available:

```text
beta_i(pi) = sum_{j: i explains j} W_j(pi) * 2^{-(|Explainers(j)| - 1)}
```

The formula keeps the key Banzhaf interpretation: a service scores high when it has large marginal explanatory contribution under the current incident phenotype.

## Incident Phenotype Inference

Extend trace extraction to retain separable trace components:

```python
@dataclass(frozen=True)
class _TraceFeatures:
    trace_scores: dict | None
    trace_error_scores: dict | None
    trace_latency_scores: dict | None
    call_graph: dict
```

`trace_scores` remains the existing max-combined signal. The two new dictionaries are optional, normalized service-level views of status-code error rate and duration ratio. Missing trace data returns `None` for all trace score dictionaries and `{}` for the call graph.

Add a small helper:

```python
def _incident_phenotype(
    cols,
    metric_ev,
    svc_anom_dict,
    trace_error_scores,
    trace_latency_scores,
    semantic_evidence,
    topology_edges,
    config,
):
    return {
        "resource": pi_resource,
        "latency": pi_latency,
        "error": pi_error,
        "propagation": pi_propagation,
    }
```

Metric type detection is lexical but generic:

- resource metrics: suffix contains `cpu`, `mem`, `memory`, `disk`, `io`, `socket`, `net`, `network`;
- latency metrics: suffix contains `lat`, `latency`, `duration`, `rt`, `response`;
- error metrics: trace error contribution from status code and error-rate time series when available.

These are telemetry semantic groups, not benchmark labels. They can be applied to new microservice datasets without knowing the injected fault type.

Propagation score uses existing bounded topology:

```text
edge_alignment = mean_{(u, v, w) in E} w * A_u * A_v
propagation = edge_alignment normalized by observed edge weights
```

This makes propagation high only when anomalous services are structurally connected, which is typical for delay/loss cascades.

## Phenotype-Conditioned Service Features

Add a helper that returns per-service semantic evidence after topology construction:

```python
def _semantic_service_evidence(
    cols,
    metric_ev,
    trace_scores,
    topology_edges,
    svc_anom_dict,
):
    return {
        svc: {
            "resource": q,
            "latency": l,
            "trace": t,
            "propagation": h,
        }
    }
```

Each component is normalized to `[0, 1]`. Missing trace data becomes zero-valued evidence, preserving a single formula without switching algorithms.

## Banzhaf Integration

Replace the current `fitness_signal` passed into deterministic Banzhaf with phenotype-conditioned payoff weights:

```text
fitness_signal[j] = W_j(pi)
```

The topology matrix and deterministic Banzhaf backend remain unchanged. This limits implementation risk and preserves the existing `O(N^2)` Banzhaf cost.

## Final Ranking

Metric ranking should remain metric-level, not collapse to service-only ranking:

```text
score(m_j) =
    alpha * root_score(service(j), pi)
  + beta  * metric_local_score(m_j)
  + eta   * root_score(service(j), pi) * metric_local_score(m_j)
  + rho   * semantic_match(m_j, pi)
```

where `semantic_match` is derived from the metric's generic semantic group:

```text
resource metric -> pi_resource
latency metric  -> pi_latency
error metric    -> pi_error
other metric    -> 0
```

This is not fault-type hardcoding because the score depends on the inferred incident phenotype, not on a supplied fault label.

## Configuration

Add conservative fixed parameters to `_GabanzhafConfig`:

```python
phenotype_temperature: float = 0.7
phenotype_gain_resource: float = 0.35
phenotype_gain_latency: float = 0.45
phenotype_gain_error: float = 0.45
phenotype_gain_propagation: float = 0.35
```

These are global constants, not dataset-specific parameters. They should be exposed through kwargs only for experiments.

## Complexity

Existing dominant costs are:

```text
metric evidence      O(T * M)
service aggregation  O(M)
topology building    O(T * N^2)
Banzhaf              O(N^2)
ranking              O(M log M)
```

The new helpers add:

```text
semantic evidence    O(M + N + E)
phenotype inference  O(N + E)
payoff construction  O(N)
semantic ranking     O(M)
```

Total complexity remains:

```text
O(T * M + T * N^2 + N^2 + M log M)
```

The added terms are lower order because `N` is already bounded by `max_candidate_svcs`, and `E` is bounded by topological pruning. No `O(M^2)`, `O(N^3)`, or `O(2^N)` step is introduced.

## Testing Strategy

Add focused unit tests for:

- semantic metric grouping does not depend on dataset names or fault labels;
- phenotype vector is normalized and finite with missing trace data;
- latency-dominant synthetic incident yields higher latency phenotype than resource phenotype;
- error-dominant synthetic trace incident yields higher error phenotype;
- propagation score increases when anomalous connected services appear on high-confidence edges;
- deterministic Banzhaf still ranks an upstream root above downstream victims when phenotype-conditioned payoff is used;
- final ranking gives semantic match a bounded boost without overriding service root evidence.

Benchmark validation:

- run `gabanzhaf` on `re2-ob`, `re2-ss`, and `re2-tt`;
- compare `AC@1`, `AC@3`, `AC@5`, `Avg@5`, wall time, and average speed against the current `benchmark_results.json`;
- inspect any regression larger than `0.05` in `AC@1` or `Avg@5`;
- verify runtime remains within the Baro-compatible budget and does not add more than a small constant overhead.

## Non-Goals

- Do not train a supervised model.
- Do not use ground-truth root cause or fault labels.
- Do not branch by dataset name, case ID, or service name.
- Do not build a full metric-level causal graph.
- Do not introduce graph neural networks, Hawkes training, or Neural Granger training in this change.
- Do not remove deterministic Banzhaf or the existing MC experimental backend.

## Implementation Scope

Primary file:

- `RCAEval/e2e/gabanzhaf.py`

Test file:

- `tests/test_gabanzhaf_improvements.py`

The implementation should keep the current orchestration structure:

```text
window split
preprocess
metric evidence
service evidence
topology
residual scoring
semantic service evidence
phenotype inference
phenotype-conditioned payoff
deterministic Banzhaf
metric ranking
```

This is a single algorithmic improvement, not a rewrite.
