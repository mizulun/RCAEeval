# Gabanzhaf Runtime Optimization Design

## Goal

Reduce `gabanzhaf` runtime on large systems such as `re2-tt` without reducing RCA accuracy, while preserving one stable production algorithm path.

## Current Problem

`gabanzhaf` is close to `baro` runtime on `re2-ob` and `re2-ss`, but `re2-tt` is roughly three times slower. The main scale difference is metric and trace volume: `re2-tt` result files contain roughly 290 metrics per case, while `re2-ob` and `re2-ss` are closer to 50 metrics per case.

The likely runtime contributors are:

- Trace CSV parsing happens twice per case: once for trace onset scores and once for call graph extraction.
- `_mc_banzhaf()` performs sampled marginal contribution estimation in Python loops.
- TT has many more metrics and services, so metric evidence, topology, Banzhaf, and final ranking costs are amplified.

## Design Principles

- Do not switch algorithms based on dataset, service count, or metric count.
- Keep one production path that works for small and large systems.
- Preserve MC Banzhaf only as an explicit experimental backend, not an automatic fallback.
- Prefer deterministic computation for production so repeated runs are stable.
- Add timing evidence before making performance claims.

## Trace Feature Extraction

Replace the two separate trace readers with one feature extractor:

```python
def _extract_trace_features(case_dir, inject_time, config):
    return {
        "trace_scores": trace_scores,
        "call_graph": call_graph,
    }
```

The extractor reads these columns once when available:

```text
serviceName, spanID, parentSpanID, startTimeMillis, statusCode, duration
```

It computes:

- `trace_scores`: per-service trace anomaly score from error rate and duration ratio.
- `call_graph`: `{caller_service: set(callee_service)}` derived from span-parent relationships.

If trace data is absent or missing optional columns, the function returns neutral values:

```python
{"trace_scores": None, "call_graph": {}}
```

If duration/error columns are missing but span columns exist, call graph extraction should still proceed. If span columns are missing but duration/error columns exist, trace scores should still proceed.

## Banzhaf Backend Design

Introduce one explicit backend selector in `_GabanzhafConfig`:

```python
banzhaf_mode: str = "deterministic"
banzhaf_samples: int = 1000
seed: int = 42
```

Add a single public internal dispatcher:

```python
def _banzhaf_power(fitness_signal, topo_matrix, config):
    if config.banzhaf_mode == "mc":
        return _mc_banzhaf(
            fitness_signal,
            topo_matrix,
            n_samples=config.banzhaf_samples,
            seed=config.seed,
        )
    return _deterministic_banzhaf_coverage(fitness_signal, topo_matrix)
```

This selector is experimental configuration only. Production defaults to deterministic mode.

## Deterministic Causal-Coverage Banzhaf

The current game is:

```text
v(S) = sum_j fitness[j] * 1(j is explained by S)
```

Service `j` is explained when `j in S` or any upstream service of `j` is in `S`.

For this coverage-style game, Banzhaf can be computed deterministically from the explainer set for each explained node. A service `i` contributes `fitness[j]` to node `j` only when:

- `i` can explain `j`, and
- no other explainer of `j` is present in the sampled coalition.

For uniform Banzhaf coalitions, this expected contribution is:

```text
fitness[j] * 0.5 ** (num_explainers[j] - 1)
```

for each `j` that `i` can explain.

Expected benefits:

- Removes sampling variance.
- Removes dependence on `seed` and `banzhaf_samples` in production.
- Reduces Banzhaf cost from sampled Python loops to matrix/vector operations.
- Keeps the same causal-coverage semantics.

Known limitation:

- The deterministic formula is valid for the current additive coverage utility. If future work adds nonlinear coalition effects, this formula must be revisited.

## Benchmark Comparison Plan

Run the same cases with three backends:

```text
A. current-mc
   banzhaf_mode="mc", banzhaf_samples=1000, seed=42

B. fast-mc
   banzhaf_mode="mc", banzhaf_samples=200 or 300, seed=42

C. deterministic
   banzhaf_mode="deterministic"
```

Compare:

- AC@1 / AC@3 / AC@5 / Avg@5 by dataset and fault type.
- Wall time and average speed by dataset.
- Stage timing for trace, preprocess, metric evidence, topology, Banzhaf, and ranking.
- Answer position changes for cases where one backend fixes or breaks top-k ranking.

Acceptance criteria for making deterministic the production default:

- `re2-tt` runtime improves by at least 30%.
- Overall AC@1 does not decrease.
- Overall Avg@5 does not decrease by more than 0.02.
- No dataset/fault AC@1 drops by more than 0.07.
- TT delay/loss do not regress relative to MC.

## Testing Strategy

Add focused tests for:

- `_extract_trace_features()` returns both trace scores and call graph from one trace file.
- Missing trace file returns neutral features.
- Missing optional trace columns degrades partially, not globally.
- `_banzhaf_power()` uses deterministic mode by default.
- MC mode remains available when explicitly configured.
- Deterministic Banzhaf matches analytic expectations on simple causal graphs.
- Deterministic Banzhaf ranks a clear upstream root above downstream victims.

## Non-Goals

- Do not add dataset-specific behavior.
- Do not change metric evidence scoring as part of this optimization.
- Do not tune OB/SS infra-service demotion in this change.
- Do not remove MC implementation; keep it for experiments.
