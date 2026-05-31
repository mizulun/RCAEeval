# Phenotype-Adaptive Gabanzhaf Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add unsupervised incident phenotype inference to `gabanzhaf` and use it to condition Banzhaf payoff and metric ranking without dataset-specific or fault-label-specific branches.

**Architecture:** Keep the existing single-file pipeline in `RCAEval/e2e/gabanzhaf.py`. Add focused helpers for trace component extraction, semantic service evidence, phenotype inference, phenotype-conditioned payoff, and semantic ranking boost; wire them through `_gabanzhaf_core()` while preserving deterministic Banzhaf and bounded candidate pools.

**Tech Stack:** Python, NumPy, pandas, pytest, existing RCAEval preprocessing and benchmark runner.

---

## File Structure

- Modify `RCAEval/e2e/gabanzhaf.py`
  - Extend `_TraceFeatures` and trace extraction helpers.
  - Extend `_GabanzhafConfig` with global phenotype parameters.
  - Add metric semantic grouping and service semantic evidence helpers.
  - Add incident phenotype inference and payoff construction.
  - Update `_rank_metrics()` and `_gabanzhaf_core()` signatures.
- Modify `tests/test_gabanzhaf_improvements.py`
  - Add unit tests for trace components, semantic grouping, phenotype inference, payoff behavior, and ranking boundedness.
- Optionally update `benchmark_results.json`
  - Only after implementation and benchmark verification.

## Task 1: Expose Trace Error and Latency Components

**Files:**
- Modify: `RCAEval/e2e/gabanzhaf.py`
- Test: `tests/test_gabanzhaf_improvements.py`

- [ ] **Step 1: Write failing tests for separable trace components**

Append near existing `_extract_trace_features` tests:

```python
def test_extract_trace_features_exposes_error_and_latency_components():
    rows = [
        ("svc-a", "span-a0", "root", 1000, 0, 10),
        ("svc-b", "span-b0", "span-a0", 1001, 0, 10),
        ("svc-a", "span-a1", "root-2", 60000, 1, 10),
        ("svc-b", "span-b1", "span-a1", 60001, 0, 300),
    ]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "traces.csv")
        _make_traces_csv(rows, path)

        features = _extract_trace_features(d, inject_time=50)

    assert features.trace_scores is not None
    assert features.trace_error_scores is not None
    assert features.trace_latency_scores is not None
    assert features.trace_error_scores["svc-a"] > features.trace_error_scores.get("svc-b", 0.0)
    assert features.trace_latency_scores["svc-b"] > features.trace_latency_scores.get("svc-a", 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
py310/bin/python -m pytest tests/test_gabanzhaf_improvements.py::test_extract_trace_features_exposes_error_and_latency_components -q
```

Expected: FAIL with an attribute error for `trace_error_scores` or `trace_latency_scores`.

- [ ] **Step 3: Extend `_TraceFeatures`**

In `RCAEval/e2e/gabanzhaf.py`, replace the dataclass with:

```python
@dataclass(frozen=True)
class _TraceFeatures:
    trace_scores: dict | None
    trace_error_scores: dict | None
    trace_latency_scores: dict | None
    call_graph: dict
```

- [ ] **Step 4: Add neutral constructors inline**

Update every neutral return:

```python
return _TraceFeatures(
    trace_scores=None,
    trace_error_scores=None,
    trace_latency_scores=None,
    call_graph={},
)
```

Update preaggregated trace return:

```python
trace_scores, trace_error_scores, trace_latency_scores = fast_trace_scores
return _TraceFeatures(
    trace_scores=trace_scores,
    trace_error_scores=trace_error_scores,
    trace_latency_scores=trace_latency_scores,
    call_graph=_extract_call_graph(traces_path),
)
```

- [ ] **Step 5: Update trace score helper return values**

Change `_trace_scores_from_chunks()` so it returns a 3-tuple:

```python
combined = {}
error_scores = {}
latency_scores = {}
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
    _norm_dict(error_scores),
    _norm_dict(latency_scores),
)
```

Change `_preaggregated_trace_scores()` to collect `combined`, `error_scores`, and `latency_scores`, and return:

```python
if not combined:
    return None
return (
    _norm_dict(combined),
    _norm_dict(error_scores) if error_scores else None,
    _norm_dict(latency_scores) if latency_scores else None,
)
```

- [ ] **Step 6: Update chunk extraction return**

In `_extract_trace_features()`, replace:

```python
trace_scores = (
    _trace_scores_from_chunks(chunks_normal, chunks_anomal)
    if saw_score_cols
    else None
)
return _TraceFeatures(trace_scores=trace_scores, call_graph=call_graph)
```

with:

```python
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
```

- [ ] **Step 7: Update existing tests for dataclass equality**

Change the missing-file assertion:

```python
assert features == _TraceFeatures(
    trace_scores=None,
    trace_error_scores=None,
    trace_latency_scores=None,
    call_graph={},
)
```

- [ ] **Step 8: Run trace tests**

Run:

```bash
py310/bin/python -m pytest tests/test_gabanzhaf_improvements.py::test_extract_trace_features_returns_scores_and_call_graph tests/test_gabanzhaf_improvements.py::test_extract_trace_features_missing_file_returns_neutral_features tests/test_gabanzhaf_improvements.py::test_extract_trace_features_exposes_error_and_latency_components -q
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add RCAEval/e2e/gabanzhaf.py tests/test_gabanzhaf_improvements.py
git commit -m "feat: expose gabanzhaf trace evidence components"
```

## Task 2: Add Semantic Metric Grouping and Service Evidence

**Files:**
- Modify: `RCAEval/e2e/gabanzhaf.py`
- Test: `tests/test_gabanzhaf_improvements.py`

- [ ] **Step 1: Write failing semantic grouping test**

Append near metric evidence tests:

```python
def test_metric_semantic_group_uses_generic_suffixes():
    from RCAEval.e2e.gabanzhaf import _metric_semantic_group

    assert _metric_semantic_group("svc-a_cpu") == "resource"
    assert _metric_semantic_group("svc-a_memory_usage") == "resource"
    assert _metric_semantic_group("svc-a_latency") == "latency"
    assert _metric_semantic_group("svc-a_duration") == "latency"
    assert _metric_semantic_group("svc-a_http_errors") == "error"
    assert _metric_semantic_group("svc-a_custom_metric") == "other"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
py310/bin/python -m pytest tests/test_gabanzhaf_improvements.py::test_metric_semantic_group_uses_generic_suffixes -q
```

Expected: FAIL because `_metric_semantic_group` is not defined.

- [ ] **Step 3: Implement semantic grouping**

Add below `_parse_svc()`:

```python
def _metric_semantic_group(col):
    svc = _parse_svc(col)
    suffix = col[len(svc) + 1:] if col.startswith(svc + "_") else col
    suffix = suffix.lower()

    resource_keys = ("cpu", "mem", "memory", "disk", "io", "socket", "net", "network")
    latency_keys = ("lat", "latency", "duration", "rt", "response")
    error_keys = ("err", "error", "fail", "failure", "status")

    if any(k in suffix for k in resource_keys):
        return "resource"
    if any(k in suffix for k in latency_keys):
        return "latency"
    if any(k in suffix for k in error_keys):
        return "error"
    return "other"
```

- [ ] **Step 4: Run semantic grouping test**

Run:

```bash
py310/bin/python -m pytest tests/test_gabanzhaf_improvements.py::test_metric_semantic_group_uses_generic_suffixes -q
```

Expected: PASS.

- [ ] **Step 5: Write failing service semantic evidence test**

Append:

```python
def test_semantic_service_evidence_combines_metrics_trace_and_topology():
    from RCAEval.e2e.gabanzhaf import _semantic_service_evidence

    cols = ["svc-a_cpu", "svc-a_latency", "svc-b_cpu", "svc-b_latency"]
    metric_ev = {
        "svc-a_cpu": {"score": 0.9},
        "svc-a_latency": {"score": 0.2},
        "svc-b_cpu": {"score": 0.1},
        "svc-b_latency": {"score": 0.8},
    }
    svc_anom = {"svc-a": 0.9, "svc-b": 0.8}
    trace_scores = {"svc-b": 1.0}
    topology_edges = [("svc-a", "svc-b", 0.7)]

    sem = _semantic_service_evidence(cols, metric_ev, trace_scores, topology_edges, svc_anom)

    assert sem["svc-a"]["resource"] > sem["svc-b"]["resource"]
    assert sem["svc-b"]["latency"] > sem["svc-a"]["latency"]
    assert sem["svc-b"]["trace"] == 1.0
    assert sem["svc-a"]["propagation"] > 0.0
    assert sem["svc-b"]["propagation"] > 0.0
```

- [ ] **Step 6: Run test to verify it fails**

Run:

```bash
py310/bin/python -m pytest tests/test_gabanzhaf_improvements.py::test_semantic_service_evidence_combines_metrics_trace_and_topology -q
```

Expected: FAIL because `_semantic_service_evidence` is not defined.

- [ ] **Step 7: Implement semantic service evidence**

Add below `_service_evidence()`:

```python
def _semantic_service_evidence(cols, metric_ev, trace_scores, topology_edges, svc_anom_dict):
    svc_values = {
        svc: {"resource": [], "latency": [], "error": [], "other": []}
        for svc in svc_anom_dict
    }
    for col in cols:
        svc = _parse_svc(col)
        if svc not in svc_values:
            svc_values[svc] = {"resource": [], "latency": [], "error": [], "other": []}
        group = _metric_semantic_group(col)
        svc_values[svc][group].append(float(metric_ev[col]["score"]))

    raw = {}
    for svc, groups in svc_values.items():
        raw[svc] = {
            "resource": max(groups["resource"], default=0.0),
            "latency": max(groups["latency"], default=0.0),
            "error": max(groups["error"], default=0.0),
            "trace": float((trace_scores or {}).get(svc, 0.0)),
            "propagation": 0.0,
        }

    prop = {svc: 0.0 for svc in raw}
    for u, v, w in topology_edges:
        if u in prop and v in prop:
            aligned = float(w) * svc_anom_dict.get(u, 0.0) * svc_anom_dict.get(v, 0.0)
            prop[u] += aligned
            prop[v] += aligned
    prop = _norm_dict(prop)
    for svc in raw:
        raw[svc]["propagation"] = prop.get(svc, 0.0)

    for key in ("resource", "latency", "error", "trace"):
        normed = _norm_dict({svc: values[key] for svc, values in raw.items()})
        for svc in raw:
            raw[svc][key] = normed.get(svc, 0.0)

    return raw
```

- [ ] **Step 8: Run semantic evidence tests**

Run:

```bash
py310/bin/python -m pytest tests/test_gabanzhaf_improvements.py::test_metric_semantic_group_uses_generic_suffixes tests/test_gabanzhaf_improvements.py::test_semantic_service_evidence_combines_metrics_trace_and_topology -q
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add RCAEval/e2e/gabanzhaf.py tests/test_gabanzhaf_improvements.py
git commit -m "feat: add gabanzhaf semantic service evidence"
```

## Task 3: Add Incident Phenotype Inference

**Files:**
- Modify: `RCAEval/e2e/gabanzhaf.py`
- Test: `tests/test_gabanzhaf_improvements.py`

- [ ] **Step 1: Write failing phenotype tests**

Append:

```python
def test_incident_phenotype_is_normalized_with_missing_trace():
    from RCAEval.e2e.gabanzhaf import _GabanzhafConfig, _incident_phenotype

    cfg = _GabanzhafConfig.from_kwargs({})
    semantic = {
        "svc-a": {"resource": 1.0, "latency": 0.0, "error": 0.0, "trace": 0.0, "propagation": 0.0},
        "svc-b": {"resource": 0.5, "latency": 0.0, "error": 0.0, "trace": 0.0, "propagation": 0.0},
    }
    pi = _incident_phenotype(
        {"svc-a": 1.0, "svc-b": 0.5},
        None,
        None,
        semantic,
        [],
        cfg,
    )

    assert set(pi) == {"resource", "latency", "error", "propagation"}
    assert all(np.isfinite(v) for v in pi.values())
    assert abs(sum(pi.values()) - 1.0) < 1e-6
    assert pi["resource"] > pi["latency"]


def test_incident_phenotype_detects_latency_error_and_propagation_dominance():
    from RCAEval.e2e.gabanzhaf import _GabanzhafConfig, _incident_phenotype

    cfg = _GabanzhafConfig.from_kwargs({})
    semantic = {
        "svc-a": {"resource": 0.0, "latency": 1.0, "error": 0.0, "trace": 0.0, "propagation": 1.0},
        "svc-b": {"resource": 0.0, "latency": 0.9, "error": 0.0, "trace": 0.0, "propagation": 1.0},
    }
    svc_anom = {"svc-a": 1.0, "svc-b": 0.9}
    pi_latency = _incident_phenotype(
        svc_anom,
        {"svc-a": 0.0, "svc-b": 0.0},
        {"svc-a": 1.0, "svc-b": 0.9},
        semantic,
        [("svc-a", "svc-b", 0.9)],
        cfg,
    )

    assert pi_latency["latency"] > pi_latency["resource"]
    assert pi_latency["propagation"] > pi_latency["resource"]

    pi_error = _incident_phenotype(
        svc_anom,
        {"svc-a": 1.0, "svc-b": 0.8},
        {"svc-a": 0.0, "svc-b": 0.0},
        semantic,
        [],
        cfg,
    )
    assert pi_error["error"] > pi_error["latency"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
py310/bin/python -m pytest tests/test_gabanzhaf_improvements.py::test_incident_phenotype_is_normalized_with_missing_trace tests/test_gabanzhaf_improvements.py::test_incident_phenotype_detects_latency_error_and_propagation_dominance -q
```

Expected: FAIL because `_incident_phenotype` or config fields are missing.

- [ ] **Step 3: Add config fields**

Extend `_GabanzhafConfig`:

```python
phenotype_temperature: float = 0.7
phenotype_gain_resource: float = 0.35
phenotype_gain_latency: float = 0.45
phenotype_gain_error: float = 0.45
phenotype_gain_propagation: float = 0.35
```

Extend `from_kwargs()`:

```python
phenotype_temperature=float(kwargs.get("phenotype_temperature", cls.phenotype_temperature)),
phenotype_gain_resource=float(kwargs.get("phenotype_gain_resource", cls.phenotype_gain_resource)),
phenotype_gain_latency=float(kwargs.get("phenotype_gain_latency", cls.phenotype_gain_latency)),
phenotype_gain_error=float(kwargs.get("phenotype_gain_error", cls.phenotype_gain_error)),
phenotype_gain_propagation=float(kwargs.get("phenotype_gain_propagation", cls.phenotype_gain_propagation)),
```

- [ ] **Step 4: Implement phenotype helper**

Add below `_semantic_service_evidence()`:

```python
def _mean_top(values, k=3):
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return 0.0
    top = np.sort(arr)[-min(k, arr.size):]
    return float(np.mean(top))


def _softmax_dict(raw, temperature):
    keys = list(raw.keys())
    vals = np.asarray([raw[k] for k in keys], dtype=float)
    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
    tau = max(float(temperature), 1e-6)
    vals = vals / tau
    vals = vals - np.max(vals)
    exp = np.exp(vals)
    denom = float(exp.sum()) if exp.size else 1.0
    return {k: float(exp[i] / max(denom, 1e-12)) for i, k in enumerate(keys)}


def _incident_phenotype(
    svc_anom_dict,
    trace_error_scores,
    trace_latency_scores,
    semantic_evidence,
    topology_edges,
    config,
):
    resource = _mean_top(v.get("resource", 0.0) for v in semantic_evidence.values())
    latency = _mean_top(v.get("latency", 0.0) for v in semantic_evidence.values())
    latency += _mean_top((trace_latency_scores or {}).values())
    error = _mean_top((trace_error_scores or {}).values())

    edge_weight_sum = 0.0
    aligned_sum = 0.0
    for u, v, w in topology_edges:
        weight = max(float(w), 0.0)
        edge_weight_sum += weight
        aligned_sum += weight * svc_anom_dict.get(u, 0.0) * svc_anom_dict.get(v, 0.0)
    propagation = aligned_sum / max(edge_weight_sum, 1e-9)

    return _softmax_dict(
        {
            "resource": resource,
            "latency": latency,
            "error": error,
            "propagation": propagation,
        },
        config.phenotype_temperature,
    )
```

- [ ] **Step 5: Run phenotype tests**

Run:

```bash
py310/bin/python -m pytest tests/test_gabanzhaf_improvements.py::test_incident_phenotype_is_normalized_with_missing_trace tests/test_gabanzhaf_improvements.py::test_incident_phenotype_detects_latency_error_and_propagation_dominance -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add RCAEval/e2e/gabanzhaf.py tests/test_gabanzhaf_improvements.py
git commit -m "feat: infer gabanzhaf incident phenotype"
```

## Task 4: Add Phenotype-Conditioned Payoff for Banzhaf

**Files:**
- Modify: `RCAEval/e2e/gabanzhaf.py`
- Test: `tests/test_gabanzhaf_improvements.py`

- [ ] **Step 1: Write failing payoff test**

Append:

```python
def test_phenotype_conditioned_fitness_boosts_matching_evidence():
    from RCAEval.e2e.gabanzhaf import _GabanzhafConfig, _phenotype_conditioned_fitness

    cfg = _GabanzhafConfig.from_kwargs({})
    cand = ["svc-a", "svc-b"]
    svc_anom = {"svc-a": 0.8, "svc-b": 0.8}
    svc_residual = {"svc-a": 0.4, "svc-b": 0.4}
    trace_scores = {"svc-a": 0.0, "svc-b": 1.0}
    semantic = {
        "svc-a": {"resource": 1.0, "latency": 0.0, "error": 0.0, "trace": 0.0, "propagation": 0.0},
        "svc-b": {"resource": 0.0, "latency": 1.0, "error": 0.0, "trace": 1.0, "propagation": 0.0},
    }
    pi = {"resource": 0.1, "latency": 0.7, "error": 0.1, "propagation": 0.1}

    fitness = _phenotype_conditioned_fitness(
        cand, svc_anom, svc_residual, trace_scores, semantic, pi, cfg
    )

    assert fitness.shape == (2,)
    assert fitness[1] > fitness[0]
    assert np.all(np.isfinite(fitness))
    assert np.all(fitness >= 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
py310/bin/python -m pytest tests/test_gabanzhaf_improvements.py::test_phenotype_conditioned_fitness_boosts_matching_evidence -q
```

Expected: FAIL because `_phenotype_conditioned_fitness` is not defined.

- [ ] **Step 3: Implement payoff helper**

Add below `_fitness_signal()`:

```python
def _phenotype_conditioned_fitness(
    cand_list,
    svc_anom_dict,
    svc_residual,
    trace_scores,
    semantic_evidence,
    phenotype,
    config,
):
    values = []
    trace_source = trace_scores or {}
    for svc in cand_list:
        sem = semantic_evidence.get(
            svc,
            {"resource": 0.0, "latency": 0.0, "error": 0.0, "trace": 0.0, "propagation": 0.0},
        )
        anomaly = svc_anom_dict.get(svc, 0.0)
        residual = svc_residual.get(svc, 0.0)
        trace = trace_source.get(svc, sem.get("trace", 0.0))
        base = 0.50 * anomaly + 0.35 * residual + 0.15 * trace
        multiplier = (
            1.0
            + config.phenotype_gain_resource * phenotype.get("resource", 0.0) * sem.get("resource", 0.0)
            + config.phenotype_gain_latency * phenotype.get("latency", 0.0) * sem.get("latency", 0.0)
            + config.phenotype_gain_error * phenotype.get("error", 0.0) * max(sem.get("error", 0.0), sem.get("trace", 0.0))
            + config.phenotype_gain_propagation * phenotype.get("propagation", 0.0) * sem.get("propagation", 0.0)
        )
        values.append(max(0.0, base * multiplier))
    return np.asarray(values, dtype=float)
```

- [ ] **Step 4: Wire payoff into `_gabanzhaf_core()`**

After `svc_residual` is computed and before Banzhaf:

```python
semantic_evidence = _semantic_service_evidence(
    cols, metric_ev, trace_scores, topology_edges, svc_anom_dict
)
phenotype = _incident_phenotype(
    svc_anom_dict,
    kwargs.get("trace_error_scores", None),
    kwargs.get("trace_latency_scores", None),
    semantic_evidence,
    topology_edges,
    config,
)
fitness_signal = _phenotype_conditioned_fitness(
    cand_list,
    svc_anom_dict,
    svc_residual,
    trace_scores,
    semantic_evidence,
    phenotype,
    config,
)
```

Delete the old `_fitness_signal(cand_list, svc_anom_dict, svc_residual, topology_edges, config)` assignment in `_gabanzhaf_core()` so `fitness_signal` is assigned exactly once by `_phenotype_conditioned_fitness(cand_list, svc_anom_dict, svc_residual, trace_scores, semantic_evidence, phenotype, config)`.

- [ ] **Step 5: Pass trace components from entry point**

In `gabanzhaf()`, add:

```python
runtime_kwargs["trace_error_scores"] = trace_features.trace_error_scores
runtime_kwargs["trace_latency_scores"] = trace_features.trace_latency_scores
```

- [ ] **Step 6: Run payoff and Banzhaf tests**

Run:

```bash
py310/bin/python -m pytest tests/test_gabanzhaf_improvements.py::test_phenotype_conditioned_fitness_boosts_matching_evidence tests/test_gabanzhaf_improvements.py::test_deterministic_banzhaf_topology_promotes_root_cause tests/test_gabanzhaf_improvements.py::test_deterministic_banzhaf_large_graph_ranks_clear_root_first -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add RCAEval/e2e/gabanzhaf.py tests/test_gabanzhaf_improvements.py
git commit -m "feat: condition gabanzhaf payoff on phenotype"
```

## Task 5: Add Phenotype-Aware Metric Ranking

**Files:**
- Modify: `RCAEval/e2e/gabanzhaf.py`
- Test: `tests/test_gabanzhaf_improvements.py`

- [ ] **Step 1: Write failing ranking test**

Append:

```python
def test_rank_metrics_uses_bounded_semantic_match():
    from RCAEval.e2e.gabanzhaf import _rank_metrics

    cols = ["svc-a_cpu", "svc-a_latency", "svc-b_latency"]
    svc_banzhaf = {"svc-a": 1.0, "svc-b": 0.2}
    svc_residual = {"svc-a": 1.0, "svc-b": 0.2}
    svc_anom = {"svc-a": 1.0, "svc-b": 0.2}
    metric_score = {"svc-a_cpu": 0.9, "svc-a_latency": 0.8, "svc-b_latency": 1.0}
    trace_scores = {}
    phenotype = {"resource": 0.05, "latency": 0.85, "error": 0.05, "propagation": 0.05}
    semantic = {
        "svc-a": {"resource": 1.0, "latency": 0.8, "error": 0.0, "trace": 0.0, "propagation": 0.0},
        "svc-b": {"resource": 0.0, "latency": 1.0, "error": 0.0, "trace": 0.0, "propagation": 0.0},
    }

    ranks = _rank_metrics(
        cols, svc_banzhaf, svc_residual, svc_anom, metric_score, trace_scores, phenotype, semantic
    )

    assert ranks[0] == "svc-a_latency"
    assert ranks.index("svc-a_latency") < ranks.index("svc-b_latency")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
py310/bin/python -m pytest tests/test_gabanzhaf_improvements.py::test_rank_metrics_uses_bounded_semantic_match -q
```

Expected: FAIL due to `_rank_metrics()` signature mismatch or missing semantic match.

- [ ] **Step 3: Add semantic match helper**

Add near ranking helpers:

```python
def _metric_semantic_match(col, phenotype):
    group = _metric_semantic_group(col)
    if group == "resource":
        return phenotype.get("resource", 0.0)
    if group == "latency":
        return phenotype.get("latency", 0.0)
    if group == "error":
        return phenotype.get("error", 0.0)
    return 0.0
```

- [ ] **Step 4: Update `_rank_metrics()` signature and score**

Change signature to:

```python
def _rank_metrics(
    all_cols,
    svc_banzhaf,
    svc_residual,
    svc_anom_dict,
    metric_score,
    trace_scores,
    phenotype=None,
    semantic_evidence=None,
):
```

Inside `_col_score()` replace the return with:

```python
semantic = _metric_semantic_match(col, phenotype or {})
return (
    0.66 * service_root
    + 0.28 * metric_local
    + 0.18 * service_root * metric_local
    + 0.08 * semantic * metric_local
)
```

Keep `semantic_evidence` in the signature for future service-level ranking diagnostics, but do not use it in this task.

- [ ] **Step 5: Pass phenotype into ranking**

In `_gabanzhaf_core()`, update:

```python
ranks = _rank_metrics(
    cols,
    svc_banzhaf,
    svc_residual,
    svc_anom_dict,
    metric_score,
    trace_scores,
    phenotype,
    semantic_evidence,
)
```

- [ ] **Step 6: Run ranking test**

Run:

```bash
py310/bin/python -m pytest tests/test_gabanzhaf_improvements.py::test_rank_metrics_uses_bounded_semantic_match -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add RCAEval/e2e/gabanzhaf.py tests/test_gabanzhaf_improvements.py
git commit -m "feat: add phenotype aware gabanzhaf ranking"
```

## Task 6: Regression Tests and Runtime Guard

**Files:**
- Modify: `tests/test_gabanzhaf_improvements.py`
- Modify: `RCAEval/e2e/gabanzhaf.py` only if tests reveal integration defects.

- [ ] **Step 1: Add integration test for `_gabanzhaf_core()` stage timing**

Append:

```python
def test_gabanzhaf_core_returns_phenotype_when_requested():
    data = pd.DataFrame({
        "time": np.arange(12),
        "svc-a_cpu": [1, 1, 1, 1, 1, 1, 8, 8, 8, 8, 8, 8],
        "svc-a_latency": [1] * 12,
        "svc-b_cpu": [1] * 12,
        "svc-b_latency": [1, 1, 1, 1, 1, 1, 5, 5, 5, 5, 5, 5],
    })

    result = _gabanzhaf_core(
        data,
        inject_time=6,
        dataset="unit",
        anomalies=None,
        kwargs={"return_stage_timing": True, "return_diagnostics": True},
    )

    assert "ranks" in result
    assert "stage_timing" in result
    assert "phenotype" in result
    assert abs(sum(result["phenotype"].values()) - 1.0) < 1e-6
```

- [ ] **Step 2: Implement optional diagnostics**

In `_gabanzhaf_core()`, after `result = {"ranks": ranks, "node_names": cols}` add:

```python
if bool(kwargs.get("return_diagnostics", False)):
    result["phenotype"] = phenotype
```

- [ ] **Step 3: Run focused tests**

Run:

```bash
py310/bin/python -m pytest tests/test_gabanzhaf_improvements.py -q
```

Expected: PASS.

- [ ] **Step 4: Run compile check**

Run:

```bash
py310/bin/python -m py_compile RCAEval/e2e/gabanzhaf.py tests/test_gabanzhaf_improvements.py
```

Expected: no output and exit code 0.

- [ ] **Step 5: Search for forbidden hardcoding**

Run:

```bash
rg -n "re2-|DELAY|LOSS|CPU|MEM|DISK|SOCKET|ground|truth|case_id|data_path.*re2" RCAEval/e2e/gabanzhaf.py tests/test_gabanzhaf_improvements.py
```

Expected: only comments, benchmark strings in tests, or generic metric semantic keywords appear. No branch should depend on dataset name, fault label, case ID, or ground truth.

- [ ] **Step 6: Commit**

```bash
git add RCAEval/e2e/gabanzhaf.py tests/test_gabanzhaf_improvements.py
git commit -m "test: cover phenotype adaptive gabanzhaf integration"
```

## Task 7: Benchmark and Tune Only Global Parameters If Needed

**Files:**
- Modify: `RCAEval/e2e/gabanzhaf.py` only if a global parameter adjustment is needed.
- Modify: `benchmark_results.json` only after final benchmark run.

- [ ] **Step 1: Run current benchmark command**

Use the same command used to generate `benchmark_results.json`. If the local convention is `run123.py`, run:

```bash
py310/bin/python run123.py --method gabanzhaf --datasets re2-ob re2-ss re2-tt
```

Expected: completes successfully and prints metrics for all three datasets.

- [ ] **Step 2: Compare against baseline**

Record:

```text
AC@1-CPU, AC@1-MEM, AC@1-DISK, AC@1-SOCKET, AC@1-DELAY, AC@1-LOSS
Avg@5-CPU, Avg@5-MEM, Avg@5-DISK, Avg@5-SOCKET, Avg@5-DELAY, Avg@5-LOSS
wall_time_sec
Avg speed
```

Expected trend:

```text
DELAY and LOSS AC@1 improve or do not regress.
CPU/MEM/DISK/SOCKET Avg@5 do not regress by more than 0.05.
Runtime overhead remains a small constant relative to the previous gabanzhaf run.
```

- [ ] **Step 3: If regression appears, adjust only global gains**

Allowed changes are limited to `_GabanzhafConfig` defaults:

```python
phenotype_gain_resource: float
phenotype_gain_latency: float
phenotype_gain_error: float
phenotype_gain_propagation: float
phenotype_temperature: float
```

Do not add dataset branches, fault-label branches, service-name branches, or benchmark-case branches.

- [ ] **Step 4: Re-run tests after any tuning**

Run:

```bash
py310/bin/python -m pytest tests/test_gabanzhaf_improvements.py -q
py310/bin/python -m py_compile RCAEval/e2e/gabanzhaf.py tests/test_gabanzhaf_improvements.py
```

Expected: PASS and compile success.

- [ ] **Step 5: Commit benchmark or tuning result**

If only benchmark output changes:

```bash
git add benchmark_results.json
git commit -m "bench: update phenotype adaptive gabanzhaf results"
```

If global parameters also changed:

```bash
git add RCAEval/e2e/gabanzhaf.py benchmark_results.json
git commit -m "tune: calibrate phenotype adaptive gabanzhaf gains"
```

## Final Verification

- [ ] Run full focused test suite:

```bash
py310/bin/python -m pytest tests/test_gabanzhaf_improvements.py tests/test_predict_banzhaf.py -q
```

Expected: PASS.

- [ ] Run syntax check:

```bash
py310/bin/python -m py_compile RCAEval/e2e/gabanzhaf.py tests/test_gabanzhaf_improvements.py
```

Expected: no output and exit code 0.

- [ ] Confirm no forbidden hardcoding:

```bash
rg -n "if .*re2|dataset.*re2|if .*DELAY|if .*LOSS|ground_truth|root_cause|case_id" RCAEval/e2e/gabanzhaf.py
```

Expected: no matches.

- [ ] Confirm worktree state:

```bash
git status --short
```

Expected: clean, or only intended benchmark artifact changes.

## Self-Review Notes

Spec coverage:

- Trace component separation is covered by Task 1.
- Semantic grouping and per-service evidence are covered by Task 2.
- Unsupervised phenotype inference is covered by Task 3.
- Phenotype-conditioned Banzhaf payoff is covered by Task 4.
- Metric-level semantic ranking is covered by Task 5.
- Diagnostics, regression tests, and hardcoding checks are covered by Task 6.
- Benchmark validation and global-only tuning are covered by Task 7.

Scope:

- The plan keeps all production logic in `RCAEval/e2e/gabanzhaf.py`, matching the existing codebase style.
- The plan does not introduce supervised training, full metric causal graphs, deep learning, or dataset-specific branches.
