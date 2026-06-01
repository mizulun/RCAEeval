# Probabilistic Causal Banzhaf Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current phenotype-conditioned binary coverage game in `gabanzhaf.py` with a single-flow probabilistic causal Banzhaf game based on noisy-OR coverage.

**Architecture:** Keep the existing window split, preprocessing, metric evidence, service evidence, trace extraction, and topology construction. Replace the service-game layer with fixed top-K candidate selection, a causal explanation probability matrix, and closed-form exact Banzhaf for noisy-OR coverage. Simplify metric ranking to service Banzhaf times local metric evidence, without phenotype or semantic ranking correction.

**Tech Stack:** Python, NumPy, Pandas, Pytest, existing RCAEval preprocessing and benchmark runner.

---

## File Structure

- Modify: `RCAEval/e2e/gabanzhaf.py`
  - Keep `_TraceFeatures`, trace extraction, metric evidence, service evidence, topology, residual scoring, and public `gabanzhaf`.
  - Add `_select_exact_candidates`, `_causal_explanation_matrix`, `_noisy_or_coverage`, `_probabilistic_banzhaf`.
  - Replace `_banzhaf_power` production path with the probabilistic closed-form score.
  - Simplify `_rank_metrics`.
  - Remove or stop using phenotype/semantic helper functions from the main path.
- Modify: `tests/test_gabanzhaf_improvements.py`
  - Add mathematical tests for noisy-OR monotonicity, diminishing returns, and closed-form Banzhaf equivalence.
  - Replace phenotype/semantic tests with tests for the new single-flow candidate game.
  - Keep trace extraction, call graph, metric evidence, timing, and no-label/no-hardcode tests.
- Modify: `benchmark_results.json`
  - Only after benchmark execution, if the user wants the new benchmark numbers persisted.

---

### Task 1: Add Noisy-OR Game Tests

**Files:**
- Modify: `tests/test_gabanzhaf_improvements.py`

- [ ] **Step 1: Write failing tests for noisy-OR coverage and diminishing returns**

Add these tests near the current Banzhaf test section:

```python
def test_noisy_or_coverage_is_monotone_and_bounded():
    from RCAEval.e2e.gabanzhaf import _noisy_or_coverage

    prob = np.array(
        [
            [1.0, 0.6, 0.0],
            [0.0, 0.5, 0.8],
        ],
        dtype=float,
    )

    empty = _noisy_or_coverage(prob, [])
    one = _noisy_or_coverage(prob, [0])
    both = _noisy_or_coverage(prob, [0, 1])

    assert np.all(empty == 0.0)
    assert np.all(one >= empty)
    assert np.all(both >= one)
    assert np.all(both <= 1.0)


def test_noisy_or_payoff_has_diminishing_marginal_returns():
    from RCAEval.e2e.gabanzhaf import _noisy_or_coverage

    prob = np.array(
        [
            [1.0, 0.8],
            [0.0, 0.7],
            [0.0, 0.6],
        ],
        dtype=float,
    )
    weights = np.array([1.0, 1.0], dtype=float)

    def payoff(coalition):
        return float(np.dot(weights, _noisy_or_coverage(prob, coalition)))

    marginal_to_empty = payoff([0]) - payoff([])
    marginal_after_overlap = payoff([0, 1, 2]) - payoff([1, 2])

    assert marginal_to_empty > marginal_after_overlap
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```bash
py310/bin/python -m pytest tests/test_gabanzhaf_improvements.py::test_noisy_or_coverage_is_monotone_and_bounded tests/test_gabanzhaf_improvements.py::test_noisy_or_payoff_has_diminishing_marginal_returns -q
```

Expected: both fail with `ImportError` because `_noisy_or_coverage` does not exist yet.

- [ ] **Step 3: Implement `_noisy_or_coverage`**

Add this helper in `RCAEval/e2e/gabanzhaf.py` near the Banzhaf helpers:

```python
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
    if len(coalition) == 0:
        return np.zeros(prob.shape[1], dtype=float)
    rows = prob[np.asarray(list(coalition), dtype=int)]
    return 1.0 - np.prod(1.0 - rows, axis=0)
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
py310/bin/python -m pytest tests/test_gabanzhaf_improvements.py::test_noisy_or_coverage_is_monotone_and_bounded tests/test_gabanzhaf_improvements.py::test_noisy_or_payoff_has_diminishing_marginal_returns -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add RCAEval/e2e/gabanzhaf.py tests/test_gabanzhaf_improvements.py
git commit -m "test: cover noisy-or causal coverage"
```

---

### Task 2: Implement Closed-Form Exact Banzhaf

**Files:**
- Modify: `RCAEval/e2e/gabanzhaf.py`
- Modify: `tests/test_gabanzhaf_improvements.py`

- [ ] **Step 1: Write failing closed-form equivalence test**

Add this test:

```python
def test_probabilistic_banzhaf_matches_bruteforce_noisy_or_game():
    from itertools import combinations
    from RCAEval.e2e.gabanzhaf import _noisy_or_coverage, _probabilistic_banzhaf

    prob = np.array(
        [
            [1.0, 0.4, 0.2],
            [0.0, 0.7, 0.5],
            [0.1, 0.2, 0.9],
        ],
        dtype=float,
    )
    weights = np.array([0.9, 0.6, 0.4], dtype=float)

    def payoff(coalition):
        return float(np.dot(weights, _noisy_or_coverage(prob, coalition)))

    brute = np.zeros(prob.shape[0], dtype=float)
    players = list(range(prob.shape[0]))
    for i in players:
        others = [p for p in players if p != i]
        total = 0.0
        count = 0
        for size in range(len(others) + 1):
            for subset in combinations(others, size):
                total += payoff(list(subset) + [i]) - payoff(list(subset))
                count += 1
        brute[i] = total / count

    closed = _probabilistic_banzhaf(prob, weights)

    assert np.allclose(closed, brute, atol=1e-10)
    assert np.all(closed >= 0.0)
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
py310/bin/python -m pytest tests/test_gabanzhaf_improvements.py::test_probabilistic_banzhaf_matches_bruteforce_noisy_or_game -q
```

Expected: fail with `ImportError` for `_probabilistic_banzhaf`.

- [ ] **Step 3: Implement `_probabilistic_banzhaf`**

Add this helper below `_noisy_or_coverage`:

```python
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

    half_complements = 1.0 - 0.5 * prob
    product_all = np.prod(half_complements, axis=0)
    scores = np.zeros(prob.shape[0], dtype=float)
    for i in range(prob.shape[0]):
        product_without_i = product_all / np.maximum(half_complements[i], 1e-12)
        scores[i] = float(np.sum(weights * prob[i] * product_without_i))
    return np.maximum(scores, 0.0)
```

- [ ] **Step 4: Run the focused Banzhaf tests**

Run:

```bash
py310/bin/python -m pytest tests/test_gabanzhaf_improvements.py::test_probabilistic_banzhaf_matches_bruteforce_noisy_or_game tests/test_gabanzhaf_improvements.py::test_noisy_or_coverage_is_monotone_and_bounded tests/test_gabanzhaf_improvements.py::test_noisy_or_payoff_has_diminishing_marginal_returns -q
```

Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add RCAEval/e2e/gabanzhaf.py tests/test_gabanzhaf_improvements.py
git commit -m "feat: add closed-form probabilistic banzhaf"
```

---

### Task 3: Add Candidate Selection and Explanation Matrix

**Files:**
- Modify: `RCAEval/e2e/gabanzhaf.py`
- Modify: `tests/test_gabanzhaf_improvements.py`

- [ ] **Step 1: Write failing tests for fixed top-K selection and `p_ij` construction**

Add these tests:

```python
def test_select_exact_candidates_uses_fixed_unsupervised_score():
    from RCAEval.e2e.gabanzhaf import _GabanzhafConfig, _select_exact_candidates

    cfg = _GabanzhafConfig.from_kwargs({"exact_candidate_k": 3})
    candidates = ["svc-a", "svc-b", "svc-c", "svc-d"]
    selected = _select_exact_candidates(
        candidates,
        svc_anom_dict={"svc-a": 0.9, "svc-b": 0.1, "svc-c": 0.5, "svc-d": 0.4},
        svc_residual={"svc-a": 0.1, "svc-b": 0.9, "svc-c": 0.4, "svc-d": 0.2},
        trace_scores={"svc-d": 1.0},
        topology_edges=[("svc-c", "svc-a", 0.8), ("svc-c", "svc-d", 0.7)],
        config=cfg,
    )

    assert len(selected) == 3
    assert selected[0] in {"svc-a", "svc-c"}
    assert "svc-b" in selected


def test_causal_explanation_matrix_builds_self_and_topology_probabilities():
    from RCAEval.e2e.gabanzhaf import _GabanzhafConfig, _causal_explanation_matrix

    cfg = _GabanzhafConfig.from_kwargs({"causal_probability_lambda": 1.5})
    cand = ["svc-a", "svc-b"]
    services = ["svc-a", "svc-b", "svc-c"]
    matrix = _causal_explanation_matrix(
        cand,
        services,
        svc_anom_dict={"svc-a": 1.0, "svc-b": 0.8, "svc-c": 0.5},
        topology_edges=[("svc-a", "svc-c", 0.6), ("svc-b", "svc-c", 0.2)],
        config=cfg,
    )

    assert matrix.shape == (2, 3)
    assert matrix[0, 0] == 1.0
    assert matrix[1, 1] == 1.0
    assert 0.0 < matrix[0, 2] <= 1.0
    assert 0.0 < matrix[1, 2] < matrix[0, 2]
    assert matrix[0, 1] == 0.0
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
py310/bin/python -m pytest tests/test_gabanzhaf_improvements.py::test_select_exact_candidates_uses_fixed_unsupervised_score tests/test_gabanzhaf_improvements.py::test_causal_explanation_matrix_builds_self_and_topology_probabilities -q
```

Expected: fail because the new helpers/config fields do not exist.

- [ ] **Step 3: Update `_GabanzhafConfig`**

Modify the dataclass and `from_kwargs` in `RCAEval/e2e/gabanzhaf.py`:

```python
    exact_candidate_k: int = 12
    causal_probability_lambda: float = 1.5
    metric_rank_epsilon: float = 0.15
```

and inside `from_kwargs`:

```python
            exact_candidate_k=int(kwargs.get("exact_candidate_k", cls.exact_candidate_k)),
            causal_probability_lambda=float(
                kwargs.get("causal_probability_lambda", cls.causal_probability_lambda)
            ),
            metric_rank_epsilon=float(kwargs.get("metric_rank_epsilon", cls.metric_rank_epsilon)),
```

Keep legacy config fields temporarily if existing tests still import them. They will be removed from the main path in Task 5.

- [ ] **Step 4: Implement `_select_exact_candidates` and `_causal_explanation_matrix`**

Add these helpers after `_candidate_services`:

```python
def _select_exact_candidates(candidates, svc_anom_dict, svc_residual, trace_scores, topology_edges, config):
    k = max(1, int(config.exact_candidate_k))
    trace_source = trace_scores or {}
    graph_participation = {svc: 0.0 for svc in candidates}
    for u, v, w in topology_edges:
        weight = _finite_nonnegative(w)
        if u in graph_participation:
            graph_participation[u] += weight
        if v in graph_participation:
            graph_participation[v] += weight
    graph_participation = _norm_trace_component_dict(graph_participation)

    def score(svc):
        return (
            0.50 * _finite_nonnegative(svc_anom_dict.get(svc, 0.0))
            + 0.30 * _finite_nonnegative(svc_residual.get(svc, 0.0))
            + 0.15 * _finite_nonnegative(trace_source.get(svc, 0.0))
            + 0.05 * _finite_nonnegative(graph_participation.get(svc, 0.0))
        )

    return sorted(candidates, key=lambda svc: (score(svc), svc), reverse=True)[:k]


def _causal_explanation_matrix(cand_list, service_list, svc_anom_dict, topology_edges, config):
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
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
py310/bin/python -m pytest tests/test_gabanzhaf_improvements.py::test_select_exact_candidates_uses_fixed_unsupervised_score tests/test_gabanzhaf_improvements.py::test_causal_explanation_matrix_builds_self_and_topology_probabilities -q
```

Expected: `2 passed`.

- [ ] **Step 6: Commit**

```bash
git add RCAEval/e2e/gabanzhaf.py tests/test_gabanzhaf_improvements.py
git commit -m "feat: build probabilistic banzhaf candidate game"
```

---

### Task 4: Integrate Probabilistic Banzhaf Into Core Pipeline

**Files:**
- Modify: `RCAEval/e2e/gabanzhaf.py`
- Modify: `tests/test_gabanzhaf_improvements.py`

- [ ] **Step 1: Replace old Banzhaf backend tests with single-flow tests**

Remove tests that assert deterministic or MC behavior:

```text
test_gabanzhaf_config_defaults_to_deterministic_banzhaf
test_banzhaf_power_uses_mc_only_when_explicit
test_deterministic_banzhaf_matches_two_node_analytic_case
test_banzhaf_power_default_is_deterministic_not_mc
test_banzhaf_power_rejects_invalid_mode
test_banzhaf_power_default_uses_same_path_for_small_and_large
test_deterministic_banzhaf_no_topology_preserves_anomaly_order
test_deterministic_banzhaf_topology_promotes_root_cause
test_deterministic_banzhaf_topology_no_negative_values
test_deterministic_banzhaf_large_graph_ranks_clear_root_first
test_deterministic_banzhaf_single_service
```

Add this replacement test:

```python
def test_banzhaf_power_uses_probabilistic_game_single_flow():
    from RCAEval.e2e.gabanzhaf import _banzhaf_power

    prob = np.array(
        [
            [1.0, 0.7],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    weights = np.array([0.9, 0.5], dtype=float)

    values = _banzhaf_power(prob, weights, _GabanzhafConfig.from_kwargs({}))

    assert values.shape == (2,)
    assert values[0] > values[1]
    assert np.all(values >= 0.0)
```

- [ ] **Step 2: Run replacement test and verify it fails**

Run:

```bash
py310/bin/python -m pytest tests/test_gabanzhaf_improvements.py::test_banzhaf_power_uses_probabilistic_game_single_flow -q
```

Expected: fail because `_banzhaf_power` still expects `(fitness_signal, topo_matrix, config)`.

- [ ] **Step 3: Replace `_banzhaf_power` implementation**

Change `_banzhaf_power` to:

```python
def _banzhaf_power(prob_matrix, anomaly_weights, config):
    return _probabilistic_banzhaf(prob_matrix, anomaly_weights)
```

Leave `_mc_banzhaf` and `_deterministic_banzhaf_coverage` only if needed by older external imports; they must not be called from `_gabanzhaf_core`.

- [ ] **Step 4: Integrate into `_gabanzhaf_core`**

Replace the block that builds `semantic_evidence`, `phenotype`, `fitness_signal`, `topo_matrix`, and calls `_banzhaf_power` with:

```python
    game_candidates = _select_exact_candidates(
        cand_list,
        svc_anom_dict,
        svc_residual,
        trace_scores,
        topology_edges,
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

    banzhaf_values = _banzhaf_power(prob_matrix, anomaly_weights, config)
    svc_banzhaf = {
        game_candidates[i]: float(banzhaf_values[i])
        for i in range(len(game_candidates))
    }
```

Remove the `phenotype` diagnostic from `_gabanzhaf_core`. If `return_diagnostics=True`, return game diagnostics instead:

```python
    if bool(kwargs.get("return_diagnostics", False)):
        result["game_candidates"] = game_candidates
```

- [ ] **Step 5: Run call graph and core tests**

Run:

```bash
py310/bin/python -m pytest tests/test_gabanzhaf_improvements.py::test_callgraph_demotes_caller_service tests/test_gabanzhaf_improvements.py::test_callgraph_edges_reach_banzhaf_topology_when_lag_corr_is_empty tests/test_gabanzhaf_improvements.py::test_gabanzhaf_core_can_return_stage_timing tests/test_gabanzhaf_improvements.py::test_banzhaf_power_uses_probabilistic_game_single_flow -q
```

Expected: all pass after adapting any monkeypatches to the new `_banzhaf_power(prob_matrix, anomaly_weights, config)` signature.

- [ ] **Step 6: Commit**

```bash
git add RCAEval/e2e/gabanzhaf.py tests/test_gabanzhaf_improvements.py
git commit -m "feat: use probabilistic causal banzhaf in gabanzhaf core"
```

---

### Task 5: Simplify Metric Ranking and Remove Phenotype/Semantic Main-Path Logic

**Files:**
- Modify: `RCAEval/e2e/gabanzhaf.py`
- Modify: `tests/test_gabanzhaf_improvements.py`

- [ ] **Step 1: Replace ranking tests**

Remove tests that depend on phenotype or semantic ranking:

```text
test_metric_semantic_group_uses_generic_suffixes
test_semantic_service_evidence_combines_metrics_trace_and_topology
test_semantic_service_evidence_preserves_singleton_positive_components
test_semantic_service_evidence_preserves_broadcast_equal_positive_components
test_incident_phenotype_is_normalized_with_missing_trace
test_incident_phenotype_detects_latency_error_and_propagation_dominance
test_incident_phenotype_uses_metric_error_without_trace_errors
test_incident_phenotype_missing_semantic_evidence_is_neutral
test_incident_phenotype_invalid_temperature_is_finite
test_incident_phenotype_pure_propagation_can_dominate
test_phenotype_conditioned_fitness_boosts_matching_evidence
test_phenotype_conditioned_fitness_handles_missing_optional_evidence
test_phenotype_conditioned_fitness_sanitizes_nonfinite_values
test_phenotype_conditioned_fitness_empty_candidates_returns_empty_vector
test_phenotype_conditioned_fitness_sanitizes_nonfinite_gain_knobs
test_rank_metrics_uses_bounded_semantic_match
test_metric_semantic_match_sanitizes_invalid_phenotype
test_rank_metrics_without_phenotype_uses_legacy_formula
test_gabanzhaf_core_returns_phenotype_when_requested
```

Add:

```python
def test_rank_metrics_uses_service_banzhaf_times_local_metric_evidence():
    from RCAEval.e2e.gabanzhaf import _rank_metrics

    cols = ["svc-a_cpu", "svc-a_mem", "svc-b_cpu"]
    ranks = _rank_metrics(
        cols,
        svc_banzhaf={"svc-a": 1.0, "svc-b": 0.2},
        metric_score={"svc-a_cpu": 0.9, "svc-a_mem": 0.1, "svc-b_cpu": 1.0},
        config=_GabanzhafConfig.from_kwargs({"metric_rank_epsilon": 0.15}),
    )

    assert ranks[0] == "svc-a_cpu"
    assert ranks.index("svc-a_mem") < ranks.index("svc-b_cpu")


def test_gabanzhaf_core_returns_game_candidates_when_requested():
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
    assert "game_candidates" in result
    assert len(result["game_candidates"]) > 0
```

- [ ] **Step 2: Run tests and verify failures**

Run:

```bash
py310/bin/python -m pytest tests/test_gabanzhaf_improvements.py::test_rank_metrics_uses_service_banzhaf_times_local_metric_evidence tests/test_gabanzhaf_improvements.py::test_gabanzhaf_core_returns_game_candidates_when_requested -q
```

Expected: fail until `_rank_metrics` signature and diagnostics are simplified.

- [ ] **Step 3: Replace `_rank_metrics`**

Replace `_rank_metrics` with:

```python
def _rank_metrics(all_cols, svc_banzhaf, metric_score, config):
    nb = _norm_dict(svc_banzhaf)
    nm = _norm_dict(metric_score)
    epsilon = _finite_nonnegative(config.metric_rank_epsilon)

    def _col_score(col):
        svc = _parse_svc(col)
        return nb.get(svc, 0.0) * (epsilon + nm.get(col, 0.0))

    return sorted(all_cols, key=_col_score, reverse=True)
```

Update the call site in `_gabanzhaf_core`:

```python
    ranks = _rank_metrics(
        cols,
        svc_banzhaf,
        metric_score,
        config,
    )
```

- [ ] **Step 4: Remove phenotype/semantic helpers from the main file**

Delete these helpers if no remaining tests import them:

```text
_metric_semantic_group
_metric_semantic_match
_semantic_service_evidence
_mean_top
_softmax_dict
_incident_phenotype
_phenotype_conditioned_fitness
_service_root_score
```

Keep `_finite_nonnegative`, `_norm_dict`, `_norm_trace_component_dict`, and trace component extraction because they are still useful.

- [ ] **Step 5: Run focused ranking tests**

Run:

```bash
py310/bin/python -m pytest tests/test_gabanzhaf_improvements.py::test_rank_metrics_uses_service_banzhaf_times_local_metric_evidence tests/test_gabanzhaf_improvements.py::test_gabanzhaf_core_returns_game_candidates_when_requested -q
```

Expected: `2 passed`.

- [ ] **Step 6: Commit**

```bash
git add RCAEval/e2e/gabanzhaf.py tests/test_gabanzhaf_improvements.py
git commit -m "refactor: simplify gabanzhaf ranking to probabilistic game output"
```

---

### Task 6: No-Hardcode Guard and Full Test Pass

**Files:**
- Modify: `tests/test_gabanzhaf_improvements.py`
- Modify: `RCAEval/e2e/gabanzhaf.py` only if this task finds violations

- [ ] **Step 1: Add implementation guard test**

Add:

```python
def test_gabanzhaf_contains_no_dataset_or_fault_specific_branches():
    source_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "RCAEval",
        "e2e",
        "gabanzhaf.py",
    )
    with open(source_path, "r", encoding="utf-8") as f:
        source = f.read()

    forbidden = [
        "ground_truth",
        "root_cause",
        "case_id",
        "if dataset",
        "re2-ob",
        "re2-ss",
        "re2-tt",
        "DELAY",
        "LOSS",
    ]
    for token in forbidden:
        assert token not in source
```

- [ ] **Step 2: Run all gabanzhaf tests**

Run:

```bash
py310/bin/python -m pytest tests/test_gabanzhaf_improvements.py tests/test_predict_banzhaf.py -q
```

Expected: all tests pass.

- [ ] **Step 3: Compile the edited files**

Run:

```bash
py310/bin/python -m py_compile RCAEval/e2e/gabanzhaf.py tests/test_gabanzhaf_improvements.py
```

Expected: exit code `0`.

- [ ] **Step 4: Search for forbidden branch patterns**

Run:

```bash
rg -n "if .*re2|dataset.*re2|if .*DELAY|if .*LOSS|ground_truth|root_cause|case_id|fallback|phenotype|semantic" RCAEval/e2e/gabanzhaf.py
```

Expected: no matches for dataset/fault/ground-truth terms. If `fallback`, `phenotype`, or `semantic` remains only in comments or dead legacy helpers, remove it unless an external import requires the helper.

- [ ] **Step 5: Commit**

```bash
git add RCAEval/e2e/gabanzhaf.py tests/test_gabanzhaf_improvements.py
git commit -m "test: verify probabilistic gabanzhaf single-flow constraints"
```

---

### Task 7: Benchmark and Result Triage

**Files:**
- Modify: `benchmark_results.json` only after benchmark numbers are accepted for persistence.

- [ ] **Step 1: Run final benchmark**

Run:

```bash
py310/bin/python run123.py --methods gabanzhaf --datasets re2-ob re2-ss re2-tt --workers 1 --output /private/tmp/probabilistic_causal_gabanzhaf.csv
```

Expected: benchmark completes and writes `/private/tmp/probabilistic_causal_gabanzhaf.csv`.

- [ ] **Step 2: Inspect metric table**

Run:

```bash
cat /private/tmp/probabilistic_causal_gabanzhaf.csv
```

Expected: review `AC@1`, `AC@3`, `AC@5`, `Avg@5`, wall time, and speed for `CPU`, `MEM`, `DISK`, `SOCKET`, `DELAY`, and `LOSS`.

- [ ] **Step 3: If benchmark regresses, inspect top ranks without adding hardcode**

Run a small diagnostic script only for analysis:

```bash
py310/bin/python - <<'PY'
import json
import pandas as pd
from pathlib import Path

path = Path('/private/tmp/probabilistic_causal_gabanzhaf.csv')
print(pd.read_csv(path).to_string(index=False))
PY
```

Expected: identify whether failures are caused by candidate truncation, topology direction, or metric projection. Do not add dataset/fault/service exceptions.

- [ ] **Step 4: Commit benchmark result only if accepted**

If the benchmark result should replace the current result file:

```bash
git add benchmark_results.json
git commit -m "bench: record probabilistic gabanzhaf results"
```

If `benchmark_results.json` already contains unrelated user changes, do not overwrite them. Report the benchmark CSV path instead.

---

## Self-Review

- Spec coverage: covered fixed top-K game, noisy-OR payoff, closed-form exact Banzhaf, simple metric projection, single-flow/no-fallback constraints, and benchmark validation.
- Placeholder scan: no unresolved placeholders or vague implementation steps remain.
- Type consistency: `_banzhaf_power(prob_matrix, anomaly_weights, config)`, `_causal_explanation_matrix(cand_list, service_list, svc_anom_dict, topology_edges, config)`, and `_rank_metrics(all_cols, svc_banzhaf, metric_score, config)` are used consistently across tasks.
