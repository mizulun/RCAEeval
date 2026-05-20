# Gabanzhaf Runtime Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce `gabanzhaf` runtime on large systems by merging trace I/O and making deterministic causal-coverage Banzhaf the production default while retaining MC as an explicit experimental backend.

**Architecture:** Keep all changes inside `RCAEval/e2e/gabanzhaf.py` and focused regression tests in `tests/test_gabanzhaf_improvements.py`. Add `_extract_trace_features()` as the single trace reader, add `_banzhaf_power()` as the single backend dispatcher, and keep `_gabanzhaf_core()` as an orchestration layer. No dataset-specific or service-count-specific algorithm switching is allowed.

**Tech Stack:** Python, pandas, numpy, pytest, existing RCAEval benchmark scripts.

---

## File Structure

- Modify `RCAEval/e2e/gabanzhaf.py`
  - Add `_TraceFeatures` data container.
  - Add `_extract_trace_features()`.
  - Keep `_trace_onset_scores()` and `_extract_call_graph()` only if needed for backwards tests, but route production through `_extract_trace_features()`.
  - Add deterministic Banzhaf backend and `_banzhaf_power()` dispatcher.
  - Add optional stage timing hooks without changing normal return schema unless explicitly enabled.
- Modify `tests/test_gabanzhaf_improvements.py`
  - Add tests for merged trace extraction.
  - Add tests for deterministic Banzhaf and MC mode.
  - Keep existing call graph and ranking regression tests.
- Optional update `benchmark_results.json`
  - Only after running the approved benchmark comparison.

---

### Task 1: Merge Trace I/O Into `_extract_trace_features`

**Files:**
- Modify: `RCAEval/e2e/gabanzhaf.py`
- Test: `tests/test_gabanzhaf_improvements.py`

- [ ] **Step 1: Write failing tests for merged trace extraction**

Append these tests near the existing `_extract_call_graph` tests in `tests/test_gabanzhaf_improvements.py`:

```python
from RCAEval.e2e.gabanzhaf import _extract_trace_features, _TraceFeatures


def test_extract_trace_features_returns_scores_and_call_graph():
    rows = [
        ("svc-a", "span-a", "root", 1000, 0, 10),
        ("svc-b", "span-b", "span-a", 1001, 0, 5),
        ("svc-b", "span-c", "span-a", 60000, 1, 200),
        ("svc-c", "span-d", "span-b", 60001, 0, 30),
    ]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "traces.csv")
        _make_traces_csv(rows, path)

        features = _extract_trace_features(d, inject_time=50)

    assert isinstance(features, _TraceFeatures)
    assert "svc-b" in features.trace_scores
    assert features.trace_scores["svc-b"] > 0.0
    assert features.call_graph == {"svc-a": {"svc-b"}, "svc-b": {"svc-c"}}


def test_extract_trace_features_missing_file_returns_neutral_features():
    features = _extract_trace_features("/nonexistent/path", inject_time=50)
    assert features == _TraceFeatures(trace_scores=None, call_graph={})


def test_extract_trace_features_keeps_call_graph_when_score_columns_missing():
    rows = [
        ("svc-a", "span-a", "root"),
        ("svc-b", "span-b", "span-a"),
    ]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "traces.csv")
        pd.DataFrame(rows, columns=["serviceName", "spanID", "parentSpanID"]).to_csv(path, index=False)

        features = _extract_trace_features(d, inject_time=50)

    assert features.trace_scores is None
    assert features.call_graph == {"svc-a": {"svc-b"}}


def test_extract_trace_features_keeps_scores_when_span_columns_missing():
    rows = [
        ("svc-a", 1000, 0, 10),
        ("svc-a", 60000, 1, 200),
    ]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "traces.csv")
        pd.DataFrame(
            rows,
            columns=["serviceName", "startTimeMillis", "statusCode", "duration"],
        ).to_csv(path, index=False)

        features = _extract_trace_features(d, inject_time=50)

    assert features.call_graph == {}
    assert features.trace_scores is not None
    assert "svc-a" in features.trace_scores
```

- [ ] **Step 2: Run tests and verify they fail for missing symbols**

Run:

```bash
py310/bin/python -m pytest tests/test_gabanzhaf_improvements.py::test_extract_trace_features_returns_scores_and_call_graph -q
```

Expected: import failure or `NameError` for `_extract_trace_features` / `_TraceFeatures`.

- [ ] **Step 3: Add `_TraceFeatures` and `_extract_trace_features()`**

In `RCAEval/e2e/gabanzhaf.py`, add this near `_GabanzhafConfig`:

```python
@dataclass(frozen=True)
class _TraceFeatures:
    trace_scores: dict | None
    call_graph: dict
```

Add this implementation below `_extract_call_graph()` or replace the production use of the two older helpers with it:

```python
def _extract_trace_features(case_dir, inject_time):
    if not case_dir:
        return _TraceFeatures(trace_scores=None, call_graph={})

    trace_path = os.path.join(case_dir, "traces.csv")
    if not os.path.exists(trace_path):
        return _TraceFeatures(trace_scores=None, call_graph={})

    try:
        header = pd.read_csv(trace_path, nrows=0)
    except Exception:
        return _TraceFeatures(trace_scores=None, call_graph={})

    available = set(header.columns)
    span_cols = ["serviceName", "spanID", "parentSpanID"]
    score_cols = ["serviceName", "startTimeMillis", "statusCode", "duration"]
    usecols = sorted((set(span_cols) | set(score_cols)) & available)

    if "serviceName" not in available or not usecols:
        return _TraceFeatures(trace_scores=None, call_graph={})

    call_graph = {}
    score_chunks_normal = []
    score_chunks_anomal = []
    anomal_rows = 0
    max_anomal_rows = 30_000
    span_to_svc = {}
    edge_rows_seen = 0
    max_edge_rows = 50_000
    inject_ms = inject_time * 1000

    try:
        for chunk in pd.read_csv(trace_path, usecols=usecols, chunksize=10_000, on_bad_lines="skip"):
            if set(span_cols).issubset(chunk.columns) and edge_rows_seen < max_edge_rows:
                edge_chunk = chunk[span_cols].head(max_edge_rows - edge_rows_seen)
                span_to_svc.update(edge_chunk.set_index("spanID")["serviceName"].to_dict())
                edge_chunk = edge_chunk.copy()
                edge_chunk["parentSvc"] = edge_chunk["parentSpanID"].map(span_to_svc)
                edges = edge_chunk[
                    edge_chunk["parentSvc"].notna()
                    & (edge_chunk["parentSvc"] != edge_chunk["serviceName"])
                ]
                for caller, callee in zip(edges["parentSvc"], edges["serviceName"]):
                    call_graph.setdefault(str(caller), set()).add(str(callee))
                edge_rows_seen += len(edge_chunk)

            if set(score_cols).issubset(chunk.columns) and anomal_rows < max_anomal_rows:
                ts = pd.to_numeric(chunk["startTimeMillis"], errors="coerce")
                is_norm = ts < inject_ms
                is_anom = ts >= inject_ms
                if is_norm.any():
                    score_chunks_normal.append(chunk.loc[is_norm, score_cols])
                if is_anom.any():
                    anomal = chunk.loc[is_anom, score_cols]
                    remaining = max_anomal_rows - anomal_rows
                    score_chunks_anomal.append(anomal.head(remaining))
                    anomal_rows += min(len(anomal), remaining)

            if edge_rows_seen >= max_edge_rows and anomal_rows >= max_anomal_rows:
                break
    except Exception:
        return _TraceFeatures(trace_scores=None, call_graph={})

    trace_scores = _trace_scores_from_chunks(score_chunks_normal, score_chunks_anomal)
    return _TraceFeatures(trace_scores=trace_scores, call_graph=call_graph)
```

Add this helper near the trace feature code:

```python
def _trace_scores_from_chunks(chunks_normal, chunks_anomal):
    if not chunks_anomal:
        return None

    anomal_df = pd.concat(chunks_anomal, ignore_index=True)
    anomal_df["is_err"] = (
        pd.to_numeric(anomal_df["statusCode"], errors="coerce").fillna(0.0) != 0.0
    )
    anomal_df["dur"] = pd.to_numeric(anomal_df["duration"], errors="coerce").fillna(0.0)
    anomal_stats = anomal_df.groupby("serviceName", sort=False).agg(
        cnt=("serviceName", "size"),
        err_cnt=("is_err", "sum"),
        dur_sum=("dur", "sum"),
    )

    normal_dur_avg = {}
    if chunks_normal:
        normal_df = pd.concat(chunks_normal, ignore_index=True)
        normal_df["dur"] = pd.to_numeric(normal_df["duration"], errors="coerce").fillna(0.0)
        normal_stats = normal_df.groupby("serviceName", sort=False).agg(
            cnt=("serviceName", "size"),
            dur_sum=("dur", "sum"),
        )
        for svc, row in normal_stats.iterrows():
            normal_dur_avg[str(svc)] = float(row["dur_sum"]) / max(int(row["cnt"]), 1)

    scores = {}
    for svc, row in anomal_stats.iterrows():
        svc = str(svc)
        count = max(int(row["cnt"]), 1)
        err_rate = float(row["err_cnt"]) / count
        anomal_avg_dur = float(row["dur_sum"]) / count
        normal_avg_dur = normal_dur_avg.get(svc, 1.0)
        dur_ratio = (anomal_avg_dur / max(normal_avg_dur, 1.0)) - 1.0
        dur_signal = max(0.0, min(dur_ratio, 10.0)) / 10.0
        scores[svc] = max(err_rate, dur_signal)

    if not scores:
        return None

    lo, hi = min(scores.values()), max(scores.values())
    span = hi - lo if hi > lo else 1.0
    return {svc: (value - lo) / span for svc, value in scores.items()}
```

Update `gabanzhaf()` to use one trace extraction:

```python
trace_features = _extract_trace_features(case_dir, _t_inject)
runtime_kwargs["trace_scores"] = trace_features.trace_scores
runtime_kwargs["call_graph"] = trace_features.call_graph
```

- [ ] **Step 4: Run trace feature tests**

Run:

```bash
py310/bin/python -m pytest tests/test_gabanzhaf_improvements.py::test_extract_trace_features_returns_scores_and_call_graph tests/test_gabanzhaf_improvements.py::test_extract_trace_features_missing_file_returns_neutral_features tests/test_gabanzhaf_improvements.py::test_extract_trace_features_keeps_call_graph_when_score_columns_missing tests/test_gabanzhaf_improvements.py::test_extract_trace_features_keeps_scores_when_span_columns_missing -q
```

Expected: all four tests pass.

- [ ] **Step 5: Run existing trace and call graph regressions**

Run:

```bash
py310/bin/python -m pytest tests/test_gabanzhaf_improvements.py::test_extract_call_graph_basic tests/test_gabanzhaf_improvements.py::test_extract_call_graph_no_self_loops tests/test_gabanzhaf_improvements.py::test_extract_call_graph_missing_file tests/test_gabanzhaf_improvements.py::test_extract_call_graph_row_limit -q
```

Expected: all existing call graph tests pass.

- [ ] **Step 6: Commit Task 1**

```bash
git add RCAEval/e2e/gabanzhaf.py tests/test_gabanzhaf_improvements.py
git commit -m "perf: merge gabanzhaf trace feature extraction"
```

If the worktree contains unrelated user changes, do not include them in this commit.

---

### Task 2: Add Deterministic Banzhaf Backend and Keep MC as Explicit Mode

**Files:**
- Modify: `RCAEval/e2e/gabanzhaf.py`
- Test: `tests/test_gabanzhaf_improvements.py`

- [ ] **Step 1: Write failing tests for backend config and deterministic coverage**

Append these tests near the existing Banzhaf tests:

```python
from RCAEval.e2e.gabanzhaf import _banzhaf_power, _deterministic_banzhaf_coverage


def test_gabanzhaf_config_defaults_to_deterministic_banzhaf():
    cfg = _GabanzhafConfig.from_kwargs({})
    assert cfg.banzhaf_mode == "deterministic"


def test_banzhaf_power_uses_mc_only_when_explicit(monkeypatch):
    calls = []

    def fake_mc(fitness_signal, topo_matrix, n_samples, seed=42):
        calls.append((len(fitness_signal), n_samples, seed))
        return np.asarray(fitness_signal, dtype=float) + 2.0

    monkeypatch.setattr(gabanzhaf_mod, "_mc_banzhaf", fake_mc)
    cfg = _GabanzhafConfig.from_kwargs({
        "banzhaf_mode": "mc",
        "banzhaf_samples": "123",
        "seed": "7",
    })

    values = _banzhaf_power(
        np.array([0.5, 0.25], dtype=float),
        np.zeros((2, 2), dtype=np.float32),
        cfg,
    )

    assert calls == [(2, 123, 7)]
    assert np.allclose(values, np.array([2.5, 2.25]))


def test_deterministic_banzhaf_matches_two_node_analytic_case():
    fitness = np.array([0.9, 0.5], dtype=float)
    topo = np.zeros((2, 2), dtype=np.float32)
    topo[0, 1] = 0.5

    values = _deterministic_banzhaf_coverage(fitness, topo)

    assert np.allclose(values, np.array([1.15, 0.25]))


def test_banzhaf_power_default_is_deterministic_not_mc(monkeypatch):
    def fail_mc(*args, **kwargs):
        raise AssertionError("MC should not be used in deterministic default mode")

    monkeypatch.setattr(gabanzhaf_mod, "_mc_banzhaf", fail_mc)
    cfg = _GabanzhafConfig.from_kwargs({})

    values = _banzhaf_power(
        np.array([0.9, 0.5], dtype=float),
        np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32),
        cfg,
    )

    assert values[0] > values[1]
```

- [ ] **Step 2: Run tests and verify they fail for missing symbols or config field**

Run:

```bash
py310/bin/python -m pytest tests/test_gabanzhaf_improvements.py::test_gabanzhaf_config_defaults_to_deterministic_banzhaf -q
```

Expected: failure because `_GabanzhafConfig` has no `banzhaf_mode` yet.

- [ ] **Step 3: Add `banzhaf_mode` to `_GabanzhafConfig`**

Modify `_GabanzhafConfig`:

```python
banzhaf_mode: str = "deterministic"
```

Modify `from_kwargs()`:

```python
banzhaf_mode=str(kwargs.get("banzhaf_mode", cls.banzhaf_mode)),
```

- [ ] **Step 4: Implement deterministic coverage Banzhaf**

Add this below `_approximate_banzhaf_topology()`:

```python
def _deterministic_banzhaf_coverage(fitness_signal, topo_matrix):
    fitness = np.asarray(fitness_signal, dtype=float).reshape(-1)
    topo = np.nan_to_num(np.asarray(topo_matrix, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    n = len(fitness)
    if n == 0:
        return np.zeros(0, dtype=float)

    explains = topo > 0
    np.fill_diagonal(explains, True)

    explainer_counts = explains.sum(axis=0).astype(float)
    explainer_counts = np.maximum(explainer_counts, 1.0)
    marginal_weight = fitness * np.power(0.5, explainer_counts - 1.0)
    values = explains @ marginal_weight
    return np.maximum(0.0, np.asarray(values, dtype=float))
```

- [ ] **Step 5: Add `_banzhaf_power()` dispatcher**

Add this near Banzhaf helpers:

```python
def _banzhaf_power(fitness_signal, topo_matrix, config):
    if config.banzhaf_mode == "mc":
        return _mc_banzhaf(
            fitness_signal,
            topo_matrix,
            n_samples=config.banzhaf_samples,
            seed=config.seed,
        )
    if config.banzhaf_mode != "deterministic":
        raise ValueError(f"Unsupported banzhaf_mode: {config.banzhaf_mode}")
    return _deterministic_banzhaf_coverage(fitness_signal, topo_matrix)
```

Update `_approximate_banzhaf_topology()` only if tests still use it:

```python
def _approximate_banzhaf_topology(fitness_signal, topo_matrix, n_samples=1000, seed=42):
    return _mc_banzhaf(fitness_signal, topo_matrix, n_samples, seed=seed)
```

This keeps old tests meaningful as MC-specific tests while production moves to `_banzhaf_power()`.

- [ ] **Step 6: Route `_gabanzhaf_core()` through `_banzhaf_power()`**

Replace:

```python
banzhaf_values = _approximate_banzhaf_topology(
    fitness_signal,
    topo_matrix,
    n_samples=config.banzhaf_samples,
    seed=config.seed,
)
```

with:

```python
banzhaf_values = _banzhaf_power(fitness_signal, topo_matrix, config)
```

- [ ] **Step 7: Run Banzhaf tests**

Run:

```bash
py310/bin/python -m pytest tests/test_gabanzhaf_improvements.py::test_gabanzhaf_config_defaults_to_deterministic_banzhaf tests/test_gabanzhaf_improvements.py::test_banzhaf_power_uses_mc_only_when_explicit tests/test_gabanzhaf_improvements.py::test_deterministic_banzhaf_matches_two_node_analytic_case tests/test_gabanzhaf_improvements.py::test_banzhaf_power_default_is_deterministic_not_mc -q
```

Expected: all four tests pass.

- [ ] **Step 8: Run full gabanzhaf improvement tests**

Run:

```bash
py310/bin/python -m pytest tests/test_gabanzhaf_improvements.py -q
```

Expected: all tests pass.

- [ ] **Step 9: Commit Task 2**

```bash
git add RCAEval/e2e/gabanzhaf.py tests/test_gabanzhaf_improvements.py
git commit -m "perf: add deterministic gabanzhaf banzhaf backend"
```

If commits are not desired in this workspace, record the exact diff and test output instead.

---

### Task 3: Add Stage Timing for Benchmark Comparison

**Files:**
- Modify: `RCAEval/e2e/gabanzhaf.py`
- Test: `tests/test_gabanzhaf_improvements.py`

- [ ] **Step 1: Write failing test for optional timing output**

Append:

```python
def test_gabanzhaf_core_can_return_stage_timing():
    df = _make_simple_dataframe("svc-root", "svc-caller")

    result = _gabanzhaf_core(
        df,
        inject_time=50,
        dataset=None,
        anomalies=None,
        kwargs={"call_graph": {}, "return_stage_timing": True},
    )

    assert "stage_timing" in result
    for key in ["preprocess", "metric_evidence", "topology", "banzhaf", "ranking"]:
        assert key in result["stage_timing"]
        assert result["stage_timing"][key] >= 0.0
```

- [ ] **Step 2: Run timing test and verify it fails**

Run:

```bash
py310/bin/python -m pytest tests/test_gabanzhaf_improvements.py::test_gabanzhaf_core_can_return_stage_timing -q
```

Expected: failure because `stage_timing` is not returned.

- [ ] **Step 3: Add lightweight timing helper**

At the top of `RCAEval/e2e/gabanzhaf.py`, add:

```python
from time import perf_counter
```

Inside `_gabanzhaf_core()` add:

```python
stage_timing = {}
last_tick = perf_counter()

def _mark_stage(name):
    nonlocal last_tick
    now = perf_counter()
    stage_timing[name] = now - last_tick
    last_tick = now
```

Call `_mark_stage()` after these blocks:

```python
_mark_stage("preprocess")
_mark_stage("metric_evidence")
_mark_stage("topology")
_mark_stage("banzhaf")
_mark_stage("ranking")
```

Return timing only when requested:

```python
result = {"ranks": ranks, "node_names": cols}
if kwargs.get("return_stage_timing", False):
    result["stage_timing"] = stage_timing
return result
```

- [ ] **Step 4: Run timing test**

Run:

```bash
py310/bin/python -m pytest tests/test_gabanzhaf_improvements.py::test_gabanzhaf_core_can_return_stage_timing -q
```

Expected: test passes.

- [ ] **Step 5: Run focused test suite**

Run:

```bash
py310/bin/python -m pytest tests/test_gabanzhaf_improvements.py tests/test_predict_banzhaf.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 3**

```bash
git add RCAEval/e2e/gabanzhaf.py tests/test_gabanzhaf_improvements.py
git commit -m "test: expose gabanzhaf stage timing for benchmarks"
```

---

### Task 4: Benchmark MC vs Deterministic Backends

**Files:**
- Modify: `benchmark_results.json` only after benchmark run is complete and reviewed.
- No production code changes in this task.

- [ ] **Step 1: Run deterministic production benchmark**

Run the existing benchmark command for `gabanzhaf` on RE2 datasets with deterministic default. Use the same command shape currently used to generate `benchmark_results.json`. If using `run123.py`, run the equivalent of:

```bash
py310/bin/python run123.py --method gabanzhaf --datasets re2-ob re2-ss re2-tt
```

Expected: benchmark completes and reports AC metrics for `re2-ob`, `re2-ss`, and `re2-tt`.

- [ ] **Step 2: Run MC 1000 comparison benchmark**

Run the same benchmark with explicit kwargs equivalent to:

```python
{"banzhaf_mode": "mc", "banzhaf_samples": 1000, "seed": 42}
```

If the runner does not support kwargs, add a temporary local invocation script under `/private/tmp/gabanzhaf_bench_compare.py` rather than changing production code.

Expected: benchmark completes and reports the same metrics.

- [ ] **Step 3: Run MC 300 comparison benchmark**

Run the same benchmark with:

```python
{"banzhaf_mode": "mc", "banzhaf_samples": 300, "seed": 42}
```

Expected: benchmark completes and reports the same metrics.

- [ ] **Step 4: Compare accuracy and runtime**

Create a short comparison table with these columns:

```text
dataset, mode, wall_time_sec, avg_speed, AC@1-CPU, AC@1-MEM, AC@1-DISK, AC@1-SOCKET, AC@1-DELAY, AC@1-LOSS, Avg@5-CPU, Avg@5-MEM, Avg@5-DISK, Avg@5-SOCKET, Avg@5-DELAY, Avg@5-LOSS
```

Acceptance criteria:

```text
deterministic runtime on re2-tt improves by at least 30%
overall AC@1 does not decrease
overall Avg@5 does not decrease by more than 0.02
no dataset/fault AC@1 drops by more than 0.07
TT delay/loss do not regress relative to MC 1000
```

- [ ] **Step 5: Update benchmark results only if accepted**

If deterministic meets acceptance criteria, update `benchmark_results.json` with the deterministic run results. If it does not meet criteria, do not update `benchmark_results.json`; instead report the failure pattern and keep MC mode available.

- [ ] **Step 6: Commit benchmark result if updated**

```bash
git add benchmark_results.json
git commit -m "bench: update gabanzhaf deterministic runtime results"
```

Skip this commit if benchmark results are not updated.

---

## Final Verification

- [ ] Run focused tests:

```bash
py310/bin/python -m pytest tests/test_gabanzhaf_improvements.py tests/test_predict_banzhaf.py -q
```

Expected: all tests pass.

- [ ] Compile changed Python files:

```bash
py310/bin/python -m py_compile RCAEval/e2e/gabanzhaf.py tests/test_gabanzhaf_improvements.py
```

Expected: exit code 0.

- [ ] Search for forbidden automatic algorithm switches:

```bash
rg -n "_exact_banzhaf|n≤20|n>20|service count|len\\(.*\\).*banzhaf|dataset.*banzhaf" RCAEval/e2e/gabanzhaf.py tests/test_gabanzhaf_improvements.py
```

Expected: no matches indicating automatic dataset-size or service-count backend switching.

---

## Self-Review Notes

- Spec coverage: trace I/O merge is Task 1, backend dispatcher and deterministic Banzhaf are Task 2, stage timing is Task 3, benchmark comparison is Task 4.
- No dataset-specific behavior is introduced.
- MC remains available only through explicit `banzhaf_mode="mc"`.
- Production default is deterministic.
