# GA-Banzhaf Ablation Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add executable GA-Banzhaf ablation switches and `run123.py` method variants for the six main studies in `docs/issues/2026-06-17-ablation-study-design.md`.

**Architecture:** Keep the ablation controls in `_GabanzhafConfig` and pass them through existing `gabanzhaf()` runtime kwargs. Trace source controls are applied immediately after trace extraction; metrics, residual, candidate-size, and ranking controls are applied inside `_gabanzhaf_core()`.

**Tech Stack:** Python, NumPy, pandas, pytest, existing `run123.py` subprocess runner.

---

### Task 1: Tests for GA-Banzhaf Ablation Switches

**Files:**
- Modify: `tests/test_gabanzhaf_improvements.py`
- Modify: `RCAEval/e2e/gabanzhaf.py`

- [ ] **Step 1: Write failing tests**

Add tests covering:

```python
def test_metric_evidence_returns_neutral_when_metrics_disabled():
    cfg = _GabanzhafConfig.from_kwargs({"use_metrics": False})
    ev = _metric_evidence(np.ones((3, 2)), np.full((3, 2), 9.0), ["a_cpu", "b_cpu"], config=cfg)
    assert all(v["score"] == 0.0 for v in ev.values())

def test_rank_metrics_banzhaf_only_uses_service_score_then_column_name():
    ranks = _rank_metrics(
        ["svc-b_mem", "svc-a_z", "svc-a_a"],
        {"svc-a": 1.0, "svc-b": 0.2},
        {"svc-b_mem": 1.0, "svc-a_z": 0.1, "svc-a_a": 0.9},
        _GabanzhafConfig.from_kwargs({"rank_mode": "banzhaf_only"}),
    )
    assert ranks == ["svc-a_a", "svc-a_z", "svc-b_mem"]
```

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m pytest tests/test_gabanzhaf_improvements.py -k "metrics_disabled or banzhaf_only or residual_disabled or ablation_env" -q`

Expected: FAIL because config fields and behavior are missing.

- [ ] **Step 3: Implement minimal behavior**

Add `use_trace_onset`, `use_call_graph`, `use_metrics`, `rank_mode`, `use_residual`, and `residual_mode` to `_GabanzhafConfig`; implement neutral metrics, Banzhaf-only ranking, residual switch behavior, and env parsing.

- [ ] **Step 4: Run tests to verify pass**

Run: `python -m pytest tests/test_gabanzhaf_improvements.py -k "metrics_disabled or banzhaf_only or residual_disabled or ablation_env" -q`

Expected: PASS.

### Task 2: Tests for `run123.py` Variants

**Files:**
- Modify: `tests/test_run123_variants.py`
- Modify: `run123.py`

- [ ] **Step 1: Write failing tests**

Add variant assertions for:

```python
for name in [
    "gabanzhaf_no_trace_onset",
    "gabanzhaf_no_call_graph",
    "gabanzhaf_no_trace",
    "gabanzhaf_no_metrics",
    "gabanzhaf_rank_banzhaf_only",
    "gabanzhaf_k6",
    "gabanzhaf_k8",
    "gabanzhaf_k10",
    "gabanzhaf_k12",
    "gabanzhaf_k16",
    "gabanzhaf_no_residual_rank",
    "gabanzhaf_no_residual_full",
]:
    base, env, meta = run123.resolve_method_variant(name)
    assert base == "gabanzhaf"
    assert meta["base_method"] == "gabanzhaf"
    assert meta["ablation_role"]
    assert env
```

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m pytest tests/test_run123_variants.py -q`

Expected: FAIL because the new variants are not registered.

- [ ] **Step 3: Implement variants**

Extend `METHOD_VARIANTS` with the names and environment variables from the issue. Add metadata keys to CSV output dynamically.

- [ ] **Step 4: Run tests to verify pass**

Run: `python -m pytest tests/test_run123_variants.py -q`

Expected: PASS.

### Task 3: Issue Log and Focused Verification

**Files:**
- Modify: `docs/issues/2026-06-17-ablation-study-design.md`

- [ ] **Step 1: Add implementation result entry**

Record the implemented switches, note that benchmark tables still require full `run123.py` execution, and list test commands.

- [ ] **Step 2: Run focused verification**

Run:

```bash
python -m pytest tests/test_gabanzhaf_improvements.py tests/test_run123_variants.py -q
```

Expected: PASS.
