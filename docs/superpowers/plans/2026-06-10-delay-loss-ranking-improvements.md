# DELAY/LOSS Ranking Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve DELAY/LOSS candidate recall and ranking without adding fault-specific branches or stacking duplicate ranking methods.

**Architecture:** Replace the Step 6 fixed weighted selector with three-source Reciprocal Rank Fusion and fix Step 10 Banzhaf normalization without a floor parameter. Keep metric, trace, and residual diagnostics separate. Trace/game fusion remains experimental because benchmark ablations exceeded the non-network-fault regression budget.

**Tech Stack:** Python, NumPy, pandas, pytest.

---

### Task 1: Three-source RRF candidate selection

**Files:**
- Modify: `tests/test_gabanzhaf_improvements.py`
- Modify: `RCAEval/e2e/gabanzhaf.py`

- [x] Replace the fixed-score candidate test with a test where a trace-rank-1 service is retained by equal-weight metric/residual/trace RRF.
- [x] Run the focused test and verify it fails against the fixed `0.50/0.40/0.05/0.05` selector.
- [x] Implement deterministic descending ranks and `sum(1 / (10 + rank))` over metric, residual, and trace sources.
- [x] Remove graph participation from `_select_exact_candidates`.
- [x] Run the focused candidate tests and verify they pass.

### Task 2: Positive-preserving Banzhaf normalization

**Files:**
- Modify: `tests/test_gabanzhaf_improvements.py`
- Modify: `RCAEval/e2e/gabanzhaf.py`

- [x] Add a test proving the smallest positive in-candidate Banzhaf score remains distinguishable from an out-of-candidate service.
- [x] Run the focused test and verify min-max normalization fails it.
- [x] Normalize positive Banzhaf values by the maximum value; keep the existing metric-only behavior for empty or tied Banzhaf values.
- [x] Run the focused ranking tests and verify they pass.

### Task 3: Trace evidence ablation

**Files:**
- Modify: `tests/test_gabanzhaf_improvements.py`
- Modify: `RCAEval/e2e/gabanzhaf.py`

- [x] Test raw `max(metric, trace)` in the call-graph gate and Banzhaf game.
- [x] Test fused anomaly weights while keeping call-graph admission and causal alignment metric-only.
- [x] Test `max(metric, trace_residual)` using the existing topology explanation.
- [x] Reject all three variants because re2-ob CPU/DISK/SOCKET/DELAY Avg@5 regressions exceeded 0.02.
- [x] Keep raw trace only for candidate rescue and diagnostics; keep the game and call-graph gate metric-only.

### Task 4: Regression and benchmark verification

**Files:**
- Modify: `docs/issues/2026-06-10-delay-loss-status-and-open-items.md`

- [x] Run `pytest tests/test_gabanzhaf_improvements.py -q`.
- [x] Run the full local repository test suite, excluding download tests and subprocess tests that require a global `python` executable.
- [x] Run clean-result benchmarks for `re2-tt`, `re2-ob`, and `re2-ss`.
- [x] Record ablation and AC@1/3/5/Avg@5 results in the issue.
- [x] Record the 11 environment-dependent full-suite failures: seven blocked Zenodo downloads and four subprocess tests without a global `python`.
