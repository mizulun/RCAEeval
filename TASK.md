# Current Task

**Plan:** `docs/superpowers/plans/2026-06-01-probabilistic-causal-banzhaf.md`
**Branch:** `codex-banzhaf`

## Status

- [x] Task 1: Add Noisy-OR Game Tests
- [x] Task 2: Implement Closed-Form Exact Banzhaf
- [x] Task 3: Add Candidate Selection and Explanation Matrix
- [x] Task 4: Integrate Probabilistic Banzhaf Into Core Pipeline
- [x] Task 5: Simplify Metric Ranking and Remove Phenotype/Semantic Main-Path Logic
- [x] Task 6: No-Hardcode Guard and Full Test Pass
- [ ] **Task 7: Benchmark and Result Triage** ← NEXT

## Task 7 Steps

- [ ] Run benchmark: `py310/bin/python run123.py --methods gabanzhaf --datasets re2-ob re2-ss re2-tt --workers 1 --output /private/tmp/probabilistic_causal_gabanzhaf.csv`
- [ ] Inspect metric table (AC@1, AC@3, AC@5, Avg@5, wall time per fault type)
- [ ] If regression: diagnose without adding dataset/fault/service exceptions
- [ ] Commit `benchmark_results.json` only if result is accepted

## Constraints (always enforce)

- No ground truth, fault labels, dataset names, case IDs, service-name exceptions
- No fallback branches — one unified flow for every incident
- K=12 fixed candidate game
- Primary files: `RCAEval/e2e/gabanzhaf.py`, `tests/test_gabanzhaf_improvements.py`

## Handoff Notes

- Claude Code handles: benchmark execution, cross-file compliance checks, result analysis
- Codex handles: code edits, refactors within gabanzhaf.py
- Sync point: commit before switching between agents
