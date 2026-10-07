# Formal-protocol MERIT rerun (main-20260930)

Raw output: `artifacts/formal-rerun-20260930/main-20260930/<variant>/<dataset>/results/`. Recompute: `PYTHONPATH=. ./py310/bin/python artifacts/formal-rerun-20260930/analyze.py main-20260930`.

Protocol: `script/run_paper_experiments.py` -> `script/run_paper_dataset.py`, evaluation_mode=full_record_retrospective, input_boundary passed, graph_source=per_case_pre_fault, seed 20260713, --allow-dirty (working tree dirty; git state in manifest.json). Errors/timeouts count as misses.

## 1. Overall (per system, n=90) and equal-system macro

| variant | system | status ok/err/timeout | AC@1 | AC@3 | AC@5 | Avg@5 | AC@1 k/n | AC@3 k/n | AC@5 k/n | task wall s |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full | re2-ss | 90/0/0 | 38.9 | 95.6 | 98.9 | 84.7 | 35/90 | 86/90 | 89/90 | 15 |
| full | re2-ob | 90/0/0 | 30.0 | 96.7 | 100.0 | 83.3 | 27/90 | 87/90 | 90/90 | 107 |
| full | re2-tt | 90/0/0 | 81.1 | 91.1 | 92.2 | 88.4 | 73/90 | 82/90 | 83/90 | 217 |
| no-trace | re2-ss | 90/0/0 | 38.9 | 95.6 | 98.9 | 84.7 | 35/90 | 86/90 | 89/90 | 14 |
| no-trace | re2-ob | 90/0/0 | 32.2 | 96.7 | 100.0 | 84.0 | 29/90 | 87/90 | 90/90 | 12 |
| no-trace | re2-tt | 90/0/0 | 78.9 | 88.9 | 92.2 | 87.3 | 71/90 | 80/90 | 83/90 | 33 |
| no-metric | re2-ss | N/A (runner: required modality absent; trace-derived evidence) | N/A | N/A | N/A | N/A | - | - | - | 0 |
| no-metric | re2-ob | 90/0/0 | 36.7 | 56.7 | 71.1 | 56.2 | 33/90 | 51/90 | 64/90 | 105 |
| no-metric | re2-tt | 90/0/0 | 28.9 | 45.6 | 61.1 | 46.2 | 26/90 | 41/90 | 55/90 | 202 |
| no-banzhaf = direct metric+trace sum | re2-ss | 90/0/0 | 38.9 | 95.6 | 98.9 | 84.7 | 35/90 | 86/90 | 89/90 | 14 |
| no-banzhaf = direct metric+trace sum | re2-ob | 90/0/0 | 65.6 | 81.1 | 93.3 | 80.0 | 59/90 | 73/90 | 84/90 | 107 |
| no-banzhaf = direct metric+trace sum | re2-tt | 90/0/0 | 81.1 | 92.2 | 95.6 | 90.2 | 73/90 | 83/90 | 86/90 | 220 |
| singleton (same matrix, no overlap discount) | re2-ss | 90/0/0 | 38.9 | 95.6 | 98.9 | 84.7 | 35/90 | 86/90 | 89/90 | 14 |
| singleton (same matrix, no overlap discount) | re2-ob | 90/0/0 | 34.4 | 96.7 | 100.0 | 84.0 | 31/90 | 87/90 | 90/90 | 107 |
| singleton (same matrix, no overlap discount) | re2-tt | 90/0/0 | 80.0 | 91.1 | 94.4 | 89.1 | 72/90 | 82/90 | 85/90 | 214 |

| variant | systems in macro | macro AC@1 | macro AC@3 | macro AC@5 | macro Avg@5 |
|---|---:|---:|---:|---:|---:|
| full | 3 | 50.0 | 94.4 | 97.0 | 85.5 |
| no-trace | 3 | 50.0 | 93.7 | 97.0 | 85.3 |
| no-metric | 2 (OB+TT only; SS N/A) | 32.8 | 51.1 | 66.1 | 51.2 |
| no-banzhaf = direct metric+trace sum | 3 | 61.9 | 89.6 | 95.9 | 85.0 |
| singleton (same matrix, no overlap discount) | 3 | 51.1 | 94.4 | 97.8 | 85.9 |

## 2. Per fault type (n=15 each): AC@1 k/15 and Avg@5 %

### full

| system | cpu | mem | disk | socket | delay | loss |
|---|---:|---:|---:|---:|---:|---:|
| re2-ss | 3/15 · 84.0 | 11/15 · 94.7 | 8/15 · 90.7 | 5/15 · 86.7 | 4/15 · 69.3 | 4/15 · 82.7 |
| re2-ob | 3/15 · 84.0 | 9/15 · 90.7 | 3/15 · 84.0 | 0/15 · 80.0 | 6/15 · 80.0 | 6/15 · 81.3 |
| re2-tt | 11/15 · 92.0 | 14/15 · 98.7 | 15/15 · 100.0 | 13/15 · 88.0 | 12/15 · 80.0 | 8/15 · 72.0 |

### no-trace

| system | cpu | mem | disk | socket | delay | loss |
|---|---:|---:|---:|---:|---:|---:|
| re2-ss | 3/15 · 84.0 | 11/15 · 94.7 | 8/15 · 90.7 | 5/15 · 86.7 | 4/15 · 69.3 | 4/15 · 82.7 |
| re2-ob | 4/15 · 85.3 | 9/15 · 89.3 | 4/15 · 85.3 | 0/15 · 80.0 | 6/15 · 80.0 | 6/15 · 84.0 |
| re2-tt | 10/15 · 90.7 | 15/15 · 100.0 | 15/15 · 100.0 | 14/15 · 98.7 | 9/15 · 70.7 | 8/15 · 64.0 |

### no-metric

| system | cpu | mem | disk | socket | delay | loss |
|---|---:|---:|---:|---:|---:|---:|
| re2-ss | N/A | N/A | N/A | N/A | N/A | N/A |
| re2-ob | 6/15 · 50.7 | 7/15 · 61.3 | 6/15 · 53.3 | 6/15 · 65.3 | 3/15 · 32.0 | 5/15 · 74.7 |
| re2-tt | 5/15 · 50.7 | 6/15 · 46.7 | 5/15 · 58.7 | 5/15 · 62.7 | 0/15 · 9.3 | 5/15 · 49.3 |

### no-banzhaf = direct metric+trace sum

| system | cpu | mem | disk | socket | delay | loss |
|---|---:|---:|---:|---:|---:|---:|
| re2-ss | 3/15 · 84.0 | 11/15 · 94.7 | 8/15 · 90.7 | 5/15 · 86.7 | 4/15 · 69.3 | 4/15 · 82.7 |
| re2-ob | 10/15 · 77.3 | 13/15 · 93.3 | 11/15 · 81.3 | 9/15 · 78.7 | 3/15 · 56.0 | 13/15 · 93.3 |
| re2-tt | 13/15 · 97.3 | 15/15 · 100.0 | 15/15 · 100.0 | 14/15 · 98.7 | 7/15 · 65.3 | 9/15 · 80.0 |

### singleton (same matrix, no overlap discount)

| system | cpu | mem | disk | socket | delay | loss |
|---|---:|---:|---:|---:|---:|---:|
| re2-ss | 3/15 · 84.0 | 11/15 · 94.7 | 8/15 · 90.7 | 5/15 · 86.7 | 4/15 · 69.3 | 4/15 · 82.7 |
| re2-ob | 4/15 · 85.3 | 11/15 · 93.3 | 4/15 · 85.3 | 0/15 · 80.0 | 6/15 · 80.0 | 6/15 · 80.0 |
| re2-tt | 10/15 · 88.0 | 15/15 · 100.0 | 15/15 · 100.0 | 13/15 · 93.3 | 12/15 · 81.3 | 7/15 · 72.0 |

## 3. Failed cases

- none: every variant x system completed 90/90 successfully.

## 4. direct metric+trace sum vs full (paired; b = direct, a = full)

Tools: `paper_evaluation.exact_mcnemar`, `paired_cluster_bootstrap` (cluster = dataset x root_service x fault, 10,000 resamples, seed 20260713), `cluster_paired_permutation` (cluster sign-flip, 10,000, seed 20260713).

| scope | full AC@1 k/n | direct AC@1 k/n | diff pp (direct-full) | 95% cluster-bootstrap CI pp | sign-flip p | McNemar full-only / direct-only | McNemar exact p |
|---|---:|---:|---:|---|---:|---|---:|
| re2-ss | 35/90 | 35/90 | +0.0 | [+0.0, +0.0] | 1.0000 | 0 / 0 | 1.0000 |
| re2-ob | 27/90 | 59/90 | +35.6 | [+17.8, +53.3] | 0.0012 | 5 / 37 | 0.0000 |
| re2-tt | 73/90 | 73/90 | +0.0 | [-10.0, +10.0] | 1.0000 | 7 / 7 | 1.0000 |
| pooled | 135/270 | 167/270 | +11.9 | [+4.4, +19.6] | 0.0038 | 12 / 44 | 0.0000 |

Pooled = 270 cases (unweighted pool; equals equal-system macro because each system has 90).

## 5. Reproducibility vs artifacts/merit-design-audit-20260908

Audit calls `merit(..., input_boundary=full_record_boundary, graph_source=per_case_pre_fault)` directly (script/merit_design_audit.py), i.e. the same protocol. Comparison uses evaluator service ranking (`metric_service`, '-db' folded) on the top-5, per case.

| this variant | audit variant | system | cases compared | identical top-5 | identical root rank (<=5) | AC@1 this / audit |
|---|---|---|---:|---:|---:|---|
| full | full | re2-ss | 90 | 90 | 90 | 35/35 |
| full | full | re2-ob | 90 | 90 | 90 | 27/27 |
| full | full | re2-tt | 90 | 90 | 90 | 73/73 |
| no-trace | no_trace | re2-ss | 90 | 90 | 90 | 35/35 |
| no-trace | no_trace | re2-ob | 90 | 90 | 90 | 29/29 |
| no-trace | no_trace | re2-tt | 90 | 90 | 90 | 71/71 |
| no-metric | no_metric | re2-ss | 0 (N/A here; audit status no_evidence) | - | - | - |
| no-metric | no_metric | re2-ob | 90 | 90 | 90 | 33/33 |
| no-metric | no_metric | re2-tt | 90 | 90 | 90 | 26/26 |
| no-banzhaf = direct metric+trace sum | direct_evidence | re2-ss | 90 | 90 | 90 | 35/35 |
| no-banzhaf = direct metric+trace sum | direct_evidence | re2-ob | 90 | 90 | 90 | 59/59 |
| no-banzhaf = direct metric+trace sum | direct_evidence | re2-tt | 90 | 90 | 90 | 73/73 |
| singleton (same matrix, no overlap discount) | singleton | re2-ss | 90 | 90 | 90 | 35/35 |
| singleton (same matrix, no overlap discount) | singleton | re2-ob | 90 | 90 | 90 | 31/31 |
| singleton (same matrix, no overlap discount) | singleton | re2-tt | 90 | 90 | 90 | 72/72 |

Cases whose root-service rank differs (case, rank here, rank in audit; None = beyond top-5):

- none

## 6. Integer check

All AC@k x 90 are integers and Avg@5 x 450 integers: True (56 checks).

## 7. Variant mapping, wall-clock, comparison with earlier numbers

- Variants are selected via `script/run_paper_experiments.py` FIXED_VARIANTS (env vars read by merit.py; merit.py not modified):
  full = no env; no-trace = `GABANZHAF_DISABLE_TRACE=1`; no-metric = `GABANZHAF_DISABLE_METRICS=1`;
  no-banzhaf and direct metric+trace sum are the SAME configuration in this codebase: `GABANZHAF_RANK_MODE=service_evidence_trace`
  (the externally archived historical no-Banzhaf report defines it this way); singleton = `GABANZHAF_BANZHAF_MODE=individual` (extra control, same matrix, linear utility).
- re2-ss no-metric: runner/protocol marks it `inapplicable` (experiment/protocol.json modality_inapplicable: RE2-SS has no traces.csv, so no trace-derived evidence); no cases executed. The design audit likewise records 90 `no_evidence`.
- Wrapper wall-clock: [run.sh] end rc=0 wall_seconds=1574 (includes ~85 s full-checksum manifest + environment validation). Per task wall seconds are in section 1.
  Separate timing probe `timing-ss-full-20260930` (re2-ss full only): 106 s wrapper, 21 s task.
- Earlier macro AC@1: 56.3% (artifacts/clean-rerun-20260807, main.py path, no input_boundary: SS 46.7 / OB 36.8 / TT 85.6 from its *_AC1.csv fault means);
  design-audit full (artifacts/merit-design-audit-20260908/summary.csv): SS 38.9 / OB 30.0 / TT 81.1 -> macro 50.0%.
