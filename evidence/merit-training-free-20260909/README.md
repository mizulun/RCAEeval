# MERIT training-free research — development evidence

270 existing RE2 cases, 90 per system, full-record retrospective protocol. Production MERIT includes victim and is unchanged. This is not independent validation. No parameters are selected using root-cause labels; no models are trained.

## Reproduce

Run from repository root with the existing `py310` environment. Use a **fresh** output directory; the runner refuses overwriting an existing run.

```sh
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 py310/bin/python script/merit_training_free.py --output artifacts/merit-training-free-rerun
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 py310/bin/python script/merit_training_free_synthetic.py --output artifacts/merit-training-free-rerun/synthetic.json
PYTHONPATH=. py310/bin/python script/summarize_merit_training_free.py --input artifacts/merit-training-free-rerun
PYTHONPATH=. py310/bin/python script/finalize_merit_training_free.py --input artifacts/merit-training-free-rerun
PYTHONPATH=. py310/bin/python -m pytest tests/test_merit_training_free.py tests/test_merit_design_audit.py -q
```

The runner defaults to prior `artifacts/merit-design-audit-20260908`; `--prior` can select another compatible artifact. It rejects production source drift and verifies every input recorded in the previous full inventory, including trace series, both before and after processing. It recomputes full MERIT solely to compare the baseline rankings, candidates, matrix, weights and scores. The finalizer additionally checks the prior harness/boundary source snapshots.

Four arms (`full`, `standard_z`, `singleton`, `shapley`) are reused verbatim with provenance. Seven new arms isolate mean/IQR, median/std, positive-IQR Gaussian scale calibration, matched persistence threshold, threshold-only calibration, raw cosine and signed baseline-centered cosine. All retain victim. No broad hyperparameter search or combined best-arm selection is performed.

`normal_iqr` divides only positive IQR by 1.3489795003921634; zero-IQR fallback remains 1. `normal_iqr_matched_threshold` uses 3*c on positive-IQR columns, 3 otherwise; `iqr_z_threshold` uses 3/c on positive-IQR columns, 3 otherwise. Original clipping guard stays unchanged. These interventions do not claim to solve the measurement-unit dependence of zero-scale fallback.

Raw/centered trace arms both bypass min-max, retain full observed time rows, lag 0–3, confidence multiplier .85, minimum confidence .3, original edge direction and downstream latency filter. Centered values retain signs; no post-only cropping, rectification or residual energy gate is added. Trace modality scores remain unchanged. SS lacks traces, so these arms are structurally inactive there.

## Outputs

- `manifest.json`, `source/`: execution-time fingerprints and snapshots. Initial tests/summary snapshots predate additional analysis checks; final analysis sources live separately in `source/analysis/` and `analysis-sources.json`, preserving initial fingerprints.
- Dataset subdirectories: every per-case record, ranks, candidates, matrix, weights, scores, baseline scale diagnostics and timing. Errors are preserved, not dropped.
- `summary.csv`: all fault types and macro metrics; `paired.json`: paired deltas, all four metric intervals, candidate recall and trigger counts inherited from the audit summary.
- `case_comparisons.csv`: root rank, candidate membership, fixed/broken, core seconds, reuse status.
- `contrasts.json`: factorial and centered-minus-raw contrasts, all four metrics with common cluster-bootstrap draws.
- `diagnostics.json`, `mechanism-cases.json`: baseline and trace diagnostic counts / per-case changes.
- `synthetic.json`: fixed-seed mechanism and null experiments, including negative examples and every synthetic p-value.
- `results-zh.md`, `findings-zh.md`: generated readable tables and interpretation.
- `verification.json`, `completion.json`: consistency checks and status counts.
- `literature-inventory.json`: supplied PDF hashes; official LagRCA code is inspected but not executed, and only a short formula excerpt is archived.

Intervals use 5,000 fault-stratified root-service×fault cluster bootstrap draws, seed 20260908; repetitions stay in their cluster. All contrasts share draws, so adding a variant does not change another variant's interval. The older report advanced one RNG across variants, so exploratory CI endpoints can differ slightly despite identical reused rankings. Intervals are not adjusted for multiple comparisons.

The cyclic-shift diagnostic enumerates the entire normal-window cyclic group, including identity, and recomputes max over lags for every shift. Cyclic invariance is stronger than ordinary stationarity. Its p-values are **never used to gate real edges**. The nominal .05 synthetic check is not a root-cause metric or a guarantee for real traces; normal service pairs may actually be dependent. Insufficient normal samples return no null statistic. Irregular trace spacing is recorded, not silently interpreted as equal elapsed-time lags.

`core_seconds` excludes shared graph/trace preparation and diagnostic null calculations. Reused timings are from a prior batch, so no end-to-end or cross-batch speed claim is supported. Smoke output is separate and is not counted as independent evidence.
