# MERIT design audit (interim, 2026-09-08)

Main method: current MERIT with victim enabled. All 23 variants use full-record retrospective windows from the existing paper protocol. No production defaults were changed.

Reproduction from repository root:

```sh
PYTHONPATH=. py310/bin/python script/merit_design_audit.py --output artifacts/merit-design-audit-20260908
PYTHONPATH=. py310/bin/python script/finalize_merit_design_audit.py
PYTHONPATH=. py310/bin/python script/summarize_merit_design_audit.py --input artifacts/merit-design-audit-20260908
PYTHONPATH=. py310/bin/python script/write_merit_design_findings.py
PYTHONPATH=. py310/bin/python -m pytest tests/test_merit_design_audit.py -q
```

Use a fresh output directory for independent reruns; the runner overwrites case artifacts if pointed at an existing directory. The finalizer currently targets this named run. Clear `GABANZHAF_*` and `MERIT_*` overrides: the runner rejects them to avoid an unfrozen configuration.

- `manifest.json`: method, configuration and runner fingerprints.
- `source/`: source snapshots, not an independently installable environment.
- `<dataset>/<service_fault>__<rep>.json`: 90 cases per dataset, all attempted variants and failures. Labels in the filename are evaluator metadata; variant scoring does not consume the root-cause label.
- `input_inventory.json`: full post-run inventory, including `tracets_lat.csv` and `tracets_err.csv`; initial metric/span/injection hashes are checked against the saved run records.
- `verification.json`: same-matrix assertions, finite bounded matrices, unique rankings and failure counts.
- `summary.csv`: all four metrics by fault and six-fault macro average; errors count as zero, absent evidence is separately N/A.
- `paired.json`: variant minus full contrasts; 5,000 fault-stratified root-service×fault cluster bootstrap replicates; seed 20260908; intervals are exploratory and unadjusted. Each cluster contains repeated injections; no rerun is treated as a new incident.

`service_banzhaf` is an inherited diagnostic field name; in the Shapley and singleton variants it contains that selected rule's scores. No Banzhaf player-target decomposition is reported as a Shapley decomposition.

`core_seconds` includes research instrumentation and repeated metric processing, but excludes shared trace preparation. It is **not** an end-to-end speedup comparison. `public_full_seconds` is only the instrumented full-method call. Source fingerprints and input hashes do not establish external validity or a clean historical Git checkout.

Coverage isolation: `rescue_only` and `edges_only` retain coverage-derived victim predecessors and parent exemptions; only candidate insertion or matrix overlay changes. `ungated_coverage` removes coverage child thresholds and permits one child, while preserving the trace edge latency gate. `standard_z` changes center and scale only, uses population std, and retains the original near-constant clipping policy. `magnitude_only` removes peak from both windows and persistence from the full window while retaining early/full fusion.

The primary method still includes victim. No new best-performing variant is automatically adopted, and this development-data audit is not independent validation.

Completed and verified 2026-09-09: 6,210 records, 6,120 ok, 90 SS/no_metric no_evidence, zero exceptions; 810 same-matrix/candidate/weight comparisons verified. Nine mechanism/evaluation tests pass.
