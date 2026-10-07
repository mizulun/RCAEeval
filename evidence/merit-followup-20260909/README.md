# MERIT follow-up: scale, robustness, architecture, RE1

Research-only frozen experiments. No model training, production/API/default change, label-dependent parameters, or old thesis table replacement.

## Protocol

- RE2: 90 cases each in SS/OB/TT. Four scale arms: original, median/std, IQR-zero-only fallback repair, positive-IQR-only std replacement. Clipping guard and all downstream settings stay fixed.
- Three selected arms (original, median/std, fallback-only) additionally run metric-baseline contamination (append floor(.05/.10/.20 × normal sample count) initial abnormal samples; the original abnormal window is retained). The added count is a fraction of the original baseline length; the contamination share in the expanded baseline is f/(1+f). This intentionally overlapping diagnostic separates baseline contamination from shortened observation.
- Detection delay: move algorithm split by 60/120/180 seconds, retaining the full record. The graph's nominal pre-fault exposure also moves, as it would with a delayed boundary. This is an overall delay intervention, not a metric-only contamination claim.
- Observation horizon: retain the full normal period and expose only 60/180/300 seconds after injection for all modalities using a common half-open boundary. If the record cannot support the specified duration or has fewer than five post samples, record unsupported.
- Backend: original and median/std evidence each compare full MERIT to pure metric-service ranking and singleton on the same candidates, matrix and weights. Production metric-level secondary/schema-order ties are preserved. No trace score is added to the direct comparator.
- RE1: 125 cases each in `data/online-boutique`, `data/sock-shop-2`, `data/train-ticket`; the actual local Sock Shop folder differs from the conversational shorthand. Apply the frozen four scale formulas without parameter selection. Prior `benchmark_re1_merit_baro.json` proves previous exposure; call this cross-dataset evidence, not untouched validation.

Each RE2 case has 35 actual arm/scenario records (4 scale + 9 pollution + 9 delay + 9 horizon + 4 backend). Each RE1 case has four. Aliases add clean/full reference rows to the viewer without rerunning or inflating the number of independent cases. Source and input hashes are frozen; all exceptions and unsupported cases are retained.

Root service and fault labels are extracted only for scoring/aggregation. Formula choice is global. Observations of outcomes never trigger a parameter update or replacement of the official method.

## Reproduce

From the repository root, using a new output directory:

```sh
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 py310/bin/python script/merit_followup.py --output artifacts/merit-followup-rerun
PYTHONPATH=. py310/bin/python script/summarize_merit_followup.py --input artifacts/merit-followup-rerun
PYTHONPATH=. py310/bin/python script/build_merit_followup_page.py --input artifacts/merit-followup-rerun
PYTHONPATH=. py310/bin/python script/audit_merit_followup.py --input artifacts/merit-followup-rerun
PYTHONPATH=. py310/bin/python -m pytest tests/test_merit_followup.py tests/test_merit_training_free.py tests/test_merit_design_audit.py -q
node script/verify_merit_followup_page.mjs artifacts/merit-followup-rerun
```

`index.html` is standalone and works offline; no CDN, model, server or external script is required. `analysis.json` is its source of truth, `cases.csv` contains every display row and identifies baseline aliases. Per-case JSON retains full matrices/rankings. `verification.json` records completeness, exceptions and exact metric-file duplicates. No byte-identical files is not proof of statistical independence.

Statistics: equal-fault macro AC@1/3/5 and AVG@5; 5000 paired fault-stratified root-service×fault cluster bootstrap draws, seed 20260909. Repeated injections remain in their cluster. No multiplicity adjustment; all intervals are descriptive. Error records count as zero; unsupported records are excluded with explicit counts. Comparisons requiring an unsupported reference are unavailable. No pooled score is used to choose a winner across datasets or metrics.

Timing caveat: the original scale/delay/horizon arm is obtained while capturing the public method result; its inner accessor timing is not algorithm runtime, so analysis marks it unavailable. Other arms store research-core timings; direct/singleton reuse matrices and therefore have different work scopes. Total per-case wall time is recorded. Do not infer end-to-end speedups from these timings.

The browser verifier uses Node 25's built-in WebSocket and local macOS Google Chrome with a temporary isolated profile. It tests 51 dataset/scenario views, case filters, search, CSV content, and mobile page width; screenshots and browser-verification.json are produced. Other machines may open index.html directly without these verification dependencies. The execution remained sequential: a proposed parallel resume was not used because the original worker had already completed.

The six unsupported records are the three scale arms for the 300-second horizon in RE2-OB checkoutservice_cpu/2 and checkoutservice_mem/2. They are two distinct cases, not six distinct incidents. Complete baseline records for these cases remain available.
