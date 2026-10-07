# MERIT frozen research protocol

> 角色更新（2026-10-02）：FROZEN：本文件與 experiment/protocol.json 的科學契約保持一致。會議時程／頁數是原凍結背景，不是目前英文碩論的格式與優先序；當前進度由 docs/thesis-workplan.md 維護。

Protocol version: `2026-07-28.14`
Protocol content frozen before the v14 retrospective rerun: **yes**; a clean Git
snapshot remains a mandatory run gate.
Scope: RCAEval RE2 `re2-ob`, `re2-ss`, and `re2-tt`, 90 cases each (270 total)

This document is the human-readable authority for the v14 full-record
retrospective experiments.
`experiment/protocol.json` is the machine-readable copy. If the two disagree,
stop the run and fix the discrepancy before producing results.

The one-click workflow is resumable only with an identical protocol, source
manifest, Git state, job matrix, and CLI configuration. Completed combinations
are validated and skipped. A partial combination is moved intact under
`interrupted-attempts/` and rerun from its first case so persistent method state
and cache semantics are never spliced across attempts.

The v14 workflow computes the 34GB dataset's full cryptographic tree
checksums once in the `main` job. Later robustness jobs derive their manifests
from that authority only after revalidating the identical Git state and every
manifest-tracked source hash; they do not reread the entire dataset solely to
duplicate an unchanged checksum.

Within the unchanged v14 method set, execution order is operational:
MERIT, quick baselines, controlled variants, and sweeps run before the
timeout-prone RCD, PC+PageRank, and CausalRCA cells. This ordering cannot change
any method input, seed, denominator, or score; it only ensures that a slow cell
does not delay preservation of already independent cells.

## Venue decision

### Primary: SANER 2027 Research Track

- Mandatory abstract: 2026-09-21 AoE; paper: 2026-09-25 AoE.
- English, PDF, `\documentclass[10pt,conference]{IEEEtran}` without `compsoc`
  or `compsocconf`.
- At most 10 pages for all main text, figures, tables, and appendices, plus at
  most 2 pages containing references only.
- Full double-anonymous review. Omit authors, affiliations, acknowledgments,
  identity-revealing repository links, absolute paths, and organization names.
- Supplementary material may be attached through EasyChair as `.zip`/`.tgz`;
  it must also be anonymized. An anonymized external link is acceptable.
- Artifact sharing is encouraged but not mandatory. The paper must contain a
  `Data Availability` section after the conclusion, either linking an anonymous
  artifact or explaining why it is unavailable.
- Generative-AI use follows the IEEE and ACM authorship policies. AI-generated
  paper content must be disclosed in acknowledgments in the camera-ready
  version; editing and grammar assistance is outside the mandatory-disclosure
  intent, though disclosure is recommended. The anonymous submission keeps the
  disclosure in supplementary metadata so it does not reveal author identity.
- Official sources: [research-track CFP](https://conf.researchr.org/track/saner-2027/saner-2027-papers),
  [AI-generated text guidance](https://conf.researchr.org/attending/saner-2027/Guidelines+for+Artificial+Intelligence+%28AI%29-Generated+Text).

### Backup: ICSA 2027 Research Papers

- Mandatory abstract: 2026-10-30 AoE; paper: 2026-11-04 AoE.
- English, IEEE conference format, at most 10 pages of main text plus 2 pages
  containing references only.
- Double-anonymous review.
- An anonymized artifact draft is expected with the paper. If none can be
  supplied, the paper must explain why. Accepted papers are expected to enter
  artifact evaluation; the listed artifact deadline is 2026-12-20.
- The venue explicitly includes microservices, observability, distributed
  tracing, architecture reliability, and lightweight evaluation methods.
- Official sources: [research-paper CFP](https://conf.researchr.org/track/icsa-2027/icsa-2027-papers),
  [artifact evaluation](https://conf.researchr.org/track/icsa-2027/icsa-artifacts-evaluation-2027).

The backup is a schedule contingency if the SANER submission gate is not met
before SANER's deadline. It is not authorization for simultaneous submission.

## Research questions and headline metric

- RQ1: effectiveness against the implemented metric-only, trace-only, and
  causal/ML baselines. No claim against multi-source peers is made.
- RQ2: attributable contribution of metric evidence, trace information,
  coverage, victim demotion, and coalition attribution.
- RQ3: numerical equivalence, complexity, and measured cost of closed-form and
  brute-force probabilistic Banzhaf.
- RQ4: robustness to missing traces, detection delay, window length, candidate
  count, and fixed global hyperparameters.

Primary metric: service-level AC@1. The abstract/intro headline is the
equal-weight macro mean of the three per-system AC@1 values. Pooled 270-case
AC@1 is reported beside it. Secondary metrics are service AC@3, AC@5, Avg@5,
execution success rate, successful-cases-only accuracy (sensitivity only),
wall-clock time, and peak memory.

Each system contributes 15 cases per fault type; pooled fault-type cells contain
45 cases and remain descriptive. Inferential tests are only per system or on
the valid pooled paired set.

The headline estimand uses the original frozen defaults and is explicitly a
retrospective benchmark evaluation. Parameter sweeps are descriptive
sensitivity analyses; they do not refit or replace headline predictions and do
not turn RE2 into a held-out test set.

## Frozen data and method assumptions

- Dataset release: RCAEval Zenodo record `14590730`, version `v2`, DOI
  [`10.5281/zenodo.14590730`](https://doi.org/10.5281/zenodo.14590730).
  Official archive MD5 values are `b9e23f8842c404b396ffd2becff15de4`
  (RE2-OB), `bd747a8fc7c5be00c613e13fbf9dd74b` (RE2-SS), and
  `a7fbcd1ada406067dcc50771ae398408` (RE2-TT). The formal manifest also
  records deterministic hashes of the extracted dataset trees.
- Exactly one injected root service per case.
- The anomaly-detection timestamp is known.
- Telemetry sources are time-synchronized and service names can be aligned.
- No cross-case graph-stability assumption is made in the primary condition.
- The scoring logic does not parse or use fault/case tokens in `data_path`.
  Ground truth and candidate recall are computed only by the offline evaluator;
  the current adapter still receives a path, so the paper does not claim that
  the method interface is mechanically identifier-free.
- The primary topology is rebuilt per case from that case's pre-injection raw
  spans. It is never shared across cases. The graph cache key records source,
  source path, boundary schema, and service schema, and an empty graph is not
  cached. First-case, full-case, and union graphs are sensitivity conditions,
  not the primary run.
- MERIT requires no model training but has fixed unsupervised hyperparameters.
- All three systems use one pipeline and one global hyperparameter set. No
  dataset-, service-, or fault-specific ranking branch is permitted.
- The observation SLI is fixed before seeing the injected service and never
  replaced by a root-specific metric: `frontend_latency` for RE2-OB,
  `front-end_latency` for RE2-SS, and `ts-ui-dashboard_latency` for RE2-TT.
- Each case freezes one benchmark-defined half-open record `[start,end)` from
  the complete metric clock. Metrics, raw spans, trace latency, and trace error
  are all filtered to this external interval. Artifacts record availability,
  actual timestamps, pre/post duration, and sample count for every modality.
  The main evaluation imposes no `--length` or `--tdelta` truncation and is an
  offline full-record diagnosis, not an online or early-RCA claim.

## Environments, repetitions, and timeouts

- MERIT and every baseline, including RCD, use the frozen `./py310` Python
  3.10.18 environment. RCD was migrated into this environment before the
  protocol was frozen; its module import and case-level execution are checked
  by the baseline smoke test. `requirements_py310.lock` is the authoritative
  environment lock for all formal runs. Before a run directory or manifest is
  created, the runner invokes that configured interpreter and requires the
  exact Python patch version plus every pinned package version to match the
  lock. Extra installed packages are recorded but do not alter the frozen
  dependency set. A mismatch is a blocking error, not a warning.
- Deterministic accuracy methods run once after smoke validation. MERIT's
  efficiency measurement runs 5 repetitions. Stochastic methods use seeds
  `20260713`, `20260714`, and `20260715`, retaining case-level ranks for every
  seed. The predeclared stochastic methods are RCD (randomized variable and
  neighborhood order) and CausalRCA (random neural-network initialization).
- `workers=1`; `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
  `MKL_NUM_THREADS=1`, and `NUMEXPR_NUM_THREADS=1`.
- Per-case timeout: 900 seconds over the whole case (worker startup/import,
  input, and method execution). Each method--dataset combination uses one
  persistent subprocess; per-case graph state is keyed by the case source and
  cannot leak to another case. The parent enforces a hard deadline and
  terminates/kills the worker on timeout; the following case starts a fresh
  worker. Dataset task budget: 24 hours, except a pre-declared infeasible
  combination. This exceeds the 90-case worst-case hard-timeout total of 22.5
  hours, so the outer budget cannot truncate required case-level artifacts.
- CIRCA on RE2-TT is `N/A (infeasible)`, based on the existing 12-case,
  12.3-hour probe and its greater-than-79-hour full-run projection. The
  projection is not reported as measured wall-clock and the remaining cases are
  not deliberately run to timeout.

## Failure and denominator policy

- The expected denominator is 90 cases per dataset. A case crash or timeout is
  a miss in overall accuracy; it remains in the denominator.
- Report execution success rate and successful-cases-only accuracy separately.
- A whole method-dataset combination declared infeasible is `N/A`, not zero.
- A trace-required method on RE2-SS is
  `N/A (required modality absent)`, not zero. MicroRank, TraceRCA, and the
  trace-only component comparator therefore have no SS or 270-case aggregate;
  MERIT full still runs because only metrics are required.
- A method gets a pooled 270-case value only if all three method-dataset
  combinations completed. It is forbidden to present 180 cases as a 270-case
  pooled result.
- Paired tests use exactly the cases for which both method-dataset combinations
  completed. Every result records the paired case set and `n`.
- Raw ranks, error type, timeout status, stage timings, and result provenance
  are retained for every expected case. Timeout artifacts contain no dummy
  ranking and record the parent-observed elapsed time and sampled peak RSS.

## Baselines

Retrospective baseline set:

1. BARO (metric-only statistical), with local results cross-checked against the
   official RCAEval/BARO report.
2. MicroRank and TraceRCA (trace-only).
3. RCD, CausalRCA, and PC+PageRank (causal/ML; PC+PageRank is labeled a
   benchmark composition, not a standalone SOTA method).
4. CIRCA (metric causal), with RE2-TT handled by the infeasibility rule above.

BARO, RCD, CausalRCA, PC+PageRank, and CIRCA receive the metric table.
MicroRank and TraceRCA receive each case's raw `traces.csv`, with the injection
timestamp converted from seconds to microseconds. All modalities are filtered
to the same frozen full-record interval. RE2-SS has no raw trace files, so
trace-required cells are declared modality-inapplicable before execution rather
than recorded as failed cases, metric fallbacks, or fabricated ranks. The
formal harness enables strict failure propagation so
the legacy `rca` decorator cannot silently turn an exception into a dummy
ranking.

MSCRED and PDiagnose/MicroCause may be added only after this complete set and
only if their official input contract is compatible. Every run performed must
then be reported.

No baseline algorithm may be modified. Integration fixes are permitted only if
they preserve the official algorithm and are documented as such. If an
environment cannot be rebuilt by the end of Week 1, use a correctly attributed
published number marked `not self-run`; never mix it into paired tests.

Implementation provenance, upstream revisions, and redistribution constraints
are recorded in `experiment/protocol.json` and `docs/related-work-sources.md`.
In particular, the RCD Python 3.10 change is an API-compatibility port of its
causal-learn calls, not an algorithm change.

PC+PageRank uses the same PC/Fisher-Z and reverse-edge PageRank composition as
the RCAEval baseline, with two frozen integration safeguards required by the
Python 3.10 lock: singular or numerically invalid correlation submatrices fall
back to a Moore--Penrose inverse and the resulting partial correlation is
clipped by one floating-point ULP to the open Fisher-transform domain;
scikit-network PageRank uses its 0.33
`fit_predict` API. Isolated telemetry nodes remain in the candidate list. These
safeguards prevent a numerical domain error or API crash; they do not use case
labels or ground truth.

If a Fisher-Z conditioning set leaves zero conditional variance even after the
pseudoinverse, the CI test is not identifiable. The frozen conservative rule is
`p=0`: retain the edge instead of declaring independence or choosing an
arbitrary ridge constant.

The blocking BARO cross-check is RCAEval README Table 6 output for RE2-TT
fault-type Avg@5: CPU 0.72, memory 0.99, disk 1.00, socket 0.83, delay 0.63,
and loss 0.64. The report is rounded to two decimals, so a self-run value must
be within 0.0051 of each displayed value. Failure blocks main-table generation
and triggers an input/version audit; it is never repaired by copying the
published numbers into the self-run row.

## Controlled variants

- `full`: frozen MERIT.
- `no_trace`: Figure-1 evidence extraction emits metric evidence only: there is
  no call graph, trace topology, trace score, or other trace-derived
  information. Candidate construction, the coverage-strength game, Banzhaf
  attribution, and final ranking still execute through the shared pipeline.
- `no_metric`: retains trace-derived service evidence, trace topology, the
  causal game, and Banzhaf attribution, but metric values do not contribute to
  candidates, anomaly weights, topology gates, coverage, or tie-breaking;
  metric column names remain only as the evaluator's output schema.
- `no_trace_topology`: retains call-graph coverage but removes time-series
  trace edges.
- `no_coverage`: changes only coverage rescue.
- `no_victim_demotion`: changes only victim demotion.
- `no_coverage_no_victim`: the missing cell in the coverage × victim `2×2`.
- `no_coverage_parent_protection`: disables protection of rescued coverage
  parents while leaving rescue itself enabled.
- `no_metric_type_dominance` and `no_early_full_fusion`: isolate the two
  metric-evidence aggregation choices.
- `trace_latency_only`, `trace_error_only`, and `trace_lag0`: isolate trace
  signal and lag choices.
- `trace_time_shuffle` and `trace_random_edges`: negative controls.
- `trace_only`: bypasses the game and ranks services directly by validated
  trace-derived evidence.
- `direct_service_evidence`: bypasses the game and ranks services directly by
  metric-derived service evidence.
- `direct_service_trace_evidence`: independently normalizes metric-derived and
  trace-derived service evidence, adds the two scores with equal weight, and
  ranks services directly without the causal game or Banzhaf attribution. If
  trace evidence is absent, the sum reduces to metric-derived evidence.
- `alternative_attribution`: keeps candidates, anomaly weights, coverage-
  strength matrix, coverage, and victim factors identical, replacing only
  coalition attribution/aggregation with singleton weighted coverage.
- Graph-source conditions: `per_case_full`, `first_case_pre_fault`, and
  `union_pre_fault`, plus three deterministic first-case order permutations.
- Correctness notice (2026-10-07): the pre-fix `first_case_pre_fault` cache key
  included the currently evaluated case boundary, so historical first-case and
  case-order sensitivity outputs do not establish their intended fixed-graph
  condition. They remain immutable historical artifacts but must be rerun with
  the runner-provided `graph_reference_case_id` before supporting a thesis
  claim. `union_pre_fault` now records the complete exposure-manifest digest and
  sorted source cases. This notice does not alter the frozen experimental grid.
- Trace dropout: 10/30/50/70/100%; trace-series and graph-edge noise:
  10/30/50%.
- `bruteforce`: numerical/efficiency validation only, not an accuracy ablation;
  use `K <= 15` and absolute tolerance `1e-9`.

The equivalence benchmark uses `K = 3, 6, 9, 12, 15`, 50 targets, five
repetitions, and seed `20260713`. It records raw JSON/CSV timings and memory and
must pass the tolerance before v14 paper results are generated.

Every v14 component variant runs on all applicable RE2 systems; cells whose
required modality is absent are N/A by contract rather than zero.

## Frozen robustness grid and global-value rule

- `victim_floor`: 0.1 through 0.9 in increments of 0.1.
- `trace_topology_conf`: 0.1 through 0.9 in increments of 0.1.
- `coverage_conf`: 0.1 through 0.9 in increments of 0.1.
- `exact_candidate_k`: 3, 6, 9, 12, 15.
- Detection delay (`--tdelta`, seconds): 0, 60, 120, 300.
- Total observation window (`--length`, split equally before and after the
  injection timestamp): 5, 10, 20, 30 minutes.

Current defaults (`victim_floor=0.5`, `trace_topology_conf=0.85`,
`coverage_conf=0.7`, `exact_candidate_k=12`) were adjusted during RE2-TT
development. They are disclosed and frozen as the retrospective headline
configuration; sweep results are never used to relabel RE2 as held-out.

The following historical selection rule is retained only to describe the
sensitivity surface and cross-system transfer; it does not select v14 headline
parameters:

1. If one value is jointly optimal within the predeclared flat region on all
   three systems, choose it and show the complete sensitivity curve.
2. Otherwise, use leave-one-system-out: choose on two systems and evaluate the
   held-out third system, rotating all three held-out systems. Within a fold,
   maximize the equal-weight macro AC@1 of the two training systems; values
   within 1/90 of that maximum use the least-intervention tie rule below. The
   final global value is the numeric median of the three fold-selected grid
   values. Held-out scores estimate generalization and never select the final
   value. Never back-fill a single-system optimum.

A flat region is any contiguous set whose per-system AC@1 is within 1/90 of the
best value. Ties are resolved toward the least intervention: largest victim
floor, smallest edge confidence, then smallest candidate K.

## Statistics

- Primary Top-1 inference uses a two-sided paired sign-flip permutation with
  one sign per `system × root service × fault type` cluster. Exact McNemar is
  retained only as a case-independence sensitivity analysis.
- AC@k and Avg@5 differences use a paired cluster bootstrap over the same
  clusters, 10,000 resamples, 95% percentile interval, RNG seed `20260713`.
- Apply Holm correction within each RQ/table family across all baseline or
  ablation comparisons. Report raw and adjusted p-values, percentage-point
  effect sizes, confidence intervals, and the exact case count.
- The paired-inference scopes are each system and a compatible pooled set.
  Compatible pooling includes only systems on which both methods have complete
  case sets and always reports the included systems and exact $n$. RQ1
  baselines, RQ2 controlled ablations, and RQ4 robustness comparisons are
  separate Holm families. Runtime
  repetitions and parameter/delay/window sweeps are descriptive and excluded
  from those multiplicity families.
- The reporting gate records the formal output path explicitly, compares exact
  metric- and service-rank sequences for all 90 RE2-SS full/no-trace pairs,
  reports five fresh-process cold runs and five measured second invocations
  after a same-case warm-up,
  and reports the observed duration distribution of timeout cases. Exact
  execution, metric-rank, and service-rank equality is a hard gate; any
  mismatch blocks graceful-degradation wording and final table generation.
- For stochastic methods, compute case metrics within each frozen seed, average
  the three seed-specific case metrics, and then resample case clusters while
  retaining the seed block. Seed-level ranks, effects, mean, and standard
  deviation remain reported.

## Efficiency protocol

- Same machine, `workers=1`, fixed threads, at least 5 repetitions.
- Report measured input-plus-method wall-clock median/IQR/p95, task wall time
  including warm-up overhead, peak resident memory, execution success rate,
  and cold-independent/warm-same-case condition.
- Separate data loading, call-graph construction/trace extraction, metric
  evidence, candidate construction, Banzhaf, and ranking; report stage shares
  and observed base/rescued/final `K` and target `N`. Projection values remain
  labeled projections.

## Submission and artifact gates

Before a v14 retrospective run, the generated manifest must record git SHA and dirty
state, checksums, Python/dependency locks, baseline revisions, hardware, thread
variables, seeds, timeouts, exact command, and a fresh output directory.

Before submission:

- every number in `paper/main.tex` resolves to a raw case artifact and generator;
- `headline-macros.tex`, `results-summary.tex`, `extended-results.tex`,
  `paired-statistics-details.tex`, and `fault-breakdown.tex` must byte-match a
  fresh render from the validated
  reproduction; the final audit rejects missing or hand-edited generated files;
- the anonymous artifact has no author names, account/repository names, absolute
  paths, acknowledgments, or VCS author metadata;
- a clean environment reproduces all tables;
- references resolve and LaTeX has no undefined references;
- the PDF is English, IEEE double-column, no more than 10 main-text pages plus
  2 reference-only pages, and contains a `Data Availability` section.
