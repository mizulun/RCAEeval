# MERIT protocol-v14 experiment plan

> 角色更新（2026-10-02）：FROZEN：保留 v14 原始設計與完整 gate，非即時待辦。現版 MERIT 的 9/30 重跑已存在；其 dirty 條件與未完成验收見 [現行工作清單](thesis-workplan.md)。不得以舊進度推定整批未跑或全部通過。

Status: implementation-frozen candidate; formal evidence remains pending until
the clean detached reproduction and all reporting gates complete.

## Study boundary and estimand

The study is a retrospective, offline diagnosis of all 270 RCAEval RE2 cases
(RE2-OB, RE2-SS, and RE2-TT; 90 each). Every method receives the same
benchmark-defined complete case record `[start,end)`. Metrics, trace series, and
raw spans are clipped to that interval; unavailable modalities remain absent.
MERIT builds the primary call graph separately for each case from spans strictly
before the benchmark event time. The current defaults are frozen historical
defaults, so the headline is a retrospective fixed-default benchmark estimate,
not a held-out generalization estimate. Parameter sweeps are descriptive and
cannot replace headline predictions.

## Research questions and falsifiable hypotheses

| RQ | Comparison and falsifiable hypothesis | Evidence needed |
|---|---|---|
| RQ1 effectiveness | For each applicable baseline, test the two-sided hypothesis that MERIT and the baseline have equal service AC@1. A non-positive interval is retained and falsifies any superiority wording. | Complete denominators, execution rate, macro and compatible-pooled AC@1, cluster interval/test, seed-aware estimates. |
| RQ2 components | Source removal and direct evidence ranking answer different questions: `no_trace` keeps the complete Figure-1 pipeline but extracts metric evidence only and forbids every downstream trace-derived input; `no_metric` also retains the game and Banzhaf, whereas the headline direct comparator independently normalizes and adds metric+trace service scores before ranking. Single-source direct rankings remain supplementary diagnostics. | Four-row source/ranking table, one-factor variants, coverage×victim 2×2, negative controls, direct/singleton comparators, candidate-stage recall and observed `(K,N)`; no-trace diagnostics must show zero call-graph/trace evidence while retaining candidates, the game, Banzhaf, and ranks; no-metric × SS is N/A and the direct sum reduces to metric evidence when trace is absent. |
| RQ3 computation | Closed form must match enumeration within absolute error `1e-9` for every frozen `K`; measured speedup applies only to the attribution kernel. End-to-end cost must separately disclose input, trace/graph, metric, candidate, Banzhaf, and ranking time. | Enumeration gate for `K={3,6,9,12,15}`, five cold and five warm repetitions, median/IQR/p95, RSS, stage shares. |
| RQ4 robustness | On naturally trace-missing RE2-SS, full and no-trace complete ranks should be identical; any mismatch blocks graceful-degradation wording. Graph-source/order, dropout/noise, candidate cap, coefficient, window, and delay effects are reported whether positive, neutral, or negative. | Exact rank comparison, graph edge Jaccard/rank change, explanation stability, cluster-aware comparisons, complete descriptive curves and realized boundaries. |

## Variables and controls

- Independent variables: method; controlled MERIT variant; graph source/order;
  trace dropout, series noise, and edge noise; fixed coefficient or candidate
  cap; supplementary observation window and event-time delay.
- Dependent variables: service AC@1 (primary), AC@3, AC@5, Avg@5, execution
  success, observed wall time, peak RSS, candidate recall, base/rescued/final
  `K`, target `N`, graph Jaccard, local faithfulness, and explanation stability.
- Controls: identical expected cases and scorer; `workers=1`; fixed BLAS threads;
  Python 3.10.18 and exact lock; three frozen stochastic seeds; 900-second
  whole-case timeout; failures remain misses; N/A only for a pre-declared
  infeasible or missing-required-modality cell.
- Ground truth is available only to the evaluator. Method code receives no root
  service, fault type, case-derived label, or directory-derived feature.

## Baselines and variant matrix

RQ1 uses BARO, MicroRank, TraceRCA, RCD, CausalRCA, PC+PageRank, and CIRCA.
Trace-required methods on RE2-SS are N/A; CIRCA on RE2-TT is a pre-declared
projected infeasibility and is not scored as zero. RCD and CausalRCA use seeds
20260713--20260715. No claim against unexecuted multi-source peers is permitted.

The main job contains 87 labels: MERIT, applicable seeded baselines, controlled
metric/trace/coverage/attribution/graph variants, perturbations, four coefficient
or candidate-cap sweeps, and five cold plus five warm runtime conditions. Seven
separate supplementary jobs cover delay 60/120/300 seconds and total windows
5/10/20/30 minutes. The frozen plan represents 25,380 method--case invocations
before pre-declared N/A short-circuiting.

## Statistical analysis

The sampling cluster is `system × root service × fault type`; repeats remain
inside their cluster. The primary AC@1 test is a paired cluster sign-flip
permutation. AC@1/3/5 and Avg@5 effects receive 10,000-resample paired cluster
bootstrap intervals. For stochastic methods, case metrics are computed within
each frozen seed, effects are averaged across seeds, and clusters are resampled
while retaining the seed block. Holm correction is separate for RQ1 baselines,
RQ2 ablations, and RQ4 robustness. Exact McNemar and independent-case bootstrap
are secondary sensitivity analyses only. Fault-type cells (15 per system, 45
pooled) are descriptive.

## Execution, recovery, and artifact flow

1. Run tests and working-draft audit in the development tree.
2. Create a durable clean anonymous Git snapshot; symlink the immutable dataset
   and frozen Python environment, then verify clean state and checksums.
3. Start a detached supervisor under `screen` plus `caffeinate`. It records a
   heartbeat and exact command, retries only the identical `--resume` plan, and
   never deletes an interrupted attempt.
4. Run experiments, evaluation, BARO cross-check, global descriptive selection,
   Banzhaf equivalence, and gated table generation.
5. Copy the five generated files into `paper/generated`, compile the final
   submission, and run anonymity, provenance, reference, overfull-box, and page
   gates. The final PDF may not exceed ten main-text pages plus two
   reference-only pages.

## Exit criteria

The campaign is complete only if every non-N/A method--dataset cell has all 90
cases, the full-record boundary and graph provenance are present per case,
environment/checksum/clean-manifest gates pass, all three Holm families are
complete, exact-rank and equivalence diagnostics pass their defined gates, all
five generated files byte-match a fresh render, and the final submission audit
has no findings. A negative method result is not a failure condition; missing
or unauditable evidence is.
