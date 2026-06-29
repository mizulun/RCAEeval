# GA-Banzhaf Ablation and Trace Diagnostics

## Aggregate benchmark

| Method | Dataset | AC@1 macro | Avg@5 macro | Wall median (s) | Range (s) |
|---|---|---:|---:|---:|---:|
| gabanzhaf | re2-ob | 0.380 | 0.838 | 7.40 | 7.30–7.70 |
| gabanzhaf | re2-ss | 0.467 | 0.872 | 8.30 | 8.00–8.40 |
| gabanzhaf | re2-tt | 0.812 | 0.883 | 26.20 | 25.80–27.10 |
| gabanzhaf_no_lag_1 | re2-ob | 0.380 | 0.838 | 7.60 | 7.40–7.70 |
| gabanzhaf_no_lag_1 | re2-ss | 0.467 | 0.872 | 8.00 | 7.90–8.40 |
| gabanzhaf_no_lag_1 | re2-tt | 0.800 | 0.882 | 26.50 | 26.30–26.70 |
| gabanzhaf_evidence_service | re2-ob | 0.380 | 0.838 | 7.30 | 7.30–7.60 |
| gabanzhaf_evidence_service | re2-ss | 0.467 | 0.872 | 8.20 | 8.00–8.50 |
| gabanzhaf_evidence_service | re2-tt | 0.800 | 0.882 | 25.90 | 25.80–26.50 |
| gabanzhaf_evidence_metric | re2-ob | 0.390 | 0.825 | 7.50 | 7.30–7.60 |
| gabanzhaf_evidence_metric | re2-ss | 0.433 | 0.865 | 8.00 | 7.90–8.50 |
| gabanzhaf_evidence_metric | re2-tt | 0.743 | 0.842 | 25.60 | 25.30–25.70 |
| gabanzhaf_k6 | re2-ob | 0.380 | 0.838 | 7.40 | 7.40–7.40 |
| gabanzhaf_k6 | re2-ss | 0.467 | 0.872 | 8.00 | 8.00–8.00 |
| gabanzhaf_k6 | re2-tt | 0.812 | 0.883 | 26.10 | 25.80–26.10 |
| gabanzhaf_k8 | re2-ob | 0.380 | 0.838 | 7.40 | 7.40–7.60 |
| gabanzhaf_k8 | re2-ss | 0.467 | 0.872 | 8.10 | 8.10–8.50 |
| gabanzhaf_k8 | re2-tt | 0.812 | 0.883 | 25.80 | 25.70–26.00 |
| gabanzhaf_k10 | re2-ob | 0.380 | 0.838 | 7.50 | 7.40–7.80 |
| gabanzhaf_k10 | re2-ss | 0.467 | 0.872 | 8.10 | 8.00–8.40 |
| gabanzhaf_k10 | re2-tt | 0.812 | 0.883 | 25.80 | 25.80–26.70 |
| gabanzhaf_k12 | re2-ob | 0.380 | 0.838 | 7.40 | 7.30–7.70 |
| gabanzhaf_k12 | re2-ss | 0.467 | 0.872 | 8.10 | 8.10–8.50 |
| gabanzhaf_k12 | re2-tt | 0.812 | 0.883 | 25.90 | 25.70–26.40 |
| gabanzhaf_k16 | re2-ob | 0.380 | 0.838 | 7.40 | 7.30–7.40 |
| gabanzhaf_k16 | re2-ss | 0.467 | 0.872 | 8.00 | 8.00–8.50 |
| gabanzhaf_k16 | re2-tt | 0.812 | 0.883 | 26.30 | 25.90–26.50 |
| with_trace_full | re2-ob | 0.390 | 0.843 | 11.30 | 11.10–12.30 |
| with_trace_full | re2-ss | 0.455 | 0.868 | 8.20 | 8.10–8.30 |
| with_trace_full | re2-tt | 0.765 | 0.862 | 35.40 | 34.60–36.50 |
| with_trace_no_scores | re2-ob | 0.390 | 0.843 | 11.40 | 11.10–11.80 |
| with_trace_no_scores | re2-ss | 0.455 | 0.868 | 8.10 | 8.10–8.60 |
| with_trace_no_scores | re2-tt | 0.765 | 0.862 | 34.70 | 34.50–35.40 |
| with_trace_no_call_graph | re2-ob | 0.380 | 0.840 | 11.10 | 11.10–11.60 |
| with_trace_no_call_graph | re2-ss | 0.455 | 0.868 | 8.40 | 8.20–8.60 |
| with_trace_no_call_graph | re2-tt | 0.765 | 0.862 | 34.80 | 34.60–35.20 |
| with_trace_no_trace | re2-ob | 0.380 | 0.840 | 7.60 | 7.60–8.00 |
| with_trace_no_trace | re2-ss | 0.455 | 0.868 | 8.20 | 8.10–8.50 |
| with_trace_no_trace | re2-tt | 0.765 | 0.862 | 26.10 | 25.70–26.30 |
| with_trace_no_residual | re2-ob | 0.390 | 0.843 | 11.40 | 11.20–11.40 |
| with_trace_no_residual | re2-ss | 0.455 | 0.868 | 8.20 | 8.10–8.60 |
| with_trace_no_residual | re2-tt | 0.765 | 0.862 | 34.80 | 34.30–35.30 |
| gabanzhaf_k1 | re2-ob | 0.380 | 0.477 | 7.40 | 7.30–7.50 |
| gabanzhaf_k1 | re2-ss | 0.467 | 0.640 | 8.20 | 8.00–8.20 |
| gabanzhaf_k1 | re2-tt | 0.800 | 0.800 | 26.00 | 26.00–26.40 |
| gabanzhaf_k2 | re2-ob | 0.380 | 0.795 | 7.50 | 7.50–7.50 |
| gabanzhaf_k2 | re2-ss | 0.467 | 0.852 | 8.10 | 8.00–8.20 |
| gabanzhaf_k2 | re2-tt | 0.812 | 0.853 | 26.10 | 26.00–26.20 |
| gabanzhaf_k3 | re2-ob | 0.380 | 0.818 | 7.40 | 7.30–7.90 |
| gabanzhaf_k3 | re2-ss | 0.467 | 0.872 | 8.00 | 8.00–8.10 |
| gabanzhaf_k3 | re2-tt | 0.812 | 0.880 | 25.80 | 25.50–26.00 |
| gabanzhaf_k4 | re2-ob | 0.380 | 0.838 | 7.50 | 7.40–7.90 |
| gabanzhaf_k4 | re2-ss | 0.467 | 0.872 | 8.20 | 8.00–8.40 |
| gabanzhaf_k4 | re2-tt | 0.812 | 0.880 | 26.30 | 26.30–26.40 |
| gabanzhaf_k5 | re2-ob | 0.380 | 0.838 | 8.00 | 8.00–8.10 |
| gabanzhaf_k5 | re2-ss | 0.467 | 0.872 | 8.30 | 8.20–8.30 |
| gabanzhaf_k5 | re2-tt | 0.812 | 0.883 | 26.20 | 25.70–26.70 |

## Paired case comparison

| Baseline | Variant | Dataset | Cases | Wins | Losses | Ties | Same top-1 | Same ordered top-5 | Candidate Jaccard | Candidate recall |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gabanzhaf | gabanzhaf_no_lag_1 | re2-ob | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 1.000 | 1.000 |
| gabanzhaf | gabanzhaf_no_lag_1 | re2-ss | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 1.000 | 1.000 |
| gabanzhaf | gabanzhaf_no_lag_1 | re2-tt | 90 | 0 | 1 | 89 | 0.989 | 0.989 | 1.000 | 0.944 |
| gabanzhaf | gabanzhaf_evidence_service | re2-ob | 90 | 0 | 0 | 90 | 0.033 | 0.011 | 1.000 | 1.000 |
| gabanzhaf | gabanzhaf_evidence_service | re2-ss | 90 | 0 | 0 | 90 | 0.100 | 0.033 | 1.000 | 1.000 |
| gabanzhaf | gabanzhaf_evidence_service | re2-tt | 90 | 0 | 1 | 89 | 0.056 | 0.000 | 1.000 | 0.944 |
| gabanzhaf | gabanzhaf_evidence_metric | re2-ob | 90 | 1 | 0 | 89 | 0.033 | 0.000 | 1.000 | 1.000 |
| gabanzhaf | gabanzhaf_evidence_metric | re2-ss | 90 | 0 | 3 | 87 | 0.089 | 0.000 | 1.000 | 1.000 |
| gabanzhaf | gabanzhaf_evidence_metric | re2-tt | 90 | 0 | 6 | 84 | 0.044 | 0.000 | 1.000 | 0.944 |
| gabanzhaf | gabanzhaf_k1 | re2-ob | 90 | 0 | 0 | 90 | 1.000 | 0.300 | 0.084 | 0.378 |
| gabanzhaf | gabanzhaf_k1 | re2-ss | 90 | 0 | 0 | 90 | 1.000 | 0.500 | 0.083 | 0.467 |
| gabanzhaf | gabanzhaf_k1 | re2-tt | 90 | 0 | 1 | 89 | 0.989 | 0.944 | 0.083 | 0.800 |
| gabanzhaf | gabanzhaf_k2 | re2-ob | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 0.167 | 0.889 |
| gabanzhaf | gabanzhaf_k2 | re2-ss | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 0.167 | 0.922 |
| gabanzhaf | gabanzhaf_k2 | re2-tt | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 0.167 | 0.867 |
| gabanzhaf | gabanzhaf_k3 | re2-ob | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 0.251 | 0.944 |
| gabanzhaf | gabanzhaf_k3 | re2-ss | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 0.250 | 0.989 |
| gabanzhaf | gabanzhaf_k3 | re2-tt | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 0.250 | 0.911 |
| gabanzhaf | gabanzhaf_k4 | re2-ob | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 0.334 | 0.989 |
| gabanzhaf | gabanzhaf_k4 | re2-ss | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 0.333 | 0.989 |
| gabanzhaf | gabanzhaf_k4 | re2-tt | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 0.333 | 0.911 |
| gabanzhaf | gabanzhaf_k5 | re2-ob | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 0.418 | 0.989 |
| gabanzhaf | gabanzhaf_k5 | re2-ss | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 0.417 | 0.989 |
| gabanzhaf | gabanzhaf_k5 | re2-tt | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 0.417 | 0.922 |
| gabanzhaf | gabanzhaf_k6 | re2-ob | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 0.501 | 0.989 |
| gabanzhaf | gabanzhaf_k6 | re2-ss | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 0.500 | 0.989 |
| gabanzhaf | gabanzhaf_k6 | re2-tt | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 0.500 | 0.933 |
| gabanzhaf | gabanzhaf_k8 | re2-ob | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 0.668 | 1.000 |
| gabanzhaf | gabanzhaf_k8 | re2-ss | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 0.667 | 1.000 |
| gabanzhaf | gabanzhaf_k8 | re2-tt | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 0.667 | 0.933 |
| gabanzhaf | gabanzhaf_k10 | re2-ob | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 0.835 | 1.000 |
| gabanzhaf | gabanzhaf_k10 | re2-ss | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 0.833 | 1.000 |
| gabanzhaf | gabanzhaf_k10 | re2-tt | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 0.833 | 0.944 |
| gabanzhaf | gabanzhaf_k12 | re2-ob | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 1.000 | 1.000 |
| gabanzhaf | gabanzhaf_k12 | re2-ss | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 1.000 | 1.000 |
| gabanzhaf | gabanzhaf_k12 | re2-tt | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 1.000 | 0.944 |
| gabanzhaf | gabanzhaf_k16 | re2-ob | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 1.000 | 1.000 |
| gabanzhaf | gabanzhaf_k16 | re2-ss | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 0.800 | 1.000 |
| gabanzhaf | gabanzhaf_k16 | re2-tt | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 0.766 | 0.967 |
| gabanzhaf | with_trace_full | re2-ob | 90 | 1 | 0 | 89 | 0.033 | 0.000 | 1.000 | 1.000 |
| gabanzhaf | with_trace_full | re2-ss | 90 | 0 | 1 | 89 | 0.089 | 0.000 | 1.000 | 1.000 |
| gabanzhaf | with_trace_full | re2-tt | 90 | 0 | 4 | 86 | 0.044 | 0.000 | 0.704 | 0.978 |
| with_trace_full | with_trace_no_scores | re2-ob | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 1.000 | 1.000 |
| with_trace_full | with_trace_no_scores | re2-ss | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 1.000 | 1.000 |
| with_trace_full | with_trace_no_scores | re2-tt | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 0.704 | 0.944 |
| with_trace_full | with_trace_no_call_graph | re2-ob | 90 | 0 | 1 | 89 | 0.989 | 0.911 | 1.000 | 1.000 |
| with_trace_full | with_trace_no_call_graph | re2-ss | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 1.000 | 1.000 |
| with_trace_full | with_trace_no_call_graph | re2-tt | 90 | 0 | 0 | 90 | 0.989 | 0.889 | 0.953 | 0.978 |
| with_trace_full | with_trace_no_trace | re2-ob | 90 | 0 | 1 | 89 | 0.989 | 0.911 | 1.000 | 1.000 |
| with_trace_full | with_trace_no_trace | re2-ss | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 1.000 | 1.000 |
| with_trace_full | with_trace_no_trace | re2-tt | 90 | 0 | 0 | 90 | 0.989 | 0.911 | 0.705 | 0.944 |
| with_trace_full | with_trace_no_residual | re2-ob | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 1.000 | 1.000 |
| with_trace_full | with_trace_no_residual | re2-ss | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 1.000 | 1.000 |
| with_trace_full | with_trace_no_residual | re2-tt | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 0.901 | 0.989 |

## Trace overlap diagnostics

| Method | Dataset | Cases | Spearman | Top-K overlap | Added seeds | Accepted call edges | Effective call edges | Trace seconds |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| with_trace_full | re2-ob | 90 | 0.345 | 1.000 | 0.00 | 0.34 | 0.34 | 0.038 |
| with_trace_full | re2-ss | 90 | 0.000 | 0.000 | 0.00 | 0.00 | 0.00 | 0.000 |
| with_trace_full | re2-tt | 90 | 0.213 | 0.607 | 9.96 | 1.30 | 1.30 | 0.100 |
| with_trace_no_scores | re2-ob | 90 | 0.000 | 0.000 | 0.00 | 0.34 | 0.34 | 0.038 |
| with_trace_no_scores | re2-ss | 90 | 0.000 | 0.000 | 0.00 | 0.00 | 0.00 | 0.000 |
| with_trace_no_scores | re2-tt | 90 | 0.000 | 0.000 | 0.00 | 1.30 | 1.28 | 0.100 |
| with_trace_no_call_graph | re2-ob | 90 | 0.345 | 1.000 | 0.00 | 0.00 | 0.00 | 0.036 |
| with_trace_no_call_graph | re2-ss | 90 | 0.000 | 0.000 | 0.00 | 0.00 | 0.00 | 0.000 |
| with_trace_no_call_graph | re2-tt | 90 | 0.213 | 0.607 | 9.96 | 0.00 | 0.00 | 0.099 |
| with_trace_no_trace | re2-ob | 90 | 0.000 | 0.000 | 0.00 | 0.00 | 0.00 | 0.000 |
| with_trace_no_trace | re2-ss | 90 | 0.000 | 0.000 | 0.00 | 0.00 | 0.00 | 0.000 |
| with_trace_no_trace | re2-tt | 90 | 0.000 | 0.000 | 0.00 | 0.00 | 0.00 | 0.000 |
| with_trace_no_residual | re2-ob | 90 | 0.345 | 1.000 | 0.00 | 0.34 | 0.34 | 0.038 |
| with_trace_no_residual | re2-ss | 90 | 0.000 | 0.000 | 0.00 | 0.00 | 0.00 | 0.000 |
| with_trace_no_residual | re2-tt | 90 | 0.213 | 0.607 | 9.96 | 1.30 | 1.30 | 0.100 |

## Experiment-supported conclusions

- Candidate K=5–6, 8, 10, 12, 16 produced identical paired top-1 and ordered top-5 rankings across the evaluated datasets.
- Candidate K=1–4 changed at least one benchmark metric; low-K effects must be interpreted as candidate truncation effects before attributing them to Banzhaf scoring.
- Removing lag-1 produced 0 wins and 1 losses across 270 paired cases.
- Removing trace scores produced 0 wins and 0 losses across 270 paired cases.
- Removing residual suppression produced 0 wins and 0 losses across 270 paired cases.
- Removing the call graph produced 0 wins and 1 losses across 270 paired cases.

## Acceptable method limitations

- Closed-form and brute-force Banzhaf are equivalent implementations of the same noisy-OR game; runtime, not accuracy, is the intended comparison.
- Deterministic name ordering makes ties reproducible, but it is not evidence-based metric identification.
- Aggregate AC metrics can hide ranking changes that do not cross the evaluation cutoff; paired top-1/top-5 rates must be reported with them.

## Issues requiring follow-up

- Lag topology currently takes a raw maximum across metrics with different units and may be dominated by the largest-scale metric.
- Concatenating normal and anomalous regimes can create shared step-change correlations that are not propagation.
- Self-explanation probability 1 can make the Banzhaf score nearly identical to service evidence when off-diagonal probabilities are weak.
- The baseline ranks metrics within a service by column name rather than local metric evidence.

## Redundant stages and measured cost

- Trace scores did not change top-1 correctness in 270 controlled paired cases.
- Residual suppression did not change top-1 correctness in 270 controlled paired cases.
- re2-ob: removing trace reduced median wall time by 3.70s (1.49×).
- re2-ss: removing trace reduced median wall time by 0.00s (1.00×).
- re2-tt: removing trace reduced median wall time by 9.30s (1.36×).

## Interpretation rules

- Treat direct comparisons between the two source files as version comparisons, not isolated trace effects.
- Attribute trace effects only from controlled variants that hold residual, ranking, and candidate logic constant.
- If accepted call edges are non-zero but effective edges are zero, the call graph is removed by candidate truncation or matrix gating.
- If candidate recall changes with K, accuracy changes cannot be attributed only to Banzhaf scoring.
