# GA-Banzhaf Ablation and Trace Diagnostics

## Aggregate benchmark

| Method | Dataset | AC@1 macro | Avg@5 macro | Wall median (s) | Range (s) |
|---|---|---:|---:|---:|---:|
| merit | re2-tt | 0.855 | 0.910 | 35.40 | 35.40–35.40 |
| merit_approx_banzhaf | re2-tt | 0.855 | 0.910 | 35.60 | 35.60–35.60 |
| merit_pure_banzhaf | re2-tt | 0.855 | 0.910 | 388.10 | 388.10–388.10 |
| merit_evidence_service | re2-tt | 0.800 | 0.882 | 35.30 | 35.30–35.30 |
| merit_evidence_metric | re2-tt | 0.743 | 0.842 | 34.40 | 34.40–34.40 |
| merit_full | re2-tt | 0.855 | 0.910 | 36.30 | 36.30–36.30 |
| merit_no_lag_1 | re2-tt | 0.855 | 0.910 | 36.30 | 36.30–36.30 |
| merit_no_trace_topology | re2-tt | 0.845 | 0.893 | 31.20 | 31.20–31.20 |
| merit_no_victim_demotion | re2-tt | 0.843 | 0.912 | 35.80 | 35.80–35.80 |
| merit_tconf_085 | re2-tt | 0.855 | 0.910 | 35.10 | 35.10–35.10 |
| merit_tconf_095 | re2-tt | 0.832 | 0.905 | 35.00 | 35.00–35.00 |
| merit_victim_03 | re2-tt | 0.855 | 0.905 | 35.50 | 35.50–35.50 |
| merit_victim_05 | re2-tt | 0.855 | 0.910 | 35.50 | 35.50–35.50 |
| merit_victim_07 | re2-tt | 0.843 | 0.915 | 35.20 | 35.20–35.20 |
| merit_k1 | re2-tt | 0.822 | 0.832 | 35.00 | 35.00–35.00 |
| merit_k2 | re2-tt | 0.855 | 0.892 | 35.00 | 35.00–35.00 |
| merit_k3 | re2-tt | 0.855 | 0.913 | 35.00 | 35.00–35.00 |
| merit_k4 | re2-tt | 0.855 | 0.912 | 34.90 | 34.90–34.90 |
| merit_k5 | re2-tt | 0.855 | 0.910 | 35.00 | 35.00–35.00 |
| merit_k6 | re2-tt | 0.855 | 0.910 | 34.90 | 34.90–34.90 |
| merit_k8 | re2-tt | 0.855 | 0.910 | 35.10 | 35.10–35.10 |
| merit_k10 | re2-tt | 0.855 | 0.910 | 35.10 | 35.10–35.10 |
| merit_k12 | re2-tt | 0.855 | 0.910 | 35.70 | 35.70–35.70 |
| merit_k16 | re2-tt | 0.843 | 0.905 | 33.90 | 33.90–33.90 |

## Delta vs Baseline

| Method | Dataset | AC@1 macro | ΔAC@1 | Avg@5 macro | ΔAvg@5 | Wall median (s) | ΔWall (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| merit_approx_banzhaf | re2-tt | 0.855 | +0.000 | 0.910 | +0.000 | 35.60 | +0.20 |
| merit_pure_banzhaf | re2-tt | 0.855 | +0.000 | 0.910 | +0.000 | 388.10 | +352.70 |
| merit_evidence_service | re2-tt | 0.800 | -0.055 | 0.882 | -0.028 | 35.30 | -0.10 |
| merit_evidence_metric | re2-tt | 0.743 | -0.112 | 0.842 | -0.068 | 34.40 | -1.00 |
| merit_full | re2-tt | 0.855 | +0.000 | 0.910 | +0.000 | 36.30 | +0.90 |
| merit_no_lag_1 | re2-tt | 0.855 | +0.000 | 0.910 | +0.000 | 36.30 | +0.90 |
| merit_no_trace_topology | re2-tt | 0.845 | -0.010 | 0.893 | -0.017 | 31.20 | -4.20 |
| merit_no_victim_demotion | re2-tt | 0.843 | -0.012 | 0.912 | +0.002 | 35.80 | +0.40 |
| merit_tconf_085 | re2-tt | 0.855 | +0.000 | 0.910 | +0.000 | 35.10 | -0.30 |
| merit_tconf_095 | re2-tt | 0.832 | -0.023 | 0.905 | -0.005 | 35.00 | -0.40 |
| merit_victim_03 | re2-tt | 0.855 | +0.000 | 0.905 | -0.005 | 35.50 | +0.10 |
| merit_victim_05 | re2-tt | 0.855 | +0.000 | 0.910 | +0.000 | 35.50 | +0.10 |
| merit_victim_07 | re2-tt | 0.843 | -0.012 | 0.915 | +0.005 | 35.20 | -0.20 |
| merit_k1 | re2-tt | 0.822 | -0.033 | 0.832 | -0.078 | 35.00 | -0.40 |
| merit_k2 | re2-tt | 0.855 | +0.000 | 0.892 | -0.018 | 35.00 | -0.40 |
| merit_k3 | re2-tt | 0.855 | +0.000 | 0.913 | +0.003 | 35.00 | -0.40 |
| merit_k4 | re2-tt | 0.855 | +0.000 | 0.912 | +0.002 | 34.90 | -0.50 |
| merit_k5 | re2-tt | 0.855 | +0.000 | 0.910 | +0.000 | 35.00 | -0.40 |
| merit_k6 | re2-tt | 0.855 | +0.000 | 0.910 | +0.000 | 34.90 | -0.50 |
| merit_k8 | re2-tt | 0.855 | +0.000 | 0.910 | +0.000 | 35.10 | -0.30 |
| merit_k10 | re2-tt | 0.855 | +0.000 | 0.910 | +0.000 | 35.10 | -0.30 |
| merit_k12 | re2-tt | 0.855 | +0.000 | 0.910 | +0.000 | 35.70 | +0.30 |
| merit_k16 | re2-tt | 0.843 | -0.012 | 0.905 | -0.005 | 33.90 | -1.50 |

## Paired case comparison

| Baseline | Variant | Dataset | Cases | Wins | Losses | Ties | Same top-1 | Same ordered top-5 | Candidate Jaccard | Candidate recall |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| merit | merit_approx_banzhaf | re2-tt | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 1.000 | 0.956 |
| merit | merit_pure_banzhaf | re2-tt | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 1.000 | 0.956 |
| merit | merit_evidence_service | re2-tt | 90 | 1 | 6 | 83 | 0.044 | 0.000 | 1.000 | 0.956 |
| merit | merit_evidence_metric | re2-tt | 90 | 0 | 10 | 80 | 0.044 | 0.000 | 1.000 | 0.956 |
| merit | merit_full | re2-tt | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 1.000 | 0.956 |
| merit | merit_no_lag_1 | re2-tt | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 1.000 | 0.956 |
| merit | merit_no_trace_topology | re2-tt | 90 | 1 | 2 | 87 | 0.967 | 0.956 | 1.000 | 0.956 |
| merit | merit_no_victim_demotion | re2-tt | 90 | 0 | 1 | 89 | 0.989 | 0.978 | 1.000 | 0.956 |
| merit | merit_tconf_085 | re2-tt | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 1.000 | 0.956 |
| merit | merit_tconf_095 | re2-tt | 90 | 0 | 2 | 88 | 0.978 | 0.967 | 1.000 | 0.956 |
| merit | merit_victim_03 | re2-tt | 90 | 0 | 0 | 90 | 1.000 | 0.989 | 1.000 | 0.956 |
| merit | merit_victim_05 | re2-tt | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 1.000 | 0.956 |
| merit | merit_victim_07 | re2-tt | 90 | 0 | 1 | 89 | 0.989 | 0.978 | 1.000 | 0.956 |
| merit | merit_k1 | re2-tt | 90 | 1 | 4 | 85 | 0.944 | 0.900 | 0.094 | 0.833 |
| merit | merit_k2 | re2-tt | 90 | 0 | 0 | 90 | 1.000 | 0.989 | 0.177 | 0.900 |
| merit | merit_k3 | re2-tt | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 0.259 | 0.933 |
| merit | merit_k4 | re2-tt | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 0.341 | 0.933 |
| merit | merit_k5 | re2-tt | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 0.423 | 0.933 |
| merit | merit_k6 | re2-tt | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 0.505 | 0.944 |
| merit | merit_k8 | re2-tt | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 0.670 | 0.944 |
| merit | merit_k10 | re2-tt | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 0.835 | 0.956 |
| merit | merit_k12 | re2-tt | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 1.000 | 0.956 |
| merit | merit_k16 | re2-tt | 90 | 0 | 1 | 89 | 0.989 | 0.978 | 0.802 | 0.956 |

## Trace overlap diagnostics

| Method | Dataset | Cases | Spearman | Top-K overlap | Added seeds | Accepted call edges | Effective call edges | Trace seconds |
|---|---|---:|---:|---:|---:|---:|---:|---:|

## Experiment-supported conclusions

- Candidate K=5–6, 8, 10, 12 produced identical paired top-1 and ordered top-5 rankings across the evaluated datasets.
- Candidate K=1–4, 16 changed at least one benchmark metric; low-K effects must be interpreted as candidate truncation effects before attributing them to Banzhaf scoring.
- Removing lag-1 produced 0 wins and 0 losses across 90 paired cases.
- Removing trace scores produced 0 wins and 0 losses across 0 paired cases.
- Removing residual suppression produced 0 wins and 0 losses across 0 paired cases.
- Removing the call graph produced 0 wins and 0 losses across 0 paired cases.

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

- Trace scores did not change top-1 correctness in 0 controlled paired cases.
- Residual suppression did not change top-1 correctness in 0 controlled paired cases.
- Runtime overhead could not be computed because benchmark timing rows were unavailable.

## Interpretation rules

- Treat direct comparisons between the two source files as version comparisons, not isolated trace effects.
- Attribute trace effects only from controlled variants that hold residual, ranking, and candidate logic constant.
- If accepted call edges are non-zero but effective edges are zero, the call graph is removed by candidate truncation or matrix gating.
- If candidate recall changes with K, accuracy changes cannot be attributed only to Banzhaf scoring.
