# MERIT re2-tt Ablation Table

Source: `benchmark_merit_re2_tt_current_ablation_2026-07-07.json`

Macro metrics are averaged across CPU, MEM, DISK, SOCKET, DELAY, and LOSS. Delta columns are relative to `merit_full`.

## Main Component Ablations

| Method | Label | AC@1 | ΔAC@1 | AC@3 | ΔAC@3 | AC@5 | ΔAC@5 | Avg@5 | ΔAvg@5 | Algo s/case | Wall s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `merit_full` | Full baseline | 0.855 | +0.000 | 0.922 | +0.000 | 0.933 | +0.000 | 0.910 | +0.000 | 0.22 | 37.2 |
| `merit_no_trace` | No trace | 0.800 | -0.055 | 0.910 | -0.012 | 0.922 | -0.012 | 0.882 | -0.028 | 0.20 | 32.3 |
| `merit_no_trace_topology` | No trace topology | 0.845 | -0.010 | 0.900 | -0.022 | 0.922 | -0.012 | 0.893 | -0.017 | 0.18 | 32.5 |
| `merit_no_coverage` | No coverage | 0.822 | -0.033 | 0.888 | -0.033 | 0.900 | -0.033 | 0.877 | -0.033 | 0.23 | 37.1 |
| `merit_no_victim_demotion` | No victim demotion | 0.843 | -0.012 | 0.933 | +0.012 | 0.933 | +0.000 | 0.912 | +0.002 | 0.22 | 36.6 |
| `merit_pure_banzhaf` | Pure Banzhaf | 0.855 | +0.000 | 0.922 | +0.000 | 0.933 | +0.000 | 0.910 | +0.000 | 4.12 | 387.9 |
| `merit_evidence_metric` | Metric evidence only | 0.743 | -0.112 | 0.855 | -0.067 | 0.888 | -0.045 | 0.842 | -0.068 | 0.21 | 34.5 |
| `merit_evidence_service` | Service evidence only | 0.800 | -0.055 | 0.910 | -0.012 | 0.922 | -0.012 | 0.882 | -0.028 | 0.22 | 35.5 |
| `merit_evidence_trace` | Trace evidence only | 0.000 | -0.855 | 0.012 | -0.910 | 0.035 | -0.898 | 0.017 | -0.893 | 0.21 | 35.3 |
| `merit_evidence_service_trace` | Service + trace evidence | 0.788 | -0.067 | 0.900 | -0.022 | 0.922 | -0.012 | 0.872 | -0.038 | 0.22 | 35.5 |
| `merit_residual` | Residual trace | 0.832 | -0.023 | 0.922 | +0.000 | 0.933 | +0.000 | 0.898 | -0.012 | 0.24 | 38.3 |

## Sensitivity Sweep

| Method | Label | AC@1 | ΔAC@1 | AC@3 | ΔAC@3 | AC@5 | ΔAC@5 | Avg@5 | ΔAvg@5 | Algo s/case | Wall s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `merit_k1` | exact_candidate_k=1 | 0.822 | -0.033 | 0.833 | -0.088 | 0.833 | -0.100 | 0.832 | -0.078 | 0.23 | 37.4 |
| `merit_k2` | exact_candidate_k=2 | 0.855 | +0.000 | 0.900 | -0.022 | 0.900 | -0.033 | 0.892 | -0.018 | 0.22 | 35.5 |
| `merit_k3` | exact_candidate_k=3 | 0.855 | +0.000 | 0.933 | +0.012 | 0.933 | +0.000 | 0.913 | +0.003 | 0.22 | 35.5 |
| `merit_k4` | exact_candidate_k=4 | 0.855 | +0.000 | 0.922 | +0.000 | 0.933 | +0.000 | 0.912 | +0.002 | 0.21 | 34.9 |
| `merit_k5` | exact_candidate_k=5 | 0.855 | +0.000 | 0.922 | +0.000 | 0.933 | +0.000 | 0.910 | +0.000 | 0.21 | 35.2 |
| `merit_k6` | exact_candidate_k=6 | 0.855 | +0.000 | 0.922 | +0.000 | 0.933 | +0.000 | 0.910 | +0.000 | 0.21 | 35.1 |
| `merit_k8` | exact_candidate_k=8 | 0.855 | +0.000 | 0.922 | +0.000 | 0.933 | +0.000 | 0.910 | +0.000 | 0.21 | 35.6 |
| `merit_k10` | exact_candidate_k=10 | 0.855 | +0.000 | 0.922 | +0.000 | 0.933 | +0.000 | 0.910 | +0.000 | 0.21 | 35.3 |
| `merit_k12` | exact_candidate_k=12 | 0.855 | +0.000 | 0.922 | +0.000 | 0.933 | +0.000 | 0.910 | +0.000 | 0.21 | 35.1 |
| `merit_k16` | exact_candidate_k=16 | 0.843 | -0.012 | 0.922 | +0.000 | 0.922 | -0.012 | 0.905 | -0.005 | 0.22 | 35.4 |
| `merit_victim_03` | victim_floor=0.3 | 0.855 | +0.000 | 0.922 | +0.000 | 0.922 | -0.012 | 0.905 | -0.005 | 0.21 | 35.3 |
| `merit_victim_05` | victim_floor=0.5 | 0.855 | +0.000 | 0.922 | +0.000 | 0.933 | +0.000 | 0.910 | +0.000 | 0.21 | 35.2 |
| `merit_victim_07` | victim_floor=0.7 | 0.843 | -0.012 | 0.933 | +0.012 | 0.933 | +0.000 | 0.915 | +0.005 | 0.21 | 35.1 |
| `merit_tconf_07` | trace_topology_conf=0.7 | 0.843 | -0.012 | 0.922 | +0.000 | 0.933 | +0.000 | 0.903 | -0.007 | 0.22 | 35.2 |
| `merit_tconf_085` | trace_topology_conf=0.85 | 0.855 | +0.000 | 0.922 | +0.000 | 0.933 | +0.000 | 0.910 | +0.000 | 0.21 | 35.1 |
| `merit_tconf_095` | trace_topology_conf=0.95 | 0.832 | -0.023 | 0.922 | +0.000 | 0.933 | +0.000 | 0.905 | -0.005 | 0.20 | 33.6 |

## Raw Artifacts

- Full raw CSV: `benchmark_merit_re2_tt_current_ablation_2026-07-07.csv`
- Full raw JSON: `benchmark_merit_re2_tt_current_ablation_2026-07-07.json`
- Summary CSV: `benchmark_merit_re2_tt_current_ablation_table_2026-07-07.csv`
