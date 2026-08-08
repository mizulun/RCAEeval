# GA-Banzhaf Ablation and Trace Diagnostics

## Aggregate benchmark

| Method | Dataset | AC@1 macro | Avg@5 macro | Wall median (s) | Range (s) |
|---|---|---:|---:|---:|---:|
| gabanzhaf | re2-ob | 0.368 | 0.835 | 17.70 | 17.70–17.70 |
| gabanzhaf | re2-ss | 0.467 | 0.872 | 8.90 | 8.90–8.90 |
| gabanzhaf | re2-tt | 0.855 | 0.910 | 35.40 | 35.40–35.40 |
| gabanzhaf_evidence_service | re2-ob | 0.380 | 0.838 | 16.90 | 16.90–16.90 |
| gabanzhaf_evidence_service | re2-ss | 0.467 | 0.872 | 8.10 | 8.10–8.10 |
| gabanzhaf_evidence_service | re2-tt | 0.800 | 0.882 | 34.90 | 34.90–34.90 |
| gabanzhaf_evidence_metric | re2-ob | 0.390 | 0.825 | 17.00 | 17.00–17.00 |
| gabanzhaf_evidence_metric | re2-ss | 0.433 | 0.865 | 8.00 | 8.00–8.00 |
| gabanzhaf_evidence_metric | re2-tt | 0.743 | 0.842 | 34.50 | 34.50–34.50 |
| gabanzhaf_k1 | re2-ob | 0.380 | 0.477 | 17.00 | 17.00–17.00 |
| gabanzhaf_k1 | re2-ss | 0.467 | 0.640 | 8.40 | 8.40–8.40 |
| gabanzhaf_k1 | re2-tt | 0.822 | 0.832 | 36.00 | 36.00–36.00 |
| gabanzhaf_k2 | re2-ob | 0.402 | 0.798 | 17.30 | 17.30–17.30 |
| gabanzhaf_k2 | re2-ss | 0.467 | 0.852 | 8.10 | 8.10–8.10 |
| gabanzhaf_k2 | re2-tt | 0.855 | 0.892 | 37.60 | 37.60–37.60 |
| gabanzhaf_k3 | re2-ob | 0.390 | 0.823 | 18.30 | 18.30–18.30 |
| gabanzhaf_k3 | re2-ss | 0.467 | 0.872 | 8.80 | 8.80–8.80 |
| gabanzhaf_k3 | re2-tt | 0.855 | 0.913 | 37.00 | 37.00–37.00 |
| gabanzhaf_k4 | re2-ob | 0.390 | 0.845 | 18.20 | 18.20–18.20 |
| gabanzhaf_k4 | re2-ss | 0.467 | 0.872 | 8.70 | 8.70–8.70 |
| gabanzhaf_k4 | re2-tt | 0.855 | 0.912 | 37.00 | 37.00–37.00 |
| gabanzhaf_k5 | re2-ob | 0.390 | 0.845 | 18.10 | 18.10–18.10 |
| gabanzhaf_k5 | re2-ss | 0.467 | 0.872 | 8.60 | 8.60–8.60 |
| gabanzhaf_k5 | re2-tt | 0.855 | 0.910 | 36.90 | 36.90–36.90 |
| gabanzhaf_k6 | re2-ob | 0.390 | 0.838 | 18.10 | 18.10–18.10 |
| gabanzhaf_k6 | re2-ss | 0.467 | 0.872 | 8.70 | 8.70–8.70 |
| gabanzhaf_k6 | re2-tt | 0.855 | 0.910 | 37.20 | 37.20–37.20 |
| gabanzhaf_k8 | re2-ob | 0.390 | 0.838 | 18.10 | 18.10–18.10 |
| gabanzhaf_k8 | re2-ss | 0.467 | 0.872 | 8.70 | 8.70–8.70 |
| gabanzhaf_k8 | re2-tt | 0.855 | 0.910 | 36.90 | 36.90–36.90 |
| gabanzhaf_k10 | re2-ob | 0.390 | 0.838 | 18.10 | 18.10–18.10 |
| gabanzhaf_k10 | re2-ss | 0.467 | 0.872 | 8.60 | 8.60–8.60 |
| gabanzhaf_k10 | re2-tt | 0.855 | 0.910 | 37.70 | 37.70–37.70 |
| gabanzhaf_k12 | re2-ob | 0.368 | 0.835 | 18.70 | 18.70–18.70 |
| gabanzhaf_k12 | re2-ss | 0.467 | 0.872 | 9.10 | 9.10–9.10 |
| gabanzhaf_k12 | re2-tt | 0.855 | 0.910 | 37.40 | 37.40–37.40 |
| gabanzhaf_k16 | re2-ob | 0.368 | 0.835 | 18.30 | 18.30–18.30 |
| gabanzhaf_k16 | re2-ss | 0.467 | 0.872 | 8.80 | 8.80–8.80 |
| gabanzhaf_k16 | re2-tt | 0.843 | 0.905 | 37.40 | 37.40–37.40 |
| gabanzhaf_no_victim_demotion | re2-ob | 0.368 | 0.835 | 18.20 | 18.20–18.20 |
| gabanzhaf_no_victim_demotion | re2-ss | 0.467 | 0.872 | 8.60 | 8.60–8.60 |
| gabanzhaf_no_victim_demotion | re2-tt | 0.843 | 0.912 | 37.60 | 37.60–37.60 |
| with_trace_full | re2-ob | 0.390 | 0.843 | 12.70 | 12.70–12.70 |
| with_trace_full | re2-ss | 0.455 | 0.868 | 9.10 | 9.10–9.10 |
| with_trace_full | re2-tt | 0.765 | 0.862 | 39.70 | 39.70–39.70 |
| with_trace_no_scores | re2-ob | 0.390 | 0.843 | 12.10 | 12.10–12.10 |
| with_trace_no_scores | re2-ss | 0.455 | 0.868 | 9.20 | 9.20–9.20 |
| with_trace_no_scores | re2-tt | 0.765 | 0.862 | 39.10 | 39.10–39.10 |
| with_trace_no_call_graph | re2-ob | 0.380 | 0.840 | 12.30 | 12.30–12.30 |
| with_trace_no_call_graph | re2-ss | 0.455 | 0.868 | 8.90 | 8.90–8.90 |
| with_trace_no_call_graph | re2-tt | 0.765 | 0.862 | 39.70 | 39.70–39.70 |
| with_trace_no_trace | re2-ob | 0.380 | 0.840 | 8.60 | 8.60–8.60 |
| with_trace_no_trace | re2-ss | 0.455 | 0.868 | 9.10 | 9.10–9.10 |
| with_trace_no_trace | re2-tt | 0.765 | 0.862 | 29.20 | 29.20–29.20 |
| with_trace_no_residual | re2-ob | 0.390 | 0.843 | 12.60 | 12.60–12.60 |
| with_trace_no_residual | re2-ss | 0.455 | 0.868 | 9.20 | 9.20–9.20 |
| with_trace_no_residual | re2-tt | 0.765 | 0.862 | 35.30 | 35.30–35.30 |

## Delta vs Baseline

| Method | Dataset | AC@1 macro | ΔAC@1 | Avg@5 macro | ΔAvg@5 | Wall median (s) | ΔWall (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| gabanzhaf_evidence_service | re2-ob | 0.380 | +0.012 | 0.838 | +0.003 | 16.90 | -0.80 |
| gabanzhaf_evidence_service | re2-ss | 0.467 | +0.000 | 0.872 | +0.000 | 8.10 | -0.80 |
| gabanzhaf_evidence_service | re2-tt | 0.800 | -0.055 | 0.882 | -0.028 | 34.90 | -0.50 |
| gabanzhaf_evidence_metric | re2-ob | 0.390 | +0.022 | 0.825 | -0.010 | 17.00 | -0.70 |
| gabanzhaf_evidence_metric | re2-ss | 0.433 | -0.033 | 0.865 | -0.007 | 8.00 | -0.90 |
| gabanzhaf_evidence_metric | re2-tt | 0.743 | -0.112 | 0.842 | -0.068 | 34.50 | -0.90 |
| gabanzhaf_k1 | re2-ob | 0.380 | +0.012 | 0.477 | -0.358 | 17.00 | -0.70 |
| gabanzhaf_k1 | re2-ss | 0.467 | +0.000 | 0.640 | -0.232 | 8.40 | -0.50 |
| gabanzhaf_k1 | re2-tt | 0.822 | -0.033 | 0.832 | -0.078 | 36.00 | +0.60 |
| gabanzhaf_k2 | re2-ob | 0.402 | +0.033 | 0.798 | -0.037 | 17.30 | -0.40 |
| gabanzhaf_k2 | re2-ss | 0.467 | +0.000 | 0.852 | -0.020 | 8.10 | -0.80 |
| gabanzhaf_k2 | re2-tt | 0.855 | +0.000 | 0.892 | -0.018 | 37.60 | +2.20 |
| gabanzhaf_k3 | re2-ob | 0.390 | +0.022 | 0.823 | -0.012 | 18.30 | +0.60 |
| gabanzhaf_k3 | re2-ss | 0.467 | +0.000 | 0.872 | +0.000 | 8.80 | -0.10 |
| gabanzhaf_k3 | re2-tt | 0.855 | +0.000 | 0.913 | +0.003 | 37.00 | +1.60 |
| gabanzhaf_k4 | re2-ob | 0.390 | +0.022 | 0.845 | +0.010 | 18.20 | +0.50 |
| gabanzhaf_k4 | re2-ss | 0.467 | +0.000 | 0.872 | +0.000 | 8.70 | -0.20 |
| gabanzhaf_k4 | re2-tt | 0.855 | +0.000 | 0.912 | +0.002 | 37.00 | +1.60 |
| gabanzhaf_k5 | re2-ob | 0.390 | +0.022 | 0.845 | +0.010 | 18.10 | +0.40 |
| gabanzhaf_k5 | re2-ss | 0.467 | +0.000 | 0.872 | +0.000 | 8.60 | -0.30 |
| gabanzhaf_k5 | re2-tt | 0.855 | +0.000 | 0.910 | +0.000 | 36.90 | +1.50 |
| gabanzhaf_k6 | re2-ob | 0.390 | +0.022 | 0.838 | +0.003 | 18.10 | +0.40 |
| gabanzhaf_k6 | re2-ss | 0.467 | +0.000 | 0.872 | +0.000 | 8.70 | -0.20 |
| gabanzhaf_k6 | re2-tt | 0.855 | +0.000 | 0.910 | +0.000 | 37.20 | +1.80 |
| gabanzhaf_k8 | re2-ob | 0.390 | +0.022 | 0.838 | +0.003 | 18.10 | +0.40 |
| gabanzhaf_k8 | re2-ss | 0.467 | +0.000 | 0.872 | +0.000 | 8.70 | -0.20 |
| gabanzhaf_k8 | re2-tt | 0.855 | +0.000 | 0.910 | +0.000 | 36.90 | +1.50 |
| gabanzhaf_k10 | re2-ob | 0.390 | +0.022 | 0.838 | +0.003 | 18.10 | +0.40 |
| gabanzhaf_k10 | re2-ss | 0.467 | +0.000 | 0.872 | +0.000 | 8.60 | -0.30 |
| gabanzhaf_k10 | re2-tt | 0.855 | +0.000 | 0.910 | +0.000 | 37.70 | +2.30 |
| gabanzhaf_k12 | re2-ob | 0.368 | +0.000 | 0.835 | +0.000 | 18.70 | +1.00 |
| gabanzhaf_k12 | re2-ss | 0.467 | +0.000 | 0.872 | +0.000 | 9.10 | +0.20 |
| gabanzhaf_k12 | re2-tt | 0.855 | +0.000 | 0.910 | +0.000 | 37.40 | +2.00 |
| gabanzhaf_k16 | re2-ob | 0.368 | +0.000 | 0.835 | +0.000 | 18.30 | +0.60 |
| gabanzhaf_k16 | re2-ss | 0.467 | +0.000 | 0.872 | +0.000 | 8.80 | -0.10 |
| gabanzhaf_k16 | re2-tt | 0.843 | -0.012 | 0.905 | -0.005 | 37.40 | +2.00 |
| gabanzhaf_no_victim_demotion | re2-ob | 0.368 | +0.000 | 0.835 | +0.000 | 18.20 | +0.50 |
| gabanzhaf_no_victim_demotion | re2-ss | 0.467 | +0.000 | 0.872 | +0.000 | 8.60 | -0.30 |
| gabanzhaf_no_victim_demotion | re2-tt | 0.843 | -0.012 | 0.912 | +0.002 | 37.60 | +2.20 |
| with_trace_full | re2-ob | 0.390 | +0.022 | 0.843 | +0.008 | 12.70 | -5.00 |
| with_trace_full | re2-ss | 0.455 | -0.012 | 0.868 | -0.003 | 9.10 | +0.20 |
| with_trace_full | re2-tt | 0.765 | -0.090 | 0.862 | -0.048 | 39.70 | +4.30 |
| with_trace_no_scores | re2-ob | 0.390 | +0.000 | 0.843 | +0.000 | 12.10 | -0.60 |
| with_trace_no_scores | re2-ss | 0.455 | +0.000 | 0.868 | +0.000 | 9.20 | +0.10 |
| with_trace_no_scores | re2-tt | 0.765 | +0.000 | 0.862 | +0.000 | 39.10 | -0.60 |
| with_trace_no_call_graph | re2-ob | 0.380 | -0.010 | 0.840 | -0.003 | 12.30 | -0.40 |
| with_trace_no_call_graph | re2-ss | 0.455 | +0.000 | 0.868 | +0.000 | 8.90 | -0.20 |
| with_trace_no_call_graph | re2-tt | 0.765 | +0.000 | 0.862 | +0.000 | 39.70 | +0.00 |
| with_trace_no_trace | re2-ob | 0.380 | -0.010 | 0.840 | -0.003 | 8.60 | -4.10 |
| with_trace_no_trace | re2-ss | 0.455 | +0.000 | 0.868 | +0.000 | 9.10 | +0.00 |
| with_trace_no_trace | re2-tt | 0.765 | +0.000 | 0.862 | +0.000 | 29.20 | -10.50 |
| with_trace_no_residual | re2-ob | 0.390 | +0.000 | 0.843 | +0.000 | 12.60 | -0.10 |
| with_trace_no_residual | re2-ss | 0.455 | +0.000 | 0.868 | +0.000 | 9.20 | +0.10 |
| with_trace_no_residual | re2-tt | 0.765 | +0.000 | 0.862 | +0.000 | 35.30 | -4.40 |

## Paired case comparison

| Baseline | Variant | Dataset | Cases | Wins | Losses | Ties | Same top-1 | Same ordered top-5 | Candidate Jaccard | Candidate recall |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gabanzhaf | gabanzhaf_evidence_service | re2-ob | 90 | 2 | 1 | 87 | 0.022 | 0.000 | 1.000 | 1.000 |
| gabanzhaf | gabanzhaf_evidence_service | re2-ss | 90 | 0 | 0 | 90 | 0.100 | 0.033 | 1.000 | 1.000 |
| gabanzhaf | gabanzhaf_evidence_service | re2-tt | 90 | 1 | 6 | 83 | 0.044 | 0.000 | 1.000 | 0.956 |
| gabanzhaf | gabanzhaf_evidence_metric | re2-ob | 90 | 2 | 0 | 88 | 0.022 | 0.000 | 1.000 | 1.000 |
| gabanzhaf | gabanzhaf_evidence_metric | re2-ss | 90 | 0 | 3 | 87 | 0.089 | 0.000 | 1.000 | 1.000 |
| gabanzhaf | gabanzhaf_evidence_metric | re2-tt | 90 | 0 | 10 | 80 | 0.044 | 0.000 | 1.000 | 0.956 |
| gabanzhaf | gabanzhaf_k1 | re2-ob | 90 | 2 | 1 | 87 | 0.956 | 0.278 | 0.084 | 0.378 |
| gabanzhaf | gabanzhaf_k1 | re2-ss | 90 | 0 | 0 | 90 | 1.000 | 0.500 | 0.083 | 0.467 |
| gabanzhaf | gabanzhaf_k1 | re2-tt | 90 | 1 | 4 | 85 | 0.944 | 0.900 | 0.094 | 0.833 |
| gabanzhaf | gabanzhaf_k2 | re2-ob | 90 | 3 | 0 | 87 | 0.956 | 0.911 | 0.167 | 0.889 |
| gabanzhaf | gabanzhaf_k2 | re2-ss | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 0.167 | 0.922 |
| gabanzhaf | gabanzhaf_k2 | re2-tt | 90 | 0 | 0 | 90 | 1.000 | 0.989 | 0.177 | 0.900 |
| gabanzhaf | gabanzhaf_k3 | re2-ob | 90 | 2 | 0 | 88 | 0.967 | 0.933 | 0.251 | 0.944 |
| gabanzhaf | gabanzhaf_k3 | re2-ss | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 0.250 | 0.989 |
| gabanzhaf | gabanzhaf_k3 | re2-tt | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 0.259 | 0.933 |
| gabanzhaf | gabanzhaf_k4 | re2-ob | 90 | 2 | 0 | 88 | 0.967 | 0.956 | 0.334 | 0.989 |
| gabanzhaf | gabanzhaf_k4 | re2-ss | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 0.333 | 0.989 |
| gabanzhaf | gabanzhaf_k4 | re2-tt | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 0.341 | 0.933 |
| gabanzhaf | gabanzhaf_k5 | re2-ob | 90 | 2 | 0 | 88 | 0.967 | 0.956 | 0.418 | 0.989 |
| gabanzhaf | gabanzhaf_k5 | re2-ss | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 0.417 | 0.989 |
| gabanzhaf | gabanzhaf_k5 | re2-tt | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 0.423 | 0.933 |
| gabanzhaf | gabanzhaf_k6 | re2-ob | 90 | 2 | 0 | 88 | 0.978 | 0.978 | 0.501 | 0.989 |
| gabanzhaf | gabanzhaf_k6 | re2-ss | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 0.500 | 0.989 |
| gabanzhaf | gabanzhaf_k6 | re2-tt | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 0.505 | 0.944 |
| gabanzhaf | gabanzhaf_k8 | re2-ob | 90 | 2 | 0 | 88 | 0.978 | 0.978 | 0.668 | 1.000 |
| gabanzhaf | gabanzhaf_k8 | re2-ss | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 0.667 | 1.000 |
| gabanzhaf | gabanzhaf_k8 | re2-tt | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 0.670 | 0.944 |
| gabanzhaf | gabanzhaf_k10 | re2-ob | 90 | 2 | 0 | 88 | 0.978 | 0.978 | 0.835 | 1.000 |
| gabanzhaf | gabanzhaf_k10 | re2-ss | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 0.833 | 1.000 |
| gabanzhaf | gabanzhaf_k10 | re2-tt | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 0.835 | 0.956 |
| gabanzhaf | gabanzhaf_k12 | re2-ob | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 1.000 | 1.000 |
| gabanzhaf | gabanzhaf_k12 | re2-ss | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 1.000 | 1.000 |
| gabanzhaf | gabanzhaf_k12 | re2-tt | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 1.000 | 0.956 |
| gabanzhaf | gabanzhaf_k16 | re2-ob | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 1.000 | 1.000 |
| gabanzhaf | gabanzhaf_k16 | re2-ss | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 0.800 | 1.000 |
| gabanzhaf | gabanzhaf_k16 | re2-tt | 90 | 0 | 1 | 89 | 0.989 | 0.978 | 0.802 | 0.956 |
| gabanzhaf | gabanzhaf_no_victim_demotion | re2-ob | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 1.000 | 1.000 |
| gabanzhaf | gabanzhaf_no_victim_demotion | re2-ss | 90 | 0 | 0 | 90 | 1.000 | 1.000 | 1.000 | 1.000 |
| gabanzhaf | gabanzhaf_no_victim_demotion | re2-tt | 90 | 0 | 1 | 89 | 0.989 | 0.978 | 1.000 | 0.956 |
| gabanzhaf | with_trace_full | re2-ob | 90 | 2 | 0 | 88 | 0.022 | 0.000 | 1.000 | 1.000 |
| gabanzhaf | with_trace_full | re2-ss | 90 | 0 | 1 | 89 | 0.089 | 0.000 | 1.000 | 1.000 |
| gabanzhaf | with_trace_full | re2-tt | 90 | 0 | 8 | 82 | 0.044 | 0.000 | 0.699 | 0.978 |
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
| with_trace_full | re2-ob | 90 | 0.345 | 1.000 | 0.00 | 0.34 | 0.34 | 0.045 |
| with_trace_full | re2-ss | 90 | 0.000 | 0.000 | 0.00 | 0.00 | 0.00 | 0.000 |
| with_trace_full | re2-tt | 90 | 0.213 | 0.607 | 9.96 | 1.30 | 1.30 | 0.114 |
| with_trace_no_scores | re2-ob | 90 | 0.000 | 0.000 | 0.00 | 0.34 | 0.34 | 0.041 |
| with_trace_no_scores | re2-ss | 90 | 0.000 | 0.000 | 0.00 | 0.00 | 0.00 | 0.000 |
| with_trace_no_scores | re2-tt | 90 | 0.000 | 0.000 | 0.00 | 1.30 | 1.28 | 0.113 |
| with_trace_no_call_graph | re2-ob | 90 | 0.345 | 1.000 | 0.00 | 0.00 | 0.00 | 0.041 |
| with_trace_no_call_graph | re2-ss | 90 | 0.000 | 0.000 | 0.00 | 0.00 | 0.00 | 0.000 |
| with_trace_no_call_graph | re2-tt | 90 | 0.213 | 0.607 | 9.96 | 0.00 | 0.00 | 0.111 |
| with_trace_no_trace | re2-ob | 90 | 0.000 | 0.000 | 0.00 | 0.00 | 0.00 | 0.000 |
| with_trace_no_trace | re2-ss | 90 | 0.000 | 0.000 | 0.00 | 0.00 | 0.00 | 0.000 |
| with_trace_no_trace | re2-tt | 90 | 0.000 | 0.000 | 0.00 | 0.00 | 0.00 | 0.000 |
| with_trace_no_residual | re2-ob | 90 | 0.345 | 1.000 | 0.00 | 0.34 | 0.34 | 0.042 |
| with_trace_no_residual | re2-ss | 90 | 0.000 | 0.000 | 0.00 | 0.00 | 0.00 | 0.000 |
| with_trace_no_residual | re2-tt | 90 | 0.213 | 0.607 | 9.96 | 1.30 | 1.30 | 0.100 |

## Experiment-supported conclusions

- Candidate K=12 produced identical paired top-1 and ordered top-5 rankings across the evaluated datasets.
- Candidate K=1–6, 8, 10, 16 changed at least one benchmark metric; low-K effects must be interpreted as candidate truncation effects before attributing them to Banzhaf scoring.
- Removing lag-1 produced 0 wins and 0 losses across 0 paired cases.
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
- re2-ob: removing trace reduced median wall time by 4.10s (1.48×).
- re2-ss: removing trace reduced median wall time by 0.00s (1.00×).
- re2-tt: removing trace reduced median wall time by 10.50s (1.36×).

## Interpretation rules

- Treat direct comparisons between the two source files as version comparisons, not isolated trace effects.
- Attribute trace effects only from controlled variants that hold residual, ranking, and candidate logic constant.
- If accepted call edges are non-zero but effective edges are zero, the call graph is removed by candidate truncation or matrix gating.
- If candidate recall changes with K, accuracy changes cannot be attributed only to Banzhaf scoring.
