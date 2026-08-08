# MERIT direct single-source ranking results (supplementary)

These rows were removed from the TANET Table 2 on 2026-08-02 when the main
direct-ranking comparator was changed to the normalized metric+trace score sum.
They remain traceable supplementary evidence and are not deleted.

Both comparators bypass the causal game and Banzhaf attribution:

- **Direct trace score**: rank services only by trace-derived evidence.
- **Direct metric score**: rank services only by metric-derived service
  evidence.

All values are service-level percentages. Each applicable dataset cell contains
90 cases; macro averages applicable systems equally.

| Comparator | Dataset | AC@1 | AC@3 | AC@5 | AVG@5 |
|---|---|---:|---:|---:|---:|
| Direct trace score | RE2-SS | N/A | N/A | N/A | N/A |
| Direct trace score | RE2-OB | 41.1 | 63.3 | 74.4 | 60.7 |
| Direct trace score | RE2-TT | 31.1 | 60.0 | 67.8 | 54.9 |
| Direct trace score | Compatible macro (OB/TT) | 36.1 | 61.7 | 71.1 | 57.8 |
| Direct metric score | RE2-SS | 38.9 | 95.6 | 98.9 | 84.7 |
| Direct metric score | RE2-OB | 32.2 | 96.7 | 100.0 | 84.0 |
| Direct metric score | RE2-TT | 78.9 | 88.9 | 92.2 | 87.3 |
| Direct metric score | Three-system macro | 50.0 | 93.7 | 97.0 | 85.3 |

Evidence source:
`artifacts/reproductions/merit-v14-no-metric-strict-20260802-a/analysis-combined-strict-20260802/dataset_summaries.json`.
The method labels are `merit_trace_only` and
`merit_direct_service_evidence`, respectively.
