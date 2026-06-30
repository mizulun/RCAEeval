# Uniform Call-Graph Construction & No-Trace Ablation

## Motivation

The method must be one uniform pipeline that runs identically on every dataset —
no per-dataset special-casing — to be publishable. The previous implementation
violated this: `_load_call_graph` branched on the dataset name and loaded a
hand-built `tt_callgraph.json` (train-ticket only). On online-boutique and
sock-shop the call graph was empty, so coverage/trace silently did nothing and
the method degraded to the metric baseline — a dataset-specific special case in
disguise.

## Change

The service call graph is now built at runtime from each case's own
`traces.csv`, with the same construction for every dataset and no precomputed
files. All in `RCAEval/e2e/gabanzhaf.py`:

- `_build_call_graph_from_traces(case_dir, service_names, cache_key)` — joins
  spans on `parentSpanID`: a span's parent service is the caller, the span's own
  service is the callee. Returns `{caller: [callees]}`, or `{}` when no
  `traces.csv` ships (e.g. sock-shop).
- `_match_service(trace_name, service_names)` — maps a trace service name to the
  metric service name by longest-prefix match (e.g. `frontendservice` →
  `frontend`). Purely data-driven; no hardcoded names.
- `gabanzhaf()` builds the graph once and passes it via `kwargs["call_graph"]`
  to both the trace-topology builder and the coverage step.
- Deleted: `_load_call_graph`, `tt_callgraph.json`, `ob_callgraph.json`, and all
  dataset-name branching.

### Caching (cost)

The call graph is system-static, so it is cached per dataset root
(`os.path.dirname(dirname(case_dir))`): only the first case of a run parses
`traces.csv`. This matters — `traces.csv` is the large raw-span file. Without
caching, re2-tt took 234 s; with it, 37 s (re2-ob 179 s → 19 s). Never parse
`traces.csv` per case.

### Equivalence verification

The aggregated auto-built tt graph is byte-identical to the old
`tt_callgraph.json` (15 callers, 55 edges, 0 diff). Per-case and first-case
graphs reproduce the curated graph's accuracy exactly on all six fault types
(`.96 / 1.0 / 1.0 / .99 / .79 / .69`). 58 unit tests pass.

## Current pipeline

```
1. Window split: normal vs anomalous window around inject_time.
2. Metric evidence: per-column anomaly score → aggregated service score svc_anom.
3. Candidates: top-k services by svc_anom.
4. Call graph: built from this case's traces.csv (uniform entry point).
5. Coverage: call-graph rule — a quiet upstream root with >=2 loud latency
   victims (the delay/loss propagation signature).
6. Trace topology edges: lagged correlation of trace time-series along call-graph
   edges → callee->caller causal edges.
7. Causal probability matrix = identity diagonal + coverage edges + trace edges.
8. Banzhaf power index over the matrix, weighted by metric anomaly.
9. Rank metric columns by Banzhaf score.
```

When trace information is absent (empty graph), steps 5 and 6 are empty, the
matrix is ~identity, Banzhaf collapses to the metric weights, and ranking
degrades to pure metric service-evidence.

## No-Trace Ablation

`GABANZHAF_DISABLE_TRACE=1` forces an empty call graph (no trace information at
all). Avg@5, full method (ON) vs no-trace (OFF):

| Fault | re2-tt ON→OFF | re2-ob ON→OFF | re2-ss ON→OFF |
| --- | --- | --- | --- |
| CPU | 0.96 → 0.96 | 0.88 → 0.88 | 0.85 → 0.85 |
| MEM | 1.00 → 1.00 | 0.92 → 0.91 | 0.96 → 0.96 |
| DISK | 1.00 → 1.00 | 0.85 → 0.85 | 0.91 → 0.91 |
| SOCKET | 0.99 → 0.97 | 0.85 → 0.85 | 0.87 → 0.87 |
| DELAY | 0.79 → 0.71 | 0.71 → 0.71 | 0.80 → 0.80 |
| LOSS | 0.69 → 0.65 | 0.81 → 0.83 | 0.84 → 0.84 |

## Interpretation

- **re2-tt — trace helps:** DELAY +0.08, LOSS +0.04, SOCKET +0.02 Avg@5. The
  gain is the call-graph coverage rescue of quiet upstream roots. Resource faults
  (CPU/MEM/DISK) are untouched — their local signal is already decisive.
- **re2-ob — trace is neutral-to-negative:** LOSS even drops 0.83 → 0.81. The
  graph is sparse (only 7/12 services have spans, mostly leaves) and ob injects
  on leaf services, so the multi-victim coverage pattern never fires.
- **re2-ss — trace has zero effect:** no `traces.csv` ships, so the graph is
  empty by construction and the ON/OFF columns are identical.

Trace information is valuable only where the system has rich traces and
propagation-style faults (train-ticket); elsewhere it is neutral. Crucially this
is now an honest, data-driven property of one uniform method, not a per-dataset
special case.

## No-Metric Ablation

The symmetric counterpart: drop metric evidence and rank purely by per-service
trace evidence (`GABANZHAF_RANK_MODE=trace_evidence`). Avg@5:

| Fault | re2-tt full → no-metric | re2-ob full → no-metric | re2-ss full → no-metric |
| --- | --- | --- | --- |
| CPU | 0.96 → 0.03 | 0.88 → 0.24 | 0.85 → 0.48 |
| MEM | 1.00 → 0.00 | 0.92 → 0.20 | 0.96 → 0.48 |
| DISK | 1.00 → 0.03 | 0.85 → 0.24 | 0.91 → 0.48 |
| SOCKET | 0.99 → 0.00 | 0.85 → 0.24 | 0.87 → 0.48 |
| DELAY | 0.79 → 0.04 | 0.71 → 0.08 | 0.80 → 0.48 |
| LOSS | 0.69 → 0.00 | 0.81 → 0.24 | 0.84 → 0.48 |

### Interpretation

- **Metric is the backbone.** Removing it collapses accuracy to ~0 on re2-tt and
  ~0.2 on re2-ob. Compared to the small no-trace drop, this shows metric does
  nearly all the localization and trace is only a marginal refiner.
- **re2-tt no-metric ≈ 0 (actively wrong):** per-service trace evidence is trace
  latency/error *magnitude*, which points at high-traffic hubs (e.g.
  ts-ui-dashboard), not the injected root. Trace magnitude alone is
  anti-informative for localization.
- **re2-ss no-metric = 0.48 flat:** sock-shop ships no trace data, so all
  services tie at 0 and ranking degrades to a fixed column-name order — a
  degenerate "no-information" baseline, not a real trace result. It exceeds
  re2-tt's 0.03 only because tt's trace evidence confidently misranks while ss
  merely guesses.

### Summary

The two ablations bracket the method: no-trace costs little (trace is a minor,
tt-only refiner), no-metric is catastrophic (metric carries the method). One
uniform pipeline, with both contributions measured honestly per dataset.
