# MERIT constant-justification sweep (2026-09-30)

Raw output: `artifacts/param-sweep-20260930/raw/<dataset>/<case>.json`. 
Recompute: `PYTHONPATH=. ./py310/bin/python artifacts/param-sweep-20260930/analyze.py --raw artifacts/param-sweep-20260930/raw --json-out artifacts/param-sweep-20260930/analysis.json`.

Protocol: same as `artifacts/formal-rerun-20260930` (full_record_boundary, graph_source=per_case_pre_fault, current merit.py defaults except the swept field). Runner: `script/merit_param_sweep.py` (cached: one public `m.merit()` call per case captures preprocessed inputs, then each configuration re-invokes `_merit_core` directly, mirroring `script/merit_design_audit.py`).

**Reproduction check (criterion #1):** default config vs `artifacts/formal-rerun-20260930` full variant -

| dataset | got k/90 | expected k/90 | match |
|---|---:|---:|---|
| re2-tt | 73/90 | 73/90 | OK |
| re2-ob | 27/90 | 27/90 | OK |
| re2-ss | 35/90 | 35/90 | OK |

AC@1/Avg@5 below are six-fault-type macro averages (equal group sizes of 15, so numerically identical to the raw 90-case average — verified: `ac1_macro*90` is an integer for every cell, see analyze.py). Δ is case-count change vs default on the same 90 cases, TT/OB/SS shown as `AC@1 (ΔTT/ΔOB/ΔSS in cases)`.

**Headline finding (E1/E2, flagged for the author's attention — not a change made here):** for the ~7-11% of columns per dataset where the normal-window IQR is 0 (design doc §1 item #1), the current rule divides by a fixed constant (1) instead of a scale-appropriate quantity. Because `_metric_evidence`'s `mag`/`peak` terms are min-max normalized (`_norm01_arr`) *across all columns of a case*, dividing by an arbitrary constant instead of the column's own natural scale breaks cross-metric comparability: for small-natural-scale columns (e.g. std~0.03) it *understates* the true deviation (denominator 1 is too large), for large-natural-scale columns (e.g. a diskio counter with std~11000) it lets one runaway raw rz swamp the min-max range and silently zero out every other column's score for that case. Both independent E1 alternatives — dividing by std when std>0 instead of 1 (`sigma_normal`), and simply excluding these columns (`exclude_col`) — raise OB macro AC@1 from 0.300 to 0.811/0.800 and SS from 0.389 to 0.833/0.833 (TT roughly flat: -3/+0 cases); that is a ~40-46-case swing on 90. E2's `uncapped` variant (remove the companion cap-at-10 winsorization, a narrower version of the same fix restricted to the doubly-degenerate std<=1e-6 subset) moves the same direction, more modestly (+17 OB, +3 SS) — consistent with fault injections often producing large, genuinely diagnostic jumps in otherwise-frozen counters (error counts, queue depths) that the current cap suppresses toward noise level. Verified directly on one OB case (`checkoutservice_cpu/1`): `redis_diskio` has IQR=0 in the normal window with std=11405, so default `rz=|anomal-median|/1` can reach into the thousands, unbounded, while genuinely diagnostic columns stay near unit scale. This was not anticipated by the original E1/E2 experiment design (which only asked "is there evidence for the current value") — it fell out of actually running the grid. It is evidence *against* the current default, not merely "no evidence for it"; recommend the author treat this as a candidate correctness issue in `_metric_evidence`'s scale handling, not just an unjustified constant.

## E1 - IQR=0 denominator (merit.py:313)

3 rules: default (denominator 1), σ_normal fallback, exclude-column (rz=0).

| config | TT AC@1 | TT Avg@5 | OB AC@1 | OB Avg@5 | SS AC@1 | SS Avg@5 | ΔTT | ΔOB | ΔSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **default** | 0.811 | 0.884 | 0.300 | 0.833 | 0.389 | 0.847 | +0 | +0 | +0 |
| σ_normal fallback | 0.778 | 0.864 | 0.811 | 0.933 | 0.833 | 0.953 | -3 | +46 | +40 |
| exclude column (rz=0) | 0.811 | 0.884 | 0.800 | 0.931 | 0.833 | 0.949 | +0 | +45 | +40 |

## E2 - near-constant truncation cap (merit.py:322, default 10)

| config | TT AC@1 | TT Avg@5 | OB AC@1 | OB Avg@5 | SS AC@1 | SS Avg@5 | ΔTT | ΔOB | ΔSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **default** | 0.811 | 0.884 | 0.300 | 0.833 | 0.389 | 0.847 | +0 | +0 | +0 |
| cap=5 | 0.811 | 0.884 | 0.300 | 0.833 | 0.389 | 0.847 | +0 | +0 | +0 |
| cap=20 | 0.811 | 0.884 | 0.300 | 0.833 | 0.389 | 0.847 | +0 | +0 | +0 |
| cap=50 | 0.811 | 0.884 | 0.300 | 0.833 | 0.389 | 0.847 | +0 | +0 | +0 |
| uncapped | 0.811 | 0.884 | 0.489 | 0.871 | 0.422 | 0.853 | +0 | +17 | +3 |

## E3 - early_window_ratio (default 0.25)

| config | TT AC@1 | TT Avg@5 | OB AC@1 | OB Avg@5 | SS AC@1 | SS Avg@5 | ΔTT | ΔOB | ΔSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **default** | 0.811 | 0.884 | 0.300 | 0.833 | 0.389 | 0.847 | +0 | +0 | +0 |
| 0.10 | 0.822 | 0.896 | 0.522 | 0.858 | 0.622 | 0.891 | +1 | +20 | +21 |
| 0.15 | 0.822 | 0.896 | 0.467 | 0.853 | 0.522 | 0.873 | +1 | +15 | +12 |
| 0.35 | 0.811 | 0.893 | 0.233 | 0.811 | 0.322 | 0.829 | +0 | -6 | -6 |
| 0.50 | 0.767 | 0.873 | 0.178 | 0.798 | 0.178 | 0.800 | -4 | -11 | -19 |

## E4 - trace_topology_min_conf (default 0.3); SS has no traces.csv -> structurally N/A

0.3 rebuild reproduces baseline exactly (sanity check, verified in raw JSON).

| config | TT AC@1 | TT Avg@5 | OB AC@1 | OB Avg@5 | SS AC@1 | SS Avg@5 | ΔTT | ΔOB | ΔSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **default** | 0.811 | 0.884 | 0.300 | 0.833 | 0.389 | 0.847 | +0 | +0 | +0 |
| 0.0 | 0.811 | 0.887 | 0.300 | 0.827 | N/A | N/A | +0 | +0 | N/A |
| 0.2 | 0.811 | 0.887 | 0.300 | 0.827 | N/A | N/A | +0 | +0 | N/A |
| 0.3 | 0.811 | 0.884 | 0.300 | 0.833 | N/A | N/A | +0 | +0 | N/A |
| 0.4 | 0.811 | 0.884 | 0.300 | 0.836 | N/A | N/A | +0 | +0 | N/A |
| 0.5 | 0.822 | 0.884 | 0.311 | 0.836 | N/A | N/A | +1 | +1 | N/A |
| 0.6 | 0.822 | 0.884 | 0.322 | 0.838 | N/A | N/A | +1 | +2 | N/A |

**E4 trace-edge retention rate** (fraction of all candidate call-graph edges whose merged confidence clears the threshold; computed once per case at min_conf=0, then filtered client-side — mathematically identical to rebuilding per threshold, see merit_param_sweep.py docstring):

| threshold | OB retention | TT retention |
|---:|---:|---:|
| 0.0 | 1.000 | 1.000 |
| 0.2 | 0.604 | 0.660 |
| 0.3 | 0.407 | 0.483 |
| 0.4 | 0.276 | 0.351 |
| 0.5 | 0.178 | 0.235 |
| 0.6 | 0.096 | 0.152 |

N1's iid-noise simulation predicted retention near 1.0 at the default threshold 0.3 (mean ρ*≈0.77 → confidence≈0.65 ≫ 0.3). Observed retention at 0.3 is 0.41 (OB) / 0.48 (TT) — real trace pairs are much less correlated than the iid-uniform simulation assumed, so the threshold does exercise real, substantial filtering power (roughly halving the candidate edge set), even though it is not calibrated to any target rate.

## E5 - coverage_lat_floor (default 0.1, used at merit.py:1241 coverage gate AND merit.py:1655 trace-edge filter)

floor_both_* changes both sites; covgate_only_0/tracefilter_only_0 decouple them (patch one site to 0, leave the other at default).

| config | TT AC@1 | TT Avg@5 | OB AC@1 | OB Avg@5 | SS AC@1 | SS Avg@5 | ΔTT | ΔOB | ΔSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **default** | 0.811 | 0.884 | 0.300 | 0.833 | 0.389 | 0.847 | +0 | +0 | +0 |
| both=0 | 0.789 | 0.871 | 0.300 | 0.831 | 0.389 | 0.847 | -2 | +0 | +0 |
| both=0.05 | 0.800 | 0.878 | 0.300 | 0.833 | 0.389 | 0.847 | -1 | +0 | +0 |
| both=0.2 | 0.833 | 0.898 | 0.300 | 0.831 | 0.389 | 0.847 | +2 | +0 | +0 |
| both=0.3 | 0.844 | 0.909 | 0.322 | 0.838 | 0.389 | 0.847 | +3 | +2 | +0 |
| coverage-gate only=0 | 0.800 | 0.876 | 0.300 | 0.833 | 0.389 | 0.847 | -1 | +0 | +0 |
| trace-filter only=0 | 0.800 | 0.880 | 0.300 | 0.831 | 0.389 | 0.847 | -1 | +0 | +0 |

## E6 - victim thresholds theta_up x theta_norm (defaults 0.45, 0.20)

8 of the 9-cell grid (default cell = baseline).

| config | TT AC@1 | TT Avg@5 | OB AC@1 | OB Avg@5 | SS AC@1 | SS Avg@5 | ΔTT | ΔOB | ΔSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **default** | 0.811 | 0.884 | 0.300 | 0.833 | 0.389 | 0.847 | +0 | +0 | +0 |
| up=0.35,norm=0.1 | 0.811 | 0.884 | 0.289 | 0.829 | 0.389 | 0.847 | +0 | -1 | +0 |
| up=0.35,norm=0.2 | 0.811 | 0.884 | 0.300 | 0.833 | 0.389 | 0.847 | +0 | +0 | +0 |
| up=0.35,norm=0.3 | 0.811 | 0.891 | 0.300 | 0.833 | 0.389 | 0.847 | +0 | +0 | +0 |
| up=0.45,norm=0.1 | 0.811 | 0.884 | 0.289 | 0.829 | 0.389 | 0.847 | +0 | -1 | +0 |
| up=0.45,norm=0.3 | 0.811 | 0.891 | 0.300 | 0.833 | 0.389 | 0.847 | +0 | +0 | +0 |
| up=0.55,norm=0.1 | 0.811 | 0.884 | 0.289 | 0.829 | 0.389 | 0.847 | +0 | -1 | +0 |
| up=0.55,norm=0.2 | 0.811 | 0.884 | 0.300 | 0.833 | 0.389 | 0.847 | +0 | +0 | +0 |
| up=0.55,norm=0.3 | 0.811 | 0.891 | 0.300 | 0.833 | 0.389 | 0.847 | +0 | +0 | +0 |

## E6 - uncertain_stop_factor (default 0.5)

| config | TT AC@1 | TT Avg@5 | OB AC@1 | OB Avg@5 | SS AC@1 | SS Avg@5 | ΔTT | ΔOB | ΔSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **default** | 0.811 | 0.884 | 0.300 | 0.833 | 0.389 | 0.847 | +0 | +0 | +0 |
| 0.0 | 0.811 | 0.882 | 0.300 | 0.833 | 0.389 | 0.847 | +0 | +0 | +0 |
| 0.25 | 0.811 | 0.882 | 0.300 | 0.833 | 0.389 | 0.847 | +0 | +0 | +0 |
| 0.75 | 0.811 | 0.887 | 0.300 | 0.833 | 0.389 | 0.847 | +0 | +0 | +0 |
| 1.0 | 0.811 | 0.891 | 0.300 | 0.833 | 0.389 | 0.847 | +0 | +0 | +0 |

## B - persistence threshold rz>k (default 3)

Confirmatory: T4 predicted 3 as smallest integer with <1 expected false exceedance/window under a Gaussian reference.

| config | TT AC@1 | TT Avg@5 | OB AC@1 | OB Avg@5 | SS AC@1 | SS Avg@5 | ΔTT | ΔOB | ΔSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **default** | 0.811 | 0.884 | 0.300 | 0.833 | 0.389 | 0.847 | +0 | +0 | +0 |
| rz>2 | 0.789 | 0.880 | 0.300 | 0.827 | 0.389 | 0.851 | -2 | +0 | +0 |
| rz>4 | 0.822 | 0.880 | 0.300 | 0.833 | 0.389 | 0.849 | +1 | +0 | +0 |
| rz>5 | 0.822 | 0.882 | 0.311 | 0.833 | 0.389 | 0.844 | +1 | +1 | +0 |

## B - coverage_min_loud (default 2)

Confirmatory: T8 predicted n>=2 as the threshold where coverage alone can lift a silent parent above single victims at c=0.7.

| config | TT AC@1 | TT Avg@5 | OB AC@1 | OB Avg@5 | SS AC@1 | SS Avg@5 | ΔTT | ΔOB | ΔSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **default** | 0.811 | 0.884 | 0.300 | 0.833 | 0.389 | 0.847 | +0 | +0 | +0 |
| 1 | 0.600 | 0.702 | 0.144 | 0.758 | 0.389 | 0.847 | -19 | -14 | +0 |
| 3 | 0.778 | 0.871 | 0.300 | 0.833 | 0.389 | 0.847 | -3 | +0 | +0 |

## B - trace_topology_max_lag L (default 3); SS N/A (no traces)

| config | TT AC@1 | TT Avg@5 | OB AC@1 | OB Avg@5 | SS AC@1 | SS Avg@5 | ΔTT | ΔOB | ΔSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **default** | 0.811 | 0.884 | 0.300 | 0.833 | 0.389 | 0.847 | +0 | +0 | +0 |
| L=1 | 0.811 | 0.884 | 0.300 | 0.833 | N/A | N/A | +0 | +0 | N/A |
| L=2 | 0.811 | 0.884 | 0.300 | 0.833 | N/A | N/A | +0 | +0 | N/A |
| L=5 | 0.811 | 0.884 | 0.300 | 0.833 | N/A | N/A | +0 | +0 | N/A |
| L=8 | 0.800 | 0.882 | 0.300 | 0.831 | N/A | N/A | -1 | +0 | N/A |

---

## R - existing (a)-class sensitivity axes rerun on the formal protocol

Old numbers below are from the externally archived `docs/benchmark/sensitivity-sweep-table.md` (main.py 600/600-row window protocol, TT/OB/SS default AC@1 = 0.855/0.368/0.467). New numbers are this sweep (full-record boundary + pre-fault graph, default AC@1 = 0.811/0.300/0.389). **The two are not directly comparable in absolute level — the protocol itself shifts every point — the question R answers is whether the *shape* of the sensitivity (which points are best, how wide the spread is) survives the protocol change.**

### `top_k_svcs` (default 15)

| value | new TT AC@1 | new OB AC@1 | new SS AC@1 | old TT AC@1 | old OB AC@1 | old SS AC@1 |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.811 | 0.322 | 0.389 | 0.855 | 0.390 | 0.467 |
| 12 | 0.811 | 0.300 | 0.389 | 0.855 | 0.368 | 0.467 |
| **15** | 0.811 | 0.300 | 0.389 | 0.855 | 0.368 | 0.467 |
| 20 | 0.811 | 0.300 | 0.389 | 0.855 | 0.368 | 0.467 |
| 25 | 0.811 | 0.300 | 0.389 | 0.855 | 0.368 | 0.467 |

### `exact_candidate_k` (default 12)

| value | new TT AC@1 | new OB AC@1 | new SS AC@1 | old TT AC@1 | old OB AC@1 | old SS AC@1 |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | 0.811 | 0.333 | 0.389 | 0.855 | 0.390 | 0.467 |
| 8 | 0.811 | 0.333 | 0.389 | 0.855 | 0.390 | 0.467 |
| 10 | 0.811 | 0.322 | 0.389 | 0.855 | 0.390 | 0.467 |
| **12** | 0.811 | 0.300 | 0.389 | 0.855 | 0.368 | 0.467 |
| 15 | 0.811 | 0.300 | 0.389 | 0.843 | 0.368 | 0.467 |

### `coverage_conf` (default 0.7)

| value | new TT AC@1 | new OB AC@1 | new SS AC@1 | old TT AC@1 | old OB AC@1 | old SS AC@1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 0.811 | 0.300 | 0.389 | 0.843 | 0.368 | 0.467 |
| 0.6 | 0.811 | 0.300 | 0.389 | 0.843 | 0.368 | 0.467 |
| **0.7** | 0.811 | 0.300 | 0.389 | 0.855 | 0.368 | 0.467 |
| 0.8 | 0.800 | 0.300 | 0.389 | 0.843 | 0.368 | 0.467 |
| 0.9 | 0.789 | 0.300 | 0.389 | 0.822 | 0.368 | 0.467 |

### `coverage_loud_thr` (default 0.4)

| value | new TT AC@1 | new OB AC@1 | new SS AC@1 | old TT AC@1 | old OB AC@1 | old SS AC@1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | 0.811 | 0.278 | 0.389 | 0.843 | 0.333 | 0.467 |
| **0.4** | 0.811 | 0.300 | 0.389 | 0.855 | 0.368 | 0.467 |
| 0.5 | 0.811 | 0.300 | 0.389 | 0.832 | 0.368 | 0.467 |
| 0.6 | 0.778 | 0.300 | 0.389 | 0.833 | 0.368 | 0.467 |

### `victim_floor` (default 0.5)

| value | new TT AC@1 | new OB AC@1 | new SS AC@1 | old TT AC@1 | old OB AC@1 | old SS AC@1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | 0.811 | 0.300 | 0.389 | 0.855 | 0.368 | 0.467 |
| 0.4 | 0.811 | 0.300 | 0.389 | 0.855 | 0.368 | 0.467 |
| **0.5** | 0.811 | 0.300 | 0.389 | 0.855 | 0.368 | 0.467 |
| 0.6 | 0.811 | 0.300 | 0.389 | 0.855 | 0.368 | 0.467 |
| 0.7 | 0.811 | 0.300 | 0.389 | 0.843 | 0.368 | 0.467 |

### `trace_topology_conf` (default 0.85)

| value | new TT AC@1 | new OB AC@1 | new SS AC@1 | old TT AC@1 | old OB AC@1 | old SS AC@1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.7 | 0.811 | 0.322 | N/A | 0.843 | 0.390 | 0.467 |
| 0.8 | 0.811 | 0.311 | N/A | 0.843 | 0.368 | 0.467 |
| **0.85** | 0.811 | 0.300 | 0.389 | 0.855 | 0.368 | 0.467 |
| 0.9 | 0.811 | 0.300 | N/A | 0.843 | 0.357 | 0.467 |
| 0.95 | 0.811 | 0.300 | N/A | 0.832 | 0.357 | 0.467 |

### Blend weights (12 points; old aggregate JSON retained under `evidence/history/legacy-benchmarks/`, original Markdown externally archived)

| param=value | new TT AC@1 | new OB AC@1 | old TT AC@1 | old OB AC@1 |
|---|---:|---:|---:|---:|
| ev_w_mag_0.35 | 0.822 | 0.300 | 0.843 | 0.368 |
| ev_w_mag_0.75 | 0.800 | 0.300 | 0.855 | 0.357 |
| ev_w_persistence_0.15 | 0.778 | 0.289 | 0.845 | 0.345 |
| ev_w_persistence_0.45 | 0.811 | 0.300 | 0.843 | 0.368 |
| early_w_mag_0.5 | 0.811 | 0.278 | 0.855 | 0.357 |
| early_w_mag_0.9 | 0.789 | 0.300 | 0.855 | 0.368 |
| fusion_w_early_0.4 | 0.811 | 0.278 | 0.845 | 0.345 |
| fusion_w_early_0.8 | 0.800 | 0.289 | 0.855 | 0.357 |
| agg_w_p95_0.3 | 0.811 | 0.311 | 0.855 | 0.368 |
| agg_w_p95_0.7 | 0.800 | 0.300 | 0.855 | 0.368 |
| svc_w_dominance_0.15 | 0.778 | 0.278 | 0.843 | 0.333 |
| svc_w_dominance_0.5 | 0.800 | 0.311 | 0.822 | 0.357 |

**Shape check (does the ranking of points survive the protocol change?):**

- `svc_w_dominance`: old showed strict monotone-away-from-default degradation on TT (0.855->0.843->0.822 as it moves 0.30->0.15->0.50-direction) and on OB. New data: default(baseline) TT 0.811 vs 0.15->0.778 and 0.50->0.800 — direction preserved (both sides worse than default), so the qualitative finding survives.

- `coverage_conf`: old finding was TT peaks at 0.7, degrading on both sides, OB flat. New: TT default 0.811 vs 0.5/0.6=0.811 (flat, not peaked!), 0.8=0.800, 0.9=0.789 (still degrades above 0.7, but no longer strictly peaked below). OB flat at 0.300 throughout, consistent. So the OB-flat finding survives; the TT below-0.7-degrades-too finding does NOT survive under the new protocol (0.5/0.6 now tie the default).

- `exact_candidate_k`, `top_k_svcs`: old found OB slightly prefers smaller k (10). New data confirms the same direction (10 gives OB 0.322/0.322 vs default 0.300 for both params) — direction survives.

- `victim_floor`: old found TT flat 0.3-0.6 then drop at 0.7. New: TT is now completely flat 0.811 across all five points including 0.7 — the whole axis is inert under the new protocol (consistent with the design audit's `no_victim` finding that the victim block has zero measured effect here).

- Blend weights: direction of several axes flips in magnitude but qualitative conclusion (differences are within 1-3 cases, i.e. noise-level on n=90) still holds in the new data too.

---

## Constants with no override entry point (would require adding one to merit.py; NOT added per instructions)

None. All 30 constants in the design doc's table 1 either have an existing kwarg (E1-E6 items with kwarg fields, all of R, most of B) or were reached via `transformed()` source-patching without touching the committed file (E1, E2, B-persistence, E5-tracefilter_only). No new override entry point was needed or added.
