# MERIT no-banzhaf ablation — RE2 (ob / ss / tt), fault-type matrix

Date: 2026-08-04 (metric-only ob addendum run 2026-08-05; no-trace / no-metric
rows added 2026-08-05)
Code state: `git rev-parse HEAD` = `b717a3be660837d28d4aeed16005f9a582b7ac37` (branch `merit`,
includes 18dfcda5 "trace重新利用後loss delay顯著提升"). **Working tree at run time also
had an uncommitted change to `RCAEval/e2e/merit.py`** (`git diff --stat` shows 913
insertions / 144 deletions, including the whole `_direct_service_score_sum` function) —
reproducing these numbers requires the same dirty working tree, not just a checkout of
this commit. `git stash`/clean checkout of b717a3be alone will NOT reproduce them.

All 6 runs executed via `run_ablation.sh` (which does `rm -rf output/results && mkdir -p
output/results` before every run, so no stale-result contamination). Raw per-run text
output and the JSON aggregate live at:
- `benchmark_merit_no_banzhaf_ablation_2026-08-04.json` (repo root)
- scratchpad: `ablation_no_banzhaf_20260804/*.txt` (combined log: `no_banzhaf_run.log`)

`no-banzhaf` = `GABANZHAF_RANK_MODE=service_evidence_trace` (bypasses the Banzhaf game;
ranks services by normalized metric+trace evidence sum). Flag read at
`RCAEval/e2e/merit.py:1842-1843`, ranking logic at `:1366-1380`.

2026-08-05 addendum modes (run via `run_ablation.sh --modes no-trace no-metric`,
same commit + same dirty merit.py as above, verified before the run):
- `no-trace` = `GABANZHAF_DISABLE_TRACE=1` — removes trace evidence, keeps the full
  metric + Banzhaf pipeline.
- `no-metric` = `GABANZHAF_DISABLE_METRICS=1` — drops metric values, ranks on trace
  evidence through the Banzhaf pipeline.
Aggregate JSON: `benchmark_merit_no_metric_no_trace_ablation_2026-08-05.json` (repo
root); raw per-run text: scratchpad `ablation_no_metric_no_trace_20260805/*.txt`.
Cross-check: the ob `no-trace` mean AC@1 here (0.380) independently reproduces the
2026-08-04 fresh-reviewer `GABANZHAF_DISABLE_TRACE=1` spot check (~0.380).

## Rerun commands and wall-clock

| dataset | mode | command | wall-clock |
|---|---|---|---:|
| re2-tt | full | `./py310/bin/python main.py --method merit --dataset re2-tt --result-dir output/results` | 321s |
| re2-tt | no-banzhaf | `GABANZHAF_RANK_MODE=service_evidence_trace ./py310/bin/python main.py --method merit --dataset re2-tt --result-dir output/results` | 332s |
| re2-ob | full | `./py310/bin/python main.py --method merit --dataset re2-ob --result-dir output/results` | 163s |
| re2-ob | no-banzhaf | `GABANZHAF_RANK_MODE=service_evidence_trace ./py310/bin/python main.py --method merit --dataset re2-ob --result-dir output/results` | 166s |
| re2-ss | full | `./py310/bin/python main.py --method merit --dataset re2-ss --result-dir output/results` | 14s |
| re2-ss | no-banzhaf | `GABANZHAF_RANK_MODE=service_evidence_trace ./py310/bin/python main.py --method merit --dataset re2-ss --result-dir output/results` | 13s |
| re2-ob | metric-only | `GABANZHAF_RANK_MODE=service_evidence ./py310/bin/python main.py --method merit --dataset re2-ob --result-dir output/results` | 162s |
| re2-tt | no-trace | `GABANZHAF_DISABLE_TRACE=1 ./py310/bin/python main.py --method merit --dataset re2-tt --result-dir output/results` | 42s |
| re2-tt | no-metric | `GABANZHAF_DISABLE_METRICS=1 ./py310/bin/python main.py --method merit --dataset re2-tt --result-dir output/results` | 318s |
| re2-ob | no-trace | `GABANZHAF_DISABLE_TRACE=1 ./py310/bin/python main.py --method merit --dataset re2-ob --result-dir output/results` | 13s |
| re2-ob | no-metric | `GABANZHAF_DISABLE_METRICS=1 ./py310/bin/python main.py --method merit --dataset re2-ob --result-dir output/results` | 164s |
| re2-ss | no-trace | `GABANZHAF_DISABLE_TRACE=1 ./py310/bin/python main.py --method merit --dataset re2-ss --result-dir output/results` | 13s |
| re2-ss | no-metric | `GABANZHAF_DISABLE_METRICS=1 ./py310/bin/python main.py --method merit --dataset re2-ss --result-dir output/results` | 9s |

Each run preceded by `rm -f output/results/*.json` per project hygiene rule.
Total wall-clock for the 6 original runs: **1009s (~16.8 min)**; the metric-only ob
addendum adds another 162s, and the six no-trace / no-metric runs add 559s
(total 1730s across the 13 runs in this document). Note the no-trace runs are much
faster (42s/13s/13s) — trace processing dominates full-run wall-clock.

Note: re2-ss has no `traces.csv` (expected, documented behaviour), so trace evidence is
empty for that dataset and the `full` vs `no-banzhaf` rows below are byte-identical —
the Banzhaf game and the direct metric+trace sum degenerate to the same metric-only
ranking when there is no trace signal to rank over.

## re2-tt

| Method | CPU AC@1 | CPU Avg@5 | MEM AC@1 | MEM Avg@5 | DISK AC@1 | DISK Avg@5 | SOCKET AC@1 | SOCKET Avg@5 | DELAY AC@1 | DELAY Avg@5 | LOSS AC@1 | LOSS Avg@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `full` | 0.80 | 0.96 | 1.00 | 1.00 | 1.00 | 1.00 | 0.93 | 0.95 | 0.80 | 0.80 | 0.60 | 0.75 |
| `no-banzhaf` | 0.87 | 0.97 | 1.00 | 1.00 | 1.00 | 1.00 | 0.93 | 0.99 | 0.47 | 0.64 | 0.60 | 0.72 |
| `no-trace` | 0.80 | 0.96 | 1.00 | 1.00 | 1.00 | 1.00 | 0.87 | 0.97 | 0.60 | 0.71 | 0.53 | 0.65 |
| `no-metric` | 0.33 | 0.51 | 0.40 | 0.47 | 0.33 | 0.59 | 0.33 | 0.61 | 0.00 | 0.09 | 0.27 | 0.48 |

## re2-ob

| Method | CPU AC@1 | CPU Avg@5 | MEM AC@1 | MEM Avg@5 | DISK AC@1 | DISK Avg@5 | SOCKET AC@1 | SOCKET Avg@5 | DELAY AC@1 | DELAY Avg@5 | LOSS AC@1 | LOSS Avg@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `full` | 0.33 | 0.87 | 0.67 | 0.92 | 0.27 | 0.85 | 0.27 | 0.85 | 0.27 | 0.71 | 0.40 | 0.81 |
| `no-banzhaf` | 0.67 | 0.81 | 0.87 | 0.93 | 0.73 | 0.83 | 0.67 | 0.81 | 0.20 | 0.47 | 0.87 | 0.93 |
| `metric-only` | 0.40 | 0.88 | 0.67 | 0.91 | 0.27 | 0.85 | 0.27 | 0.85 | 0.27 | 0.71 | 0.40 | 0.83 |
| `no-trace` | 0.40 | 0.88 | 0.67 | 0.91 | 0.27 | 0.85 | 0.27 | 0.85 | 0.27 | 0.71 | 0.40 | 0.83 |
| `no-metric` | 0.40 | 0.51 | 0.47 | 0.63 | 0.40 | 0.53 | 0.40 | 0.65 | 0.20 | 0.32 | 0.33 | 0.75 |

## re2-ss

| Method | CPU AC@1 | CPU Avg@5 | MEM AC@1 | MEM Avg@5 | DISK AC@1 | DISK Avg@5 | SOCKET AC@1 | SOCKET Avg@5 | DELAY AC@1 | DELAY Avg@5 | LOSS AC@1 | LOSS Avg@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `full` | 0.27 | 0.85 | 0.80 | 0.96 | 0.53 | 0.91 | 0.40 | 0.87 | 0.47 | 0.80 | 0.33 | 0.84 |
| `no-banzhaf` | 0.27 | 0.85 | 0.80 | 0.96 | 0.53 | 0.91 | 0.40 | 0.87 | 0.47 | 0.80 | 0.33 | 0.84 |
| `no-trace` | 0.27 | 0.85 | 0.80 | 0.96 | 0.53 | 0.91 | 0.40 | 0.87 | 0.47 | 0.80 | 0.33 | 0.84 |
| `no-metric` | 0.20 | 0.48 | 0.20 | 0.48 | 0.20 | 0.48 | 0.20 | 0.48 | 0.20 | 0.48 | 0.20 | 0.48 |

## Mean across the 6 fault types

| mode | tt AC@1 | tt Avg@5 | ob AC@1 | ob Avg@5 | ss AC@1 | ss Avg@5 |
|---|---:|---:|---:|---:|---:|---:|
| `full` | 0.855 | 0.910 | 0.368 | 0.835 | 0.467 | 0.872 |
| `no-banzhaf` | 0.812 | 0.887 | 0.668 | 0.797 | 0.467 | 0.872 |
| `metric-only` | — | — | 0.380 | 0.838 | — | — |
| `no-trace` | 0.800 | 0.882 | 0.380 | 0.838 | 0.467 | 0.872 |
| `no-metric` | 0.277 | 0.458 | 0.367 | 0.565 | 0.200 | 0.480 |

(metric-only was run only for re2-ob, to disambiguate the ob finding below; tt/ss don't
need it — see reasoning. no-trace / no-metric added 2026-08-05.)

Degeneracy notes for the 2026-08-05 rows:
- **ss `no-trace` ≡ `full`** (all 12 cells identical) — expected: re2-ss has no
  `traces.csv`, so disabling trace removes nothing.
- **ss `no-metric` is a flat constant** (every fault type 0.20 / 0.48) — with no
  metrics and no traces there is no evidence left; the ranking degenerates to a fixed
  order. This row measures the fallback, not the method.
- **ob `no-trace` ≡ ob `metric-only`** (all 12 cells identical at this rounding) —
  running the Banzhaf pipeline on metric-only evidence reproduces the direct metric
  ranking on ob, consistent with the known Banzhaf-degeneracy-on-sparse-matrix
  behaviour; the two modes differ in mechanism but not outcome here.

## Cross-dataset finding — Banzhaf's real contribution is DELAY, and the ob "no-banzhaf win" is a trace-utilization gap, not evidence Banzhaf itself hurts

The hypothesis in scope for this ablation ("Banzhaf 主要貢獻在 delay/loss 這類緩慢傳播的
fault") holds only for **DELAY**, on both tt and ob (full beats no-banzhaf: tt 0.80 vs
0.47 AC@1; ob 0.27 vs 0.20 AC@1). It does not extend to LOSS: on tt, LOSS AC@1 is now
tied at 0.60 = 0.60 for `full` and `no-banzhaf` (the earlier 0.53 in the pre-18dfcda5
table `merit-ablation-re2-tt-matrix-table.md` is stale), so there is no LOSS gain to attribute to
Banzhaf on tt either. **The Banzhaf-specific, reproducible gain in this whole ablation is
DELAY only** (tt and ob); nothing else in this document should be read as "Banzhaf helps
X".

The initial read of re2-ob — "Banzhaf mechanism is net-negative overall" (mean AC@1 0.368
for `full` vs 0.668 for `no-banzhaf`) — was an over-attribution. A fresh-reviewer check
ran `full` with `GABANZHAF_DISABLE_TRACE=1` on ob and got mean AC@1 ≈ 0.380, essentially
identical to plain `full` (0.368). The `metric-only` row added above (`GABANZHAF_RANK_MODE
=service_evidence`, mean AC@1 0.380) confirms the same thing independently: **trace
evidence is inert inside the Banzhaf/coverage path on ob**. Three-way comparison, ob mean
AC@1:

- `metric-only` (0.380) ≈ `full`/Banzhaf (0.368) — trace makes essentially no difference
  once routed through the Banzhaf/coverage machinery.
- `full`/Banzhaf (0.368) ≪ `no-banzhaf` direct metric+trace sum (0.668) — trace makes a
  large difference when summed directly, bypassing Banzhaf/coverage.

So the gap is not "Banzhaf hurts ob" — it is that **the Banzhaf/coverage stage fails to
propagate trace evidence into the ranking on ob**, while the direct-sum path lets that
same trace evidence through unfiltered. The fix target is the Banzhaf/coverage stage's
handling of trace evidence on ob, not removing Banzhaf.

**Implication for the paper claim**: state "Banzhaf's measured, reproducible benefit in
this ablation is DELAY (tt, ob)." Do not claim a Banzhaf/LOSS benefit (tt is now tied,
0.53 was stale) and do not claim "Banzhaf is net-negative on ob" — the correct framing is
that Banzhaf/coverage on ob is not yet using trace evidence that the direct-sum path can
use, which is a mechanism gap, not evidence that Banzhaf itself is harmful.

## 2026-08-05 addendum finding — modality ablation supports the "Banzhaf for slowly-propagating faults" story on DELAY only

Hypothesis under test: "Banzhaf 是用來處理緩慢變化/緩慢傳播的異常（DELAY/LOSS），
突變型資源故障靠 metric 證據就夠了。" The four-way tt comparison on DELAY AC@1:

- `full` 0.80 → `no-trace` 0.60 (−0.20) → `no-banzhaf` 0.47 (−0.33) → `no-metric` 0.00.

Removing the Banzhaf attribution costs DELAY more than removing the trace modality
itself, and `no-metric` collapses entirely — DELAY needs all three components, with
Banzhaf the single most expensive removal. By contrast, for the abrupt resource faults
(CPU/MEM/DISK/SOCKET) on tt, `no-trace` ≈ `full` (metric evidence alone suffices) and
`no-banzhaf` is equal or slightly better (CPU 0.87 vs 0.80) — Banzhaf contributes
nothing there. This is the clean supporting evidence for the hypothesis, **but only
for DELAY**:

- **LOSS does not fit the story**: tt LOSS AC@1 is `full` 0.60 = `no-banzhaf` 0.60 >
  `no-trace` 0.53 — trace evidence helps LOSS, but the Banzhaf attribution adds nothing
  over the direct sum. Write the claim as "slowly-propagating DELAY faults", not
  "DELAY/LOSS".
- **ob is weak support only**: DELAY `full` 0.27 vs `no-banzhaf` 0.20 vs `no-trace`
  0.27 — a one-case gap, and trace is inert inside the Banzhaf path on ob (see the
  section above), so ob cannot carry the argument on its own.
- **`no-metric` collapses everywhere** (tt 0.277, ob 0.367, ss degenerate 0.200 mean
  AC@1) — metric evidence is the load-bearing modality; trace + Banzhaf refine it.
  Frame Banzhaf as re-attributing metric+trace evidence, not as an alternative
  evidence source.
