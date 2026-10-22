# MERIT method definitions and closed-form derivation

This document is the source of truth for the paper's problem definition,
notation, and complexity claims. It describes the implementation jointly
defined by the seven `RCAEval/e2e/merit*.py` modules listed in
[`merit-ablation.md`](merit-ablation.md); it does not introduce a second algorithm.

## Problem and assumptions

For one diagnosed incident, the input is a multivariate metric time series
`X`, an anomaly-detection timestamp `t_a`, and, when available, distributed
trace records from the same system. The output is an ordered list of
service-metric indicators. Service-level evaluation removes the metric suffix
and duplicates from that list.

The retrospective evaluation assumes:

1. one injected root-cause service per case;
2. `t_a` is supplied by the benchmark (delay sensitivity perturbs it by the
   pre-registered `tdelta` values);
3. telemetry clocks are sufficiently aligned for pre/post and multi-lag
   comparisons;
4. metric and trace service names can be aligned by exact or data-driven prefix
   matching; and
5. each case uses only its own spans strictly before `t_a` to build the request
   graph; a case without usable pre-fault spans has an explicitly empty graph.
   Cache keys include dataset, case, boundary, and graph source, so graph state
   is never shared across cases.

MERIT is training-free: it does not fit a supervised or reusable predictive
model. It does have fixed unsupervised hyperparameters, shared by all three
systems.

## Evidence, directions, and candidate truncation

Metric evidence retains every metric row in the complete case record and splits
it at `t_a`. It robustly standardizes every post-`t_a` row against all available
pre-`t_a` rows using median/IQR statistics, then combines magnitude,
persistence, peak, and early/full-post behavior. Unequal pre/post lengths are
intentional. Metric scores are aggregated into a service anomaly weight
`w_j >= 0`.

All game edges use the direction "candidate cause explains affected target."
For a request edge `caller -> callee`, a fault in the callee can affect the
caller-visible latency, so dynamic propagation evidence is stored as
`callee -> caller`. Coverage rescue retains the request direction
`caller -> callee`: under its separately gated signature, a quiet caller that
has at least two loud latency callees is allowed to explain those callees.
Thus dynamic trace edges and coverage edges can point in opposite directions
for the same request pair. The call graph, dynamic propagation graph, and
coverage explanation edges must not be described as if they shared one edge
semantics.

Let `C` be the anomaly-ranked candidate pool and let `K_0` be its frozen base
cutoff. The game uses those base candidates plus any separately qualified
coverage parents, so the observed player count `K` can exceed `K_0`. Candidate
construction occurs before attribution and therefore sets an
upper bound on achievable service recall: if the true service is absent from
the game, no attribution rule can recover it. Candidate recall is computed only
by the offline evaluator and is reported separately, so a candidate-selection
gain is not credited to Banzhaf attribution.

## 零 IQR 分母政策

狀態：可選 kwarg，預設行為不變；討論與追蹤見 [Issue #2](https://github.com/mizulun/RCAEeval/issues/2)。

**問題。** robust z 以 normal 期（`t_a` 之前）的中位數與 IQR 標準化每一列 post 資料：`rz = |x - median| / IQR`。若某 metric 欄在 normal 期 `q75 - q25 = 0`，原實作以 1 作分母。IQR 為 0 不代表欄位恆定：稀疏欄（例如 90% 為 0、10% 為正值）同樣 IQR 為 0，此時分母 1 使 raw 偏差取決於該 metric 的量測單位。

**定義。** `metric_zero_iqr_policy` 只依 normal 期統計量（`q75 - q25 <= 0`）決定作用欄位，不使用資料集、服務、故障類型或任何標籤：

| 政策 | IQR > 0 的欄 | IQR = 0 的欄 |
|---|---|---|
| `unit`（預設） | 分母 = IQR | 分母 = 1（歷史行為） |
| `std` | 分母 = IQR | normal 期母體標準差 `np.nanstd(normal, axis=0)`（ddof=0）為正時用之，否則 1 |
| `exclude` | 分母 = IQR | 分母同 `unit`；在既有 near-constant cap 之後把該欄 `rz` 設為 0 |

既有順序不變：`nan_to_num` → near-constant cap（IQR = 0 且 std `<= 1e-6` 的欄截在 10）→ 僅 `exclude` 的零化 → magnitude／persistence／peak 與 early/full 摘要。`exclude` 是零證據補值，不是刪除欄位，也不是正式棄權：欄位仍以 0 出現在 metric evidence、min-max 正規化與 service 聚合中。`scorer='cusum'` 只支援 `unit`；非 `unit` 搭配 `cusum`，或未知值（含非字串），都會丟出 `ValueError`，不會靜默退回預設。

**用法。**

```python
from RCAEval.e2e.merit import merit

out = merit(data, inject_time=inject_time,
            metric_zero_iqr_policy="std",  # 或 "exclude"；省略即 "unit"
            return_diagnostics=True)
out["diagnostics"]["configuration"]["metric_zero_iqr_policy"]  # "std"
```

**raw 與最終證據的區別。** IQR > 0 的欄，其 raw `rz`（例如每欄 p95）在三種政策下相同；但最終 metric evidence 經跨欄 min-max 正規化，所以零 IQR 欄的值改變時，正 IQR 欄的最終分數、service 證據與排名仍可能改變。

**相容性。** 預設 `unit`；不傳此 kwarg 時分數與排名不變。`unit` 路徑保留原本的分母程式行，供以 `inspect.getsource` 做文字替換的研究腳本使用。near-constant cap 與 `1e-6` 門檻、各組權重、正規化、service 聚合、candidate、coverage game 與 trace 處理皆未修改；diagnostics 只新增 `configuration.metric_zero_iqr_policy`。

**風險與限制。**

- IQR 為 0 可能來自稀疏而非恆定的欄位；三種政策對同一組欄位作不同假設，沒有一種在所有資料上都正確。
- `std` 對 normal 期離群值敏感，少數尖峰即可決定分母；std 很小但高於 `1e-6` 時，`rz` 可被大幅放大且不受 cap 限制。IQR 與 std 皆為 0 時仍用分母 1，而固定 cap（10）與 `1e-6` 門檻都是絕對值，因此仍依 metric 單位而定。
- `exclude` 會丟失「normal 期平坦、注入後出現新偏移」以及稀疏欄的訊號；這些欄仍以 0 計入 service 聚合（p95／top-3／max），而非從聚合中移除。
- 這些政策沒有一般性的準確率或因果保證；不得在看到結果後才選定政策作為 headline（見 [CONTRIBUTING](../CONTRIBUTING.md) 的結果回報規則）。

**未完成事項。** 逐系統 × 故障類型的 AC@1／Avg@5 配對比較、結果解讀與接受判準仍待完成。既有的初步表格不等同已完成的詳細分析；完成前不改預設，也不據以宣稱任一政策較佳。

## Directed coverage-strength game

Let `\mathcal P = {1, ..., K}` be candidate players and
`\mathcal T = {1, ..., N}` be target
services. For candidate `i` and target `j`, `a_ij in [0, 1]` is a bounded,
directed evidence strength saying how strongly `i` covers `j` in the designed
game. It is not a calibrated probability or structural-causal effect. The
implemented matrix combines:

- diagonal self-coverage;
- call-graph coverage rescue;
- time-series-derived trace propagation; and
- victim demotion applied to the self-coverage of a likely downstream victim.

For coalition `S subseteq \mathcal P`, the noisy-OR-inspired coverage transform is

```text
q_j(S) = 1 - product_{l in S} (1 - a_lj).
```

The characteristic function is weighted explained anomaly:

```text
v(S) = sum_{j=1}^N w_j q_j(S).
```

This is monotone for nonnegative `w_j` and bounded above by `sum_j w_j`.
It also has diminishing marginal coverage: once a target is well explained by
other players, another player's marginal contribution to that target shrinks.

MERIT uses the unnormalized Banzhaf value, i.e., the uniformly averaged
marginal contribution over coalitions not containing player `i`:

```text
beta_i = 2^{-(K-1)} sum_{S subseteq \mathcal P \\ {i}}
         [v(S union {i}) - v(S)].
```

"Unnormalized" here means that the values are not rescaled to sum to
`v(\mathcal P)`.
The final ranking code rescales positive service scores by their maximum only
for ranking; that is not a normalized Banzhaf index.

## Closed-form lemma

**Lemma.** For the weighted noisy-OR game above,

```text
beta_i = sum_{j=1}^N w_j a_ij
         product_{l in \mathcal P \\ {i}} (1 - a_lj / 2).
```

**Proof.** For a fixed target `j`, adding `i` to coalition `S` increases its
coverage by

```text
q_j(S union {i}) - q_j(S)
  = a_ij product_{l in S} (1 - a_lj).
```

Uniformly averaging over all subsets of `\mathcal P \\ {i}` is equivalent to including
each other player independently with probability one half. Independence gives

```text
E_S[product_{l in S} (1 - a_lj)]
  = product_{l != i} [(1/2) * 1 + (1/2) * (1 - a_lj)]
  = product_{l != i} (1 - a_lj / 2).
```

Multiplying by `w_j a_ij`, summing over targets, and using linearity of
expectation proves the expression. The brute-force validator enumerates the
same `2^(K-1)` coalitions per player and must agree within the frozen absolute
tolerance `1e-9` for every tested `K <= 15`. QED.

## Complexity

The closed form constructs `1 - a/2`, one product per target, and one weighted
sum per player. Its time complexity is `O(KN)`. The vectorized implementation
stores the coverage-strength matrix and a `K x N` product-without-player
temporary,
so its implemented space complexity is `O(KN)`; an algebraically equivalent
streaming implementation could use `O(N)` auxiliary space.

Direct Banzhaf enumeration evaluates `K * 2^(K-1)` player-coalition marginal
contributions. With cached or incrementally maintained coalition coverage,
this is `O(K 2^K N)` time and `O(N)` auxiliary space in addition to the input.
The deliberately simple validation implementation recomputes noisy-OR products
for each coalition and can incur an additional factor up to `K`; empirical
runtime claims refer to that measured implementation and never replace the
algorithmic bound with a projection.

The individual-attribution comparator uses the same players, targets, weights,
coverage-strength matrix, coverage, and victim factors, but assigns singleton
weighted coverage `sum_j w_j a_ij`. It is a controlled attribution replacement,
not a claim that Banzhaf is the only possible game-theoretic score.
