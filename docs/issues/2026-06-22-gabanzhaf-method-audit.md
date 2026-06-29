# GA-Banzhaf 方法流程、公式與邏輯審計

**日期：** 2026-06-22  
**正式方法：** `RCAEval/e2e/gabanzhaf.py`  
**歷史 trace 對照：** `RCAEval/e2e/banzhaf_with_trace.py`

## 1. 正式方法資料流

1. 依 `inject_time` 將輸入切成正常與異常視窗，裁成相同長度。
2. 將兩個視窗串接後執行共同 preprocessing，再切回正常與異常矩陣。
3. 對每個 metric 計算 robust anomaly evidence。
4. 將 metric evidence 聚合為 service evidence。
5. 取 service evidence 前 15 名作 seeds，並在擴大的 service pool 上建立 lag-1 topology。
6. 將 seeds 與其 lag upstream 合併，再依 service evidence 截成最多 `K=12` 個 game candidates。
7. 建立 candidate 對所有 service 的 causal explanation probability matrix。
8. 使用 noisy-OR payoff 的 closed-form exact Banzhaf 計算 service power。
9. 正式 baseline 依 service Banzhaf score 排序；同 service metrics 以欄位名稱排序。

## 2. 公式

### Metric evidence

正常視窗中位數與 IQR：

```text
rz_m(t) = |x_m(t) - median_normal(m)| / IQR_normal(m)
```

正常視窗完全固定時，`rz` 截斷至 10。其餘 evidence：

```text
magnitude_m   = norm(P95(rz_m))
persistence_m = norm(mean(rz_m > 3))
peak_m        = norm(max(rz_m))

full_m = norm(
    0.55 * magnitude_m
  + 0.30 * persistence_m
  + 0.15 * peak_m
)

early_m = norm(
    0.70 * P95(rz_m in early window)
  + 0.30 * max(rz_m in early window)
)

E_m = norm(0.60 * early_m + 0.40 * full_m)
```

### Service evidence

對 service `s` 的 metrics：

```text
aggregate_s =
    0.50 * P95(E_m)
  + 0.30 * mean(top3(E_m))
  + 0.20 * max(E_m)
```

每種 metric type 的最高 service 再取得相對第二名的 margin bonus：

```text
A_s = norm(0.70 * aggregate_s + 0.30 * dominance_s)
```

### Lag-1 topology

service time series 目前定義為該 service 所有原始 metric 的逐時刻最大值：

```text
z_s(t) = max_{m belongs to s} x_m(t)
C_ij   = corr(z_i(t), z_j(t+1))
w_i→j  = max(0, C_ij - C_ji)
```

只保留 `w > 0.08`，每個節點最多保留三個 incoming parents。

### Causal explanation probability

candidate 對自己的 explanation probability 固定為 1。對 topology edge：

```text
P_ij = 1 - exp(-lambda * w_i→j * sqrt(A_i * A_j))
lambda = 1.5
```

### Noisy-OR payoff 與 Banzhaf

coalition `S` 對 service `j` 的 coverage：

```text
coverage_j(S) = 1 - product_{i in S}(1 - P_ij)
v(S) = sum_j A_j * coverage_j(S)
```

Banzhaf：

```text
beta_i = average_{S subset of N without i} [v(S union {i}) - v(S)]
```

noisy-OR game 的 exact closed form：

```text
beta_i =
  sum_j A_j * P_ij
  * product_{l != i}(1 - P_lj / 2)
```

因此 closed-form 與 coalition enumeration 計算的是同一個 game；這是 equivalence/efficiency study，不是兩種不同方法的 accuracy ablation。

## 3. 已確認合理的部分

- Metric evidence 僅使用正常/異常視窗，不使用 benchmark 標註。
- Selective Winsorization 避免完全固定 baseline 因分母設為 1 而產生無界 evidence。
- Closed-form Banzhaf 公式與 brute-force enumeration 已有單元測試驗證至浮點誤差範圍。
- Candidate、service 與 probability matrix 都有重複名稱檢查。
- 所有排序 tie 都有 deterministic name ordering，可重現。

## 4. 高風險邏輯與待驗證假設

### 4.1 Lag topology 混合不同量綱

`z_s(t)` 直接取 CPU、memory、latency、disk、workload 等原始 metric 的最大值。除 memory 轉為 MB 外，各 metric 未先做 column-wise scaling。因此最大量綱 metric 可能長期支配 service series，lag edge 不一定代表綜合 service anomaly。

**判定：需要後續修正。** 可比較「每 metric robust scaling 後再 max」或「使用 metric evidence 加權 series」，但不得在本輪 ablation 中修改 baseline。

### 4.2 Normal/anomalous 串接可能產生共同 step-change edge

lag correlation 使用 normal 與 anomalous 串接序列。注入邊界只有一個跨視窗 pair，但多個 service 在異常視窗共同升高時，整段 regime shift 仍可能造成高相關，且非真正 propagation。

**本輪驗證：** no-lag-1、edge 數量、candidate 變化與 paired ranking flip。  
**後續驗證：** 分別在 normal、anomalous window 估 edge，或排除邊界附近樣本。

### 4.3 Top-15 seeds 與 K=12 截斷削弱 topology rescue

topology upstream 先加入 candidate pool，之後仍完全依 service evidence 排序並截成 K。若 upstream evidence 不在前 K，edge 不會進入 game。當 service 數不超過 15 時，trace 或 topology 也無法增加新的 seed service。

**本輪 diagnostics：** `candidate_pool`、`game_candidates`、root candidate recall、trace added seeds。  
**判定方式：** 有 edge 但 candidate set 不變時，不能把 accuracy 不變解讀成 topology 無資訊。

### 4.4 Self-explanation 可能使 Banzhaf 接近 evidence ranking

每個 candidate 的 `P_ii=1`，而大部分非對角 probability 通常較小。若 matrix 接近 identity：

```text
beta_i = A_i
```

此時 Banzhaf 只是 service evidence 的單調轉換。`evidence_service` 與 full method 的 paired top-1/top-5 一致率可直接量化這個問題。

### 4.5 最終 metric identity 使用字典序

正式 baseline 只依 service Banzhaf 排序，同 service 內不使用原始 metric evidence，而以欄位名稱排序。service-level 評估仍合理，但 metric-level AC@k 會受到命名順序影響。

**判定：方法限制。** 本輪 `evidence_service` 使用 metric evidence 作同 service tie-break；若其 metric-level 指標提升且 service-level 不下降，應在後續版本調整正式 ranking。

### 4.6 全部 Banzhaf score 相等時忽略 evidence

`_rank_metrics()` 在 Banzhaf 空集合或所有值相等時，把所有 service score 設為 1，最後全域依欄位名稱排序。此行為可重現，但不是 evidence-based ranking。

**判定：需後續修正。** 本輪保持 baseline 不變，並用 evidence-only variants 顯示影響。

### 4.7 K 對 closed-form runtime 的理論影響有限

closed-form 複雜度約為 `O(KN)`，沒有 `2^K`。因此 K sensitivity 的主要意義是 candidate recall 與 ranking，而不是大幅加速。只有 brute-force equivalence run 會呈指數成本。

**判定：原先將 K 描述成 closed-form 主要 runtime 控制並不精確。**

## 5. Trace 流程為何容易與 metrics 重複

legacy trace score 由 error rate 與 latency ratio形成。正式 metric 集合本身已含 error、latency、workload 等 service metrics，因此兩者預期高度相關。

trace score 的使用位置有限：

1. 將 trace top services 加入 seeds。
2. 在 metric evidence、residual、trace 三來源 reciprocal-rank fusion 中影響 top-K candidates。

trace score沒有直接成為 noisy-OR anomaly weight，也沒有直接進入最終 metric score。若 metric seeds 已覆蓋 trace services，trace score不會增加資訊。

call graph 的限制：

1. caller 與 callee 都必須先位於 metric-derived pool。
2. caller/callee metric anomaly 必須通過固定門檻。
3. 與 lag edge 重複的方向不再加入。
4. edge source 必須通過 exact K 截斷才會進入 causal matrix。
5. call graph 使用固定 confidence 0.5，trace 本身沒有提供 case-specific propagation strength。

residual score只參與候選融合，不作 anomaly weight，也不直接進入最終 ranking。因此 residual 改變後若 top-K candidates 不變，輸出必然不變。

### 完整 RE2 受控實驗

完整結果見 `docs/benchmark/gabanzhaf-ablation-and-trace.md`。每個
method × dataset 以單 worker 重跑三次，runtime 報中位數。

- 關閉 trace scores 後，270 個 case 的 top-1 與 ordered top-5 全部不變。
- 關閉 residual suppression 後，270 個 case 的 top-1 與 ordered top-5 全部不變。
- 關閉 call graph 後，只有 RE2-OB 一個 case 的 top-1 correctness 下降；沒有 paired win。
- 關閉全部 trace 後，accuracy 與 no-call-graph 相同；RE2-OB wall time
  從 11.3 秒降至 7.6 秒，RE2-TT 從 35.4 秒降至 26.1 秒。
- RE2-OB 的平均 metric/trace top-K overlap 為 1.0，trace 沒有新增 seeds。
- RE2-TT 的平均 overlap 為 0.607，trace 平均新增約 9.96 個 seeds，
  但關閉 trace scores 後最終排名仍完全相同。
- RE2-TT 關閉 trace scores 會使 candidate set 相對 full trace 的平均
  Jaccard 降至約 0.70，但輸出排名不變。這表示 trace 改變了候選組成，
  卻沒有產生足以超越 metric/Banzhaf 主訊號的 power。

因此「trace 與 metrics 重複」只完整解釋 RE2-OB；RE2-TT 更精確的結論是：
trace 提供不同候選資訊，但該資訊在 causal matrix、anomaly weights 與最終
ranking 中影響不足。RE2-SS 的 trace diagnostics 為零，需另行區分
「trace 值為零、service 名稱無法對齊、或資料解析路徑未產生 evidence」。

## 6. 冗餘與維護性

- `banzhaf_with_trace.py` 超過 1,200 行，同時包含 trace I/O、feature extraction、topology、candidate selection、game 與 ranking，責任過多。
- `_extract_trace_features()` 對多種讀檔錯誤使用廣泛 `except Exception` 並回傳空 evidence，會讓 trace 缺失與解析失敗難以區分。
- preaggregated trace score 與 raw span score 是兩條不同資料路徑，但輸出共用同一介面；報告需記錄實際使用來源。
- `_noisy_or_coverage()` 與 brute-force Banzhaf 不在正式 closed-form 路徑，只應保留作 equivalence 測試。
- `top_k_svcs=15`、lag threshold 0.08、max incoming edges 3 等設定散落於函式預設值，尚未集中至 config。

## 7. 實驗命令

Component 與 K sensitivity：

```bash
py310/bin/python run123.py \
  --methods gabanzhaf gabanzhaf_no_lag_1 \
    gabanzhaf_evidence_service gabanzhaf_evidence_metric \
    gabanzhaf_k6 gabanzhaf_k8 gabanzhaf_k10 gabanzhaf_k12 gabanzhaf_k16 \
  --datasets re2-ob re2-ss re2-tt \
  --workers 1 --repetitions 3 --diagnostics \
  --output benchmark_gabanzhaf_component_ablation.csv
```

Trace 受控比較：

```bash
py310/bin/python run123.py \
  --methods with_trace_full with_trace_no_scores \
    with_trace_no_call_graph with_trace_no_trace with_trace_no_residual \
  --datasets re2-ob re2-ss re2-tt \
  --workers 1 --repetitions 3 --diagnostics \
  --output benchmark_gabanzhaf_trace_controlled.csv
```

報告：

```bash
py310/bin/python script/analyze_gabanzhaf_ablation.py \
  --benchmark benchmark_gabanzhaf_component_ablation.json \
  --baseline gabanzhaf \
  --variants gabanzhaf_no_lag_1 gabanzhaf_evidence_service \
    gabanzhaf_evidence_metric gabanzhaf_k6 gabanzhaf_k8 gabanzhaf_k10 \
    gabanzhaf_k12 gabanzhaf_k16 with_trace_full with_trace_no_scores \
    with_trace_no_call_graph with_trace_no_trace with_trace_no_residual \
  --output docs/benchmark/gabanzhaf-ablation-and-trace.md
```
