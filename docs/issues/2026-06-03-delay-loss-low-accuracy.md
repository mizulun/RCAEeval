# Issue: DELAY / LOSS 的 AC@1 與 Avg@5 準確率偏低

**日期：** 2026-06-03 15:15  
**狀態：** 未解決  
**影響範圍：** `gabanzhaf.py` — Step 3 metric evidence、Step 4 topology、Step 7 residual  

---

## 症狀

當前 benchmark 結果（`benchmark_results.json`，2026-06-03）：

| Dataset  | AC@1-DELAY | AC@3-DELAY | AC@5-DELAY | Avg@5-DELAY | AC@1-LOSS | AC@3-LOSS | AC@5-LOSS | Avg@5-LOSS |
|----------|-----------|-----------|-----------|------------|----------|----------|----------|-----------|
| re2-ob   | **0.00**  | 0.73      | 0.93      | 0.65       | **0.53** | 0.87     | 1.00     | 0.84      |
| re2-ss   | **0.07**  | 0.93      | 0.93      | 0.72       | **0.33** | 1.00     | 1.00     | 0.83      |
| re2-tt   | **0.60**  | 0.73      | 0.80      | 0.73       | **0.53** | 0.67     | 0.73     | 0.64      |

對照同期正常故障類型（re2-tt 為例）：

| Fault  | AC@1 | AC@5 | Avg@5 |
|--------|------|------|-------|
| CPU    | 0.73 | 1.00 | 0.95  |
| MEM    | 1.00 | 1.00 | 1.00  |
| DISK   | 1.00 | 1.00 | 1.00  |
| SOCKET | 0.80 | 1.00 | 0.96  |
| DELAY  | 0.60 | 0.80 | 0.73  |
| LOSS   | 0.53 | 0.73 | 0.64  |

**關鍵觀察**：DELAY 的根因在 re2-ob 有 93% 的案例出現在 Top-5，但 AC@1 是 0.00——根因存在於候選清單中卻被 1–2 個受害者壓在前面。這不是偵測問題，是**排名問題**。

---

## 根本原因假設

**緩慢累積的受害者（slowly drifting victims）被誤判為根因。**

網路層故障（DELAY/LOSS）注入後：
- 根因服務 R：網路指標立即尖峰（onset 快，但持續時間可能集中）
- 受害者 V1, V2, ...：上游 R 異常後，回應時間緩慢累積上升，長時間維持高於基線的狀態

---

## 瓶頸定位（三層連鎖失效）

### 瓶頸 1（主因）：Step 3 `_metric_evidence` — persistence 項偏袒受害者

**程式位置**：`gabanzhaf.py:474`

```python
raw = 0.55 * mag + 0.30 * persistence + 0.15 * peak
# persistence = mean(rz > 3.0, axis=0)
```

- 受害者指標長時間漂移，大量時間點跨越 3σ → `persistence` 高
- 根因指標快速尖峰但持續時間可能較短 → `persistence` 未必最高
- persistence 佔 30% 權重，足以讓受害者總分超越根因

### 瓶頸 2（次因）：Step 4 `_build_topology` — lag-1 非對稱在網路故障下失靈

**程式位置**：`gabanzhaf.py:655`

```python
confidence = np.maximum(0., C - C.T)  # 非對稱性
```

- CPU/MEM：資源耗盡漸進傳播，根因 leads 受害者數個時間步 → lag-1 非對稱清晰
- DELAY/LOSS：網路層傳播幾乎瞬時（lag-0），C[R,V] ≈ C[V,R] → 非對稱趨近 0 → **拓樸邊方向錯誤或缺失**
- re2-ob/re2-ss 有複雜的扇出圖，多受害者同時響應，進一步稀釋非對稱信號

### 瓶頸 3（連帶）：Step 7 `_residual_scores` — 因拓樸方向錯誤而失效

**程式位置**：`gabanzhaf.py:845–859`

```python
explained = max(
    svc_anom_dict.get(u, 0.) * min(edge_conf.get((u, svc), 0.) * 2.0, 0.95)
    for u in upstream_map_residual.get(svc, set())
), default=0.)
residual[svc] = max(0., score - explained)
```

- 若拓樸方向錯誤（V→R 而非 R→V），`upstream_map_residual[V]` 不包含 R
- 受害者的 `residual ≈ raw score`，抑制幾乎無效
- 最終 `_select_exact_candidates` 的 `0.50*raw + 0.40*residual` 讓受害者排在根因之前

### 為何 re2-tt DELAY 相對較好（AC@1=0.60）

Train Ticket 是鏈狀/星狀拓樸，受害者數量少、傳播路徑單一 → lag-1 非對稱信號較強 → 殘差抑制部分有效。  
re2-ob/re2-ss 是複雜扇出圖，多點同時受害 → 信號被稀釋，AC@1 崩潰至 0.00–0.07。

---

## 結論：演算法瓶頸，非資料集問題

根因的特徵確實存在（AC@5 ≥ 0.80），資料集具備可用的辨識資訊。問題在於管道內部的排名機制在網路故障場景下系統性地偏向受害者。

---

## 改善方向（優先序）

1. **Onset time 優先排名**  
   偵測每個服務指標最初超閾值的時間點。根因應有最早 onset——這是 delay/loss 與受害者最可靠的區分依據，比 lag-1 correlation 更直接。

2. **Call graph 殘差增強**  
   Trace 的 call graph 方向（callee→caller）在網路故障時比 lag-1 更可靠。對 call graph 邊在殘差計算中給更高的 suppression 係數，或在 residual 中讓 call graph edge 優先參與 victim suppression。

3. **Persistence 項差異化**  
   對 delay/loss 語境降低 persistence 的權重（目前 30%），考慮引入 onset speed（異常初期的上升速率）替代，以懲罰緩慢漂移的受害者模式。

---

## 相關檔案

- `RCAEval/e2e/gabanzhaf.py`：主管道
- `benchmark_results.json`：當前數據（本 issue 記錄時間點）
- `docs/superpowers/specs/2026-05-31-phenotype-adaptive-gabanzhaf-design.md`：設計規格

---

## 回覆格式規範

> 本檔案的回覆遵循以下規則：

**解法（Solution）**
- 必須寫清楚用什麼演算法、公式或步驟，不能只說「怎麼做」
- 不使用「適當」、「足夠」、「大概」等模糊字眼，除非是後續需要實驗調參才能確定的數值（此時須明確標注「待調參」）
- 不使用「fallback」一詞；需要降級處理時，改寫成具體條件與對應行為
- 不使用「ground truth」

**風險（Risk）**
- 放在該步驟解法描述的下方
- 格式：`> **[風險]** ...`

**風險回覆（Risk Response）**
- 緊接在對應風險的下方
- 格式：`> **[回覆]** ...`
- 同樣遵守解法的寫作規則：必須具體、不模糊

---

## 目前解法

TORAI-style mean/std severity 只作為參考，不作為主線替代方案。若它不能提高準確率、精簡程式碼或降低時間成本，就不應為了貼近 TORAI 而引入額外 pipeline。

目前主線應聚焦在兩件事：

1. 修正 DELAY/LOSS 下 full-window evidence 偏袒 victim 的問題。
2. 強化 trace/call graph 對 victim suppression 的作用。

建議 pipeline：

```text
early/onset-aware robust service evidence
→ trace/onset-aware candidate rescue
→ trace call graph suppression
→ GA-Banzhaf final ranking
```

### 1. 不直接用 TORAI-style mean/std 取代現有篩選

TORAI 的可用部分是「normal-vs-abnormal severity framing」，但不建議裸用：

```text
z = abs(x - mean) / std
```

直接替換目前 `_cheap_service_filter` 風險偏高，因為 flat metric 的 `std≈0` 會造成 z-score 膨脹，問題只是從 IQR denominator 平移到 std denominator。

較合理的方向是保留現有 robust denominator 思路，將 `_cheap_service_filter` 升級成 early-aware robust severity prefilter：

```text
robust_severity =
  abs(anomal_value - normal_center)
  / max(normal_iqr, baseline_floor, global_iqr_floor)
```

其中 TORAI-style mean/std 只作為可比較的 ablation，不作為預設主線。

此階段只回答：

```text
哪些服務看起來異常？
```

不直接回答：

```text
哪個服務是 root cause？
```

因此它不應取代後續 topology、residual suppression 與 Banzhaf ranking。

> **[風險]** 與現有 `_cheap_service_filter` 高度重疊，增益不確定。現有 filter 已處理 flat metric 的 IQR=0 問題（用 baseline_floor），改成 mean/std 後 std≈0 的 flat metric 同樣會膨脹 z-score——問題平移而非解決。建議不要全部取代，而是在現有 filter 內加入 early-window 維度，保留原有的 denominator 邏輯。
>
> **[回覆]** 風險成立。因此目前解法不採用「mean/std 直接取代」。只保留 TORAI 的 severity framing，實作上沿用 robust denominator/floor，並把改善重點放在 early/onset 維度。

### 2. Severity 需要加入 early/onset 訊號

若只用 full abnormal window 計算 severity，仍可能偏袒長時間漂移的受害者。因此 prefilter/service evidence 應加入 early-window 或 onset-aware 訊號：

```text
service_prefilter_score =
  0.6 * early_window_severity
+ 0.4 * full_window_severity
```

或更完整地納入：

```text
service_score =
  early_severity
+ onset_priority
+ onset_speed
+ full_window_severity
```

目標是偏向「早發生且快速異常」的服務，而不是「長時間受害」的服務。

> **[風險 1]** early window 長度需要調參。若 anomal window 只有 20–30 個時間點，「前 25%」只剩 5–7 點，雜訊會主導結果。建議用比例（前 25–33%）而非固定長度，並設最小保護長度（如至少 5 點才計算 early score）。
>
> **[回覆]** `early_len = max(round(anomal_len * 0.25), 1)`。觸發條件：當 `early_len < 5`（即 `anomal_len < 20`）時，`early_window_severity` 直接取 `full_window_severity` 的值，不單獨計算 early window。`ratio=0.25`、`min_len=5` 為**待調參**數值，需在各 dataset 上做 ablation 確認。
>
> **[風險 2]** onset 偵測閾值與 persistence 共用 `rz > 3.0`。若某 dataset 的正常波動偏大，這個閾值會使 onset 完全偵測不到（所有服務 onset_time 都是 NaN）。考慮對 onset 偵測用更低閾值（如 `rz > 1.5`），與 persistence 分開設定。
>
> **[回覆]** onset 使用獨立閾值 `rz > 1.5`（與 persistence 的 3.0 分開設定），並要求連續 2 個時間點超過閾值才記錄為 onset，避免單點雜訊。計算方式：對每個服務的每條 metric，找出第一個滿足「連續 2 點 rz > 1.5」的起始索引，取所有 metric 中的最小值作為該服務的 `onset_idx`。若該服務所有 metric 在整個 anomal window 內都未出現連續 2 點超閾值，`onset_idx = anomal_len + 1`（設為 anomal window 長度加一，在 onset-aware 排名中排最後，視為未發作）。`threshold=1.5`、連續點數 2 為**待調參**數值。
>
> `metric_onset_idx` 與 `trace_onset_idx` 需分開命名。Metric onset 來自 metric rz；trace onset 來自 trace latency/error time series。若兩者後續要合併，使用 `min(metric_onset_idx, trace_onset_idx)` 作為 service-level onset，但 diagnostics 必須保留兩個原始欄位，避免無法判斷是哪個 source 造成排序變化。

### 3. 不全局弱化 metric lag-correlation topology

目前 `_build_topology` 依賴 lag-1 asymmetric correlation。這對 CPU/MEM/DISK 可能有效，但 DELAY/LOSS 常接近同步傳播，lag-1 方向訊號容易缺失或錯向。

因此不應直接把 lag-corr 整體降權。更安全的做法是在統一流程中加入 trace/call graph direction prior：

```text
lag-corr edge 保留
call graph / trace onset 在有證據時補強或覆蓋方向
```

也就是不要依賴 fault type 分支，而是使用一致的證據融合規則。


> **[風險]** Runtime 不知道 fault type，無法只對 DELAY/LOSS 弱化 lag-corr。若直接降低 lag-corr 權重，CPU/MEM 的準確率可能退化（lag-corr 在那些 fault type 是有效的）。實際可行的做法是：當 call graph 邊與 lag-corr 邊方向衝突時，以 call graph 為主；而非直接調低 lag-corr 整體權重。這樣對所有 fault type 都安全。
>
> **[回覆]** 風險成立，且符合 `.cursor/rules/gabanzhaf-workflow.mdc` 的限制：不能加入 fault-type-specific branch。規則如下：當 call graph 提供邊 `(callee, caller)` 且同時滿足 `svc_anom_dict[callee] >= 0.2`（對應現有 `call_graph_callee_min_score`）與 `svc_anom_dict[caller] >= 0.1`（對應現有 `call_graph_caller_min_score`，`gabanzhaf.py:736`），該 call graph 邊方向視為可信。此時若 lag-corr 給出相反方向的邊 `(caller, callee)`，則該 lag-corr 邊只保留在 `upstream_map_full`（用於 candidate 擴展），不納入 `upstream_map_residual`（不參與殘差抑制）。lag-corr 同向邊（方向與 call graph 一致）不受影響。

### 4. 用 trace onset + call graph 做輕量因果線索

Trace 可轉成 service-level time series：

```text
error_rate(t)
avg_latency(t)
p95_latency(t)
request_count(t)
```

再計算每個服務的：

```text
trace_severity
trace_onset_time
trace_onset_speed
```

若 call graph 中存在：

```text
caller -> callee
```

則 RCA 解釋方向可視為：

```text
callee -> caller
```

若 `callee_onset <= caller_onset`，則 callee 更可能是 root 或 upstream cause，caller 更可能是 victim。

> **[風險]** onset_time 的時間解析度受限於 `tracets_lat.csv` / `tracets_err.csv` 的 row 頻率。若 trace 每 15 秒一個資料點，onset 精度最多 15 秒；對快速傳播的 DELAY/LOSS 故障（root→victim 傳播可能在幾秒內完成），onset 差異可能落在同一個 bucket → callee_onset == caller_onset → 無法區分。需確認各 dataset 的 trace 取樣頻率是否能支撐 onset 比較。
>
> **[回覆]** 風險成立。Onset 不作為唯一排序依據。當 `callee_onset_idx == caller_onset_idx`（同 bucket）時，不直接用 onset 判斷誰更早；改用以下條件決定 call graph edge 是否可進入 `upstream_map_residual`：
>
> 1. call graph 提供 `callee -> caller` 的 RCA 解釋方向
> 2. `early_window_severity[callee] > early_window_severity[caller]`
> 3. `svc_anom_dict[callee] >= svc_anom_dict[caller] * 0.8`
>
> 三個條件同時滿足時，允許同 bucket 的 `callee -> caller` 進入 residual suppression；否則同 bucket 只保留在 candidate/topology context，不參與 residual suppression。這樣避免「同 bucket 永遠不能 suppression」導致 DELAY/LOSS 同步傳播案例完全得不到改善。

> **[補充風險]** Call graph direction 不永遠等於 root-cause direction。某些 caller-side fault 可能造成 callee 異常，若固定使用 `callee -> caller`，可能壓低真正 root。
>
> **[回覆]** Call graph 邊 `(callee, caller)` 觸發 suppression 的條件必須同時滿足：
> 1. `callee_onset_idx < caller_onset_idx`，或同 bucket 時 `early_window_severity[callee] > early_window_severity[caller]`
> 2. `svc_anom_dict[callee] >= svc_anom_dict[caller] * 0.8`（callee 的 anomaly score 至少是 caller 的 80%，避免輕微異常的 callee 壓制嚴重異常的 caller）
>
> 兩個條件均滿足時，才對 caller 套用 call graph suppression。`0.8` 為**待調參**數值。

### 5. 強化 residual suppression

DELAY/LOSS 的 AC@1 問題主要來自 victim 未被壓低。若某個服務的異常可由 upstream/root 解釋，應降低其 residual score：

```text
victim_residual = max(0, victim_raw_score - explained_by_upstream)
```

對 trace call graph edge 可給比 metric lag-correlation edge 更高的 confidence，因為網路故障下 call graph 方向通常比 lag-1 correlation 更可靠。

> **[風險]** 若 traces 不存在（無 `traces.csv` 也無 preaggregated 檔案），call graph 為空，call graph suppression 不會執行，退回現有 lag-corr residual——對 DELAY/LOSS 等同於未改善。
>
> **[回覆]** Call graph suppression 以增強項加入：當 `call_graph` 為空 dict 時，`_call_graph_topology_edges` 已回傳 `[]`（`gabanzhaf.py:723`），`upstream_map_residual` 只由 lag-corr 邊構成，現有 residual 邏輯完整保留，不需要額外處理。此路徑不需要改動，測試時只需確認 `call_graph={}` 時 residual 輸出與現行版本一致。

> **[補充風險]** Suppression 過強會把真正 root 壓低，尤其是 call graph prior 方向錯誤時。
>
> **[回覆]** Suppression 觸發需同時滿足以下條件（以 `(callee, caller)` 邊為例）：
> 1. 現有 absolute guard（`gabanzhaf.py:736`）：`svc_anom_dict[callee] >= 0.2` 且 `svc_anom_dict[caller] >= 0.1`
> 2. 新增 relative guard：`svc_anom_dict[callee] >= svc_anom_dict[caller] * 0.8`
>
> 條件 2 確保 callee（supposed upstream）的 anomaly score 至少達到 caller（supposed victim）的 80%。若 callee 遠比 caller 輕微，代表 call graph 方向可疑，此時不觸發 suppression，保留 caller 的 raw residual score。`0.8` 為**待調參**數值。

### 6. 第一版暫時不用 log

Log source 需要額外處理：

```text
log parsing
template extraction
bucket by time
template count time series
```

這會增加實作複雜度與 runtime。當前瓶頸是 DELAY/LOSS 的 service ranking，而不是缺少 log evidence，因此第一版先使用 metrics + traces/call graph 做 ablation 較合理。

> **[風險]** 不用 log 可能錯過只有 log 明顯異常的服務。
>
> **[回覆]** 風險存在，但不是目前 AC@1 bottleneck 的主因。加入 log 需要 template parsing、bucket aggregation 與 service mapping，會增加時間成本與 failure mode。第一版先不使用 log，等 metrics + traces 的 ranking 問題收斂後再做 ablation。

### 7. 保留 GA-Banzhaf 做最終排名

Severity prefilter 不應取代 GA-Banzhaf。Severity 只知道誰異常，Banzhaf 才負責判斷：

```text
誰最能解釋其他服務的異常？
```

最終方法應是：

```text
metrics:
  robust severity + early/onset score

traces:
  latency/error severity + onset time + call graph

candidate:
  top severity services + trace/call graph rescue

ranking:
  residual suppression + Banzhaf
```

核心原則：

```text
不為了 TORAI 而 TORAI；
保留現有 robust evidence 架構，
加入 early/onset-aware 訊號，
用 trace call graph 改善 victim suppression，
最後仍用 GA-Banzhaf 做 root cause ranking。
```

---

## 實作結果格式規範

每次實作完成後，在「實作結果紀錄」區塊新增一筆，格式如下：

```
### 第N次實作 — YYYY-MM-DD HH:MM

**實作內容：** 本次實作了哪些解法步驟（對應「目前解法」章節編號）

**Benchmark 對照：**
（表格：baseline vs 實作後，標注退步項目）

**演算法執行時間：**
（表格：dataset、實作前、實作後、baro 參考值）

**是否解決 issue：** 已解決 / 部分解決 / 未解決
（條列每個 AC@1 / Avg@5 目標的達成狀況）

**後續問題：**
（每個問題獨立一行，格式：Q編號. 問題描述）
```

---

## 實作結果紀錄

### 第 1 次實作 — 2026-06-03

**實作內容：** 解法 2（early/onset-aware metric evidence）、解法 3（call graph 方向控制）、解法 5（residual suppression onset gate）

具體變更：
- `_metric_evidence`：加入 early window（前 25%）與 `_first_sustained_onset`，新分數公式 `0.60 * early + 0.40 * full`
- `_upstream_maps`：lag-corr 與 call graph 方向衝突時，lag-corr 邊排除出 `upstream_map_residual`
- `_call_graph_residual_allowed`：加入 onset gate，`callee_onset > victim_onset` 時封鎖 suppression

**Benchmark 對照（baseline：2026-06-03 15:15）：**

| Dataset | 指標 | Baseline | 實作後 | 變化 |
|---------|------|---------|--------|------|
| re2-ob | AC@1-DELAY | 0.00 | **0.27** | ↑ +0.27 |
| re2-ob | AC@1-LOSS  | 0.53 | 0.40 | ↓ -0.13 |
| re2-ob | Avg@5-DELAY | 0.65 | **0.72** | ↑ +0.07 |
| re2-ob | Avg@5-LOSS  | 0.84 | 0.83 | → |
| re2-ss | AC@1-DELAY | 0.07 | **0.47** | ↑ +0.40 |
| re2-ss | AC@1-LOSS  | 0.33 | 0.33 | → |
| re2-ss | Avg@5-DELAY | 0.72 | **0.80** | ↑ +0.08 |
| re2-ss | Avg@5-LOSS  | 0.83 | 0.84 | → |
| re2-tt | AC@1-DELAY | 0.60 | 0.60 | → |
| re2-tt | AC@1-LOSS  | 0.53 | 0.53 | → |
| re2-tt | Avg@5-DELAY | 0.73 | **0.69** | ↓ -0.04 |
| re2-tt | Avg@5-LOSS  | 0.64 | 0.64 | → |
| re2-tt | AC@3-DELAY | 0.73 | **0.67** | ↓ -0.06 |
| re2-tt | AC@3-SOCKET | 1.00 | **0.93** | ↓ -0.07 |

**演算法執行時間：**

| Dataset | 實作前 | 實作後 | baro | 實作後 vs baro |
|---------|--------|--------|------|--------------|
| re2-ob  | 41.8s  | 28.1s  | 9.8s | **2.9x** 慢 |
| re2-ss  | 24.9s  | 12.1s  | 10.3s | 1.2x 慢 |
| re2-tt  | ~47s   | 69.2s  | 35.3s | **2.0x** 慢 |

**gabanzhaf vs baro 完整精準度對照（實作後）：**

*re2-ob*

| 指標 | gabanzhaf | baro | 差距 |
|------|-----------|------|------|
| AC@1-CPU | 0.40 | 0.00 | +0.40 |
| AC@1-MEM | 0.73 | 0.33 | +0.40 |
| AC@1-DISK | 0.27 | 0.13 | +0.14 |
| AC@1-SOCKET | 0.27 | 0.00 | +0.27 |
| AC@1-DELAY | 0.27 | 0.00 | +0.27 |
| AC@1-LOSS | 0.40 | 0.40 | 0 |
| Avg@5-CPU | 0.88 | 0.64 | +0.24 |
| Avg@5-DELAY | 0.72 | 0.67 | +0.05 |
| Avg@5-LOSS | 0.83 | 0.80 | +0.03 |

*re2-ss*

| 指標 | gabanzhaf | baro | 差距 |
|------|-----------|------|------|
| AC@1-CPU | 0.27 | 0.00 | +0.27 |
| AC@1-MEM | 0.80 | 0.20 | +0.60 |
| AC@1-DISK | 0.53 | 0.00 | +0.53 |
| AC@1-SOCKET | 0.33 | 0.00 | +0.33 |
| AC@1-DELAY | 0.47 | 0.00 | +0.47 |
| AC@1-LOSS | 0.33 | 0.20 | +0.13 |
| Avg@5-DELAY | 0.80 | 0.68 | +0.12 |
| Avg@5-LOSS | 0.84 | 0.80 | +0.04 |

*re2-tt*

| 指標 | gabanzhaf | baro | 差距 |
|------|-----------|------|------|
| AC@1-CPU | 0.73 | 0.47 | +0.26 |
| AC@1-MEM | 1.00 | 0.93 | +0.07 |
| AC@1-DISK | 1.00 | 1.00 | 0 |
| AC@1-SOCKET | 0.73 | 0.60 | +0.13 |
| AC@1-DELAY | 0.60 | 0.47 | +0.13 |
| AC@1-LOSS | 0.53 | 0.53 | **0** |
| AC@3-DELAY | 0.67 | 0.67 | 0 |
| AC@5-DELAY | 0.80 | 0.67 | +0.13 |
| AC@5-LOSS | 0.73 | 0.73 | 0 |
| Avg@5-CPU | 0.95 | 0.72 | +0.23 |
| Avg@5-SOCKET | 0.89 | 0.83 | +0.06 |
| Avg@5-DELAY | 0.69 | 0.63 | +0.06 |
| Avg@5-LOSS | 0.64 | 0.64 | **0** |

**Trade-off 判斷：**

| 狀況 | 結論 |
|------|------|
| re2-tt LOSS：2x 時間換到的精準度提升 | 0（完全沒換到）|
| re2-tt DELAY：2x 時間換到的 AC@1 提升 | +0.13，仍距目標 0.9 很遠 |
| re2-tt CPU/MEM：2x 時間換到的提升 | +0.26 / +0.07，相對合理 |
| re2-ss：1.2x 時間換到的 DELAY AC@1 提升 | +0.47，時間成本低、收益高 |

**是否解決 issue：** 部分解決

- re2-ob / re2-ss DELAY AC@1：顯著提升 ✓
- re2-tt DELAY AC@1：無變化，DELAY/LOSS 整體仍低於目標 ✗
- re2-tt Avg@5-DELAY 與 AC@3-DELAY/SOCKET：退步 ✗
- re2-tt 執行時間：47s → 65.5s，超過 baro（34s）✗

**後續問題：**

Q1. re2-tt DELAY/LOSS Avg@5 下降（0.73→0.69）且 AC@3-SOCKET 退步（1.00→0.93）

> 根本原因：`_metric_evidence` 新公式 `0.60*early + 0.40*full` 將 persistence 有效權重從 30% 降至 12%，early-window mag 有效權重升至 42%。SOCKET 是漸進積累型故障（socket 耗盡），根因服務的指標早期不一定有明顯尖峰 → early score 偏低 → 根因排名下滑。DELAY 的 AC@3 退步原因相同：`_call_graph_residual_allowed` 的 onset gate 在 onset_idx 偵測不準確時（threshold 1.5 + 連續 2 點未觸發），將正確的 suppression 封鎖，受害者未被壓低。
>
> **[回覆]** 這個判斷方向成立。第 1 次實作把 evidence 主導權從 full-window / persistence 拉向 early-window，因此 re2-ob、re2-ss 的 DELAY AC@1 上升，但 re2-tt SOCKET 與 re2-tt DELAY AC@3 / Avg@5 退步。這表示 early signal 對 fan-out DELAY 有幫助，但目前 `0.60 * early + 0.40 * full` 權重過強，對漸進積累型故障不穩。
>
> 下一步不應直接擴大 early/onset 權重，而應做 ablation：
>
> ```text
> A. full-only baseline
> B. 0.30 * early + 0.70 * full
> C. 0.40 * early + 0.60 * full
> D. early 只作為 bonus / tie-break，不取代 full-window 主分數
> ```
>
> 同時要拆開比較：
>
> ```text
> early scoring only
> call graph onset gate only
> early scoring + call graph onset gate
> ```
>
> 這樣才能判斷 re2-tt 退步主要來自 early scoring，還是來自 suppression gate 封鎖了正確的 call graph edge。

Q2. re2-tt 執行時間從 47s 增加至 69.2s，為 baro（35.3s）的 2.0x，且 DELAY/LOSS 精準度提升有限

> 根本原因：`_first_sustained_onset`（`gabanzhaf.py:467`）使用雙層 Python 迴圈（`for j in n_cols: for i in n_rows`），時間複雜度 O(n\_cols × n\_rows)，未向量化。re2-tt metric 數量最多，迴圈執行次數最大，因此在 re2-tt 上的開銷最顯著。re2-ob / re2-ss 反而因為 early-onset 分數讓候選集更精準，後段計算減少，整體變快。
>
> **[回覆]** onset loop 是否為主因目前沒有 stage timing 證據，不應先投入 vectorization。更關鍵的問題是 **2x 時間成本換到的精準度是否值得**：
>
> | 狀況 | 實際結果 |
> |------|---------|
> | re2-tt LOSS：2x 時間換到的 AC@1 提升 | **0**（0.53 vs baro 0.53） |
> | re2-tt DELAY：2x 時間換到的 AC@1 提升 | +0.13（0.60 vs baro 0.47），仍距目標 0.9 很遠 |
> | re2-tt CPU：2x 時間換到的 AC@1 提升 | +0.26（0.73 vs baro 0.47），相對合理 |
> | re2-ss：1.2x 時間換到的 DELAY AC@1 提升 | +0.47（0.47 vs baro 0.00），時間成本低、收益高 |
>
> re2-tt 是問題的核心：gabanzhaf 在 re2-tt 的 2x 時間成本主要由架構決定（服務數多 → `_build_topology` 更重、trace 檔案大 → I/O 更重），不是最近的 onset 改動造成的。這個時間差在 onset 加入之前就已存在。
>
> 下一步應先用 `return_stage_timing=True` 對 re2-tt 做 case-level profiling，確認 69.2s 分佈在哪一段：
>
> ```text
> preprocess / metric_evidence / topology / banzhaf / ranking
> ```
>
> 這決定優化方向：若時間集中在 topology，應縮小 pool_svcs 或降低 lag-corr 計算量；若集中在 metric_evidence，才考慮向量化 onset loop。在拿到 stage timing 之前，不應對任何單一函式做假設性優化。

> **[Profiling 結果 — 2026-06-03]** 已執行 `GABANZHAF_PROFILE=1 py310/bin/python main.py --method gabanzhaf --dataset re2-tt`，對 re2-tt 90 個 case 做 stage-level timing，結果如下：
>
> | Stage | 總耗時 | 平均/case | 占比 |
> |-------|--------|-----------|------|
> | `metric_evidence` | 28.6s | 0.317s | **53%** |
> | trace I/O | 20.8s | 0.231s | **39%** |
> | `preprocess` | 3.5s | 0.039s | 7% |
> | `topology` | 0.8s | 0.009s | 1% |
> | `banzhaf` | ~0s | ~0s | ~0% |
> | `ranking` | ~0s | ~0s | ~0% |
> | **case-level 合計** | **53.7s** | — | 100% |
> | framework overhead | ~15.5s | — | — |
> | **wall time** | **69.2s** | — | — |
>
> **推論（更新後）：**
>
> 原假設「topology 或 Banzhaf 是主因」完全不成立。topology 只佔 1%，banzhaf 幾乎為零。真正的瓶頸有兩個：
>
> 1. **`metric_evidence`（53%）**：`_first_sustained_onset` 的雙層 Python 迴圈（O(n_cols × n_rows)）確實是主因。re2-tt 服務數與 metric 數遠多於 re2-ob/re2-ss，這個迴圈在每個 case 耗掉 0.317s，乘以 90 case = 28.6s，超過 `topology` 的 35 倍。
>
> 2. **trace I/O（39%）**：每個 case 都重新讀取 `tracets_lat.csv` / `tracets_err.csv`，無任何快取。re2-tt 的 trace 檔案比 re2-ob/re2-ss 大，I/O 成本 0.231s/case × 90 = 20.8s。
>
> **下一步優化方向（依優先序）：**
>
> 1. **向量化 `_first_sustained_onset`**：改用 numpy sliding window 或 `np.where` + rolling minimum，消除雙層 Python 迴圈。預期可將 `metric_evidence` 從 28.6s 壓縮至 5s 以內，節省 ~23s。
>
> 2. **Trace I/O 快取**：在 dataset-level 初始化時一次性讀入所有 case 的 trace files，存成 dict（`case_id → DataFrame`），case-level 呼叫只做 dict lookup。預期節省 15–20s。
>
> 3. **Topology 不需要優化**（1% 占比，動它的 ROI 幾乎為零）。

Q3. 第 1 次實作尚未真正使用 trace onset

> **[回覆]** 文件中的目標是「trace onset + call graph」，但目前第 1 次實作只使用了 metric-derived onset：
>
> ```text
> metric_onset_idx: 已實作
> trace_onset_idx: 尚未實作
> ```
>
> 因此目前方法實際上是：
>
> ```text
> metric early/onset evidence
> + call graph residual gate
> ```
>
> 還不是完整的 trace time-series onset 方法。這可能是 re2-ob / re2-ss DELAY 有改善，但 re2-tt 與 SOCKET 仍不穩的原因之一。後續若要繼續沿用文件主線，應先補 diagnostics，分別輸出 `metric_onset_idx` 與 `trace_onset_idx`，再決定是否把 trace onset 納入 residual gate。

---

### 第 2 次實作 — 2026-06-08 (Q2 效能優化)

**實作內容：** 處理 Q2（re2-tt 執行時間為 baro 的 2.0x）。對應 Profiling 結果列出的兩個瓶頸：`metric_evidence`（53%）與 trace I/O（39%）。

具體變更：

- **向量化 `_first_sustained_onset`（`gabanzhaf.py:470`）**：移除原本的雙層 Python 迴圈（`for j in n_cols: for i in n_rows`，O(n_cols × n_rows)），改用 `numpy.lib.stride_tricks.sliding_window_view`：
  - `windows = sliding_window_view(active, run_len, axis=0)` → shape `(n_starts, n_cols, run_len)`
  - `window_all = windows.all(axis=2)` → 每個 (start, col) 是否連續 `run_len` 點皆超閾值
  - `first_start = window_all.argmax(axis=0)`，以 `has_run = window_all.any(axis=0)` 守門，未發作的服務回傳 sentinel `n_rows + 1`
  - 語意與原迴圈逐位元等價（見下方驗證）。
- **trace I/O 不加快取**：原文件 Q2 回覆建議「dataset-level 一次讀入所有 case 的 trace files，存成 `case_id → DataFrame` dict」。實測後**不採用**，理由具體如下。

**演算法執行時間（re2-tt，90 cases，stage-level）：**

| Stage | 第 1 次實作 | 第 2 次實作 | 變化 |
|-------|------------|------------|------|
| `metric_evidence` | 28.6s（0.317s/case） | **9.8s（0.109s/case）** | ↓ -18.8s（-66%） |
| trace I/O | 20.8s | 21.3s | →（無變動，未動此路徑） |
| `preprocess` | 3.5s | 3.4s | → |
| `topology` | 0.8s | 0.5s | → |
| **case-level 合計** | **53.7s** | **35.0s** | ↓ -18.7s（-35%） |

Micro-benchmark（re2-tt scale，150 rows × 3000 cols，20 次平均）：迴圈版 44.9ms/call → 向量化 0.42ms/call，**107x** 加速。

**是否解決 issue：** 部分解決（僅針對 Q2）

- Q2 主瓶頸（`metric_evidence` 雙層迴圈）已消除：case-level 合計 53.7s → 35.0s。對應 wall time 預期由 ~69s 降至 ~50s。✓
- trace I/O（現為剩餘最大宗，21.3s / 61%）**未優化**，原因見下方 Q4。
- 本次**不觸及準確率**：`_first_sustained_onset` 輸出逐位元等價，故 AC@k / Avg@5 與第 1 次實作的 pipeline 完全一致。

**準確率不變的驗證方式：**

- 新增 `test_first_sustained_onset_matches_reference_on_random_matrices`：以 50 組隨機矩陣（隨機 rows/cols/threshold/consecutive）比對向量化版與保留原雙層迴圈的 reference 實作，`np.array_equal` 全數通過。
- 新增 `test_first_sustained_onset_edge_cases`：涵蓋 empty、1-D 輸入、`n_rows < run_len`、整段無連續 run 四種邊界，sentinel 行為與原版一致。
- `tests/test_gabanzhaf_improvements.py` 全檔 54 tests pass。

**後續問題：**

Q4. trace I/O（21.3s，現為 re2-tt 剩餘最大宗）不採用原文件建議的「case 快取」

> **[回覆]** 原 Q2 回覆假設「重複讀取同一 case 的 trace files」，實測不成立：
>
> 1. 在單趟 benchmark（`main.py --dataset re2-tt`）中，每個 case 目錄只被 `_extract_trace_features` 呼叫一次，且各 case 目錄互異。以 `case_id → DataFrame` 為 key 的快取在單趟流程中命中率為 0，無任何增益。
> 2. 真正的成本來源是 `_extract_call_graph` 掃描 `traces.csv`：re2-tt 每個 case 的 `traces.csv` 為 **166 MB**，而 preaggregated 的 `tracets_lat.csv` / `tracets_err.csv` 僅 56–68 KB。call graph 必須來自 `traces.csv` 的 span/parent 欄位。
> 3. 該讀取**已經是受限讀取**：`_extract_call_graph`（`gabanzhaf.py:358`）已使用 `usecols`（只取 serviceName / spanID / parentSpanID / traceID）加 `nrows=50_000`，C parser 讀滿 50K 列即停。再縮減列數會改變 call graph → 影響 residual suppression → 改動準確率，已超出「純效能、不動準確率」的 Q2 範圍。
>
> 結論：trace I/O 的 21.3s 在單趟流程中屬於不可消除的固有成本。若未來改為多趟（多 seed / ablation 重跑同一批 case），才值得做 dataset-level 快取；屆時 key 應為 `case_dir`，並只快取已解析的 call graph dict 與 preaggregated scores，而非原始 DataFrame。

Q5. 現行 working tree 的 AC@1 全面低於第 1 次實作紀錄，與本次 Q2 改動無關

> **[回覆]** 本次以 `main.py --dataset re2-tt` 實測，AC@1-CPU=0.24、AC@1-DELAY=0.22，遠低於第 1 次實作表格（AC@1-CPU=0.73、AC@1-DELAY=0.60）。此差異**不是** Q2 向量化造成的（onset 輸出逐位元等價）。成因是第 1 次實作之後的 pipeline 重構（commits `ae044232` 改用 probabilistic causal banzhaf、`dfc2ba0b` 簡化 ranking、`c8d1a97b` 加入 `_cheap_service_filter` 實驗），這些變更把 ranking 機制整個換掉，造成 AC@1 全面下滑。
>
> 這代表 issue 主線的 Q1（early/full 權重 ablation）已被 pipeline 重構**取代語境**：在重新討論 DELAY/LOSS 排名前，需先釐清 probabilistic banzhaf 重構是否為當前的目標 pipeline。此為獨立議題，不在 Q2 範圍內。

Q6. re2-tt DELAY/LOSS Avg@5 未超過 0.8 的主要瓶頸

> **[版本說明]** 下列分析同時使用兩類證據：
>
> 1. `output/runs/gabanzhaf/re2-tt/results` 中 2026-06-08 的 30 個 DELAY/LOSS case 結果，用於統計最終服務名次。
> 2. 目前 working tree 的 case-level instrumentation，用於觀察 `svc_anom_dict`、`svc_residual`、K=12 game candidate、trace score 與 Banzhaf 名次。
>
> 現存結果檔與目前 working tree 對少數 case 的輸出不同，因此以下結論著重於失敗模式與 stage-level 原因，不把兩個版本的單一 case 名次視為同一次 benchmark。

**現存 30-case 名次分布：**

| Fault | 第 1 名 | 第 2–5 名 | 第 6 名以後 |
|-------|--------:|----------:|------------:|
| DELAY | 9 / 15 | 3 / 15 | 3 / 15 |
| LOSS | 8 / 15 | 3 / 15 | 4 / 15 |

DELAY 的主要失敗集中在：

```text
ts-auth-service: rank 19, 19
ts-travel-service: rank 4, 5, 9
```

LOSS 的主要失敗集中在：

```text
ts-auth-service: rank 4, 18, 7
ts-travel-service: rank 61, 62, 3
```

這不是所有服務都無法辨識，而是特定服務在 candidate selection 與 final ranking 出現排名斷崖。

### Q6.1 固定 K=12 前的 candidate selection 淘汰有效候選

目前 `_select_exact_candidates` 使用：

```text
candidate_score =
  0.50 * metric_anomaly
+ 0.40 * residual
+ 0.05 * trace
+ 0.05 * graph_participation
```

case-level 診斷：

| Case | Metric anomaly rank | Trace rank | 是否進入初始 candidate | 是否進入 K=12 |
|------|--------------------:|-----------:|------------------------|--------------|
| `ts-auth-service_delay/2` | 29 | 11 | 是（SLI rescue） | 否 |
| `ts-auth-service_delay/3` | 28 | 4 | 是（SLI rescue） | 否 |
| `ts-auth-service_loss/2` | 53 | 1 | 是（SLI rescue） | 否 |

根因服務即使已被 SLI 或 topology 帶入 candidate，仍會因 metric anomaly 與 residual 佔 90% 而在固定 K=12 前被淘汰。對 LOSS 而言，trace 已將根因排在第 1，但 5% 權重不足以保留候選。

> **[風險]** 直接提高 trace 權重可能在 trace latency 廣泛上升時，把多個受害者同時送入 K=12，排擠局部資源故障的真正候選。
>
> **[回覆]** 不直接以單一加權總分提高 trace 權重。K=12 改用 evidence-source quota 建立：
>
> ```text
> metric/residual top candidates
> ∪ trace latency/error top candidates
> ∪ topology upstream candidates
> → 去重
> → 以跨來源支持數與來源內分位數排序
> → 截取 K=12
> ```
>
> 建議第一組 ablation 使用 `metric/residual 6 + trace 3 + topology 3`，去重後不足 12 個時，再依統一的跨來源分數補滿。`6/3/3` 為**待調參**數值；所有 incident 使用同一規則，不依 dataset、fault 或 service 分支。

### Q6.2 `_rank_metrics` 將 K=12 最低 Banzhaf 候選歸零

目前 `_rank_metrics` 對 K=12 的 Banzhaf 分數做 min-max normalization：

```python
nb = _norm_dict(finite_banzhaf)
score(metric) = nb[service] * (epsilon + metric_evidence)
```

K=12 中 Banzhaf 最低的服務會被轉成 `0`；未進入 K=12 的服務也由 `nb.get(service, 0.0)` 得到 `0`。兩者因此失去區別。

`ts-travel-service_loss/2` 的根因已進入 K=12，但 Banzhaf 位於候選尾端；正規化後 service multiplier 接近或等於 0，最終服務名次掉到第 62。這是 final ranking 的數值斷崖，不是根因完全沒有 evidence。

> **[風險]** 單純加入固定 floor 會讓低可信服務重新獲得過多分數，削弱 Banzhaf 的篩選作用。
>
> **[回覆]** 使用保序且有下界的轉換：
>
> ```text
> normalized_banzhaf =
>   banzhaf_floor
> + (1 - banzhaf_floor) * minmax(banzhaf)
> ```
>
> K=12 內候選使用 `banzhaf_floor > 0`；K=12 外服務維持 0。第一組 ablation 使用 `banzhaf_floor=0.10`，數值為**待調參**。這保留 K=12 內的排序，同時避免最低候選與未入選服務完全等價。

### Q6.3 Trace evidence 已有辨識力，但沒有進入核心 game

目前 trace pipeline 已產生：

```text
trace_scores
trace_error_scores
trace_latency_scores
```

但實際使用方式是：

```text
trace_scores: candidate selection 中占 5%
trace_error_scores: 讀取後未進核心計算
trace_latency_scores: 讀取後未進核心計算
anomaly_weights: 只使用 svc_anom_dict
```

LOSS 失敗 case 中，`ts-auth-service_loss/2`、`ts-auth-service_loss/3`、`ts-travel-service_loss/1`、`loss/2`、`loss/3` 的 trace rank 都是第 1，但此訊號沒有進入 Banzhaf anomaly weights。

> **[風險]** 將 trace latency 直接加進 anomaly weight 會提高 caller/victim 的權重，因為 end-to-end latency 也會沿 call path 傳播。
>
> **[回覆]** anomaly weight 不使用 trace latency 的絕對值直接相加，改用來源一致性與 residual trace evidence：
>
> ```text
> trace_residual(s) =
>   trace_score(s)
>   - max upstream_trace_explanation(s)
>
> anomaly_weight(s) =
>   w_metric * metric_score(s)
> + w_trace_residual * trace_residual(s)
> + w_cross * sqrt(metric_score(s) * trace_score(s))
> ```
>
> 第一組 ablation 使用 `w_metric=0.60`、`w_trace_residual=0.25`、`w_cross=0.15`，數值為**待調參**。`trace_error_scores` 與 `trace_latency_scores` 必須保留為兩個 diagnostics，確認增益來自 error 還是 latency。

### Q6.4 Call graph edge 被 metric-only gate 封鎖

目前 call graph edge 需要同時滿足：

```text
callee metric anomaly >= 0.2
caller metric anomaly >= 0.1
```

但 `ts-auth-service_delay/2` 與 `/3` 的根因 metric anomaly 約為 `0.106`、`0.088`。即使 trace 顯示服務異常，call graph edge 仍無法進入 topology，後續 residual suppression 與 causal explanation matrix 都無法使用該方向資訊。

> **[風險]** 移除 absolute guard 會將低活動服務的稀疏 call edge 加入 topology，增加錯誤 suppression。
>
> **[回覆]** 不移除 guard，改成 metric/trace 聯合 evidence：
>
> ```text
> endpoint_evidence(s) =
>   max(metric_score(s), trace_score(s))
>
> edge_allowed(u, v) =
>   endpoint_evidence(u) >= callee_threshold
>   and endpoint_evidence(v) >= caller_threshold
>   and edge 在取樣 call graph 中出現
> ```
>
> threshold 初始沿用 `0.2/0.1`，只替換 evidence source。另輸出每條 edge 的 metric、trace 與通過原因，確認沒有因 trace normalization 造成大量邊同時通過。

### Q6.5 Probabilistic Banzhaf 被 self-explanation 主導

目前 causal explanation matrix 固定：

```text
P(service explains itself) = 1.0
```

而 anomaly weights 只使用 metric anomaly。當 topology edge 稀少或 confidence 偏低時，Banzhaf 的主要收益來自解釋自己的異常，結果會接近 metric anomaly 排名。

`ts-travel-service` DELAY 的診斷結果：

| Case | Metric anomaly rank | Banzhaf rank |
|------|--------------------:|-------------:|
| delay/1 | 3 | 3 |
| delay/2 | 5 | 5 |
| delay/3 | 9 | 9 |

這表示目前 game 沒有充分獎勵「可解釋多個下游服務」的候選。

> **[風險]** 直接降低 self-explanation probability 會傷害沒有明確 propagation 的 CPU/MEM/DISK case。
>
> **[回覆]** 不使用 fault-type branch。將 self-explanation 拆成 local 與 propagated coverage，並保留 local evidence：
>
> ```text
> payoff(s) =
>   local_weight * local_coverage(s)
> + propagated_weight * downstream_coverage(s)
> ```
>
> 先做 `local_weight/propagated_weight` ablation：
>
> ```text
> A. 1.00 / 1.00（目前語意）
> B. 0.75 / 1.00
> C. 0.50 / 1.00
> ```
>
> 只有在 CPU/MEM/DISK Avg@5 不下降超過 0.02 且 DELAY/LOSS Avg@5 上升時，才接受降低 local dominance。`0.02` 為本輪 regression budget。

### Q6.6 方法限制判斷

目前證據不支持「Banzhaf 方法本身無法讓 re2-tt DELAY/LOSS Avg@5 超過 0.8」。

主要原因是：

1. 根因可能在 Banzhaf 前被 K=12 淘汰。
2. 已擷取的 trace latency/error 沒有進入 anomaly weights。
3. call graph 被 metric-only threshold 封鎖。
4. 進入 K=12 的最低候選會在 final ranking 被歸零。
5. topology 較弱時，self-explanation 讓 Banzhaf 退化成 metric anomaly 排名。

因此目前屬於 **pipeline 與 game formulation 尚未充分利用可用 evidence**，不能先歸因為 Banzhaf 的理論限制。

### Q6.7 下一步實驗優先序

每次只改一個變數，依序執行：

```text
Experiment 1:
  修正 K=12 內最低 Banzhaf 候選被歸零

Experiment 2:
  trace-aware K=12 candidate construction

Experiment 3:
  metric + trace residual anomaly weights

Experiment 4:
  metric/trace 聯合 call graph gate

Experiment 5:
  local vs propagated coverage ablation

Experiment 6:
  early/full evidence 權重 ablation
```

每個實驗至少記錄：

```text
root metric rank
root trace rank
root residual rank
是否進入 candidate pool
是否進入 K=12
root Banzhaf rank
final service rank
call graph edges accepted/rejected 與原因
```

接受條件：

```text
re2-tt Avg@5-DELAY > 0.80
re2-tt Avg@5-LOSS > 0.80
CPU/MEM/DISK/SOCKET 任一 Avg@5 下降不超過 0.02
不加入 dataset、fault、case 或 service-specific branch
K=12 維持不變
```

優先做 Experiment 1 與 Experiment 2。這兩項直接對應已觀察到的排名斷崖與候選淘汰，且不需要改變 fault-independent 的單一路徑。
