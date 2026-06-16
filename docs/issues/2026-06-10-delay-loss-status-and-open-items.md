# Issue: DELAY / LOSS 準確率提升 — 已解決事項與未解決事項彙整

**日期：** 2026-06-10 00:00  
**狀態：** 進行中  
**影響範圍：** `gabanzhaf.py` — Step 5 candidate、Step 6 exact candidates、Step 8 causal matrix、Step 9 Banzhaf、Step 10 ranking；`main.py` 評估流程  
**前身 issue：** `docs/issues/2026-06-03-delay-loss-low-accuracy.md`（完整診斷、Q1–Q8、Q6.1–6.8、三次實作紀錄都在該檔）

---

## 回覆格式規範

> 本檔案的回覆遵循以下規則：

**解法（Solution）**
- 必須寫清楚用什麼演算法、公式或步驟，不能只說「怎麼做」
- 不使用「適當」、「足夠」、「大概」等模糊字眼，除非是後續需要實驗調參才能確定的數值（此時須明確標注「待調參」）
- 不使用「fallback」一詞；需要降級處理時，改寫成具體條件與對應行為

**風險（Risk）** — `> **[風險]** ...`，放在解法下方  
**風險回覆（Risk Response）** — `> **[回覆]** ...`，緊接在對應風險下方

---

## 當前流程（Pipeline Overview）

```
Step 1. Window split         — _split_raw_windows()
Step 2. Preprocess           — _preprocess_windows()
Step 3. Metric evidence      — _metric_evidence(), _service_evidence()
Step 4. Topology             — _build_topology(), _call_graph_topology_edges()
Step 5. Candidate selection  — _candidate_services()
Step 6. Exact candidates     — _select_exact_candidates()      ← 主瓶頸 (Q6.1)
Step 7. Residual scores      — _residual_scores()
Step 8. Causal matrix        — _causal_explanation_matrix()    ← Q6.5
Step 9. Banzhaf power        — _probabilistic_banzhaf()
Step 10. Metric ranking      — _rank_metrics()                 ← Q6.2
```

---

## 當前 Benchmark 結果（基準）

**跑測：** gabanzhaf，`--length 20`，現行程式（call graph nrows=10000）  
**重要前置條件：** 每次 `main.py` 執行前必須 `rm -f output/results/*.json`。結果檔名為 `{service}_{metric}_{case}.json`（不含 method/dataset 前綴），評估階段 glob 全部 `*.json`，殘留檔會汙染數字。  
**數值已驗證等同 committed `benchmark_results.json`；nrows 50000→10000 對全部 72 格指標零變動。**

### AC@1

| Dataset | CPU | MEM | DISK | SOCKET | DELAY | LOSS |
|---------|----:|----:|-----:|-------:|------:|-----:|
| re2-ob  | 0.40 | 0.73 | 0.27 | 0.27 | **0.27** | **0.40** |
| re2-ss  | 0.27 | 0.80 | 0.53 | 0.33 | **0.47** | **0.33** |
| re2-tt  | 0.73 | 1.00 | 1.00 | 0.73 | **0.60** | **0.53** |

### DELAY / LOSS 的 AC@3 / AC@5

| Dataset | AC@3-DELAY | AC@5-DELAY | AC@3-LOSS | AC@5-LOSS |
|---------|-----------:|-----------:|----------:|----------:|
| re2-ob  | 0.80 | 0.93 | 0.87 | 1.00 |
| re2-ss  | 0.93 | 0.93 | 1.00 | 1.00 |
| re2-tt  | 0.67 | 0.80 | 0.67 | 0.73 |

### Avg@5

| Dataset | CPU | MEM | DISK | SOCKET | DELAY | LOSS |
|---------|----:|----:|-----:|-------:|------:|-----:|
| re2-ob  | 0.88 | 0.93 | 0.85 | 0.85 | **0.72** | **0.83** |
| re2-ss  | 0.85 | 0.96 | 0.91 | 0.85 | **0.80** | **0.84** |
| re2-tt  | 0.95 | 1.00 | 1.00 | 0.89 | **0.69** | **0.64** |

### 執行時間（clean run）

| Dataset | Avg speed（s/case） | 對照 baro |
|---------|--------------------:|----------:|
| re2-tt  | **0.36**（wall 36.8s）| 0.34（35.3s）|
| re2-ob  | 0.10 | 0.07 |
| re2-ss  | 0.06 | 0.08 |

---

## 已解決事項

### S1. re2-tt 執行時間（前身 Q2）— 已解決
69s → 53.7s（onset 向量化）→ **36.8s（call graph nrows 50000→10000）**，逼近 baro 35.3s。
- `_first_sustained_onset`：雙層 Python 迴圈改 `numpy.lib.stride_tricks.sliding_window_view`，輸出逐位元等價。
- `_extract_call_graph`：`nrows` 50000→10000（`gabanzhaf.py:379`，env `GABANZHAF_CG_NROWS` 可覆寫），全部 72 格指標零變動。
- 54 tests pass。

### S2. Benchmark 重現性（前身 Q5 / Q7 / 資料快照誤判）— 已解決
先前觀察到的「AC@1 全面變低 / baro 0.47→0.16 / 資料快照不符」**全是 `output/results` 殘留汙染**，非程式退步、非資料問題。清空 output 後：baro re2-tt 重現作者 Table 6（Avg@5-CPU 0.72），gabanzhaf 重現 `benchmark_results.json`（DELAY AC@1 0.60）。
- 操作規範：每次跑前 `rm -f output/results/*.json`。
- baro 的 Table 6 數字為對照基準，不變動。

### S3. trace I/O 是否需要 case 快取（前身 Q4）— 已解決（排除）
單趟 benchmark 每個 case 只讀一次，快取命中率為 0。真正槓桿是 call-graph nrows（見 S1）。

### S4. 注入服務 SLI 是否影響結果（前身 Q6.6 / Exp1）— 已釐清
`main.py:240-241` 把 SLI 覆寫成 `{注入服務}_latency`。清空 output 後實測：關閉 gabanzhaf 的 SLI seed，re2-ob/re2-ss/re2-tt 全部 benchmark 指標逐項相同。**該 seed 對結果 inert**，且已於 2026-06-10 16:27 CST 從 gabanzhaf candidate construction 移除；不修改共用的 `main.py`。

> **[風險]** 雖然結果 inert，程式路徑仍消費了注入服務資訊；嚴格合規應移除。
>
> **[回覆]** 已直接移除 gabanzhaf 的 SLI seed，不保留環境變數分支。三個 RE2 dataset 的 clean benchmark 均與 committed baseline 逐項相同。

---

## 未解決事項

### U0.（核心）DELAY / LOSS 的 AC@1 / Avg@5 仍遠低於目標
re2-tt DELAY AC@1 **0.60**、Avg@5 **0.69**；LOSS AC@1 **0.53**、Avg@5 **0.64**。目標為各 Avg@5 ≥ 0.80（待最終目標確認）。U1/U2 已實作但 aggregate benchmark 未變；U3/U4 的三種 trace fusion ablation 因跨 fault regression 被拒絕。以下保留各項診斷、實作結果與未解風險。

### U1. Step 6 candidate selection 在固定 K=12 前淘汰有效候選（前身 Q6.1）
`_select_exact_candidates` 以 `0.50*metric + 0.40*residual + 0.05*trace + 0.05*graph` 排序後截 K=12，根因即使被帶入 candidate pool 仍被淘汰。

> **[解法]** 改用多來源 Reciprocal Rank Fusion：`Σ_source w_source / (c + rank_source(s))`，缺來源者以該來源服務數加一作為 rank。第一組 `c=10`、四來源權重各 1.0 為**待調參**。對應前身 Experiment 2。

> **[Codex 建議｜2026-06-10 15:51 CST]** RRF 可行，但應直接取代 `_select_exact_candidates` 的固定加權公式，不要在舊公式之外再增加一層排序。RRF 只保留 `metric rank`、`residual rank`、`trace rank` 三個互補來源；移除 `graph participation rank`，因為服務參與較多 edge 可能只代表它是高流量 hub，不代表更接近根因。Topology 的責任維持為 residual suppression 與 Banzhaf propagation，不再重複作為候選排名票源。

> **[實作結果｜2026-06-10 16:27 CST]** 已完成。Step 6 改為 `Σ 1/(10+rank_source)`，來源固定為 metric/residual/trace；移除 graph participation。Metric top-k 與 trace top-k 共同 seed，Step 5 不再以 metric-only cap 提前淘汰 rescued candidate，唯一計算邊界保留在 Step 6 的 K=12。三個 RE2 dataset 的 72 格 aggregate 指標均與 baseline 相同，表示此修改修正候選資料流，但目前沒有改變最終 benchmark 名次。

### U2. Step 10 `_rank_metrics` 把 K=12 最低 Banzhaf 候選歸零（前身 Q6.2）
`nb = _norm_dict(finite_banzhaf)` 把最低候選正規化為 0，與未入選服務（`nb.get(s,0)=0`）失去區別，根因名次出現斷崖。

> **[解法]** 保序有下界轉換：`banzhaf_floor + (1-banzhaf_floor)*minmax(banzhaf)`，K=12 內候選用 `banzhaf_floor>0`，K=12 外維持 0。第一組 `banzhaf_floor=0.10` 為**待調參**。對應 Experiment 3。

> **[Codex 建議｜2026-06-10 15:51 CST]** 不新增 `banzhaf_floor` 與待調參權重。直接把 K=12 候選的正規化改為 `normalized_banzhaf(s) = banzhaf(s) / max_banzhaf`；當 `max_banzhaf <= 0` 或全部候選同分時，沿用現有的 metric evidence 排序。這個轉換保序、正值不會因 min-max 被強制歸零，K=12 外服務仍維持 0，可直接取代 `_norm_dict(finite_banzhaf)`。

> **[實作結果｜2026-06-10 16:27 CST]** 已完成。正值候選不再因 min-max 被強制歸零；空集合與全同分情況維持 metric evidence 排序。三個 RE2 dataset aggregate 指標與 baseline 相同，未產生 regression，也尚未提升 DELAY/LOSS。

### U3. trace error/latency evidence 未進入核心 game（前身 Q6.3）
`anomaly_weights` 只用 `svc_anom_dict`（metric），已擷取的 `trace_error_scores`/`trace_latency_scores` 未使用。

> **[解法]** anomaly weight 改為 `w_metric*metric + w_trace_residual*trace_residual + w_cross*sqrt(metric*trace)`，其中 `trace_residual = trace_score - max 上游 trace 解釋`。第一組 `0.60/0.25/0.15` 為**待調參**。對應 Experiment 4。

> **[Codex 建議｜2026-06-10 15:51 CST]** 與 U4 合併，不新增 `trace_residual`、cross term 與三個權重。先建立唯一的聯合服務證據 `service_evidence(s) = max(metric_evidence(s), trace_evidence(s))`，再用同一份 `service_evidence` 取代 Banzhaf `anomaly_weights` 的 metric-only 輸入。RRF 內仍保留 metric rank 與 trace rank 兩個來源，避免在融合前丟失來源互補性；Banzhaf game 則只接收一份聯合 evidence，避免重複加權。
>
> **[風險｜Claude 已驗證 2026-06-10 16:01 CST]** `max(metric, trace)` 取消了原解法 `trace_residual` 對 victim 的抑制。端到端 latency 會沿 call path 傳到 victim，`max` 不扣除上游解釋，故傳染性高 latency 的 victim 會獲得偏高 anomaly weight。實測（re2-tt 12 個 DELAY/LOSS case，以 `_extract_trace_features` 的 combined trace score 為準）：平均每 case 有 **1.0 個非根因服務 trace ≥ 0.5**，fan-out LOSS（`ts-order-service_loss/2`、`/3`）達 **3 個**。原本由 `trace_residual` 擋下的這批 victim，改成 `max` 後完全交由 Step 9 Banzhaf game 與 Step 7 residual 抑制。
>
> **[回覆｜Claude]** 不否決 `max` 方案——同一實測也顯示 `max` 會救回弱 metric 根因（`ts-auth-service_loss/2`：root metric rank 53 → trace rank 1；`/delay/2`、`/3`：metric rank 29/28 → trace rank 11/4），這正是 U3 要的。但因為 victim 抑制的責任被整批移交給下游，採用 `max` 時**必須在 case-level diagnostics 加一欄「非根因且 service_evidence ≥ root 的服務數」**，並要求接上 U3/U4 後重跑時，DELAY/LOSS 的 AC@1 不得因 victim 自排而下降；若下降，再回頭引入 `trace_residual = trace − max 上游 trace 解釋` 作為 anomaly weight 的輸入（只在此處，不擴散成多權重公式）。此為待驗證項，不是預設關閉。
>
> **[風險｜Claude 程式事實]** Codex 命名的 `service_evidence` 與現有函式 `_service_evidence()`（`gabanzhaf.py:554`）及其輸出 `svc_anom_dict` 撞名。實作前須明定：聯合 evidence 是**取代** `svc_anom_dict` 在 anomaly_weights / call graph gate 的角色（同一個量改定義），而非新增第二個服務分數，否則 Step 3 與 Step 8/9 會出現兩套服務證據。建議直接重定義 `_service_evidence` 的輸出為 `max(metric, trace)`，全管道單一來源。
>
> **[實驗結果｜2026-06-10 16:27 CST]** U3 未採用。已依序測試：
>
> 1. `max(metric, raw_trace)` 同時接入 call-graph gate、causal alignment、Banzhaf weights：re2-tt DELAY/LOSS Avg@5 升至 `0.71/0.73`，但 re2-ob CPU/DISK/SOCKET/DELAY Avg@5 降至 `0.80/0.77/0.81/0.61`。
> 2. 只把 `max(metric, raw_trace)` 接入 Banzhaf weights：re2-tt 升至 `0.71/0.79`，但 re2-ob CPU/DISK/SOCKET/DELAY 降至 `0.80/0.80/0.83/0.61`。
> 3. Banzhaf weights 改用 `max(metric, trace_residual)`：re2-tt 為 `0.71/0.75`，re2-ob 仍為 `0.80/0.80/0.83/0.61`。
>
> 三種方案都超過非目標 fault Avg@5 下降 0.02 的 regression budget，因此 production code 已撤回 trace/game fusion。現行程式只保留 raw trace 的 candidate rescue 與 diagnostics；Banzhaf anomaly weights 維持 metric-only。

### U4. call graph edge 被 metric-only gate 擋掉（前身 Q6.4）
edge gate 用 `callee metric>=0.2、caller metric>=0.1`；低 metric 但 trace 異常的根因邊進不了 topology。

> **[解法]** 先驗證 graph extraction coverage（10K 取樣是否已含該 edge），再把 gate 的 evidence 來源改為 `max(metric_score, trace_score)`，threshold 沿用 0.2/0.1。對應 Experiment 5。

> **[Codex 建議｜2026-06-10 15:51 CST]** 保留「先驗證 10K graph extraction coverage」的步驟；確認 edge 已被擷取後，直接使用 U3 定義的 `service_evidence = max(metric_evidence, trace_evidence)` 取代 call graph gate 的 metric-only evidence，threshold 暫時沿用 `0.2/0.1`。U3 與 U4 因此是一次資料流替換：同一份聯合 evidence 同時供 call graph gate 與 Banzhaf 使用，不建立兩套融合公式。
>
> **[風險｜Claude 已驗證 2026-06-10 16:01 CST]** 「threshold 暫時沿用 0.2/0.1」不是中性沿用，而是單調放寬 gate。因 `max(metric, trace) ≥ metric` 恆成立，通過 gate 的端點集合是原本（metric-only）的**超集**。實測（re2-tt 12 個 DELAY/LOSS case，現行 `svc_anom_dict` 對比 `max(svc_anom_dict, combined trace)`）：以原 threshold，平均每 case **新增 +1.3 個 callee（≥0.2）、+1.4 個 caller（≥0.1）** 通過，fan-out LOSS case 達 +4。更多端點通過 → 更多 call graph edge 進入 topology → Step 7 residual suppression 的誤抑制面變大，正是前身 Q6.4 原 `[風險]` 指出的問題。
>
> **[回覆｜Claude]** 換 evidence 來源後 threshold 不可假設等價，需重新校準。具體做法：在接上 U3/U4 的同一次實驗中，輸出每個 case 的「gate 通過 edge 數（metric-only vs max）」與「被這些 edge 觸發 suppression 的服務數」，若 root 因新增 edge 被誤抑制（root residual 下降且 final rank 退步），則對 trace 端點採用較高 threshold（如 callee 0.35 / caller 0.2 為**待調參**），而非與 metric 共用 0.2/0.1。先量測再決定，不預設沿用。
>
> **[實驗結果｜2026-06-10 16:27 CST]** U4 未採用。聯合 gate 與 raw trace Banzhaf weights 的組合雖改善 re2-tt DELAY/LOSS，但造成 re2-ob 多個 fault regression；只關閉聯合 gate 仍未消除 regression，證明主要問題是 raw trace anomaly weight 的 victim 汙染，而非單獨的 gate threshold。Call-graph gate 維持 metric-only，後續若重啟 U4，必須先有可驗證的 trace victim suppression，不直接調高 threshold 補償。

### U5. Probabilistic Banzhaf 被 self-explanation 主導（前身 Q6.5）
`_causal_explanation_matrix` 對角線固定 1.0，topology 弱時 Banzhaf 退化成 metric 排名。

> **[解法]** 把 payoff 拆成 `local_weight*local_coverage + propagated_weight*downstream_coverage`，做 `1.0/1.0 → 0.75/1.0 → 0.50/1.0` ablation。**接受條件：CPU/MEM/DISK/SOCKET 任一 Avg@5 下降不超過 0.02**。對應 Experiment 6。

> **[Codex 建議｜2026-06-10 15:51 CST]** 暫不實作。先完成 U1 的候選選擇替換、U2 的正規化修正，以及合併後的 U3/U4 evidence 接線，再重新產生 case-level diagnostics。只有在 topology 與 trace 已確實進入 game 後，仍觀察到 Banzhaf rank 幾乎等同 metric rank，才進行 local/propagated ablation；否則降低 self-explanation 可能只是在補償上游 evidence 未接通，並額外增加權重與 regression 風險。

### U6. 其餘開放項
- **U6a（前身 Q1）**：第 1 次實作後 re2-tt SOCKET / DELAY 的 AC@3 曾退步，need 在乾淨基準上重核。
- **U6b（前身 Q3）**：trace onset 尚未真正納入（目前只有 metric onset）。
- **U6c（前身 Q8）— 已解決**：SLI seed 已移除；re2-ob/re2-ss/re2-tt clean benchmark 全部逐項等同 baseline。

---

## 結論

障礙與基準問題已清除：執行時間 ≈ baro、重現性根因（output 汙染）已定位、注入服務 SLI seed 已移除。U1/U2 已以替換方式實作，沒有增加平行 ranking pipeline，也沒有造成 regression；但三個 RE2 dataset aggregate 指標仍等同 baseline。

U3/U4 的 `max(metric, trace)`、raw trace weights、trace residual weights 均未通過 regression budget。下一步不應繼續增加融合權重或調高 gate threshold，而應先回答：為何現有 topology 無法把 re2-ob 的 trace victim 從 anomaly weights 中扣除。建議下一輪只做 diagnostics：逐 case 記錄 raw trace rank、trace residual rank、實際 upstream edge、edge confidence、root/victim residual 變化；確認缺陷是 call-graph coverage、edge direction，或 trace score 本身缺乏 root/victim 可識別性後，再選擇單一替代算法。

每個實驗至少記錄：root 的 metric/trace/residual rank、是否進 candidate pool、是否進 K=12、Banzhaf rank、final rank，並同時看 AC@1/3/5 與 Avg@5，確保 CPU/MEM/DISK/SOCKET 不退步超過 0.02。

---

## 相關檔案

- `RCAEval/e2e/gabanzhaf.py`：主管道
- `benchmark_results.json`：基準數據（gabanzhaf 自跑、baro 為作者 Table 6）
- `docs/issues/2026-06-03-delay-loss-low-accuracy.md`：完整診斷與三次實作紀錄
- `README.md`：作者 Table 6（baro Train Ticket Avg@5 對照）
