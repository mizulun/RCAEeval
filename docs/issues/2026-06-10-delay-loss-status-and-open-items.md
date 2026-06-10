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
`main.py:240-241` 把 SLI 覆寫成 `{注入服務}_latency`，gabanzhaf 於 `gabanzhaf.py:1075-1076` 把它 seed 進 candidate。清空 output 後實測：以 `GABANZHAF_NO_SLI_SEED=1` 關閉 seed，re2-tt 逐項相同（DELAY 0.60、LOSS 0.53）。**該 seed 對結果 inert**——seed 進 candidate pool 後在 Step 6 的 K=12 trim 被丟棄。

> **[風險]** 雖然結果 inert，程式路徑仍消費了注入服務資訊；嚴格合規應移除。
>
> **[回覆]** 把 `GABANZHAF_NO_SLI_SEED=1` 設為預設（移除 seed），re2-tt 零影響；不動 `main.py`（與 baro 等對照共用）。套用前先以清空 output 的對照確認 re2-ob/re2-ss 同樣 inert（對應未解 U5）。

---

## 未解決事項

### U0.（核心）DELAY / LOSS 的 AC@1 / Avg@5 仍遠低於目標
re2-tt DELAY AC@1 **0.60**、Avg@5 **0.69**；LOSS AC@1 **0.53**、Avg@5 **0.64**。目標為各 Avg@5 ≥ 0.80（待最終目標確認）。S1–S4 都是清除障礙與基準校正，**尚未對排名演算法做任何提升**。以下 U1–U5 是已診斷、未實作的瓶頸（細節與 risk/回覆見前身 issue Q6.1–6.5）。

### U1. Step 6 candidate selection 在固定 K=12 前淘汰有效候選（前身 Q6.1）
`_select_exact_candidates` 以 `0.50*metric + 0.40*residual + 0.05*trace + 0.05*graph` 排序後截 K=12，根因即使被帶入 candidate pool 仍被淘汰。

> **[解法]** 改用多來源 Reciprocal Rank Fusion：`Σ_source w_source / (c + rank_source(s))`，缺來源者以該來源服務數加一作為 rank。第一組 `c=10`、四來源權重各 1.0 為**待調參**。對應前身 Experiment 2。

### U2. Step 10 `_rank_metrics` 把 K=12 最低 Banzhaf 候選歸零（前身 Q6.2）
`nb = _norm_dict(finite_banzhaf)` 把最低候選正規化為 0，與未入選服務（`nb.get(s,0)=0`）失去區別，根因名次出現斷崖。

> **[解法]** 保序有下界轉換：`banzhaf_floor + (1-banzhaf_floor)*minmax(banzhaf)`，K=12 內候選用 `banzhaf_floor>0`，K=12 外維持 0。第一組 `banzhaf_floor=0.10` 為**待調參**。對應 Experiment 3。

### U3. trace error/latency evidence 未進入核心 game（前身 Q6.3）
`anomaly_weights` 只用 `svc_anom_dict`（metric），已擷取的 `trace_error_scores`/`trace_latency_scores` 未使用。

> **[解法]** anomaly weight 改為 `w_metric*metric + w_trace_residual*trace_residual + w_cross*sqrt(metric*trace)`，其中 `trace_residual = trace_score - max 上游 trace 解釋`。第一組 `0.60/0.25/0.15` 為**待調參**。對應 Experiment 4。

### U4. call graph edge 被 metric-only gate 擋掉（前身 Q6.4）
edge gate 用 `callee metric>=0.2、caller metric>=0.1`；低 metric 但 trace 異常的根因邊進不了 topology。

> **[解法]** 先驗證 graph extraction coverage（10K 取樣是否已含該 edge），再把 gate 的 evidence 來源改為 `max(metric_score, trace_score)`，threshold 沿用 0.2/0.1。對應 Experiment 5。

### U5. Probabilistic Banzhaf 被 self-explanation 主導（前身 Q6.5）
`_causal_explanation_matrix` 對角線固定 1.0，topology 弱時 Banzhaf 退化成 metric 排名。

> **[解法]** 把 payoff 拆成 `local_weight*local_coverage + propagated_weight*downstream_coverage`，做 `1.0/1.0 → 0.75/1.0 → 0.50/1.0` ablation。**接受條件：CPU/MEM/DISK/SOCKET 任一 Avg@5 下降不超過 0.02**。對應 Experiment 6。

### U6. 其餘開放項
- **U6a（前身 Q1）**：第 1 次實作後 re2-tt SOCKET / DELAY 的 AC@3 曾退步，need 在乾淨基準上重核。
- **U6b（前身 Q3）**：trace onset 尚未真正納入（目前只有 metric onset）。
- **U6c（前身 Q8）**：re2-ob/re2-ss 的 SLI seed 是否 inert 未驗證（清空 output 後可補測，連動 S4 的合規移除）。

---

## 結論

障礙與基準問題已清除：執行時間 ≈ baro、重現性根因（output 汙染）已定位、注入服務 SLI 對結果 inert。**目前是乾淨且可信的 baseline，但 DELAY/LOSS 排名準確率的演算法提升一項都還沒做。**

下一步切入點：在「每次清空 output」的乾淨流程上，依序執行單變數實驗——**先 U1（RRF 候選重建）**，因為它直接打 re2-tt 觀察到的 rank 斷崖（根因被 K=12 trim 淘汰），且不需改 fault-independent 的單一路徑。每個實驗至少記錄：root 的 metric/trace/residual rank、是否進 candidate pool、是否進 K=12、Banzhaf rank、final rank，並同時看 AC@1/3/5 與 Avg@5，確保 CPU/MEM/DISK/SOCKET 不退步超過 0.02。

---

## 相關檔案

- `RCAEval/e2e/gabanzhaf.py`：主管道
- `benchmark_results.json`：基準數據（gabanzhaf 自跑、baro 為作者 Table 6）
- `docs/issues/2026-06-03-delay-loss-low-accuracy.md`：完整診斷與三次實作紀錄
- `README.md`：作者 Table 6（baro Train Ticket Avg@5 對照）
