# Issue: GA-Banzhaf 消融實驗（Ablation Study）設計與待補元件

**日期：** 2026-06-17 00:00
**狀態：** 進行中
**影響範圍：** `RCAEval/e2e/gabanzhaf.py`（Step 3、Step 4、Step 9）、`run123.py`（`METHOD_VARIANTS`）、`benchmark_banzhaf_ablation.json`

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

---

## 當前流程（Pipeline Overview）

```
Step 1. Window split         — _split_raw_windows()
Step 2. Preprocess           — _preprocess_windows()
Step 3. Metric evidence      — _metric_evidence(), _service_evidence()
Step 4. Topology             — _build_topology(), _call_graph_topology_edges()
Step 5. Candidate selection  — _candidate_services()
Step 6. Exact candidates     — _select_exact_candidates()
Step 7. Residual scores      — _residual_scores()
Step 8. Causal matrix        — _causal_explanation_matrix()
Step 9. Banzhaf power        — _probabilistic_banzhaf()  (banzhaf_mode: closed_form | bruteforce)
Step 10. Metric ranking      — _rank_metrics()
```

**本次 issue 涉及步驟：** Step 3（metric 訊號）、Step 4（trace 呼叫圖）、Step 9（Banzhaf 計算模式）

**兩類證據來源（消融的兩個切入軸）：**
- **Metrics 訊號**：`_metric_evidence()` / `_service_evidence()`（Step 3），驅動候選排序與殘差。
- **Trace 訊號**：由 `_extract_trace_features()` 產生 `_TraceFeatures`，內含兩種互相獨立的子訊號——
  - `trace_scores` / `trace_error_scores` / `trace_latency_scores`：trace onset 異常分數（Step 3 種子，`_trace_seed_services()`）。
  - `call_graph`：服務呼叫圖拓樸（Step 4，`_call_graph_topology_edges()`、`_call_graph_residual_allowed()`）。

---

## 當前 Benchmark 結果

> 對照基準。`benchmark_banzhaf_ablation.json` 已含 `approx_banzhaf`（closed_form）在 re2-ob / re2-ss 的初步數據；其餘待補。

**跑測日期：** 待補（重跑後填入）
**方法：** gabanzhaf（closed_form baseline）

### AC@1

| Dataset | AC@1-CPU | AC@1-MEM | AC@1-DISK | AC@1-SOCKET | AC@1-DELAY | AC@1-LOSS |
|---------|---------|---------|---------|------------|-----------|----------|
| re2-ob  | 0.40    | 0.73    | 0.27    | 0.27       | 0.27      | 0.40     |
| re2-ss  | 0.27    | 0.80    | 0.53    | 0.33       | 0.47      | 0.33     |
| re2-tt  | 待補     | 待補     | 待補     | 待補        | 待補       | 待補      |

（Avg@5 等其餘指標於正式重跑後補齊，沿用 benchmark_banzhaf_ablation.json 的欄位。）

---

## 當前問題

### 症狀

論文需要消融實驗，量化（a）我們的近似 Banzhaf 相對純枚舉 Banzhaf 的準確率/時間取捨，以及（b）trace 與 metrics 兩類證據各自的貢獻。目前只有 Banzhaf 模式（Step 9）有開關，trace 與 metrics 來源（Step 3 / Step 4）尚無消融開關，無法單獨關閉。

### 瓶頸定位

#### Step 9 — Banzhaf 模式開關 — 已存在

**程式位置**：`gabanzhaf.py:752-756`、`run123.py:48-83`

```python
if config.banzhaf_mode == "closed_form":
    ...
if config.banzhaf_mode == "bruteforce":
    ...
raise ValueError(f"Unsupported banzhaf_mode: {config.banzhaf_mode}")
```

`run123.py` 已定義 `gabanzhaf_approx_banzhaf` / `gabanzhaf_closed_form`（closed_form）與 `gabanzhaf_pure_banzhaf` / `gabanzhaf_bruteforce`（bruteforce），透過 `GABANZHAF_BANZHAF_MODE` 環境變數注入。**消融 1 的機制已備齊**，只需補 re2-tt 與全指標的完整數據。

#### Step 3 / Step 4 — trace 與 metrics 來源無消融開關 — 待補

**程式位置**：`gabanzhaf.py:236-351`（`_extract_trace_features`）、`gabanzhaf.py:494-556`（`_metric_evidence` / `_service_evidence`）

目前 `_extract_trace_features()` 只在檔案不存在時才回傳全 `None` 的 `_TraceFeatures`；無法用設定強制關閉。metrics 證據同理無開關。**消融 2、消融 3 缺少注入點。**

---

## 解法

### 1. 消融 1 — 純 Banzhaf vs 近似 Banzhaf（準確率 + 執行時間）

直接使用既有變體執行配對比較：

```
python run123.py --methods gabanzhaf_approx_banzhaf gabanzhaf_pure_banzhaf \
    --re2 --workers 1 --output benchmark_banzhaf_ablation.csv
```

- `approx_banzhaf` = `banzhaf_mode=closed_form`（我們的機率因果 Banzhaf 閉式近似）。
- `pure_banzhaf` = `banzhaf_mode=bruteforce`（對 `_select_exact_candidates()` 選出的候選集合做 2^k 子集枚舉的精確 Banzhaf）。
- 比較欄位：所有 AC@1/AC@3/AC@5、Avg@5，以及 `wall_time_sec`（每資料集牆鐘時間）。`--workers 1` 確保時間量測不受並行干擾。
- 輸出表需同時呈現「準確率差異（pure − approx）」與「時間倍率（pure / approx）」，論證閉式近似在準確率損失 ≤ 待調參門檻下換取的加速。

> **[風險]** bruteforce 在候選數 k 較大時為 2^k 級複雜度，re2-tt 上可能極慢或記憶體爆。
>
> **[回覆]** bruteforce 僅作用於 `_select_exact_candidates()` 輸出的 k 個候選（`exact_candidate_k` 預設 12，見 `gabanzhaf.py:36`）。先以 `exact_candidate_k` ∈ {8, 10, 12}（待調參）量測時間曲線；若 re2-tt 在 k=12 超過 600s，於消融表中標注該格為「逾時」並以較小 k 補點，不更動 baseline 預設值。

### 2. 消融 2 — 有無 trace 資訊（準確率比較）

新增環境變數 `GABANZHAF_DISABLE_TRACE`，在 `_GabanzhafConfig` 增 `use_trace: bool = True` 欄位，並於 `gabanzhaf.py:1254` 附近的 env 讀取區塊（與既有 `GABANZHAF_BANZHAF_MODE` 同位置）解析。當 `use_trace=False` 時，於 `_extract_trace_features()` 入口直接回傳：

```python
_TraceFeatures(trace_scores=None, trace_error_scores=None,
               trace_latency_scores=None, call_graph={})
```

如此同時關閉 trace onset 種子（Step 3）與呼叫圖拓樸（Step 4）。`run123.py` 新增變體 `gabanzhaf_no_trace`（`env={"GABANZHAF_DISABLE_TRACE": "1"}`，`metadata.ablation_role="no_trace"`）。比較對象為 closed_form baseline（`approx_banzhaf`）。

> **[風險]** trace 同時餵入兩個獨立子訊號（onset 分數與呼叫圖），一次全關無法區分各自貢獻，且 re2 部分資料集 trace 檔可能本就缺失，導致「有/無」差異被資料覆蓋率稀釋。
>
> **[回覆]** 拆成三個變體量測：(a) `no_trace_onset`（只清 `trace_*_scores`，保留 `call_graph`）、(b) `no_call_graph`（只清 `call_graph`，保留 onset 分數）、(c) `no_trace`（兩者全清）。對應在 `_GabanzhafConfig` 增 `use_trace_onset` 與 `use_call_graph` 兩個布林欄位分別控制。每資料集另記錄 trace 檔案存在比例，於表下註明覆蓋率，避免將缺資料誤判為消融效果。

### 3. 消融 3 — 有無 metrics 資訊（準確率比較）

新增環境變數 `GABANZHAF_DISABLE_METRICS`，在 `_GabanzhafConfig` 增 `use_metrics: bool = True`。當 `use_metrics=False` 時，`_metric_evidence()` 對所有欄位回傳中性證據（異常分數設為 0、IQR 分母維持既有非零最小值以避免除零），使候選排序與殘差完全改由 trace 訊號驅動。`run123.py` 新增變體 `gabanzhaf_no_metrics`（`metadata.ablation_role="no_metrics"`）。

> **[風險]** re2 資料集以 metrics 為主訊號，完全關閉 metrics 後管道可能退化到近乎隨機，AC@1 全面崩潰，使該消融只能得出「metrics 不可或缺」的平庸結論。
>
> **[回覆]** 此結論本身即論文要的對照（證明 metrics 為主導訊號）。為避免退化為無意義隨機，量測時固定亂數種子並回報相對隨機基準（1/服務數）的提升幅度；若全 0 證據導致候選選擇無法排序，明訂行為為「依服務名稱字典序穩定排序」，使結果可重現而非隨機。

### 4. 消融 4 — 排序層策略：blended / service_first / banzhaf_only（準確率比較）

原始最終 column 分數為兩項相乘：

```python
def _col_score(col):
    svc = _parse_svc(col)
    return nb.get(svc, 0.0) * (epsilon + nm.get(col, 0.0))
```

其中 `nb` = 正規化的 Banzhaf 服務分數、`nm` = 正規化的 metric-local 證據、`epsilon = metric_rank_epsilon`。此設計會讓 metric-local evidence 同時影響「服務之間」與「同服務內 metric 之間」的排序。`benchmark_banzhaf_ablation.json` 顯示 `rank_banzhaf_only` 在 re2-ss、re2-tt 對部分 fault type 優於原 baseline，代表 metric-local evidence 可能在服務層排序中引入干擾。

本次將預設排序改為 `service_first`：

```python
return sorted(
    all_cols,
    key=lambda col: (
        -nb.get(_parse_svc(col), 0.0),
        -nm.get(col, 0.0),
        col,
    ),
)
```

演算法意義：

1. 服務之間只由 Banzhaf service score 排序，避免 metric-local evidence 把低 Banzhaf 服務推到高 Banzhaf 服務之前。
2. 同一服務內仍由 metric-local evidence 排 metric，保留 metric-level AC@k 的區分能力。
3. 若 Banzhaf 分數全相同或沒有候選服務分數，`nb` 會全設為 1，此時排序自然退回 metric-local evidence 主導，保持無服務訊號時的可重現行為。

新增 / 保留三個 rank mode：

- `service_first`：新的預設 baseline。服務層用 Banzhaf，服務內用 metric evidence。
- `blended`：舊 baseline 策略，保留為 `gabanzhaf_rank_blended` 對照。
- `banzhaf_only`：完全移除 metric-local evidence，同服務內以 column 名稱字典序固定排序。

`run123.py` 對應變體：

- `gabanzhaf_approx_banzhaf`：使用新的 `service_first` 預設。
- `gabanzhaf_rank_blended`：`env={"GABANZHAF_RANK_MODE": "blended"}`，`metadata.ablation_role="rank_blended_previous"`。
- `gabanzhaf_rank_banzhaf_only`：`env={"GABANZHAF_RANK_MODE": "banzhaf_only"}`，`metadata.ablation_role="rank_banzhaf_only"`。

> **[風險]** `service_first` 可能降低少數 metric-level 指標，因為低 Banzhaf 服務中的高 metric-local evidence 不再能跨服務提前。
>
> **[回覆]** 這是預期取捨：根因服務排序優先於跨服務 metric 局部分數。若論文表格同時報 service-level 與 metric-level AC@k，需將 `service_first`、`blended`、`banzhaf_only` 三者並列，說明 `service_first` 是折衷策略：服務層不受 metric evidence 干擾，同服務內仍保留 metric 區分。

### 5. 消融 5 — 候選集合大小 `exact_candidate_k` 敏感度（準確率 + 執行時間）

量化 Step 6 預過濾（`_select_exact_candidates()`,`gabanzhaf.py:949`）保留候選數對結果的影響。`exact_candidate_k` 決定進入 Banzhaf 賽局的候選數,直接左右 bruteforce 的 2^k 成本與 closed_form 的矩陣規模。

新增環境變數 `GABANZHAF_EXACT_CANDIDATE_K`,在 env 區塊（`gabanzhaf.py:1254` 附近）解析後注入 `_GabanzhafConfig.exact_candidate_k`（欄位已存在,`gabanzhaf.py:36`,預設 12）。`run123.py` 新增掃描變體 `gabanzhaf_k6` / `gabanzhaf_k8` / `gabanzhaf_k10` / `gabanzhaf_k12` / `gabanzhaf_k16`（`env={"GABANZHAF_EXACT_CANDIDATE_K": "<k>"}`,`metadata.ablation_role="exact_candidate_k=<k>"`），k ∈ {6, 8, 10, 12, 16}（待調參）。輸出準確率與 `wall_time_sec` 隨 k 的曲線,論證預過濾在準確率幾乎持平下對時間的節省。

> **[風險]** k 太小時真正根因可能在 Step 6 被剔除而落到候選集外,使準確率驟降;k 與 bruteforce 模式（消融 1）耦合,雙重掃描組合數爆量。
>
> **[回覆]** 為每個 k 記錄「根因落在候選集內的比例」（recall@candidates）作為輔助欄位,將準確率下降明確歸因於候選漏選而非排序。本消融固定 `banzhaf_mode=closed_form` 單獨掃 k;k × bruteforce 的交叉僅在消融 1 的時間分析中以 re2-tt 單資料集補點,不做全資料集全 fault type 笛卡兒積。

### 6. 消融 6 — 拓樸/殘差修正 on/off（準確率比較）

驗證呼叫圖拓樸與殘差約束（Step 4 `_call_graph_topology_edges()`、Step 7 `_residual_scores()` / `_call_graph_residual_allowed()`）對抑制「受害者誤判」的貢獻。此消融與消融 2 的 `use_call_graph` 不同:消融 2 關閉的是 trace 呼叫圖「來源」,本消融關閉的是「拓樸/殘差修正邏輯本身」（即使有 lag-corr 推得的邊也不套用殘差約束）。

新增環境變數 `GABANZHAF_DISABLE_RESIDUAL`,在 `_GabanzhafConfig` 增 `use_residual: bool = True`。當 `use_residual == False` 時,`_residual_scores()` 直接回傳原始 `svc_anom_dict`（不套用 `upstream_map_residual` 與 `residual_call_graph_ratio`,`gabanzhaf.py:33` 的呼叫圖約束),使候選排序不受拓樸抑制。`run123.py` 新增變體 `gabanzhaf_no_residual`（`metadata.ablation_role="no_residual"`）。預期關閉後上游受害者服務的排名上升,AC@1 在具明確呼叫鏈的 fault type（DELAY/LOSS/SOCKET）下降。

> **[風險]** `_residual_scores()` 的輸出同時被 `_select_exact_candidates()`（Step 6）與最終排序使用,直接回傳原始異常分數可能改變候選集合組成,使「拓樸修正」與「候選選擇」兩個效果混淆。
>
> **[回覆]** 分兩段量測:(a) `no_residual_rank`——候選選擇仍用殘差,僅最終排序改用原始異常分數,隔離排序層效果;(b) `no_residual_full`——候選與排序皆用原始分數,量測整體效果。兩者差值即「殘差對候選選擇」的貢獻,使效果可分離歸因。

### 7. 其他建議的消融項目

針對論文完整性，建議再補以下消融（依預期貢獻度排序，皆透過 `_GabanzhafConfig` 加布林/數值開關 + `run123.py` 變體實作）：

1. **因果解釋矩陣 on/off**（Step 8）：以原始證據分數直接排序、跳過 `_causal_explanation_matrix()`，量測因果修正帶來的增益。
2. **Metric ranking epsilon 敏感度**（Step 10）：`metric_rank_epsilon` ∈ {0.05, 0.10, 0.15, 0.25}（`gabanzhaf.py:39`，待調參），對 DELAY/LOSS 這類弱訊號 fault type 的 AC@3/AC@5 影響。
3. **Trace onset 三分量分解**：分別只用 error、只用 latency、兩者合併，定位哪一種 trace 分量對 SOCKET/DELAY/LOSS 最關鍵。
4. **Phenotype-adaptive 模組 on/off**：對照 spec `docs/superpowers/specs/2026-05-31-phenotype-adaptive-gabanzhaf-design.md`，若該自適應邏輯已落地，需消融驗證其對各 fault type 的差異化貢獻。

> **[風險]** 消融組合過多（變體 × 3 資料集 × 6 fault type）會使實驗時間與表格爆量，難以在論文八頁內呈現。
>
> **[回覆]** 主文放消融 1–6 的主結果表；第 7 節的敏感度分析（epsilon）與第 5 節的 k 掃描以單一資料集（re2-tt，服務數最多、最具代表性）的曲線圖呈現，其餘移入附錄。每個變體沿用 `run123.py` 既有的 `ablation_role` metadata 欄位標記，統一輸出至 `benchmark_*_ablation.csv/json` 供匯總。

---

## 結論

消融 1（Banzhaf 模式）的執行機制已完整（`banzhaf_mode` + `run123.py` 變體 + `benchmark_banzhaf_ablation.json`），只缺 re2-tt 與全指標的完整數據與時間配對表。消融 2、3（trace / metrics）為演算法瓶頸——目前無法單獨關閉這兩類證據來源，需在 `_GabanzhafConfig` 新增 `use_trace_onset` / `use_call_graph` / `use_metrics` 布林開關，並於 env 讀取區塊（`gabanzhaf.py:1254` 附近）解析、在 `run123.py` 補對應變體。消融 4（只用 Banzhaf 分數排序）為純排序層開關（`rank_mode`），驗證 metric-local 證據對 metric-level AC@k 的貢獻。消融 5（`exact_candidate_k` 敏感度）與消融 6（拓樸/殘差修正 on/off）分別量化 Step 6 預過濾與 Step 4/7 拓樸抑制的貢獻，皆需新增對應 config 開關（`exact_candidate_k` 已存在欄位、新增 `use_residual`）。其餘因果矩陣、epsilon、trace 分量等列入第 7 節附錄。主文聚焦六大消融（1–6）。

---

## 實作結果格式規範

每次實作完成後，在「實作結果紀錄」區塊新增一筆，格式如下：

```
### 第N次實作 — YYYY-MM-DD HH:MM

**實作內容：** 本次實作了哪些解法步驟（對應「解法」章節編號）

**Benchmark 對照：**
（表格：baseline vs 實作後，標注退步項目）

**演算法執行時間：**
（表格：dataset、實作前、實作後、baro 參考值）

**是否解決 issue：** 已解決 / 部分解決 / 未解決

**後續問題：**
（格式：Q編號. 問題描述，問題下方直接寫根本原因分析）
```

---

## 實作結果紀錄

### 第1次實作 — 2026-06-17 03:21

**實作內容：** 已完成解法 1–6 的執行開關與 `run123.py` 變體註冊。包含：
- `GABANZHAF_BANZHAF_MODE=closed_form|bruteforce`：沿用既有 Banzhaf 模式切換。
- `GABANZHAF_DISABLE_TRACE_ONSET=1`、`GABANZHAF_DISABLE_CALL_GRAPH=1`、`GABANZHAF_DISABLE_TRACE=1`：分別關閉 trace onset、call graph、兩者全關。
- `GABANZHAF_DISABLE_METRICS=1`：`_metric_evidence()` 對所有 metric 回傳中性證據，分數皆為 0。
- `GABANZHAF_RANK_MODE=banzhaf_only`：最終 metric 排序只使用服務層 Banzhaf 分數，服務內以 column 名稱字典序固定排序。
- `GABANZHAF_EXACT_CANDIDATE_K=<k>`：注入 `exact_candidate_k`，`run123.py` 已註冊 k ∈ {6, 8, 10, 12, 16}。
- `GABANZHAF_DISABLE_RESIDUAL=rank|full|1`：`rank` 註冊為 `no_residual_rank` 變體；`full`/`1` 使 `_residual_scores()` 回傳原始 service anomaly 分數，候選選擇不套用殘差抑制。
- `run123.py` 新增 `gabanzhaf_no_trace_onset`、`gabanzhaf_no_call_graph`、`gabanzhaf_no_trace`、`gabanzhaf_no_metrics`、`gabanzhaf_rank_banzhaf_only`、`gabanzhaf_k6`、`gabanzhaf_k8`、`gabanzhaf_k10`、`gabanzhaf_k12`、`gabanzhaf_k16`、`gabanzhaf_no_residual`、`gabanzhaf_no_residual_rank`、`gabanzhaf_no_residual_full`。

**Benchmark 對照：**

| 項目 | baseline | 實作後 |
|------|----------|--------|
| 消融 1：Banzhaf 模式 | `gabanzhaf_approx_banzhaf` / `gabanzhaf_pure_banzhaf` 已存在 | 保留並通過變體解析測試 |
| 消融 2：trace | 無法單獨關閉 | 可分別關閉 onset、call graph、全 trace |
| 消融 3：metrics | 無法關閉 | 可回傳中性 metric evidence |
| 消融 4：rank mode | 只支援 blended | 支援 `banzhaf_only` |
| 消融 5：candidate k | config 欄位存在但 env 無注入 | 支援 env 注入與 k 掃描變體 |
| 消融 6：residual | 無變體 | 支援 `rank_only` 標記與 `full` 關閉候選殘差抑制 |

正式 AC@k / Avg@5 benchmark 尚未重跑；本次僅完成可執行消融機制與單元驗證。

**演算法執行時間：**

| dataset | 實作前 | 實作後 | baro 參考值 |
|---------|--------|--------|-------------|
| re2-ob | 待正式 benchmark | 待正式 benchmark | 待正式 benchmark |
| re2-ss | 待正式 benchmark | 待正式 benchmark | 待正式 benchmark |
| re2-tt | 待正式 benchmark | 待正式 benchmark | 待正式 benchmark |

**是否解決 issue：** 部分解決。已完成所有消融項目的程式開關與 runner 變體；尚需執行完整 `run123.py` benchmark 以填入正式結果表。

**後續問題：**

Q1. 為什麼尚未填入正式 benchmark 數據？

本次實作先補齊消融注入點與變體註冊，避免 benchmark 指令只能跑既有 baseline。正式數據仍需執行 `python run123.py --methods ... --re2 --workers 1 --output benchmark_banzhaf_ablation.csv`，其中 bruteforce 與 k 掃描可能耗時較長。

Q2. `no_residual_rank` 與 `no_residual_full` 的差異如何解讀？

目前管道中的 residual score 只直接進入 `_select_exact_candidates()`，最終 `_rank_metrics()` 不直接讀取 residual score。因此 `no_residual_rank` 先作為可追蹤變體標記保留候選殘差選擇；`no_residual_full` 則實際關閉候選選擇階段的殘差抑制。若後續把 residual score 接入最終排序，`no_residual_rank` 可直接承接該排序層消融。

**驗證：**
- `python3.10 -m py_compile RCAEval/e2e/gabanzhaf.py run123.py tests/test_gabanzhaf_improvements.py tests/test_run123_variants.py`：通過。
- `python3.10 -m pytest tests/test_run123_variants.py -q`：5 passed。
- 直接載入 `RCAEval/e2e/gabanzhaf.py` 並檢查 `use_metrics=False`、`rank_mode=banzhaf_only`、`use_residual=False`：通過。
- `python3.10 -m pytest tests/test_gabanzhaf_improvements.py -k "metrics_disabled or banzhaf_only or residual_disabled or ablation_env or full_residual_disable or rank_only_residual" -q`：未完成；collection 會載入 `RCAEval/e2e/__init__.py` 的其他 RCA 方法，目前環境會因 `causalrca` 需要 `torch` 而中止。
- `python3.10 run123.py --methods gabanzhaf_no_metrics --datasets re2-ob --test --workers 1 --skip-failed --output /tmp/gabanzhaf_ablation_smoke.csv`：未完成；`main.py` 載入 `RCAEval.e2e` 時同樣因 `causalrca` 需要 `torch` 而中止，尚未進入 `gabanzhaf` 本體。

### 第2次實作 — 2026-06-17 16:35

**實作內容：** 根據已跑完的排序層消融結果，將預設 `rank_mode` 從 `blended` 改為 `service_first`。新增 `gabanzhaf_rank_blended` 變體保留舊策略作為對照，`gabanzhaf_rank_banzhaf_only` 繼續保留作為完全移除 metric-local evidence 的排序消融。

**Benchmark 對照：**

| Dataset | `rank_banzhaf_only` 相對舊 baseline 觀察 | 本次決策 |
|---------|------------------------------------------|----------|
| re2-ob | 24 個指標中 2 個退步，總差異 -0.08 | 不直接採用純 Banzhaf |
| re2-ss | 24 個指標中 2 個進步，總差異 +0.09 | 支持降低 metric evidence 跨服務影響 |
| re2-tt | 24 個指標中 9 個進步，總差異 +0.61 | 支持 Banzhaf service score 作為服務層主排序 |

本次沒有直接採用 `banzhaf_only` 作為預設，原因是該模式在同服務內沒有 metric 區分能力。新的 `service_first` 使用 Banzhaf 排服務，並只在同一服務內使用 metric evidence 排 metric。

**演算法執行時間：**

| dataset | 實作前 | 實作後 | baro 參考值 |
|---------|--------|--------|-------------|
| re2-ob | 待重跑 `service_first` benchmark | 待重跑 `service_first` benchmark | 待正式 benchmark |
| re2-ss | 待重跑 `service_first` benchmark | 待重跑 `service_first` benchmark | 待正式 benchmark |
| re2-tt | 待重跑 `service_first` benchmark | 待重跑 `service_first` benchmark | 待正式 benchmark |

**是否解決 issue：** 部分解決。排序策略已依消融結果更新；仍需重跑 `gabanzhaf_approx_banzhaf`、`gabanzhaf_rank_blended`、`gabanzhaf_rank_banzhaf_only` 產生新的正式對照表。

**後續問題：**

Q1. 為什麼不直接把 `banzhaf_only` 當 baseline？

`banzhaf_only` 可改善部分 service-level 結果，但同一服務內所有 metric 分數相同，只能靠 column 名稱固定排序。這會削弱 metric-level AC@k 的解釋力。`service_first` 保留 Banzhaf 的服務層優勢，同時保留 metric-local evidence 的服務內 metric 排序功能。

Q2. 新 baseline 要如何重跑？

執行 `python3.10 run123.py --methods gabanzhaf_approx_banzhaf gabanzhaf_rank_blended gabanzhaf_rank_banzhaf_only --re2 --workers 1 --output benchmark_rankmode_ablation.csv`。其中 `gabanzhaf_approx_banzhaf` 現在代表 `closed_form + service_first`。

**驗證：**
- `python3.10 -m pytest tests/test_gabanzhaf_ranking_direct.py -q`：3 passed。
- `python3.10 -m pytest tests/test_run123_variants.py -q`：5 passed。

---

## 相關檔案

- `RCAEval/e2e/gabanzhaf.py`：主管道（Step 3/4/9 為消融注入點）
- `run123.py`：`METHOD_VARIANTS` 消融變體定義（`gabanzhaf_pure_banzhaf` / `gabanzhaf_approx_banzhaf` 已存在）
- `benchmark_banzhaf_ablation.json`：消融 1 初步數據
- `tests/test_run123_variants.py`：變體解析測試
- `docs/superpowers/specs/2026-05-31-phenotype-adaptive-gabanzhaf-design.md`：設計規格
