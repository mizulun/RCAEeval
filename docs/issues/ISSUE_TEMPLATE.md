# Issue: {{標題}}

**日期：** YYYY-MM-DD HH:MM  
**狀態：** 未解決 / 進行中 / 已解決  
**影響範圍：** `檔案名稱` — 哪些 Step 或函式  

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

## 當前流程（Pipeline Overview）

> 描述目前 gabanzhaf 管道的執行步驟，聚焦在與本 issue 相關的部分。

```
Step 1. Window split         — _split_raw_windows()
Step 2. Preprocess           — _preprocess_windows()
Step 3. Metric evidence      — _metric_evidence(), _service_evidence()
Step 4. Topology             — _build_topology(), _call_graph_topology_edges()
Step 5. Candidate selection  — _candidate_services()
Step 6. Exact candidates     — _select_exact_candidates()
Step 7. Residual scores      — _residual_scores()
Step 8. Causal matrix        — _causal_explanation_matrix()
Step 9. Banzhaf power        — _probabilistic_banzhaf()
Step 10. Metric ranking      — _rank_metrics()
```

**本次 issue 涉及步驟：** Step X, Step Y

---

## 當前 Benchmark 結果

> 記錄當下跑出的數字，作為對照用。

**跑測日期：** YYYY-MM-DD  
**方法：** gabanzhaf  

### AC@K

| Dataset | AC@1-CPU | AC@1-MEM | AC@1-DISK | AC@1-SOCKET | AC@1-DELAY | AC@1-LOSS |
|---------|---------|---------|---------|------------|-----------|----------|
| re2-ob  |         |         |         |            |           |          |
| re2-ss  |         |         |         |            |           |          |
| re2-tt  |         |         |         |            |           |          |

| Dataset | AC@3-DELAY | AC@5-DELAY | AC@3-LOSS | AC@5-LOSS |
|---------|-----------|-----------|----------|----------|
| re2-ob  |           |           |          |          |
| re2-ss  |           |           |          |          |
| re2-tt  |           |           |          |          |

### Avg@5

| Dataset | Avg@5-CPU | Avg@5-MEM | Avg@5-DISK | Avg@5-SOCKET | Avg@5-DELAY | Avg@5-LOSS |
|---------|----------|----------|----------|-------------|------------|-----------|
| re2-ob  |          |          |          |             |            |           |
| re2-ss  |          |          |          |             |            |           |
| re2-tt  |          |          |          |             |            |           |

---

## 當前問題

### 症狀

> 描述在 benchmark 數字或執行行為上觀察到的異常，例如：某 fault type 的 AC@1 明顯低於其他類型。

### 根本原因假設

> 初步推測的失效機制，例如：受害者的某個特徵被誤判為根因。

### 瓶頸定位

> 具體到哪個函式、哪幾行、哪個計算邏輯造成問題。格式如下：

#### Step X — `函式名稱()` — 問題描述

**程式位置**：`gabanzhaf.py:行號`

```python
# 相關程式碼片段
```

說明為什麼這段邏輯在此情境下失效。

---

## 解法

> 每個解法步驟的結構如下：
> 1. 解法描述（演算法 / 公式 / 步驟）
> 2. `> **[風險]**` 緊接在解法下方
> 3. `> **[回覆]**` 緊接在對應風險下方

### 1. {{解法名稱}}

{{解法描述，含演算法、公式、程式位置}}

> **[風險]** {{描述此解法潛在的失效情境}}
>
> **[回覆]** {{具體的對應方式，含條件、公式或步驟；需調參的數值標注「待調參」}}

### 2. {{解法名稱}}

{{解法描述}}

> **[風險]** {{風險描述}}
>
> **[回覆]** {{回覆}}

---

## 結論

> 說明此 issue 的核心判斷：是演算法瓶頸還是資料集限制，以及改善的主要切入點。

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
（條列每個 AC@1 / Avg@5 目標的達成狀況，✓ / ✗）

**後續問題：**
（格式：Q編號. 問題描述，問題下方直接寫根本原因分析）
```

---

## 實作結果紀錄

### 第 1 次實作 — YYYY-MM-DD HH:MM

**實作內容：**

**Benchmark 對照（baseline：YYYY-MM-DD）：**

| Dataset | 指標 | Baseline | 實作後 | 變化 |
|---------|------|---------|--------|------|
|         |      |         |        |      |

**演算法執行時間：**

| Dataset | 實作前 | 實作後 | baro（參考） |
|---------|--------|--------|------------|
| re2-ob  |        |        |            |
| re2-ss  |        |        |            |
| re2-tt  |        |        |            |

**是否解決 issue：**

**後續問題：**

Q1.

---

## 相關檔案

- `RCAEval/e2e/gabanzhaf.py`：主管道
- `benchmark_results.json`：當前數據
- `docs/issues/YYYY-MM-DD-{{slug}}.md`：相關 issue
- `docs/superpowers/specs/...`：設計規格
