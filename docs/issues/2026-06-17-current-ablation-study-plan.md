# Issue: 現行 Probabilistic Causal GA-Banzhaf 消融實驗計畫

**日期：** 2026-06-17 00:00  
**狀態：** 進行中  
**影響範圍：** `RCAEval/e2e/gabanzhaf.py`（Step 3、Step 4、Step 5、Step 8、Step 9、Step 10）、`run123.py`（`METHOD_VARIANTS`）、`benchmark_clean_result.csv`、`benchmark_clean_result.json`

---

## 回覆格式規範

> 本檔案的回覆遵循以下規則：

**解法（Solution）**
- 必須寫清楚用什麼演算法、公式或步驟，不能只說「怎麼做」。
- 不使用「適當」、「足夠」、「大概」等模糊字眼，除非是後續需要實驗調參才能確定的數值，此時須明確標注「待調參」。
- 不使用「fallback」一詞；需要降級處理時，改寫成具體條件與對應行為。
- 不使用「ground truth」一詞；需要描述標註資料時，使用「benchmark 標註」。

**風險（Risk）**
- 放在該步驟解法描述的下方。
- 格式：`> **[風險]** ...`

**風險回覆（Risk Response）**
- 緊接在對應風險的下方。
- 格式：`> **[回覆]** ...`

---

## 當前流程（Pipeline Overview）

現行 `gabanzhaf` 已移除 trace onset、call graph、residual 與 phenotype/semantic 主路徑，主管道如下：

```
Step 1. Window split         - _split_raw_windows()
Step 2. Preprocess           - _preprocess_windows()
Step 3. Metric evidence      - _metric_evidence(), _service_evidence()
Step 4. Lag-1 topology       - _build_topology()
Step 5. Candidate selection  - _candidate_services(), _select_exact_candidates()
Step 6. Causal matrix        - _causal_explanation_matrix()
Step 7. Banzhaf power        - _probabilistic_banzhaf(), _bruteforce_probabilistic_banzhaf()
Step 8. Metric ranking       - _rank_metrics()
```

**本次 issue 涉及步驟：** Step 3、Step 4、Step 5、Step 6、Step 7、Step 8

**現行程式位置：**
- `_metric_evidence()`：`RCAEval/e2e/gabanzhaf.py:83`
- `_service_evidence()`：`RCAEval/e2e/gabanzhaf.py:126`
- `_probabilistic_banzhaf()`：`RCAEval/e2e/gabanzhaf.py:227`
- `_bruteforce_probabilistic_banzhaf()`：`RCAEval/e2e/gabanzhaf.py:256`
- `_build_topology()`：`RCAEval/e2e/gabanzhaf.py:309`
- `_candidate_services()`：`RCAEval/e2e/gabanzhaf.py:404`
- `_select_exact_candidates()`：`RCAEval/e2e/gabanzhaf.py:415`
- `_causal_explanation_matrix()`：`RCAEval/e2e/gabanzhaf.py:424`
- `_rank_metrics()`：`RCAEval/e2e/gabanzhaf.py:472`
- `gabanzhaf()` env 入口：`RCAEval/e2e/gabanzhaf.py:578`
- 既有 run variants：`run123.py:48`

---

## 當前 Benchmark 基準

**跑測日期：** 2026-06-17  
**方法：** `gabanzhaf`  
**結果檔：** `benchmark_clean_result.csv`

### RE2 AC@1

| Dataset | CPU | MEM | DISK | SOCKET | DELAY | LOSS | Avg speed |
|---------|----:|----:|-----:|-------:|------:|-----:|----------:|
| re2-ob  | 0.40 | 0.67 | 0.27 | 0.27 | 0.27 | 0.40 | 0.05 |
| re2-ss  | 0.27 | 0.80 | 0.53 | 0.40 | 0.47 | 0.33 | 0.06 |
| re2-tt  | 0.87 | 1.00 | 1.00 | 0.87 | 0.60 | 0.53 | 0.27 |

### RE2 Avg@5

| Dataset | CPU | MEM | DISK | SOCKET | DELAY | LOSS |
|---------|----:|----:|-----:|-------:|------:|-----:|
| re2-ob  | 0.88 | 0.91 | 0.85 | 0.85 | 0.71 | 0.83 |
| re2-ss  | 0.85 | 0.96 | 0.91 | 0.87 | 0.80 | 0.84 |
| re2-tt  | 0.97 | 1.00 | 1.00 | 0.97 | 0.71 | 0.65 |

---

## 當前問題

### 症狀

論文需要重新設計消融實驗，以符合現行大幅簡化後的 `gabanzhaf` 管道。舊消融檔 `docs/issues/2026-06-17-ablation-study-design.md` 仍包含 trace onset、call graph、residual、rank mode 等已不在主流程中的變體，無法直接作為現行論文主表。

使用者目前想保留三類消融：

1. `no-lag-1`
2. `no-metrics`
3. probabilistic Banzhaf closed-form 與 pure/bruteforce Banzhaf 的準確率與執行時間比較

本 issue 重新定義主文與附錄消融組合，使每個變體都對應到現行程式中的單一演算法因素。

### 瓶頸定位

#### Step 3 - metrics 是主管道唯一觀測來源

`_metric_evidence()` 產生 column-level 異常證據；`_service_evidence()` 將其彙整為 service-level 證據。現行管道沒有 trace/log 輸入，因此「完全 no-metrics」會使方法失去主要觀測訊號。

> **[風險]** 若主表放完全 `no-metrics`，結果可能近乎依名稱排序，實驗只能證明「沒有觀測訊號時方法不可用」，論文資訊量低。
>
> **[回覆]** 主文改放 `no_metric_type_dominance` 與 `full_window_only`，分別測 `_service_evidence()` 的 metric-type dominance 與 `_metric_evidence()` 的 early-window evidence。完全 `no_metrics` 保留為 sanity check 或附錄表。

#### Step 4 - lag-1 topology 同時影響候選擴張與 causal matrix

`_build_topology()` 建立 asymmetric lag-1 edge；`_candidate_services()` 用 edge upstream 擴張 seed；`_causal_explanation_matrix()` 用 edge confidence 寫入 probabilistic causal game 的非對角解釋機率。

> **[風險]** 單一 `no-lag-1` 同時關閉候選擴張與 propagation probability，無法判斷是哪一段造成變化。
>
> **[回覆]** 主文放整體 `no_lag_1`，附錄可拆成 `no_lag_candidate_expansion` 與 `no_lag_propagation`。主文先回答「lag-1 topology 是否有用」，附錄再回答「用在哪裡最有用」。

#### Step 7 - closed-form 與 bruteforce 理論上計算同一個 noisy-OR Banzhaf

`_probabilistic_banzhaf()` 是 noisy-OR game 的閉式計算；`_bruteforce_probabilistic_banzhaf()` 是枚舉所有 coalition 的直接計算。兩者應得到相同 Banzhaf ordering，主要差異是執行時間。

> **[風險]** 將 closed-form vs bruteforce 寫成一般 ablation，可能被誤讀為兩種不同演算法貢獻。
>
> **[回覆]** 將此實驗命名為 efficiency/equivalence study。報告欄位包含 AC@1/AC@3/AC@5/Avg@5、`wall_time_sec`、`Avg speed`、closed-form speedup ratio，以及每資料集 ranking 是否一致。

---

## 解法

### 1. 主表 A0 - Full method

作為所有消融的 baseline。設定如下：

```
banzhaf_mode = closed_form
exact_candidate_k = 12
top_k_svcs = 15
causal_probability_lambda = 1.5
early_window_ratio = 0.25
early_window_min_len = 5
```

輸出欄位：

- AC@1 / AC@3 / AC@5 / Avg@5 by fault type
- `wall_time_sec`
- `Avg speed`

> **[風險]** benchmark 結果會被舊 output 殘留污染。
>
> **[回覆]** 一律使用 `run123.py` 的 per-method/per-dataset result dir：`output/runs/<method>/<dataset>/results`。每個 `(method, dataset)` 執行前由 `clear_result_dir()` 清除該 result dir。

### 2. 主表 A1 - no-lag-1

新增 config 欄位：

```
use_lag_topology: bool = True
```

新增 env：

```
GABANZHAF_DISABLE_LAG_TOPOLOGY=1
```

當 `use_lag_topology=False` 時：

1. `_gabanzhaf_core()` 不呼叫 `_build_topology()`。
2. `topology_edges = []`。
3. `_candidate_services(seed_svcs, topology_edges, svc_names)` 只回傳 seed services。
4. `_causal_explanation_matrix()` 只保留 self-explanation 對角線 `P(candidate explains itself)=1.0`。

`run123.py` 新增：

```
gabanzhaf_no_lag_1
metadata.ablation_role = "no_lag_1"
```

比較目標：量化 lag-1 topology 對候選擴張與 propagation probability 的整體貢獻。

> **[風險]** 若 Full 與 no-lag-1 結果相同，不能直接推論 topology 無用，可能是 `exact_candidate_k=12` 截斷後新增 upstream 未進入 game。
>
> **[回覆]** 每次跑測同步記錄 diagnostics：`len(topology_edges)`、`len(cand_list)`、`game_candidates`。若 Full 有 topology edges 但 `game_candidates` 不變，於分析表標記「lag edge 未進入 exact candidate game」。

### 3. 主表 A2 - evidence-only / no-Banzhaf

新增 rank mode：

```
rank_mode = "banzhaf" | "evidence_only"
```

`evidence_only` 的排序規則：

```
service_score(s) = svc_anom_dict[s]
rank(col) = (-service_score(parse_svc(col)), -metric_ev[col], col)
```

此變體不計算 probabilistic Banzhaf，直接使用 Step 3 的 service evidence 與 metric evidence 排名。

`run123.py` 新增：

```
gabanzhaf_evidence_only
metadata.ablation_role = "no_banzhaf"
```

比較目標：證明 Step 6/7 的 causal matrix + Banzhaf power 是否相對純 evidence ranking 帶來增益。

> **[風險]** evidence-only 省略 Banzhaf 後 runtime 會變快，準確率與時間同時變動，難以只談準確率。
>
> **[回覆]** 主表同時報準確率與時間。若 evidence-only 準確率接近 Full，但時間更快，需在結論中明確承認 Banzhaf 的主要價值不足；若 Full 準確率更高，則以準確率差異解釋 Banzhaf 成本。

### 4. 主表 A3 - no metric-type dominance

新增 config 欄位：

```
use_metric_type_dominance: bool = True
```

現行 `_service_evidence()`：

```
svc_score = 0.70 * svc_agg + 0.30 * dom_norm
```

當 `use_metric_type_dominance=False` 時：

```
svc_score = svc_agg
```

其中：

```
svc_agg = 0.50 * percentile95(metrics_of_service)
        + 0.30 * mean(top3(metrics_of_service))
        + 0.20 * max(metrics_of_service)
```

`run123.py` 新增：

```
gabanzhaf_no_metric_type_dominance
metadata.ablation_role = "no_metric_type_dominance"
```

比較目標：驗證 `_service_evidence()` 中 30% metric-type dominance 是否對 CPU/MEM/DISK 等廣播型異常有貢獻。

> **[風險]** dominance 對不同 fault type 的效果可能相反，macro average 會掩蓋 CPU/MEM/DISK 與 DELAY/LOSS 的差異。
>
> **[回覆]** 表格按 fault type 報 AC@1 與 Avg@5，不只報 macro average。分析時分成 resource faults（CPU/MEM/DISK）與 communication faults（SOCKET/DELAY/LOSS）兩組。

### 5. 主表 A4 - full-window-only / no early-window evidence

新增 config 欄位：

```
use_early_window: bool = True
```

現行 `_metric_evidence()`：

```
full  = norm(0.55 * mag + 0.30 * persistence + 0.15 * peak)
early = norm(0.70 * early_mag + 0.30 * early_peak)
score = norm(0.60 * early + 0.40 * full)
```

當 `use_early_window=False` 時：

```
score = full
```

`run123.py` 新增：

```
gabanzhaf_full_window_only
metadata.ablation_role = "no_early_window"
```

比較目標：驗證 early-window evidence 是否能減少下游傳播污染，尤其是 DELAY/LOSS。

> **[風險]** 若 anomalous window 很短，early window 會等同 full window，導致此消融沒有差異。
>
> **[回覆]** diagnostics 記錄 `anomal_len`、`early_len`、`use_full_for_early`。當 `use_full_for_early=True` 時，該 case 在 no early-window 分析中標記為「early 等同 full」。

### 6. 主表 A5 - closed-form vs bruteforce Banzhaf efficiency/equivalence

使用既有 variants：

```
gabanzhaf_approx_banzhaf  -> GABANZHAF_BANZHAF_MODE=closed_form
gabanzhaf_pure_banzhaf    -> GABANZHAF_BANZHAF_MODE=bruteforce
```

執行命令：

```
py310/bin/python run123.py --methods gabanzhaf_approx_banzhaf gabanzhaf_pure_banzhaf \
    --datasets re2-ob re2-ss re2-tt --workers 1 \
    --output benchmark_banzhaf_equivalence.csv
```

比較欄位：

- AC@1 / AC@3 / AC@5 / Avg@5
- `wall_time_sec`
- `Avg speed`
- speedup ratio：`bruteforce wall_time_sec / closed_form wall_time_sec`
- ranking consistency：每 case top-1 service 是否一致（需另加 diagnostics 或 case-level output script）

> **[風險]** `exact_candidate_k=12` 時 bruteforce 每 case 需枚舉 `2^(k-1)` coalitions per player，re2-tt 可能過慢。
>
> **[回覆]** 先以 `--test` smoke run 估計 re2-tt 時間，再跑完整 re2。若完整 re2-tt 超過 600 秒，主表標記 bruteforce re2-tt 為 timeout，附錄補 `exact_candidate_k=8` 的 equivalence/time curve。

### 7. 附錄 B1 - exact_candidate_k sensitivity

使用既有 variants：

```
gabanzhaf_k6
gabanzhaf_k8
gabanzhaf_k10
gabanzhaf_k12
gabanzhaf_k16
```

固定：

```
banzhaf_mode = closed_form
```

比較目標：量化 `_select_exact_candidates()` 的 K 對準確率與時間的影響。主線預設為 `K=12`，掃描 K 只作敏感度分析，不把最佳 K 反寫為 baseline。

> **[風險]** K 掃描若用全部 RE1/RE2/RE3，表格會過大。
>
> **[回覆]** 附錄先放 RE2 三個資料集；若篇幅不足，只保留 re2-tt，因為 re2-tt 服務數最多且 wall time 最大。

### 8. 附錄 B2 - complete no-metrics sanity check

此項回應使用者原始想法，但不建議放主文主表。

新增 config 欄位：

```
use_metrics: bool = True
```

當 `use_metrics=False` 時：

```
metric_ev[col] = 1.0 for every col
svc_anom_dict[svc] = 1.0 for every svc
topology_edges = []
svc_banzhaf scores are equal or empty
rank(col) is stable by service/column name
```

`run123.py` 新增：

```
gabanzhaf_no_metrics
metadata.ablation_role = "no_metrics_sanity"
```

比較目標：確認沒有 metrics observation 時，現行 metric-only 管道退化為 deterministic name-based ordering。此結果只作 sanity check，不作核心貢獻論證。

> **[風險]** 完全 no-metrics 可能被解讀為不公平對照。
>
> **[回覆]** 論文文字明確標註為 sanity check；核心貢獻由 A3 與 A4 代表 metrics 內部元件的可解釋消融。

### 9. 附錄 B3 - lag topology split

若 A1 顯示 no-lag-1 有顯著差異，附錄拆分兩個變體：

1. `no_lag_candidate_expansion`
   - `_candidate_services()` 不使用 upstream expansion。
   - `_causal_explanation_matrix()` 仍使用 topology edges。

2. `no_lag_propagation`
   - `_candidate_services()` 使用 upstream expansion。
   - `_causal_explanation_matrix()` 接收 `topology_edges=[]`，只保留 self-explanation。

> **[風險]** 拆分後變體多，會增加 run time 與整理成本。
>
> **[回覆]** 只有當 A1 與 Full 的任一 RE2 Avg@5 fault type 差異絕對值 >= 0.03 時才跑 B3。若差異低於 0.03，A1 足以說明 lag-1 影響小。

---

## 建議主文表格

### Table A - Component ablation on RE2

| Method | Role | re2-ob Avg@5 macro | re2-ss Avg@5 macro | re2-tt Avg@5 macro | Avg speed macro |
|--------|------|-------------------:|-------------------:|-------------------:|----------------:|
| gabanzhaf | Full | 待補 | 待補 | 待補 | 待補 |
| gabanzhaf_no_lag_1 | no lag-1 topology | 待補 | 待補 | 待補 | 待補 |
| gabanzhaf_evidence_only | no Banzhaf | 待補 | 待補 | 待補 | 待補 |
| gabanzhaf_no_metric_type_dominance | no metric-type dominance | 待補 | 待補 | 待補 | 待補 |
| gabanzhaf_full_window_only | no early-window evidence | 待補 | 待補 | 待補 | 待補 |

主文另保留 fault-type table，至少報 AC@1 與 Avg@5 的 CPU/MEM/DISK/SOCKET/DELAY/LOSS 六欄，避免 macro average 掩蓋差異。

### Table B - Closed-form Banzhaf equivalence and runtime

| Dataset | Closed-form Avg@5 macro | Bruteforce Avg@5 macro | Delta | Closed-form wall time | Bruteforce wall time | Speedup |
|---------|-------------------------:|------------------------:|------:|----------------------:|---------------------:|--------:|
| re2-ob  | 待補 | 待補 | 待補 | 待補 | 待補 | 待補 |
| re2-ss  | 待補 | 待補 | 待補 | 待補 | 待補 | 待補 |
| re2-tt  | 待補 | 待補 | 待補 | 待補 | 待補 | 待補 |

---

## 執行順序

1. 實作 env/config 開關：`use_lag_topology`、`rank_mode=evidence_only`、`use_metric_type_dominance`、`use_early_window`、`use_metrics`。
2. 在 `run123.py` 新增 variants：`gabanzhaf_no_lag_1`、`gabanzhaf_evidence_only`、`gabanzhaf_no_metric_type_dominance`、`gabanzhaf_full_window_only`、`gabanzhaf_no_metrics`。
3. 跑 smoke test：

```
py310/bin/python run123.py --methods gabanzhaf gabanzhaf_no_lag_1 gabanzhaf_evidence_only \
    --datasets re2-ob --test --workers 1 --output /private/tmp/gabanzhaf_ablation_smoke.csv
```

4. 跑主表：

```
py310/bin/python run123.py --methods gabanzhaf gabanzhaf_no_lag_1 gabanzhaf_evidence_only \
    gabanzhaf_no_metric_type_dominance gabanzhaf_full_window_only \
    --datasets re2-ob re2-ss re2-tt --workers 1 \
    --output benchmark_current_ablation.csv
```

5. 跑 closed-form vs bruteforce：

```
py310/bin/python run123.py --methods gabanzhaf_approx_banzhaf gabanzhaf_pure_banzhaf \
    --datasets re2-ob re2-ss re2-tt --workers 1 \
    --output benchmark_banzhaf_equivalence.csv
```

6. 視主表結果決定是否跑附錄 B1/B2/B3。

---

## 結論

現行主文消融應聚焦五個可解釋因素：lag-1 topology、Banzhaf game、metric-type dominance、early-window metric evidence、closed-form Banzhaf efficiency/equivalence。完全 `no-metrics` 保留為 sanity check，不建議作為主表核心消融，因為現行管道沒有 trace/log 替代訊號。舊版 trace/call graph/residual 消融不放主文，除非論文需要補一段「為何最終移除這些 stage」的歷史實驗說明。

---

## 實作結果紀錄

### 第 1 次實作 - YYYY-MM-DD HH:MM

**實作內容：**

**Benchmark 對照：**

| Method | Dataset | Avg@5 macro | Avg speed | 備註 |
|--------|---------|------------:|----------:|------|
|        |         |             |           |      |

**是否解決 issue：**

**後續問題：**

Q1.

---

## 相關檔案

- `RCAEval/e2e/gabanzhaf.py`：現行主管道
- `run123.py`：benchmark variants 與輸出
- `benchmark_clean_result.csv`：現行 clean baseline
- `benchmark_clean_result.json`：現行 clean baseline JSON
- `benchmark_trace_ablation.json`：舊版 trace/call graph/residual 相關對照
- `docs/issues/2026-06-17-ablation-study-design.md`：舊版 ablation 設計，保留歷史脈絡
