# GA-Banzhaf → Exact Banzhaf 重構設計

**日期：** 2026-05-12  
**Branch：** metric-banzhaf-ssob  
**目標：** 修復 ss/ob AC@1=0 問題，同時加速所有 dataset

---

## 問題陳述

| Dataset | 服務數 | Trace | AC@1 CPU | AC@1 SOCKET | AC@1 DELAY |
|---------|--------|-------|----------|-------------|------------|
| tt      | 72     | 有    | 0.93     | 0.87        | 0.60       |
| ss      | 22     | 無    | **0.0**  | **0.0**     | 0.13       |
| ob      | 17     | 有    | **0.0**  | **0.0**     | **0.0**    |

**根因：** GA 的 EMA Banzhaf 估計在小系統（17-22 服務）variance 過高。
pop=20 × gen=10 = 200 次 fitness 評估對平坦 fitness landscape 不足以區分服務，
導致 EMA 收斂到近似均勻分佈，排名趨近隨機。

---

## 設計決策

### 決策 1：用確定性 Banzhaf 取代 GA

**選擇：** 完全移除 GA，改用精確枚舉或 Monte Carlo 取樣估計 Banzhaf power index。

**理由：**
- GA 的 topology-aware fitness 設計是正確的，問題在估計方法，非 fitness 定義
- 小系統（n≤22）可做精確計算，大系統用 MC 取樣
- 同一套 code path 處理所有 dataset，無條件分支

### 決策 2：Topology 嵌入 v(S)（不做 pre-filter 或 post-adjust）

**選擇：** `v(S) = coverage(S) − alpha_topo × topo_redundancy(S)`，
取樣時直接用此 v(S) 計算邊際貢獻。

**理由：**
- 數學一致性最好：Banzhaf 本身衡量「含拓撲意識的」邊際貢獻
- Pre-filter 在 ss（無 trace，lag-corr 雜訊高）風險過大
- Post-adjust 無法逆轉基礎排名錯誤

### 決策 3：Exact 計算門檻 n ≤ 20，超過則 MC 取樣 500 次

**理由：**
- 2^20 = 1M 子集，numpy 向量化 < 0.2s
- 實際 max_cand 預設 = top_k_svcs(15) + 5 = 20，涵蓋所有 dataset 的預設配置
- ss(22 服務)在候選集縮減後通常也 ≤ 20；若未縮減則走 MC fallback

---

## 架構變更

### 移除

| 元件 | 說明 |
|------|------|
| `_approximate_banzhaf_ga_svc` | 整個 GA 函式 |
| `_eval_fitness_svc` | GA fitness 評估 |
| `high_sev_mask` | 只有 GA 初始化需要 |
| kwargs: `n_generations`, `pop_size`, `ema_alpha` | GA 超參數 |

### 新增

**`_exact_banzhaf_topology(svc_anom_arr, topo_matrix, alpha_topo, n_samples=500)`**

```
Input:
  svc_anom_arr  : (n,) float  — 每個候選服務的 anomaly score
  topo_matrix   : (n, n) float — topo_matrix[u, v] = confidence of u→v
  alpha_topo    : float       — topology redundancy 懲罰係數
  n_samples     : int         — MC fallback 的取樣數（n > 20 時使用）

Output:
  banzhaf       : (n,) float ≥ 0 — 各服務的 Banzhaf power index 估計值
```

**計算流程：**

```
Step 1: upstream_masks[i] = bitmask(all u where topo_matrix[u,i] > 0)

Step 2: 決定路徑
  if n ≤ 20: exact（枚舉 2^n 子集）
  else: MC（均勻取樣 n_samples 個子集 per service）

Step 3: 計算 v(S) for all S（向量化）
  coverage[S]  = dot(bits(S), svc_anom_arr)
  topo_red[S]  = Σ_{i∈S: S & upstream_masks[i] ≠ 0} svc_anom[i]
  v[S]         = coverage[S] - alpha_topo × topo_red[S]

Step 4: Banzhaf per service
  S_base = 所有不含 i 的子集
  β_i    = mean(v[S_base | (1<<i)] - v[S_base])
  return clip(β, 0, ∞)
```

### 不動的部分

- `_metric_evidence`
- `_service_evidence`
- `_build_topology`
- `_trace_onset_scores` / `_extract_call_graph`
- `_gabanzhaf_core` 中的 window split、topology 建構、final `_col_score`

---

## 效能預估

| Dataset | n_cand | 子集數 | 路徑     | 預估時間 | 現在時間 |
|---------|--------|--------|----------|----------|----------|
| ob      | 17     | 131K   | exact    | < 0.1s   | 72.4s    |
| ss      | ≤20    | ≤1M    | exact    | ~0.2s    | 16.1s    |
| tt      | ≤20    | ≤1M    | exact    | ~0.2s    | 112.3s   |

注意：ob 的 72.4s 主要來自 preprocessing/trace I/O，不是 GA 本身。
Banzhaf 計算的加速會有，但 ob 的整體時間受 preprocessing 主導。

---

## 預期精度改善

**為何 ss/ob 的 CPU/SOCKET AC@1 應該改善：**
- Exact Banzhaf 消除 GA 的估計 variance → 排名穩定
- CPU/SOCKET fault 的 topology 信號能正確懲罰下游 victim
- 即使各服務 svc_anom 相近，v(S) 的拓撲懲罰可區分根因 vs 受害者

**tt 不退步的保證：**
- max_cand ≤ 20 → 走 exact 路徑
- 移除 sparsity 項對 tt 是正向的（不再輕微懲罰正確子集）
- v(S) 語義與 GA fitness 一致

**ob DELAY=0.0 的處理：**
此次設計不承諾修復（可能是 metric_evidence 對 latency 的獨立問題）。
Exact Banzhaf 至少不讓 DELAY 變更差。

---

## 超參數

| 參數 | 值 | 說明 |
|------|-----|------|
| `alpha_topo` | 0.3 | 與現有 GA 一致 |
| `exact_threshold` | 20 | n_cand ≤ 20 走 exact |
| `n_samples` | 500 | MC fallback 取樣數 |

---

## 實作範圍

**只改 `RCAEval/e2e/gabanzhaf.py`：**
1. 刪除 `_eval_fitness_svc` 函式
2. 刪除 `_approximate_banzhaf_ga_svc` 函式
3. 新增 `_exact_banzhaf_topology` 函式
4. 修改 `_gabanzhaf_core`：移除 GA 相關參數與呼叫，改呼叫 `_exact_banzhaf_topology`
5. 移除 `high_sev_mask` 計算區塊

**不改：**
- `main.py`
- `gabanzhaf` entry point 的 signature
- 其他 e2e 方法
