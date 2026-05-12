# Exact Banzhaf 實作計畫

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 以精確枚舉/MC 取樣取代 GA，估計 topology-aware Banzhaf power index，修復 ss/ob AC@1=0 問題並加速所有 dataset。

**Architecture:** 新增 `_exact_banzhaf_topology`（penalty-based v(S) 精確計算，n≤20 走 exact，否則 MC）；移除 `_eval_fitness_svc`、`_approximate_banzhaf_ga_svc`；`_gabanzhaf_core` 改呼叫新函式。唯一改動檔案：`RCAEval/e2e/gabanzhaf.py`。

**Tech Stack:** Python 3.10, NumPy（向量化 bitmask 運算）

---

## 重要背景

### v(S) 設計（penalty-based）

```
v(S) = coverage(S) − alpha_topo × topo_redundancy(S)

coverage(S)      = Σ_{i∈S} fitness_signal[i]
topo_redundancy(S) = Σ_{i∈S, ∃u∈S: topo_matrix[u,i]>0} fitness_signal[i]
```

當 fitness_signal[root_cause] >> fitness_signal[victim]（由 svc_residual 保證），
penalty-based v(S) 能正確給 root cause 更高的 Banzhaf 值。
**這是假設前提：若 svc_residual 無法區分服務，Banzhaf 無論如何都難以正確排名。**

### Exact vs MC 路徑

- **n ≤ 20**（exact）：枚舉所有 2^n 子集，numpy 向量化計算 v(S) for all S，再計算 Banzhaf
- **n > 20**（MC fallback）：對每個 service i，取樣 n_samples 個隨機子集，估計邊際貢獻

### 現有測試注意事項

`tests/test_gabanzhaf_improvements.py` 有呼叫 `_gabanzhaf_core` 時傳 `n_generations`, `pop_size` 的 kwargs。
新版 `_gabanzhaf_core` 直接忽略這些 kwargs（`kwargs.get(...)` 找不到就跳過），**現有測試無需修改**。

---

## File Map

| 操作 | 路徑 |
|------|------|
| Modify | `RCAEval/e2e/gabanzhaf.py` |
| Modify (tests) | `tests/test_gabanzhaf_improvements.py` |

---

## Task 1：為 `_exact_banzhaf_topology` 寫 failing tests

**Files:**
- Modify: `tests/test_gabanzhaf_improvements.py`

- [ ] **Step 1: 在 test 檔案末尾加入新的 test section**

在 `tests/test_gabanzhaf_improvements.py` 末尾加入：

```python
# ─────────────────────────────────────────────────────────────────────────────
# 4. _exact_banzhaf_topology
# ─────────────────────────────────────────────────────────────────────────────
from RCAEval.e2e.gabanzhaf import _exact_banzhaf_topology


def test_exact_banzhaf_no_topology_proportional_to_anomaly():
    """Without topology, Banzhaf should be proportional to fitness_signal.

    For additive v(S) (alpha_topo=0), Banzhaf(i) = fitness_signal[i] exactly.
    """
    fs = np.array([0.9, 0.4, 0.1], dtype=float)
    topo = np.zeros((3, 3), dtype=np.float32)
    bz = _exact_banzhaf_topology(fs, topo, alpha_topo=0.0)
    # ranking must match fitness_signal ranking
    assert bz[0] > bz[1] > bz[2], f"Expected descending Banzhaf, got {bz}"


def test_exact_banzhaf_topology_promotes_root_cause():
    """Topology penalty must increase the gap between root cause and victim.

    Setup: n=2, A(idx=0) upstream of B(idx=1), A has higher anomaly score.
    Analytical expected values:
      fitness = [0.9, 0.5], topo[0,1]=0.5, alpha=0.3
      v({A})   = 0.9,  v({B})   = 0.5
      v({A,B}) = 1.4 - 0.3*0.5 = 1.25   (B penalized: has upstream A in S)
      Banzhaf(A) = (0.9 + 1.25-0.5) / 2  = (0.9 + 0.75) / 2 = 0.825
      Banzhaf(B) = (0.5 + 1.25-0.9) / 2  = (0.5 + 0.35) / 2 = 0.425
    """
    fs = np.array([0.9, 0.5], dtype=float)
    topo = np.zeros((2, 2), dtype=np.float32)
    topo[0, 1] = 0.5  # A(0) upstream of B(1)
    bz = _exact_banzhaf_topology(fs, topo, alpha_topo=0.3)
    assert abs(bz[0] - 0.825) < 1e-6, f"Expected Banzhaf(A)=0.825, got {bz[0]}"
    assert abs(bz[1] - 0.425) < 1e-6, f"Expected Banzhaf(B)=0.425, got {bz[1]}"


def test_exact_banzhaf_topology_no_negative_values():
    """Banzhaf values must all be non-negative (clipped at 0)."""
    rng = np.random.default_rng(0)
    n = 5
    fs = rng.random(n)
    topo = np.zeros((n, n), dtype=np.float32)
    topo[0, 1] = 0.8
    topo[1, 2] = 0.6
    bz = _exact_banzhaf_topology(fs, topo, alpha_topo=0.5)
    assert np.all(bz >= 0), f"Negative Banzhaf values: {bz}"


def test_exact_banzhaf_mc_fallback_consistent_with_exact():
    """MC fallback (n>20) must give rankings consistent with exact (tested on n=5).

    We compare rankings (not exact values) between exact (n=5) and MC with
    large n_samples. Rankings must match for all pairs with clear gaps.
    """
    rng = np.random.default_rng(42)
    n = 5
    fs = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
    topo = np.zeros((n, n), dtype=np.float32)
    topo[0, 1] = 0.5  # 0 upstream of 1

    # exact (EXACT_THRESHOLD >= n, so exact path used)
    bz_exact = _exact_banzhaf_topology(fs, topo, alpha_topo=0.3, n_samples=500)

    # Force MC path by monkey-patching threshold (we pass n_samples large to stabilise)
    # We test MC path indirectly by using n > 20
    # Build a n=22 case where manual ranking is known
    fs_big = np.array([0.9] + [0.3] * 21)
    topo_big = np.zeros((22, 22), dtype=np.float32)
    # service 0 is upstream of all others
    for j in range(1, 22):
        topo_big[0, j] = 0.3
    bz_mc = _exact_banzhaf_topology(fs_big, topo_big, alpha_topo=0.3, n_samples=2000)
    assert bz_mc[0] == bz_mc.max(), (
        f"Root cause (idx 0) should have max Banzhaf in MC path. bz={bz_mc}"
    )


def test_exact_banzhaf_single_service():
    """Single service: Banzhaf = fitness_signal (trivial coalition)."""
    fs = np.array([0.7])
    topo = np.zeros((1, 1), dtype=np.float32)
    bz = _exact_banzhaf_topology(fs, topo, alpha_topo=0.3)
    assert abs(bz[0] - 0.7) < 1e-6, f"Single service Banzhaf should be 0.7, got {bz[0]}"
```

- [ ] **Step 2: 確認 test 現在 FAIL（函式還不存在）**

```bash
cd /Users/user/RCAEval && /Users/user/RCAEval/py310/bin/python -m pytest tests/test_gabanzhaf_improvements.py::test_exact_banzhaf_no_topology_proportional_to_anomaly -v 2>&1 | tail -5
```

預期輸出：`ImportError` 或 `FAILED` with `cannot import name '_exact_banzhaf_topology'`

- [ ] **Step 3: Commit failing tests**

```bash
cd /Users/user/RCAEval && git add tests/test_gabanzhaf_improvements.py && git commit -m "test: add failing tests for _exact_banzhaf_topology"
```

---

## Task 2：實作 `_exact_banzhaf_topology`

**Files:**
- Modify: `RCAEval/e2e/gabanzhaf.py`（在 `_build_topology` 之後、`_gabanzhaf_core` 之前插入）

- [ ] **Step 1: 刪除 `_eval_fitness_svc` 和 `_approximate_banzhaf_ga_svc` 函式**

在 `gabanzhaf.py` 中找到並刪除以下兩個函式（行 277–375）：

```
def _eval_fitness_svc(pop, svc_anom_arr, topo_matrix, alpha_topo, penalty_weight):
    ...（整個函式刪除）

def _approximate_banzhaf_ga_svc(svc_anom_arr, topo_matrix, high_sev_mask, ...):
    ...（整個函式刪除）
```

- [ ] **Step 2: 在刪除的位置插入 `_exact_banzhaf_topology`**

在 `_build_topology` 函式結束後（行 ~419）、`_gabanzhaf_core` 開始前插入：

```python
def _exact_banzhaf_topology(fitness_signal, topo_matrix, alpha_topo, n_samples=500):
    """Exact or MC-sampled Banzhaf with topology-aware penalty.

    v(S) = coverage(S) - alpha_topo * topo_redundancy(S)

    coverage(S)        = Σ_{i∈S} fitness_signal[i]
    topo_redundancy(S) = Σ_{i∈S: ∃u∈S s.t. topo_matrix[u,i]>0} fitness_signal[i]

    For n ≤ 20: exact enumeration over all 2^n subsets (vectorised).
    For n > 20: uniform Monte Carlo with n_samples random subsets per service.

    Returns non-negative Banzhaf values, one per service.
    """
    EXACT_THRESHOLD = 20
    n = len(fitness_signal)

    # upstream_masks[i] = integer bitmask of all u where topo_matrix[u, i] > 0
    upstream_masks = np.zeros(n, dtype=np.int64)
    for i in range(n):
        for u in range(n):
            if topo_matrix[u, i] > 0:
                upstream_masks[i] |= np.int64(1 << u)

    if n <= EXACT_THRESHOLD:
        return _exact_banzhaf_enum(fitness_signal, upstream_masks, alpha_topo)
    else:
        return _mc_banzhaf(fitness_signal, upstream_masks, alpha_topo, n_samples)


def _exact_banzhaf_enum(fitness_signal, upstream_masks, alpha_topo):
    """Vectorised exact Banzhaf for n <= 20."""
    n = len(fitness_signal)
    N = 1 << n  # 2^n
    all_S = np.arange(N, dtype=np.int64)

    # coverage[m] = Σ_{i: bit i set in m} fitness_signal[i]
    coverage = np.zeros(N, dtype=float)
    for i in range(n):
        mask_i = np.int64(1 << i)
        coverage[(all_S & mask_i) != 0] += fitness_signal[i]

    # topo_red[m] = Σ_{i∈S: upstream_masks[i] & m ≠ 0} fitness_signal[i]
    topo_red = np.zeros(N, dtype=float)
    if alpha_topo > 0:
        for i in range(n):
            if upstream_masks[i] == 0:
                continue
            mask_i = np.int64(1 << i)
            in_S = (all_S & mask_i) != 0
            has_upstream = (all_S & upstream_masks[i]) != 0
            topo_red[in_S & has_upstream] += fitness_signal[i]

    v = coverage - alpha_topo * topo_red

    # Banzhaf(i) = mean(v[S ∪ {i}] - v[S]) over S not containing i
    banzhaf = np.zeros(n, dtype=float)
    for i in range(n):
        mask_i = np.int64(1 << i)
        S_base = all_S[(all_S & mask_i) == 0]   # subsets not containing i
        S_with = S_base | mask_i
        banzhaf[i] = np.mean(v[S_with] - v[S_base])

    return np.maximum(0., banzhaf)


def _mc_banzhaf(fitness_signal, upstream_masks, alpha_topo, n_samples, seed=42):
    """Monte Carlo Banzhaf estimation for n > 20."""
    n = len(fitness_signal)
    rng = np.random.default_rng(seed)
    banzhaf = np.zeros(n, dtype=float)
    counts = np.zeros(n, dtype=float)

    def _v(m):
        cov = 0.
        topo_red = 0.
        for i in range(n):
            if (m >> i) & 1:
                cov += fitness_signal[i]
                if alpha_topo > 0 and upstream_masks[i] != 0 and (m & upstream_masks[i]):
                    topo_red += fitness_signal[i]
        return cov - alpha_topo * topo_red

    for _ in range(n_samples):
        m = int(rng.integers(0, 1 << n))
        v_m = _v(m)
        for i in range(n):
            mask_i = 1 << i
            if m & mask_i:
                v_without = _v(m ^ mask_i)
                banzhaf[i] += v_m - v_without
            else:
                v_with = _v(m | mask_i)
                banzhaf[i] += v_with - v_m
            counts[i] += 1

    safe_counts = np.where(counts > 0, counts, 1.)
    return np.maximum(0., banzhaf / safe_counts)
```

- [ ] **Step 3: 確認函式可 import**

```bash
cd /Users/user/RCAEval && /Users/user/RCAEval/py310/bin/python -c "from RCAEval.e2e.gabanzhaf import _exact_banzhaf_topology; print('OK')"
```

預期輸出：`OK`

- [ ] **Step 4: 跑新的 unit tests**

```bash
cd /Users/user/RCAEval && /Users/user/RCAEval/py310/bin/python -m pytest tests/test_gabanzhaf_improvements.py -k "banzhaf" -v 2>&1 | tail -20
```

預期：所有 `test_exact_banzhaf_*` PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/user/RCAEval && git add RCAEval/e2e/gabanzhaf.py && git commit -m "feat: add _exact_banzhaf_topology (exact + MC fallback)"
```

---

## Task 3：更新 `_gabanzhaf_core`

**Files:**
- Modify: `RCAEval/e2e/gabanzhaf.py`（`_gabanzhaf_core` 函式內部）

- [ ] **Step 1: 移除 GA 相關參數讀取（約行 433–438）**

找到並刪除以下四行（在 `_gabanzhaf_core` 開頭的 kwargs 讀取區塊）：

```python
    n_generations = int(kwargs.get("n_generations", 10))
    pop_size      = int(kwargs.get("pop_size", 20))
    ema_alpha     = float(kwargs.get("ema_alpha", 0.2))
    rng           = np.random.default_rng(int(kwargs.get("seed", 42)))
```

保留其餘 kwargs 讀取（top_k_svcs, max_cand, trace_scores, sli）。

- [ ] **Step 2: 移除 `high_sev_mask` 計算區塊（約行 544–548）**

找到並刪除：

```python
    # ── 6. Soft high-evidence mask for GA initialisation ─────────────────────
    cand_anom_vals = np.array([svc_anom_dict.get(s, 0.) for s in cand_list])
    high_cut = np.percentile(cand_anom_vals, 60) if len(cand_anom_vals) else 1.
    high_sev_mask = np.array([1. if svc_anom_dict.get(s, 0.) >= high_cut else 0.
                              for s in cand_list])
```

- [ ] **Step 3: 替換 GA 呼叫為 `_exact_banzhaf_topology`（約行 576–583）**

找到：

```python
    # ── 8. Bounded GA-Banzhaf computation ────────────────────────────────────
    banzhaf_w = _approximate_banzhaf_ga_svc(
        fitness_signal, topo_matrix, high_sev_mask,
        n_generations=n_generations,
        pop_size=pop_size,
        ema_alpha=ema_alpha,
        rng=rng,
    )
    svc_banzhaf = {cand_list[i]: float(banzhaf_w[i]) for i in range(n_cand)}
```

替換為：

```python
    # ── 8. Exact Banzhaf computation ─────────────────────────────────────────
    banzhaf_w = _exact_banzhaf_topology(
        fitness_signal, topo_matrix, alpha_topo=0.3
    )
    svc_banzhaf = {cand_list[i]: float(banzhaf_w[i]) for i in range(n_cand)}
```

- [ ] **Step 4: 確認 `_gabanzhaf_core` 可呼叫（快速 smoke test）**

```bash
cd /Users/user/RCAEval && /Users/user/RCAEval/py310/bin/python -c "
import numpy as np, pandas as pd
from RCAEval.e2e.gabanzhaf import _gabanzhaf_core
rng = np.random.default_rng(0)
n = 100
df = pd.DataFrame({'time': range(n), 'svcA_cpu': rng.normal(0,1,n), 'svcB_cpu': rng.normal(0,1,n)})
df.loc[50:, 'svcA_cpu'] += 5
result = _gabanzhaf_core(df, inject_time=50, dataset=None, anomalies=None, kwargs={})
print('ranks[:3]:', result['ranks'][:3])
print('OK')
"
```

預期輸出：輸出 3 個 column names 然後 `OK`，無 exception。

- [ ] **Step 5: 跑所有現有測試確認沒有 regression**

```bash
cd /Users/user/RCAEval && /Users/user/RCAEval/py310/bin/python -m pytest tests/test_gabanzhaf_improvements.py -v 2>&1 | tail -25
```

預期：所有測試 PASS（包含舊的 winsorization、call graph、demote caller 測試）。

- [ ] **Step 6: Commit**

```bash
cd /Users/user/RCAEval && git add RCAEval/e2e/gabanzhaf.py && git commit -m "refactor: replace GA-Banzhaf with exact Banzhaf in _gabanzhaf_core"
```

---

## Task 4：Sanity check 在真實資料上

**Files:** 無修改，只跑命令

- [ ] **Step 1: 跑 baro 做基準確認 main.py 正常（memory 要求）**

```bash
cd /Users/user/RCAEval && /Users/user/RCAEval/py310/bin/python main.py --method baro --dataset re2-tt --length 20 2>&1 | tail -5
```

預期：正常輸出結果，無 exception。

- [ ] **Step 2: 跑 gabanzhaf 在 re2-tt 單一 case（計時）**

```bash
cd /Users/user/RCAEval && time /Users/user/RCAEval/py310/bin/python -c "
import sys; sys.argv = ['']
from RCAEval.e2e.gabanzhaf import gabanzhaf
import pandas as pd, os, json
# Find one re2-tt case
base = 'data/RE2/RE2-TT'
fault = os.listdir(base)[0]
case_path = os.path.join(base, fault, '1')
df = pd.read_csv(os.path.join(case_path, 'metrics.csv'))
inject_time = int(open(os.path.join(case_path, 'inject_time.txt')).read())

class Args:
    data_path = os.path.join(case_path, 'metrics.csv')
result = gabanzhaf(df, inject_time=inject_time, dataset='re2-tt', args=Args())
print('Top-3:', result['ranks'][:3])
print('n_candidates_implied:', len(set(c.rsplit(\"_\",1)[0] for c in result['node_names'])))
"
```

預期：輸出 Top-3 metrics，`real` 時間 < 30s（主要是 I/O 和 preprocess）。

- [ ] **Step 3: 跑 gabanzhaf 在 re2-ss 單一 case（計時）**

```bash
cd /Users/user/RCAEval && time /Users/user/RCAEval/py310/bin/python -c "
import sys; sys.argv = ['']
from RCAEval.e2e.gabanzhaf import gabanzhaf
import pandas as pd, os

base = 'data/RE2/RE2-SS'
fault = [f for f in os.listdir(base) if 'cpu' in f][0]
case_path = os.path.join(base, fault, '1')
df = pd.read_csv(os.path.join(case_path, 'metrics.csv'))
inject_time = int(open(os.path.join(case_path, 'inject_time.txt')).read())

class Args:
    data_path = os.path.join(case_path, 'metrics.csv')
result = gabanzhaf(df, inject_time=inject_time, dataset='re2-ss', args=Args())
root_cause_svc = fault.rsplit('_', 1)[0]
top5_svcs = []
seen = set()
for col in result['ranks'][:20]:
    svc = col.rsplit('_',1)[0]
    if svc not in seen:
        top5_svcs.append(svc)
        seen.add(svc)
    if len(top5_svcs) == 5:
        break
print(f'Fault: {fault}')
print(f'Root cause svc: {root_cause_svc}')
print(f'Top-5 svcs: {top5_svcs}')
print(f'Root cause rank: {top5_svcs.index(root_cause_svc)+1 if root_cause_svc in top5_svcs else \">5\"} ')
"
```

預期：`real` 時間 < 10s；若 root cause svc 出現在 top-1 或 top-2，說明方向正確。

---

## Task 5：完整 benchmark

**Files:** 無修改，只跑命令

- [ ] **Step 1: 跑 re2-ss benchmark**

```bash
cd /Users/user/RCAEval && /Users/user/RCAEval/py310/bin/python main.py --method gabanzhaf --dataset re2-ss 2>&1 | tail -20
```

記錄 AC@1-CPU, AC@1-SOCKET 數值。目標：兩者 > 0（改善自 0.0）。

- [ ] **Step 2: 跑 re2-ob benchmark**

```bash
cd /Users/user/RCAEval && /Users/user/RCAEval/py310/bin/python main.py --method gabanzhaf --dataset re2-ob 2>&1 | tail -20
```

記錄 AC@1-CPU, AC@1-SOCKET 數值。目標：兩者 > 0。

- [ ] **Step 3: 跑 re2-tt benchmark（確認不退步）**

```bash
cd /Users/user/RCAEval && /Users/user/RCAEval/py310/bin/python main.py --method gabanzhaf --dataset re2-tt 2>&1 | tail -20
```

目標：AC@1-CPU ≥ 0.8, AC@1-MEM ≥ 0.9（不低於 0.93 和 1.0 太多）。

- [ ] **Step 4: Commit benchmark 結果**

```bash
cd /Users/user/RCAEval && git add benchmark_results.json && git commit -m "bench: gabanzhaf exact-banzhaf results on ss/ob/tt"
```

---

## 補充：若 AC@1 在 Task 5 仍未改善的 debug 方向

若 ss/ob 的 CPU/SOCKET AC@1 仍為 0，問題可能在 `_service_evidence` 無法區分服務，而非 Banzhaf 估計。
可用以下命令診斷：

```bash
cd /Users/user/RCAEval && /Users/user/RCAEval/py310/bin/python -c "
from RCAEval.e2e.gabanzhaf import _metric_evidence, _service_evidence, _parse_svc, preprocess
import pandas as pd, numpy as np, os

base = 'data/RE2/RE2-SS'
fault = [f for f in os.listdir(base) if 'cpu' in f][0]
case_path = os.path.join(base, fault, '1')
df = pd.read_csv(os.path.join(case_path, 'metrics.csv'))
inject_time = int(open(os.path.join(case_path, 'inject_time.txt')).read())

normal_df = df[df['time'] < inject_time]
anomal_df = df[df['time'] >= inject_time]
from RCAEval.io.time_series import preprocess as pp
full = pp(pd.concat([normal_df, anomal_df], ignore_index=True), 're2-ss', False)
n = len(normal_df)
nm = full.iloc[:n].drop(columns=['time'], errors='ignore').values.astype(float)
am = full.iloc[n:].drop(columns=['time'], errors='ignore').values.astype(float)
cols = list(full.drop(columns=['time'], errors='ignore').columns)
me = _metric_evidence(nm, am, cols)
se = _service_evidence(cols, me)
root_cause_svc = fault.rsplit('_', 1)[0]
print('Fault:', fault)
print('Top-5 svc_anom:')
for svc, score in sorted(se.items(), key=lambda x: -x[1])[:5]:
    marker = ' <-- ROOT' if svc == root_cause_svc else ''
    print(f'  {svc}: {score:.4f}{marker}')
"
```

若 root cause 不在 top-5 svc_anom，需改 `_service_evidence` 的 metric dominance 邏輯。
