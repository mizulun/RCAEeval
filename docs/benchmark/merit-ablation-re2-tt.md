# MERIT ablation study — re2-tt（2026-07-08）

論文用三成分 ablation：**no-metric**、**no-trace**、**no-banzhaf**，對照 **full**（現行 default）。

## 變體定義

| 表中名稱 | 環境變數 | 意義 |
|---|---|---|
| full | （無，default） | 完整 MERIT：metric evidence + trace topology + Banzhaf 排序 |
| no-metric | `GABANZHAF_RANK_MODE=trace_evidence` | 最終排序只用 per-service trace evidence，不用 metric |
| no-trace | `GABANZHAF_DISABLE_TRACE=1` | call graph 清空，coverage 與 trace-topology edges 全部歸零 |
| no-banzhaf | `GABANZHAF_RANK_MODE=service_evidence_trace` | metric+trace 等權正規化和直接排序，跳過 Banzhaf game（script mode 名為 `metric+trace`） |

## AC@1

| fault | full | no-metric | no-trace | no-banzhaf |
| --- | --- | --- | --- | --- |
| CPU | 0.80 | 0.00 | 0.80 | 0.80 |
| MEM | 1.00 | 0.00 | 1.00 | 1.00 |
| DISK | 1.00 | 0.00 | 1.00 | 1.00 |
| SOCKET | 0.93 | 0.00 | 0.87 | 0.87 |
| DELAY | 0.80 | 0.00 | 0.60 | 0.53 |
| LOSS | 0.60 | 0.00 | 0.53 | 0.53 |
| **mean** | **0.86** | **0.00** | **0.80** | **0.79** |

## Avg@5

| fault | full | no-metric | no-trace | no-banzhaf |
| --- | --- | --- | --- | --- |
| CPU | 0.96 | 0.03 | 0.96 | 0.96 |
| MEM | 1.00 | 0.00 | 1.00 | 1.00 |
| DISK | 1.00 | 0.03 | 1.00 | 1.00 |
| SOCKET | 0.95 | 0.00 | 0.97 | 0.97 |
| DELAY | 0.80 | 0.04 | 0.71 | 0.65 |
| LOSS | 0.75 | 0.00 | 0.65 | 0.65 |
| **mean** | **0.91** | **0.02** | **0.88** | **0.87** |

## 解讀

- **metric 是不可或缺的基底**：no-metric 幾乎全零（trace evidence 單獨無法定位根因，
  排序近似隨機甚至更差）。此結果與 2026-07-07 舊 run 完全一致，非單次 artifact。
- **trace 的貢獻集中在 DELAY/LOSS**：no-trace 使 DELAY AC@1 0.80→0.60、LOSS 0.60→0.53；
  資源型故障（CPU/MEM/DISK）不受影響，符合機制預期（trace topology 幫助分離傳播路徑上的受害者）。
- **Banzhaf 排序優於直接分數和**：no-banzhaf 在 DELAY AC@1 再掉到 0.53（比 no-trace 更差），
  顯示 metric+trace 的價值需要透過 Banzhaf game 的聯盟結構才能兌現，naive 等權和反而劣化。

## 附註（審稿防禦）

- no-metric 模式下，候選池挑選（`RCAEval/e2e/merit.py:990-1004`）仍以 metric 分數選 top-k，
  但 `trace_evidence` 分支（`merit.py:895-906`）的最終排序不使用該候選池，
  故輸出結果等效於完全不用 metric。
- no-trace 除清空 call graph 外，連帶關閉依賴 call graph 的 coverage（`merit.py:1015-1019`）
  與 trace-topology edges（`merit.py:552-554`），是乾淨的「無 trace 資訊」對照。

## 重現

```bash
cd /Users/user/RCAEval
bash run_ablation.sh --datasets re2-tt --modes full no-trace no-metric metric+trace
```

- 執行時間（wall, 2026-07-08）：full 35s、no-trace 29s、no-metric 34s、metric+trace 33s（合計約 2.5 分鐘）。
- 每輪開頭自動 `rm -rf output/results`（無殘留污染）；使用 `./py310/bin/python`。
- 原始輸出：`ablation_results/re2-tt__{full,no-metric,no-trace,metric+trace}.txt`（timestamp 20260708-192906）。
- 回歸檢查：full 的 DELAY AC@1 0.80 / LOSS 0.60 與現行 default 基準一致。
