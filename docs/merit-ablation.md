# MERIT Ablation and Experiment Semantics

狀態：ACTIVE；核對日期：2026-10-07。本文件定義 MERIT 比較與實驗語義；執行狀態以各批 manifest 與相關 issue 為準。

## 程式分工（2026-10-07）

- `RCAEval/e2e/merit.py`：只協調共用流程、公開 `merit(...)` 與舊私有名稱的相容轉接。
- `RCAEval/e2e/merit_config.py`：`_GabanzhafConfig`、原預設值及 kwargs 轉換。
- `RCAEval/e2e/merit_ablation.py`：環境變數開關、no-metric／no-trace 控制、替代排序／singleton／bruteforce／CUSUM、trace 擾動與 graph-source 選擇；需要核心運算時以 callback 傳入，不反向匯入核心。
- `RCAEval/e2e/merit_evidence.py`：視窗前處理、metric／service evidence、正規化與 evidence ranking。
- `RCAEval/e2e/merit_trace.py`：trace I/O、call graph、topology、cache、case input 與 `GraphArtifact` provenance。
- `RCAEval/e2e/merit_game.py`：候選、coverage、victim、矩陣、閉式 Banzhaf 與 faithfulness。
- `RCAEval/e2e/merit_diagnostics.py`：診斷 payload、可相加的 stage timing schema 與結果組裝。

環境變數名稱、kwargs、結果欄位與預設值不變。一般 kwargs 優先；`GABANZHAF_DISABLE_TRACE=1` 仍強制清除所有 trace 輸入，明確 `use_trace=False` 也在任何 trace I/O 前清除輸入；`GABANZHAF_WRITE_DIAGNOSTICS=1` 仍強制開啟診斷。coverage candidate augmentation 與 trace latency filter 使用可 monkeypatch hook，不再對 `_merit_core` 做文字替換。

這是拆檔重構加四項 correctness 修正，不是新方法或重跑結果。新研究批次記錄七檔 `method_source_sha256`，正式 manifest 的 `file_sha256` 同樣涵蓋七檔。舊 manifest 不得通過新來源集合的重用檢查；重現歷史結果須用該批快照，不能改寫舊 hash 來取得通過。固定重構前行為基準位於 `tests/fixtures/merit_refactor.json`；測試保留它作歷史基準，只排除明確修正的 first-case、provenance、stage total 與 Python `use_trace=False` 行為。

計時輸出只以 `trace`、`preprocess`、`metric_evidence`、`candidates`、`banzhaf`、`ranking` 作可加總頂層；trace 細項只在 `trace_stage_timing`。evaluator 讀舊資料時若已有 `trace` 即忽略細項，否則只合成一次。這些回歸檢查不代表重新驗證全資料集準確率或成本。

## 共同契約

主線以 script/run_paper_experiments.py → script/run_paper_dataset.py 傳共同 full-record boundary 與每案例 pre-fault graph。run_ablation.sh／run123.py 是歷史或探索 runner；沒有核對 boundary、graph、scorer 與失敗政策前，其結果不能與正式 protocol-aligned batch 混用。必要負面結果見 [精選歷史證據](../evidence/history/negative-results.md)。

Primary 為 service AC@1；另報 AC@3/5、AVG@5、有效分母、execution success、wall time。逐 case 配對，統計及 seed 條件見 [凍結協定](research-protocol.md)。開發期 5,000-resample exploratory interval 不冒充 v14 10,000-resample 與完整 Holm family。

## 比較矩陣

| 比較 | 實際介入 | 要回答的問題／避免的混淆 |
|---|---|---|
| full | frozen defaults；closed_form、victim on | 每個 batch 都保存 config／source，名稱相同不保證版本相同 |
| no_trace | GABANZHAF_DISABLE_TRACE=1；清空全部 trace-derived inputs，後段 game 保留 | 模態移除；不是只關動態邊 |
| no_metric | GABANZHAF_DISABLE_METRICS=1；metric values 不作證據，欄名作輸出 schema | SS 無 trace 時 N/A/no_evidence；不以 tie 命中當準確率 |
| direct sum／no-banzhaf | GABANZHAF_RANK_MODE=service_evidence_trace；各自正規化的 metric+trace sum，跳過整個 game | 架構比較，不是只關閉 Banzhaf 的單元消融；metric-only direct 另列 |
| singleton | GABANZHAF_BANZHAF_MODE=individual；同 candidates、A、w、coverage、victim | 隔離重疊折減／歸因規則，與 direct sum 不同 |
| Shapley | research runner 同矩陣歸因 | 不改官方 default，也不宣稱是 production CLI 的另一預設 |
| early／dominance／aggregation | 單一前端因素替換 | 固定其餘權重及 boundary，保留無收益與退步 |
| coverage／victim 2×2 | full、no_coverage、no_victim、no_C_no_V | 說明 parent protection、矩陣與候選的聯動；不要把歷史 factorial 當現版 matched |
| rescue_only／edges_only | audit 固定 coverage-derived victim 邊與 parent protection | 區分候選召回与矩陣重排；記 base/rescued/final K 與 recall |
| lag0／shuffle／trace perturbation | 使用對應 runner 的明確控制 | 相似度不是因果；negative control 未顯示差異時收縮主張 |
| brute force | 同 game 對照 closed form | 數值等價及 kernel timing；不作新定位方法的 accuracy baseline |

可用組態名以 [正式 runner](../script/run_paper_experiments.py)、[設計 audit](../script/merit_design_audit.py)、[follow-up](../script/merit_followup.py) 為準；跨 runner 名稱不可互換。已完成的主要比較見 [9/30 摘要](../artifacts/formal-rerun-20260930/summary.md)。

## 穩健性、敏感度與成本

- follow-up 已含污染 5/10/20%、delay 60/120/180 秒、horizon 60/180/300 秒；與 frozen v14 的 delay/window job grid 不同，不能合併分母或進度。所有模態依相應邊界同步處理，unsupported 明列。
- [9/30 常數掃描](../artifacts/param-sweep-20260930/summary.md) 使用現版來源；default reproduction、零 IQR 與其餘係數分開報。掃描最佳值不回填 headline。
- 歷史混合權重與 factorial 的少量聚合輸入保存在 [`evidence/history/legacy-benchmarks/`](../evidence/history/legacy-benchmarks/)；原生成表已外部封存。它們屬不同版本的開發證據，不能據此宣稱現版組件必然有效。
- 成本分 attribution kernel、含 I/O 的 E2E、task wall 與 cold/warm；重跑前先估 case 數×組態×repetitions，固定硬體／workers／BLAS，不把多 worker 35 秒當單行程成本。

先核對現存產物與 V-01/V-03 缺口再決定是否補跑；本輪文件整合不新增或自動執行實驗。
