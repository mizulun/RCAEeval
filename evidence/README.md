# MERIT 精選研究證據

狀態：EVIDENCE；核對日期：2026-10-05。這裡保存可 commit 的聚合證據，**不是目前方法、待辦或預設參數的權威來源**。

目前方法先讀 `docs/method-theory.md`、`docs/figures/merit-flow.md` 與程式；目前工作先讀 `docs/thesis-workplan.md`；凍結協定以 `experiment/protocol.json` 與 `docs/research-protocol.md` 為準。此目錄只可用來支持已標明批次、版本、分母與限制的敘述。

## 收錄範圍

- `formal-rerun-20260930/`：正式重跑的摘要、案例、paired comparison、manifest 與 run config。
- `merit-design-audit-20260908/`：設計稽核的聚合表、配對統計及驗證紀錄。
- `merit-followup-20260909/`：follow-up 摘要、窗口診斷與完成／驗證紀錄；不含逐案例 raw JSON。
- `merit-training-free-20260909/`：受控比較、聚合表與驗證紀錄。
- `param-sweep-20260930/`、`ob-diagnosis-20260930/`、`merit-thesis-readiness-20260930/`：敏感度、失敗診斷及來源稽核。
- `data/`：仍被論文生成器使用的小型聚合 JSON，不是重新執行方法所得的新結果。
- `history/`：歷史限制、負面結果及可重建舊生成表的少量聚合輸入。

## 禁止事項

- 不可把不同批次、runner、窗口、分母或程式版本的數字拼成新主表。
- 不可把 `INTERIM`、allow-dirty 或開發資料結果改稱 frozen reproduction 或獨立驗證。
- 不可由歷史檔名中的 `full`、`current`、`baseline` 推定它等於現版 MERIT。
- 不在此目錄新增待辦或方法規格；新實驗先建立 manifest，再由工作清單登錄。

完整舊 raw results、逐案例產物和被取代文件已移至 `/Users/user/RCAEval-cleanup-archive-20261005/`；`targets.tsv` 與 `checksums.sha256` 提供還原與校驗資訊。

本目錄 46 個收錄檔案的 commit 前校驗值列於 `source-manifest.sha256`；內容改動後必須重新產生該清單，不能沿用舊雜湊。
