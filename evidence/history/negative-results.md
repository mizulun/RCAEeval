# MERIT 歷史負面結果與不可外推邊界

狀態：HISTORICAL EVIDENCE；整理日期：2026-10-07。這是從已封存的混合版本文件提煉出的最小保留紀錄，不是目前待辦、方法規格或採納建議。完整原文及舊表在 `/Users/user/RCAEval-cleanup-archive-20261005/files/docs/`。

## 必須保留的負面結果

- 2026-06-10 的三種 trace/game fusion 在當時 OB CPU、DISK、SOCKET、DELAY 上超過 0.02 AVG@5 regression budget，因此沒有採用；這是舊版本結果，不能宣稱為現版效果。
- 2026-06-22 controlled audit 的 270 cases 中，關閉 trace scores 或 residual suppression 沒有改變 top-1 與 ordered top-5；TT 候選集合雖變，排名未變。這只否證當時實作的有效性，不代表任何 trace 設計都無效。
- 2026-06-29 trace/frontier 實驗在部分 SOCKET／LOSS top-k 有改善，但 CPU／LOSS AC@1 退步；90 個 TT cases 中 frontier bonus 啟動 4 次、rescue candidate 3 次、新增候選 0 次。不能外推成 frontier 概念普遍無效。
- 2026-06-30 first-case graph cache 與「每個 case 不重讀圖」的建議，已被 per-case pre-fault graph 契約取代；舊 234→37 秒不能當作現行圖建構成本。
- 2026-09-01 零模型分析只指出 TT 失敗可能涉及 service schema 與近零尺度，沒有支持普遍的 locally-normal-root 假說；後續應以 2026-09-30 OB diagnosis 和 parameter sweep 為準。
- trace 白名單可能移除真根因；conditional Granger、call-rate、conditional victim 與 BRF 都是未採納探索，不是現版缺少但待補的元件。
- 舊結果曾暴露輸出污染與 dummy fallback 風險，因此不同方法輸出必須隔離，失敗不得偽裝成排名，ground truth 不得參與方法打分。

## 2026-09-09 training-free 受控比較

- 全面改用 standard deviation 在污染或延遲條件下會明顯退步；只處理 `IQR=0` 在 SS／OB 有收益，但 TT 沒有同步改善，因此不更換現行預設。
- 替換中心值、保留 IQR 時，SS 與 OB 的 AC@1 分別下降 11.11 與 8.89 個百分點；trace 中心化沒有帶來 AC@1 改善。
- median/std 在 SS／OB 的開發資料上提升很大，但 TT 的 AC@1 略降，且沒有獨立留出資料；它是後續研究訊號，不是正式採納證據。
- p-value edge filtering 在 shared-trend null 下 397/400 次拒絕，不能把顯著位移直接稱為因果邊。
- 同矩陣的 singleton／Shapley 比較與 1080 筆舊結果等價檢查已記錄於 `evidence/merit-training-free-20260909/`。

## 現版設計稽核限制

- 2026-09-08 design audit 顯示取消 coverage threshold 在 OB 與 TT 分別造成約 16.67、23.33 個百分點 AC@1 退步，但 SS 無變化；只支持該批次的門檻機制，不證明參數最優。
- coverage rescue 在 TT 新增候選，但該批次沒有提高真根因 candidate recall；不得寫成已實證救回根因。
- Shapley 在 TT 多修正一個案例；現行維持 closed-form Banzhaf 是版本與計算設計選擇，不是「所有資料上準確率最佳」的結論。

## 2026-10-07 correctness 稽核

- 修正前 `first_case_pre_fault` 的 cache key 含目前評估案例 boundary，無法固定第一案例圖；既有 first-case 與三個 case-order sensitivity 結果不得支持原固定圖主張，原始 artifact 與 manifest 保持不改寫，須由修正版另批重跑。
- graph provenance 過去會以目前案例冒充共享圖來源；現版 first-case 記固定 reference，union 記完整排序後來源集合與 exposure-manifest digest。
- Python API 的 `use_trace=False` 過去仍可能讀 trace；env-based `GABANZHAF_DISABLE_TRACE=1` 正式結果不受此入口缺陷影響。只有直接使用 kwargs 的批次需另行盤點。
- 以修正後 evaluator 重新讀取 9/30 `timing-ss-full-20260930` 的 90 份 raw diagnostics（未重跑案例、未改寫 artifact）：canonical stage-total median 為 `0.037757073529064655` 秒、p95 為 `0.04217985058203339` 秒；舊式把 `trace` 與四個 trace substages 同時計入時分別為 `0.037861330434679985` 與 `0.04229198945686221` 秒。個別 stage median、stage share、method wall-clock 與 end-to-end 數值不因這項加總修正作廢。

## 來源與版本

- 舊合併沿革：`docs/historical-evidence.md`，SHA-256 `022a7e39051d8f9d173463956e16997319a2fff1e2962c56c50d18a2bd3dd107`（已封存）。
- training-free 報告：`docs/merit-training-free-improvements-zh.md`，SHA-256 `b98ad7b3d790edeb54970381156056177e95c13ff1078a9c75b3d2f171d7c3b2`（已封存）；正式保留 manifest SHA-256 `b5445078cb86936a0739c0b50c8dd17c14dbceaaea4dd96e3b86c7ceba58ba76`。
- design-audit manifest SHA-256 `0f1e4b0dc7ee56f3b99018b371bbccc89ac36e2b0c4877b743534e03e8f0c812`。
- formal rerun main manifest SHA-256 `a2b55f5779f33da78c46db7510a4f431803ef37d690c0b8fd65d3009d596672c`。
- parameter-sweep raw manifest SHA-256 `c4e2f4b89a2b09b12e069755470f7fd6a83e3e3fc755bacc72b45c3736029d1b`。

原始來源的逐檔雜湊以外部 archive 的 `checksums.sha256` 為準；此摘要不得取代各批次 manifest 的版本、輸入與驗收欄位。
