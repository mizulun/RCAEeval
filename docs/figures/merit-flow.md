# GaBanzhaf 流程圖

論文用的整體流程概覽（精簡版）。可在 mermaid.live 或支援 Mermaid 的 markdown 檢視器渲染，匯出 SVG/PNG。

```mermaid
flowchart TD
    IN["① 輸入<br/>Metrics 時序 + Traces 呼叫紀錄"]

    EV["② 異常證據<br/>切窗(注入前/後)，算每個服務的異常程度"]

    TOPO["③ 因果拓樸<br/>用時序相關 + 呼叫關係<br/>建立「誰影響誰」的有向圖"]

    GAME["④ 機率因果賽局<br/>挑出候選根因，用 noisy-OR + Banzhaf<br/>衡量每個服務「解釋全系統異常」的貢獻度"]

    RANK["⑤ 根因排名<br/>Banzhaf 貢獻度 × metric 異常分數"]

    OUT(["根因排名清單"])

    IN --> EV --> TOPO --> GAME --> RANK --> OUT
```

## 對應程式碼

| 步驟 | 主要函式 (`RCAEval/e2e/gabanzhaf.py`) |
|---|---|
| ① 輸入 + Trace 特徵 | `_extract_trace_features` / `_extract_call_graph` |
| ② 異常證據 | `_split_raw_windows` → `_metric_evidence` → `_service_evidence` |
| ③ 因果拓樸 | `_build_topology` + `_call_graph_topology_edges` → `_upstream_maps` |
| ④ 機率因果賽局 | `_residual_scores` → `_select_exact_candidates` → `_causal_explanation_matrix` → `_probabilistic_banzhaf` |
| ⑤ 根因排名 | `_rank_metrics` |
