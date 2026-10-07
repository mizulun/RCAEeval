# RE2-OB 診斷：為何完整 MERIT（27/90）遠低於 direct-sum（59/90）

日期：2026-09-30。資料來源：`artifacts/formal-rerun-20260930/main-20260930/`（正式 protocol 重跑，未重跑 benchmark）。
所有數字皆可由本目錄腳本重算（見 §8）；除「CAP 反事實」外，其餘數字皆直接取自已存輸出。

## 0. 一句話結論

OB 的 63 個 full miss **全部**是同一個 infra 偽服務（redis，快取）排在第 1 名；原因不在 Banzhaf，也不在 dominance，而是出在更上游的 metric 證據層：
redis 的 `diskio` 欄位是稀疏突波計數器（正常期中位數 0、IQR = 0，但 std ≈ 7000），robust z 的分母退回 1，
偏離量因此以原始單位（bytes/s，約 2×10⁴）計算，壓過所有真實服務（根因 CPU 的 rz 約 90）。direct-sum 之所以贏，是因為
redis 沒有 spans，拿不到 trace 分數，**trace 加法等於隱性地過濾掉 infra 偽服務**；並不是 trace 找根因比 Banzhaf 準。
SS 也有同樣的 artifact（session-db_diskio），只是 SS 沒有 trace，full 與 direct 一起受害，所以看不出差距。

## 1. 資料與正確性核對

| 核對項目 | 結果 |
|---|---|
| 腳本重算的 AC@1（full / direct / no-trace / singleton / no-metric） | 27 / 59 / 29 / 31 / 33，與 summary.md 一致 |
| 以 `merit._service_evidence` 從已存的 `metric_evidence` 重算 w | 90/90 完全一致（OB、SS、TT 皆是） |
| 以 runner 的 `prepare_case` + `_preprocess_windows` + `_metric_evidence` 重算 metric evidence | 90/90 完全一致（三個資料集皆是） |
| 以 `_merit_core` 加上已存的 trace 輸入重跑 game | full ranking 與已存輸出逐欄一致：OB/SS/TT 皆 90/90；singleton AC@1 重現 31/35/72 |
| direct-sum 的 metric 項等於 full 的 w | 90/90（`direct_metric_service_evidence` ≡ `metric_service_evidence`） |

## 2. Q1：full 輸、direct 贏的 case 中，排在根因前面的是誰

- 配對結果：direct-only 命中 **37**、full-only 命中 **5**（McNemar 在 paired-direct-vs-full.json：a_only 5 / b_only 37）。
- **37/37 的 full 第 1 名都是 redis**（infra 偽服務：沒有 spans、不在呼叫圖中、不是根因的 victim，也不是 coverage parent）。
  根因在 full 的名次：第 2 名 34 件、第 3 名 3 件。
- 擴大到全部 63 個 full miss：**63/63 的第 1 名都是 redis**；根因名次為 2（55 件）、3（5）、4（2）、5（1）。這也解釋了為什麼 AC@3 仍有 96.7%。

依 fault type 分組：

| fault | direct-only（full 第 1 名 = redis） | full-only | full miss 中 redis 為第 1 名 | OB w-argmax 是 redis |
|---|---:|---:|---:|---:|
| cpu | 7 | 0 | 12 | 11/15 |
| mem | 5 | 1 | 6 | 6/15 |
| disk | 8 | 0 | 12 | 11/15 |
| socket | 9 | 0 | 15 | 15/15 |
| delay | 1 | 4 | 9 | 9/15 |
| loss | 7 | 0 | 9 | 8/15 |
| 合計 | 37 | 5 | 63 | 60/90 |

在 37 件 direct-only 中，redis 自己的最高分欄位有 36 件是 `diskio`、1 件是 `mem`；全 case 最高分的 metric 欄位是 `redis_diskio` 的有 33 件。

### 逐 case 證據（w、β、trace 皆取自 diagnostics；β 未正規化；trace 為 direct 的正規化 trace 項；direct = w + trace）

| case | fault | root rank full / direct / by w | full #1 | #1 w / β / trace / direct | root w / β / trace / direct | top metric column | full rank under CAP |
|---|---|---|---|---|---|---|---:|
| checkoutservice_cpu_2 | cpu | 2 / 1 / 1 | redis | 0.549 / 0.549 / 0.000 / 0.549 | 1.000 / 0.432 / 0.150 / 1.150 | checkoutservice_cpu | 1 |
| checkoutservice_loss_1 | loss | 3 / 1 / 2 | redis | 1.000 / 1.000 / 0.000 / 1.000 | 0.178 / 0.114 / 1.000 / 1.178 | redis_diskio | 1 |
| checkoutservice_loss_3 | loss | 3 / 1 / 2 | redis | 1.000 / 1.000 / 0.000 / 1.000 | 0.204 / 0.118 / 1.000 / 1.204 | redis_diskio | 2 |
| checkoutservice_mem_2 | mem | 2 / 1 / 1 | redis | 0.451 / 0.451 / 0.000 / 0.451 | 1.000 / 0.328 / 1.000 / 2.000 | checkoutservice_mem | 1 |
| currencyservice_cpu_1 | cpu | 2 / 1 / 2 | redis | 1.000 / 1.000 / 0.000 / 1.000 | 0.180 / 0.180 / 1.000 / 1.180 | redis_diskio | 3 |
| currencyservice_cpu_2 | cpu | 2 / 1 / 2 | redis | 1.000 / 1.000 / 0.000 / 1.000 | 0.227 / 0.253 / 1.000 / 1.227 | redis_diskio | 4 |
| currencyservice_cpu_3 | cpu | 2 / 1 / 2 | redis | 1.000 / 1.000 / 0.000 / 1.000 | 0.212 / 0.212 / 1.000 / 1.212 | redis_diskio | 4 |
| currencyservice_delay_3 | delay | 2 / 1 / 2 | redis | 1.000 / 1.000 / 0.000 / 1.000 | 0.113 / 0.128 / 0.949 / 1.062 | redis_diskio | 1 |
| currencyservice_disk_1 | disk | 2 / 1 / 2 | redis | 1.000 / 1.000 / 0.000 / 1.000 | 0.204 / 0.227 / 1.000 / 1.204 | redis_diskio | 1 |
| currencyservice_disk_2 | disk | 2 / 1 / 2 | redis | 1.000 / 1.000 / 0.000 / 1.000 | 0.226 / 0.248 / 1.000 / 1.226 | redis_diskio | 1 |
| currencyservice_disk_3 | disk | 2 / 1 / 2 | redis | 1.000 / 1.000 / 0.000 / 1.000 | 0.223 / 0.241 / 1.000 / 1.223 | redis_diskio | 1 |
| currencyservice_loss_1 | loss | 2 / 1 / 2 | redis | 1.000 / 1.000 / 0.000 / 1.000 | 0.265 / 0.301 / 1.000 / 1.265 | redis_diskio | 1 |
| currencyservice_loss_3 | loss | 2 / 1 / 3 | redis | 1.000 / 1.000 / 0.000 / 1.000 | 0.144 / 0.190 / 1.000 / 1.144 | redis_diskio | 3 |
| currencyservice_mem_1 | mem | 3 / 1 / 3 | redis | 1.000 / 1.000 / 0.000 / 1.000 | 0.173 / 0.189 / 1.000 / 1.173 | redis_diskio | 2 |
| currencyservice_mem_2 | mem | 2 / 1 / 2 | redis | 1.000 / 1.000 / 0.000 / 1.000 | 0.348 / 0.370 / 1.000 / 1.348 | redis_diskio | 2 |
| currencyservice_mem_3 | mem | 2 / 1 / 3 | redis | 1.000 / 1.000 / 0.000 / 1.000 | 0.176 / 0.195 / 1.000 / 1.176 | redis_diskio | 1 |
| currencyservice_socket_1 | socket | 2 / 1 / 2 | redis | 1.000 / 1.000 / 0.000 / 1.000 | 0.482 / 0.482 / 1.000 / 1.482 | redis_mem | 2 |
| currencyservice_socket_2 | socket | 2 / 1 / 2 | redis | 1.000 / 1.000 / 0.000 / 1.000 | 0.292 / 0.315 / 1.000 / 1.292 | redis_diskio | 2 |
| currencyservice_socket_3 | socket | 2 / 1 / 2 | redis | 1.000 / 1.000 / 0.000 / 1.000 | 0.252 / 0.278 / 1.000 / 1.252 | redis_diskio | 2 |
| emailservice_loss_1 | loss | 2 / 1 / 2 | redis | 1.000 / 1.000 / 0.000 / 1.000 | 0.427 / 0.427 / 1.000 / 1.427 | redis_diskio | 1 |
| emailservice_loss_2 | loss | 2 / 1 / 2 | redis | 1.000 / 1.000 / 0.000 / 1.000 | 0.523 / 0.553 / 1.000 / 1.523 | redis_diskio | 1 |
| emailservice_loss_3 | loss | 2 / 1 / 2 | redis | 1.000 / 1.000 / 0.000 / 1.000 | 0.578 / 0.633 / 1.000 / 1.578 | redis_diskio | 1 |
| productcatalogservice_cpu_1 | cpu | 2 / 1 / 2 | redis | 1.000 / 1.000 / 0.000 / 1.000 | 0.261 / 0.312 / 1.000 / 1.261 | redis_diskio | 1 |
| productcatalogservice_cpu_3 | cpu | 2 / 1 / 2 | redis | 1.000 / 1.000 / 0.000 / 1.000 | 0.306 / 0.306 / 1.000 / 1.306 | redis_diskio | 1 |
| productcatalogservice_disk_1 | disk | 2 / 1 / 2 | redis | 1.000 / 1.000 / 0.000 / 1.000 | 0.364 / 0.423 / 1.000 / 1.364 | redis_diskio | 1 |
| productcatalogservice_disk_2 | disk | 2 / 1 / 2 | redis | 1.000 / 1.000 / 0.000 / 1.000 | 0.326 / 0.340 / 1.000 / 1.326 | redis_diskio | 1 |
| productcatalogservice_disk_3 | disk | 2 / 1 / 2 | redis | 1.000 / 1.000 / 0.000 / 1.000 | 0.368 / 0.398 / 1.000 / 1.368 | redis_diskio | 1 |
| productcatalogservice_mem_3 | mem | 2 / 1 / 2 | redis | 1.000 / 1.000 / 0.000 / 1.000 | 0.621 / 0.595 / 0.848 / 1.469 | redis_diskio | 1 |
| productcatalogservice_socket_1 | socket | 2 / 1 / 2 | redis | 1.000 / 1.000 / 0.000 / 1.000 | 0.271 / 0.337 / 1.000 / 1.271 | redis_diskio | 1 |
| productcatalogservice_socket_2 | socket | 2 / 1 / 2 | redis | 1.000 / 1.000 / 0.000 / 1.000 | 0.253 / 0.290 / 1.000 / 1.253 | redis_diskio | 1 |
| productcatalogservice_socket_3 | socket | 2 / 1 / 2 | redis | 1.000 / 1.000 / 0.000 / 1.000 | 0.257 / 0.292 / 1.000 / 1.257 | redis_diskio | 1 |
| recommendationservice_cpu_3 | cpu | 2 / 1 / 2 | redis | 1.000 / 1.000 / 0.000 / 1.000 | 0.336 / 0.240 / 0.767 / 1.103 | redis_diskio | 1 |
| recommendationservice_disk_2 | disk | 2 / 1 / 1 | redis | 0.662 / 0.662 / 0.000 / 0.662 | 1.000 / 0.656 / 0.629 / 1.629 | recommendationservice_mem | 1 |
| recommendationservice_disk_3 | disk | 2 / 1 / 2 | redis | 1.000 / 1.000 / 0.000 / 1.000 | 0.756 / 0.492 / 0.596 / 1.352 | redis_diskio | 1 |
| recommendationservice_socket_1 | socket | 2 / 1 / 2 | redis | 1.000 / 1.000 / 0.000 / 1.000 | 0.667 / 0.483 / 1.000 / 1.667 | redis_diskio | 1 |
| recommendationservice_socket_2 | socket | 2 / 1 / 2 | redis | 1.000 / 1.000 / 0.000 / 1.000 | 0.420 / 0.272 / 1.000 / 1.420 | redis_diskio | 1 |
| recommendationservice_socket_3 | socket | 2 / 1 / 2 | redis | 1.000 / 1.000 / 0.000 / 1.000 | 0.364 / 0.259 / 1.000 / 1.364 | redis_diskio | 1 |
| | | | | | | | |
| checkoutservice_mem_3 | mem | 1 / 2 / 1 | root (direct #1 = currencyservice) | — | 1.000 / 1.000 / 0.000 / 1.000 | checkoutservice_diskio | 1 |
| emailservice_delay_2 | delay | 1 / 2 / 1 | root (direct #1 = currencyservice) | — | 1.000 / 1.000 / 0.000 / 1.000 | emailservice_latency | 1 |
| emailservice_delay_3 | delay | 1 / 2 / 1 | root (direct #1 = currencyservice) | — | 1.000 / 1.000 / 0.000 / 1.000 | emailservice_latency | 1 |
| recommendationservice_delay_2 | delay | 1 / 2 / 1 | root (direct #1 = currencyservice) | — | 1.000 / 1.000 / 0.000 / 1.000 | recommendationservice_latency | 1 |
| recommendationservice_delay_3 | delay | 1 / 2 / 1 | root (direct #1 = currencyservice) | — | 1.000 / 1.000 / 0.000 / 1.000 | recommendationservice_latency | 1 |
| | | | | | | | |

說明：checkoutservice_cpu_2、checkoutservice_mem_2、recommendationservice_disk_2 這 3 件的根因在 w 上是第 1 名，是在 Banzhaf 層掉到第 2 名（見 §3 e）；其餘 34 件在 w 上就已輸給 redis。
「full rank under CAP」是 §4 反事實的結果，只用來診斷。

## 3. Q2：差異來自哪一層（逐層排除）

| 層 | 假設 | 證據 | 判定 |
|---|---|---|---|
| (a) metric 聚合（w 的定義） | full 與 direct 的 metric 項不同 | 兩者的 metric 項在 90/90 case 中完全相同 | 不是差異來源 |
| (a') dominance 項 D_j | 小系統的 dominance 懲罰根因 | 關掉 dominance 後，w 的 AC@1 只從 29 變 30；37 件中有 34 件光看聚合項 a_j 就已是 a_redis > a_root（D 為 33 件）；redis 為 w-argmax 的 60 件中，a 的中位數 redis 0.755、root 0.192 | 否定 |
| (a'') **metric evidence（robust z）** | 退化的 baseline 使用原始單位 | `redis_diskio` 在 90/90 case 中 IQR 為 0 且 std 大於 1e-6，分母退回 1；它是 60/90 case 的最高分欄位；例：currencyservice_socket_1 的 redis_diskio 正常期中位數 0、IQR 0、std 7257、非零比例 0.18，異常期 rz p95 為 2.74×10⁴、rz>3 的比例為 0.12（突波頻率與正常期相當，**並非真的異常**）；productcatalogservice_cpu_1 的根因 CPU rz p95 為 87.4、持續 99%，但 evidence 只有 0.163，redis_diskio 則為 1.000 | **主因** |
| (b) trace 證據 | direct 多加了 trace 分數，full 沒有 | full 的 w 本來就不含 trace 服務證據（設計如此：main_zh.tex:182；merit.py:1716-1719 的 anomaly_weights 只取 metric；trace_service_evidence 只在 merit.py:1472-1485 的 evidence 排序模式中使用）；redis 在 90/90 case 中沒有 trace key，也不在 call graph 或 trace 邊中；37 件中根因的 trace 名次為第 1 名 27 件、第 2 名 9 件、第 3 名 1 件 | direct 贏的**機制**：trace 項等於隱性的 infra 過濾器 |
| (c) victim demotion／候選截斷 | 根因被降權或被截掉 | 根因被 demote：0/90；根因不在 game 中：0/90（OB 的 12 個服務全部進入 game） | 否定 |
| (d) tie-breaking | 第 1、2 名同分 | β 前兩名同分：0/90 | 否定 |
| (e) Banzhaf 重疊折扣 | game 本身把根因往後推 | full 27、w 29、singleton 31；full 與 w 的 top-1 判定只在 4 件不同：3 件輸（根因的 callee 有 trace-topology 邊指向根因，例如 currency/email/payment → checkout，強度 0.33–0.66，β_root 從 1.000 降到 0.432/0.328/0.656，低於孤立的 redis 的 β = w = 0.549/0.451/0.662），1 件贏（productcatalogservice_mem_2） | 次要（淨 −2）；結構性質：**β_redis = w_redis，90/90** |

direct 的 5 件 full-only（4 件 DELAY、1 件 MEM）：根因的 w 為 1.000（w 排名第 1），但它自己的 trace 證據正規化後為 0（email/recommendation 的 delay：自身 span latency evidence = 0.0）；
currencyservice 的 trace 為 1.0，加總後 currencyservice 被推到第 1 名。可能的原因：網路延遲注入後，延遲出現在 caller 的 client span，而不是根因自己的 server span。**這點未驗證**。

## 4. 反事實 CAP（只用於診斷，**不是**建議的修法，也不能當成結果報告）

作法是把「IQR 為 0 且 std 大於 1e-6」的欄位也套上既有的 rz ≤ 10 上限（程式對 std ≤ 1e-6 本來就這樣處理），其餘設定全部不動。
full/singleton 由 `_merit_core` 搭配已存的 trace 輸入重跑。限制：trace-topology 邊用的是已存的、經 latency-floor 過濾後的清單。

| 資料集 | 設定 | w（metric-only） | direct w+trace | full MERIT | singleton |
|---|---|---:|---:|---:|---:|
| OB | 原始 | 29 | 59 | 27 | 31 |
| OB | CAP | 75 | 70 | 72 | 72 |
| SS | 原始 | 35 | 35 | 35 | 35 |
| SS | CAP | 75 | 75 | 75 | 75 |
| TT | 原始 | 71 | 73 | 73 | 72 |
| TT | CAP | 71 | 73 | 73 | 72 |

OB 在 CAP 下的逐 fault AC@1 k/15（cpu/mem/disk/socket/delay/loss）：w 12/15/15/12/10/11；direct 11/13/15/15/5/11；full 12/13/15/12/10/10。
37 件 direct-only 中，有 27 件在 CAP 下變成 full 命中。**artifact 拿掉之後，direct-sum（70）低於 full（72）與 metric-only（75）**，DELAY 則是 full 10 對 direct 5。
這是事後在測試資料上做的分析，只能證明因果機制，不能拿來選設定。若要改方法，必須換新的 protocol，並重跑全部方法與 ablation；這需要作者決定。
使用同一種 robust-z 的 baseline（例如 BARO）是否也受影響，**本分析沒有查**。

## 5. Q3：為什麼 TT／SS 沒有差距

- **TT**：也有 zero-IQR 欄位（低流量服務的 socket），但原始偏離量很小。30 個 case 中，各 case 零 IQR 欄位 rz-p95 最大值的中位數只有 2（OB 為 2.66×10⁴，SS 為 2.82×10⁴）。因此它們從未成為最高分欄位（0/90），CAP 也不改變任何結果（w 71→71、full 73→73）。
  TT 的 infra 偽服務（mongo/mysql）只在 2/90 case 是 w-argmax。direct-only 7 件、full-only 7 件，排在前面的主要是其他應用服務，屬於一般的排序誤差。
- **SS**：**同一個 artifact 也存在**。`session-db_diskio` 在 90/90 case 中 IQR 為 0，是 58/90 case 的最高分欄位；55 個 miss 中有 51 件的第 1 名是 session（evaluator 會剝掉 `-db` 後綴，見 paper_evaluation.py:127-130），CAP 下 35→75。
  但 SS 沒有 traces.csv，trace 證據為空，所以 direct ≡ metric-only；也沒有任何邊，所以 β ≡ w（90/90 同序）。三者**一起**失敗，才會看起來「相同」。
- 所以「SS／TT 相同」的意義不同：TT 是沒有 artifact；SS 是 artifact 存在，但沒有 trace 模態能遮住它。

## 6. Q4：是否為方法的結構性質（可寫進論文的限制或取捨）

1. **主因在證據層，不在 Banzhaf，也不在 dominance**：63/63 個 OB miss 都來自 robust z 在「IQR = 0 且 std > 0」時退回原始單位，失去尺度不變性；再加上 merit.py:329-335 的跨欄位 min–max 正規化，單一原始單位欄位會壓縮所有其他欄位。
   這個限制在 thesis_expanded_zh.tex:210、244 已經描述（zero IQR 改成 1、會破壞尺度不變性），這次是第一次量化出它能主導某些系統的準確率。
   它取決於系統中是否有「閒置、偶發突波、以大單位計量」的 infra 元件，跟系統大小無關；「小系統中 dominance 懲罰真根因」的假設已被否定。
2. **Attribution 的結構取捨**：coverage game 只能在「有邊相連」的服務之間重新分配責任。孤立節點只有自我覆蓋，所以 β = w（OB 90/90），game 無法修正圖外元件（沒有 spans 的 infra）的證據偏差。
   重疊折扣只會降低相連的候選，因此 OB 中有 3 件相連的根因輸給孤立的偽服務（淨 −2 vs w）。這可以誠實寫成取捨。
3. **模態不對稱**：direct-sum 的優勢來自「infra 只有 metrics、沒有 spans」這個觀測性質，等於一個隱性的候選過濾器，不代表加法融合在歸因上優於 Banzhaf。
   反過來說，當 artifact 被中和時（§4），加法融合在 DELAY 上明顯較差（5 對 10）。

## 7. Bug 與描述落差（未修改任何檔案）

- **實作與 thesis 描述不一致**：未發現。zero-IQR 的處理與 thesis_expanded_zh.tex:210 一致；full 的 w 不含 trace 服務證據，也與 main_zh.tex:182 一致。
- 誤導性註解：merit.py:317-322 說 IQR 為 0 但 std 不小的欄位「rz is still meaningful」；實際上此時 rz 是原始單位，與 thesis:244 自己的描述矛盾。只是註解問題，行為沒有錯。
- 短版論文描述不完整：main_zh.tex:150-151「正常期無變異的欄位改以替代基準計算，並對偏離倍數設定上限」，容易被讀成所有 zero-IQR 欄位都有上限；實際上只有 std ≤ 1e-6 才有上限。建議與 expanded 版對齊（由作者決定）。
- 名稱未對齊（輕微）：trace 服務證據使用 span 的原始名稱（merit.py:1210-1215），在 direct-sum／no-metric 中以 metric 前綴查找（merit.py:1455-1485、1497-1523），而既有的 `_match_service`（merit.py:633）沒有套用。
  因此 OB 的 `frontendservice` 對不到 `frontend`，frontend 的 trace 證據在這兩個 variant 中被丟掉（另有 `health`、`traceservice` 無對應）。frontend 在 OB 不是根因，所以**不影響任何 AC@1**。

## 8. 重算方式（repo root 執行，唯讀、不跑 benchmark；總耗時約 1–2 分鐘）

```bash
S=/private/tmp/claude-501/-Users-user-RCAEval/bb8f5965-5c02-4754-a46d-1f72dbc66b54/scratchpad
for d in re2-ob re2-ss re2-tt; do
  PYTHONPATH=. ./py310/bin/python $S/ob_diag.py $d     > $S/ob_diag_$d.out      # 逐 case 名次、排在前面者與角色 → rows_<ds>.json
  PYTHONPATH=. ./py310/bin/python $S/ob_wdecomp.py $d  > $S/wdecomp_$d.out      # w = a/D 分解、去 infra、去 dominance
  PYTHONPATH=. ./py310/bin/python $S/unitdenom.py $d   > $S/unitdenom_$d.out    # zero-IQR 稽核 + CAP/DROP（metric 層）
  PYTHONPATH=. ./py310/bin/python $S/game_cf.py $d     > $S/game_cf_$d.out      # _merit_core 重現 + CAP（full/singleton）
done
PYTHONPATH=. ./py310/bin/python $S/redis_raw.py        # 原始視窗統計（§3 a''）
PYTHONPATH=. ./py310/bin/python $S/banzhaf_layer.py    # Banzhaf 層 4 件
PYTHONPATH=. ./py310/bin/python $S/join37.py           # 37 件合併統計
PYTHONPATH=. ./py310/bin/python $S/unit_mag.py         # 三資料集 zero-IQR 欄位量級
PYTHONPATH=. ./py310/bin/python $S/make_table.py       # §2 逐 case 表
```
（ob_diag 的 infra 判定只用名稱規則 redis/mongo/mysql；SS 的 `session-db` 不在規則內，所以 SS 的數字改由 unitdenom／rows 的第 1 名統計。）

## 9. 英文 Discussion 草稿（不含服務名與檔名）

**Why does direct evidence summation outperform MERIT on one system?**
On one benchmark system, summing normalized metric and trace evidence placed the root cause first in 59 of 90 cases, whereas MERIT did so in 27. A per-case analysis attributes this gap to the evidence layer rather than to the attribution game. In all 63 cases that MERIT missed, the same infrastructure component, a key–value cache that emits resource metrics but no spans, was ranked first, and the root cause was ranked second in 55 of them. The cache's evidence came from a sparse disk-I/O counter. In the normal window, its median and interquartile range were both zero, but its standard deviation was not. Our robust scorer replaces a zero IQR with one, so deviations in this column were measured in raw units, on the order of 10^4, rather than relative to the baseline. After cross-column normalization, this column outranked genuine resource anomalies whose robust deviations were on the order of 10^2, even though its burst frequency in the fault window was similar to that in the normal window. The direct sum avoided this failure only because the cache produced no trace evidence: adding a trace score to every instrumented service implicitly filtered out uninstrumented components. The same artifact appeared on a second system, where a session store's disk-I/O counter was ranked first in 51 of 55 misses. Because that system provides no traces, the direct sum and MERIT failed in the same cases. The third system was unaffected because its zero-IQR columns were low-magnitude counters.

These results suggest two limitations. First, the robust deviation loses scale invariance for sparse, bursty counters. Section~X already notes this limitation, and these results show that it can determine top-1 accuracy on systems that contain idle infrastructure components. In a post hoc diagnostic that was not used to select any reported configuration, capping such columns in the same way as constant columns raised MERIT's top-1 hits on the affected system from 27 to 72 of 90, whereas the direct sum rose only to 70. Thus, the apparent advantage of additive fusion disappeared once the artifact was neutralized. Second, the coverage game cannot correct evidence errors for components outside the trace graph. An isolated candidate has only self-coverage, so its Banzhaf value equals its evidence weight, whereas the overlap discount can lower a connected root cause. This occurred in three cases, where lagged latency edges from the root's callees shared the root's own evidence. Attribution therefore redistributes responsibility among connected services but inherits any bias in the evidence weights of disconnected ones. Checking evidence quality for each component, for example detecting degenerate baselines before attribution, is a natural direction for future work.
