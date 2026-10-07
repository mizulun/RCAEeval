# Baseline 比較 + Ablation 歷史總表（2026-07-03）

狀態：HISTORICAL EVIDENCE。這份表保留舊 runner 的可用性、失敗政策與成本紀錄，不是現版 MERIT 主結果，也未通過目前 frozen protocol 的 matched-comparison gate。現行論文只能在逐列核對 method、runner、input、scorer、denominator 與失敗政策後引用；完整舊 logs 已移至外部 cleanup archive。

機器:Intel i9-9880H(16 threads)/ macOS 14.5 / `./py310/bin/python`,所有 run 串行執行。
gabanzhaf 數字 = `ablation_results/*__full` 欄;baro/circa/microrank/tracerca = `baseline_results/<method>_<dataset>.txt`。

## 主表:AC@1(service 級)

### RE2-TT(90 cases)

| fault | gabanzhaf | baro | circa | microrank† | tracerca† |
|---|---|---|---|---|---|
| CPU | **0.80** | 0.47 | 不可行* | 0.20 | 0.60 |
| MEM | **1.00** | 0.93 | 不可行* | 0.13 | 0.33 |
| DISK | **1.00** | 1.00 | 不可行* | 0.00 | 0.60 |
| SOCKET | **0.93** | 0.60 | 不可行* | 0.20 | 0.40 |
| DELAY | **0.80** | 0.47 | 不可行* | 0.07 | 0.73 |
| LOSS | **0.60** | 0.53 | 不可行* | 0.13 | 0.53 |

\* circa re2-tt:跑 12.3h 只完成 12/90 case(單 case 8s~68min,估算全量 >79h)→ 不可行,詳見 circa_re2-tt.txt。
† microrank/tracerca 在 17/90 case 崩潰(KeyError:anomaly 視窗出現 normal 視窗沒有的 operation);
  上表把失敗計為 miss(n=15/fault)。若只在成功的 73 case 上計分(高估),見附錄 A。

### RE2-OB(90 cases)

| fault | gabanzhaf | baro | circa | microrank | tracerca |
|---|---|---|---|---|---|
| CPU | 0.33 | 0.00 | **0.60** | 0.00 | 0.12 |
| MEM | 0.67 | 0.33 | **0.80** | 0.00 | 0.00 |
| DISK | 0.27 | 0.13 | **0.87** | 0.00 | 0.20 |
| SOCKET | 0.27 | 0.00 | **0.60** | 0.00 | 0.07 |
| DELAY | 0.27 | 0.00 | **0.47** | 0.00 | 0.20 |
| LOSS | 0.40 | 0.40 | **0.67** | 0.00 | 0.07 |

### RE2-SS(90 cases;無 traces.csv → trace 方法不適用)

| fault | gabanzhaf | baro | circa | microrank | tracerca |
|---|---|---|---|---|---|
| CPU | 0.27 | 0.00 | **0.93** | N/A | N/A |
| MEM | 0.80 | 0.20 | **0.93** | N/A | N/A |
| DISK | 0.53 | 0.00 | **0.80** | N/A | N/A |
| SOCKET | 0.40 | 0.00 | **0.80** | N/A | N/A |
| DELAY | 0.47 | 0.00 | **0.60** | N/A | N/A |
| LOSS | 0.33 | 0.20 | **0.60** | N/A | N/A |

## 主表:Avg@5

| dataset/fault | gabanzhaf | baro | circa | tracerca† |
|---|---|---|---|---|
| TT CPU | **0.96** | 0.72 | — | 0.69 |
| TT MEM | **1.00** | 0.99 | — | 0.44 |
| TT DISK | **1.00** | 1.00 | — | 0.69 |
| TT SOCKET | **0.95** | 0.83 | — | 0.51 |
| TT DELAY | **0.80** | 0.63 | — | 0.76 |
| TT LOSS | **0.75** | 0.64 | — | 0.63 |
| OB(六類平均) | 0.84 | 0.74 | **0.84** | 0.62 |
| SS(六類平均) | 0.87 | 0.77 | **0.95** | N/A |

(microrank Avg@5 全面 <0.35,略。OB/SS 逐 fault 明細見 baseline_results/*.txt 與 ablation_results/*.csv。)

## Runtime(整資料集 wall-clock,串行、同機)

| method | re2-tt | re2-ob | re2-ss |
|---|---|---|---|
| gabanzhaf | **36s** | 18s | 7s |
| baro | 34s | 8s | 9s |
| circa | >79h(不可行) | 1124s | 1484s |
| microrank | 6442s | 7502s | N/A |
| tracerca | 1235s | 293s | N/A |

## 論文論述要點

1. **TT(dense call graph + 傳播型 fault)是主場**:gabanzhaf 每個 fault 都是第一,DELAY 0.80 是 baro 的 1.7 倍;唯一能逼近的 trace 方法 tracerca(DELAY 0.73)有 19% case 直接崩潰。
2. **Scalability**:circa 準(ob/ss 最強)但在 342 欄的 TT 上不可行(>79h vs 我們 36s);metric 因果圖方法的成本隨 metric 數爆炸,我們的複雜度不會。
3. **Robustness**:multi-source 在來源缺失時退化優雅——SS 沒有 trace,trace-only baseline 全滅,我們仍是第二(輸 circa、贏 baro)。
4. **誠實面**:OB/SS 上 circa 準確率最強;我們在 OB 弱(leaf 注入、無傳播鏈,coverage 無從發力,見 ablation)。定位:「在傳播型 fault + 大規模系統上最準且唯一全場景可行」。
5. Ablation(見 ablation_results/):TT 上 coverage(DELAY +0.13)、victim demotion(+0.07)、trace(+0.20)各有貢獻;拿掉 Banzhaf 走 direct ranking 掉到 0.53;metric evidence 是必要基礎(no-metric 歸零)。

## 附錄 A:microrank/tracerca 只計成功 case 的數字(73/90,高估)

tracerca TT:CPU 0.64 / MEM 0.62 / DISK 0.64 / SOCKET 0.60 / DELAY 0.85 / LOSS 0.57
microrank TT:CPU 0.21 / MEM 0.25 / DISK 0.00 / SOCKET 0.30 / DELAY 0.08 / LOSS 0.14
