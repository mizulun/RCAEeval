#!/usr/bin/env python
"""Factorial ablation analysis for MERIT.

Reads the JSON produced by run123.py and reports, per fault type, each
factor's main effect (matched pairs) and each factor pair's second-order
interaction. Positive interaction = the two steps reinforce each other;
negative = they are redundant or fight each other.

Usage:
    ./py310/bin/python script/analyze_merit_factorial.py \
        --input evidence/history/legacy-benchmarks/benchmark_merit_factorial_2026-08-31.json \
        --output output/generated-docs/merit-factorial-ablation.md
"""
import argparse
import itertools
import json
import os
from collections import defaultdict

FAULT_TYPES = ["CPU", "MEM", "DISK", "SOCKET", "DELAY", "LOSS"]
FACTORS = ["M", "T", "C", "V", "B"]
# T is ordinal: off < graph < full. Others are binary. None = structural no-op
# in that cell (see the "structural no-op" section of the generated report).
T_ORDER = {"off": 0, "graph": 1, "full": 2}

CELLS = {
    "merit_full":                    (1, "full", 1, 1, 1),
    "merit_no_victim_demotion":      (1, "full", 1, 0, 1),
    "merit_no_coverage":             (1, "full", 0, 1, 1),
    "merit_fx_noC_noV":              (1, "full", 0, 0, 1),
    "merit_no_trace_topology":       (1, "graph", 1, 1, 1),
    "merit_fx_noTT_noV":             (1, "graph", 1, 0, 1),
    "merit_fx_noTT_noC":             (1, "graph", 0, 1, 1),
    # V is a no-op here: with C=0 and no topology edges, causal_edges is empty.
    "merit_fx_noTT_noC_noV":         (1, "graph", 0, None, 1),
    "merit_no_trace":                (1, "off", None, None, 1),
    "merit_no_metric":               (0, "full", None, 1, 1),
    "merit_fx_noM_noV":              (0, "full", None, 0, 1),
    "merit_fx_noM_noTT":             (0, "graph", None, None, 1),
    "merit_evidence_metric":         (1, "off", None, None, 0),
    "merit_evidence_trace":          (0, "full", None, None, 0),
    "merit_evidence_service_trace":  (1, "full", None, None, 0),
}

# Cells whose evidence is identically zero, so the ranking is a degenerate tie
# rather than a weaker method. Listed in the grid for completeness but excluded
# from every effect estimate -- averaging them in would inflate the other
# factor's apparent contribution. Verified by probe: svc_anom_dict has 0
# non-zero entries for merit_fx_noM_noTT on re2-tt (M=0 removes metric
# evidence, T=graph removes trace evidence, leaving only the call graph).
DEGENERATE = {"merit_fx_noM_noTT"}

FACTOR_DOC = """## 因子定義

| 代號 | 因子 | Level | pipeline 位置 | 關掉時 fallback |
|---|---|---|---|---|
| **M** | metric evidence | 1 / 0 | `merit.py:_merit_core`、`merit_evidence.py` | 服務分數改由 trace service evidence 提供 |
| **T** | trace 資訊 | full / graph / off | `merit_trace.py`、`merit.py:_prepare_case_inputs` | 見下 |
| **C** | call-graph coverage | 1 / 0 | `merit_game.py:_coverage_parents` | 無 quiet-root 救援邊 |
| **V** | victim demotion | 1 / 0 | `merit_game.py:_victim_factors` | 自我覆蓋維持 1.0，不因上游異常收縮 |
| **B** | Banzhaf 聚合 | 1 / 0 | `merit.py:_merit_core`、`merit_game.py` | 直接以證據分數排序，不建 game |

### T 的三個 level 實際差在哪（與旗標名稱不一致，容易誤解）

`GABANZHAF_DISABLE_TRACE_TOPOLOGY=1` 的名字看起來只關 topology 邊，實際上
`_trace_topology_from_case_dir` 會直接回傳**完全空的** feature bundle
（`merit.py:_prepare_case_inputs`），trace service evidence 一併消失。三個 level 的真實內容：

| Level | trace 證據 | topology 邊 | call graph（供 coverage 用） |
|---|---|---|---|
| `full` | 有 | 有 | 有 |
| `graph` (`DISABLE_TRACE_TOPOLOGY=1`) | **無** | 無 | **有** |
| `off` (`DISABLE_TRACE=1`) | 無 | 無 | 無 |

所以 `graph` 與 `off` 的差別只在 call graph 是否留給 coverage 使用，
**不是**「保留 trace 證據但不用 topology」。

### B=0 那三格的 M / T level 是什麼意思

`B=0`（`rank_mode` 換成 evidence 直接排序）不是靠 `use_metrics` / `use_trace`
關閉模態，而是**排序公式本身只讀其中一種證據**（`merit.py` 的 metric_evidence /
trace_evidence / service_evidence_trace 分支）。表中仍標成 M/T 的 0/1，語意統一為
「該模態的資訊有沒有進入最終排名」；但要注意 B=1 與 B=0 兩層的「關掉」是不同機制
（config 開關 vs 公式忽略），跨層比較時請以同層內的對比為主。

## 退化格

`merit_fx_noM_noTT`（M=0 且 T=graph）同時失去 metric 證據與 trace 證據，
`svc_anom_dict` 全為 0（已用 probe 實測：re2-tt 上 68 個服務 nonzero=0），
排名退化成**同分打平**——最終順序由 tie-break 決定，不反映任何方法能力。

- re2-tt：24 個指標（AC@1/3/5、Avg@5 × 6 fault type）全部精確為 0。
- re2-ob：AC@1 全部為 0，但 AC@3≈0.13-0.2、AC@5≈0.6、Avg@5≈0.21-0.24 **不為 0**。
  這不是它在 ob 上比較有效，而是 ob 的服務數少，打平順序被 top-3／top-5 撈中的
  機率本來就高——正好說明這些數字是 tie-break 的產物。

因此本格列在格表中（標 ⚠︎退化）但**排除於所有主效應與交互作用計算**：
把它平均進去會虛增 M 的表觀貢獻（re2-tt AC@1 上為 +0.626 vs 實際 +0.553）。

## 結構性 no-op（未跑的格子）

下列組合在**程式邏輯上**註定與對照格完全同值，因此不列入格表。這是程式結構決定的，
不是實驗發現，也不是遺漏。已用 `re2-tt --test` 實測驗證（四個配置輸出檔 hash 完全相同）。

| 條件 | 後果 | 依據 |
|---|---|---|
| `B=0`（rank_mode != banzhaf） | C / V / T-topology 全部不影響排名 | `merit.py:_merit_core` 的 `rank_mode == "banzhaf"` 分支 |
| `T=off` | call graph 清空 → coverage_parents 恆空 → causal_edges 恆空 → C 與 V 皆失效 | `merit.py:_prepare_case_inputs, _merit_core` |
| `M=0` | coverage 結構性失效（`if config.use_metrics and use_coverage`）；V 仍可經 trace topology 邊生效 | `merit.py:_merit_core` |
| `M=0` 且 `T=off` | 證據全零、退化排名，無資訊 → 排除 | `merit_ablation.py:metric_free_evidence` |
| `T=graph` 且 `C=0` | call graph 只透過 coverage 進入方法，C=0 時它無處可用 → 整格等同 `T=off`；且 causal_edges 的兩個來源（coverage 邊、topology 邊）皆空 → V 一併失效 | `merit.py:_merit_core` |

最後一條是本輪實測發現的（不是事前推導）：`merit_fx_noTT_noC`、
`merit_fx_noTT_noC_noV`、`merit_no_trace` 三者在 re2-tt 跑出**逐位元相同**的
24 個指標。三格都留在格表裡（數據已跑），但 `merit_fx_noTT_noC_noV` 的 V 標為 `–`，
不讓那組恆為 0 的差異稀釋 V 的主效應。

表中 `–` 代表該因子在那一格是上述 no-op，不是沒測。
"""


def load(paths):
    records = []
    for path in ([paths] if isinstance(paths, str) else paths):
        with open(path, encoding="utf-8") as f:
            records.extend(json.load(f))
    # {(dataset, method): {metric: value}}
    out = {}
    for r in records:
        if str(r.get("status", "OK")).lower() not in ("ok", "success"):
            continue
        out[(r["dataset"], r["method"])] = r
    return out


def score(rec, metric, fault):
    """One scalar: a specific fault type, or the mean over all present ones."""
    if fault != "ALL":
        return rec.get(f"{metric}-{fault}")
    vals = [rec[f"{metric}-{f}"] for f in FAULT_TYPES if f"{metric}-{f}" in rec]
    return sum(vals) / len(vals) if vals else None


# A cell where a factor is a structural no-op behaves exactly like the default
# level of that factor, so the two are comparable. C and V default to on.
DEFAULT_LEVEL = {FACTORS.index("C"): 1, FACTORS.index("V"): 1}


def comparable(a, b, idx):
    """True if cells a and b differ only at factor idx.

    A None (structural no-op) on one side matches that factor's default level
    on the other side -- e.g. merit_evidence_service_trace (B=0, where C/V
    cannot act at all) is the right partner for merit_full (B=1, C=1, V=1).
    """
    for i in range(len(FACTORS)):
        if i == idx or a[i] == b[i]:
            continue
        if a[i] is None and b[i] == DEFAULT_LEVEL.get(i):
            continue
        if b[i] is None and a[i] == DEFAULT_LEVEL.get(i):
            continue
        return False
    return a[idx] != b[idx]


def level_rank(idx, val):
    return T_ORDER[val] if idx == FACTORS.index("T") else val


def main_effects(scores, idx):
    """Matched pairs for one factor: [(hi_method, lo_method, delta), ...]."""
    pairs = []
    for m1, m2 in itertools.combinations(sorted(CELLS), 2):
        if m1 in DEGENERATE or m2 in DEGENERATE:
            continue
        c1, c2 = CELLS[m1], CELLS[m2]
        if c1[idx] is None or c2[idx] is None or not comparable(c1, c2, idx):
            continue
        if m1 not in scores or m2 not in scores:
            continue
        hi, lo = (m1, m2) if level_rank(idx, c1[idx]) > level_rank(idx, c2[idx]) else (m2, m1)
        if scores[hi] is None or scores[lo] is None:
            continue
        pairs.append((hi, lo, scores[hi] - scores[lo]))
    return pairs


def interactions(scores, ix, iy):
    """(11-10)-(01-00) over every context where the full 2x2 exists."""
    out = []
    by_context = defaultdict(dict)
    for method, cell in CELLS.items():
        if method in DEGENERATE or cell[ix] is None or cell[iy] is None:
            continue
        if method not in scores:
            continue
        if scores[method] is None:
            continue
        context = tuple(
            DEFAULT_LEVEL.get(i, cell[i]) if cell[i] is None else cell[i]
            for i in range(len(FACTORS)) if i not in (ix, iy)
        )
        by_context[context][(cell[ix], cell[iy])] = (method, scores[method])
    for context, corners in by_context.items():
        xs = sorted({k[0] for k in corners}, key=lambda v: level_rank(ix, v))
        ys = sorted({k[1] for k in corners}, key=lambda v: level_rank(iy, v))
        if len(xs) < 2 or len(ys) < 2:
            continue
        for xlo, xhi in itertools.combinations(xs, 2):
            for ylo, yhi in itertools.combinations(ys, 2):
                quad = [(xlo, ylo), (xhi, ylo), (xlo, yhi), (xhi, yhi)]
                if not all(q in corners for q in quad):
                    continue
                v00, v10, v01, v11 = (corners[q][1] for q in quad)
                out.append((context, f"{xhi}v{xlo}", f"{yhi}v{ylo}",
                            (v11 - v10) - (v01 - v00)))
    return out


def fmt(v):
    return "–" if v is None else f"{v:.3f}"


def summarise(data, datasets, metrics, out):
    """Up-front table: average main effect and interaction per factor."""
    faults = ["ALL"] + FAULT_TYPES
    out.append("\n## 摘要：主效應與交互作用\n")
    out.append("主效應 = 只有該因子不同的所有 matched pair 的平均 Δ（括號為 pair 數）。"
               "交互作用 = `(11−10)−(01−00)` 的平均；正=相輔相成，負=冗餘或互斥。\n")
    for dataset in datasets:
        present = [m for m in CELLS if (dataset, m) in data]
        if len(present) < len(CELLS):
            out.append(f"\n> **{dataset} 非完整格表**：15 格中只跑了 {len(present)} 格"
                       f"（{', '.join('`' + m + '`' for m in present)}），"
                       "下表的主效應可能僅由單一 matched pair 得出，括號內即 pair 數。\n")
        for metric in metrics:
            out.append(f"\n### {dataset} — {metric} 主效應\n")
            out.append("| 因子 | " + " | ".join(
                "總體" if f == "ALL" else f for f in faults) + " |")
            out.append("|---" * (len(faults) + 1) + "|")
            for idx, name in enumerate(FACTORS):
                cells = []
                for fault in faults:
                    sc = {m: (score(data[(dataset, m)], metric, fault)
                              if (dataset, m) in data else None) for m in CELLS}
                    pairs = main_effects(sc, idx)
                    cells.append(f"{sum(p[2] for p in pairs) / len(pairs):+.3f} ({len(pairs)})"
                                 if pairs else "–")
                out.append(f"| **{name}** | " + " | ".join(cells) + " |")

            rows = []
            for (ix, nx), (iy, ny) in itertools.combinations(list(enumerate(FACTORS)), 2):
                cells, any_val = [], False
                for fault in faults:
                    sc = {m: (score(data[(dataset, m)], metric, fault)
                              if (dataset, m) in data else None) for m in CELLS}
                    inter = interactions(sc, ix, iy)
                    if inter:
                        any_val = True
                        cells.append(f"{sum(i[3] for i in inter) / len(inter):+.3f} ({len(inter)})")
                    else:
                        cells.append("–")
                if any_val:
                    rows.append(f"| **{nx}×{ny}** | " + " | ".join(cells) + " |")
            if rows:
                out.append(f"\n### {dataset} — {metric} 二階交互作用\n")
                out.append("| 因子對 | " + " | ".join(
                    "總體" if f == "ALL" else f for f in faults) + " |")
                out.append("|---" * (len(faults) + 1) + "|")
                out.extend(rows)


def render(data, datasets, metric, out):
    for dataset in datasets:
        out.append(f"\n## {dataset} — {metric}\n")
        for fault in ["ALL"] + FAULT_TYPES:
            scores = {}
            for method in CELLS:
                rec = data.get((dataset, method))
                scores[method] = score(rec, metric, fault) if rec else None
            if all(v is None for v in scores.values()):
                continue
            label = "總體平均" if fault == "ALL" else fault
            out.append(f"\n### {dataset} / {metric} / {label}\n")
            out.append("| # | M | T | C | V | B | method | " + metric + " |")
            out.append("|---|---|---|---|---|---|---|---|")
            for i, (method, cell) in enumerate(CELLS.items(), 1):
                lv = " | ".join("–" if c is None else str(c) for c in cell)
                mark = " ⚠︎退化" if method in DEGENERATE else ""
                out.append(f"| {i} | {lv} | `{method}`{mark} | {fmt(scores[method])} |")

            out.append("\n**主效應**（matched pairs，同一列其餘因子相同）\n")
            out.append("| 因子 | 對比 | Δ | 平均 Δ |")
            out.append("|---|---|---|---|")
            for idx, name in enumerate(FACTORS):
                pairs = main_effects(scores, idx)
                if not pairs:
                    continue
                avg = sum(p[2] for p in pairs) / len(pairs)
                for j, (hi, lo, d) in enumerate(pairs):
                    cellavg = f"{avg:+.3f}" if j == 0 else ""
                    out.append(f"| {name if j == 0 else ''} | `{hi}` − `{lo}` | {d:+.3f} | {cellavg} |")

            rows = []
            for (ix, nx), (iy, ny) in itertools.combinations(list(enumerate(FACTORS)), 2):
                for context, xl, yl, val in interactions(scores, ix, iy):
                    rows.append((f"{nx}×{ny}", f"{xl} / {yl}", str(context), val))
            if rows:
                out.append("\n**二階交互作用** `(11−10)−(01−00)`；正=相輔相成，負=冗餘/互斥\n")
                out.append("| 因子對 | 對比 level | 其餘因子 context | 交互作用 |")
                out.append("|---|---|---|---|")
                for r in rows:
                    out.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:+.3f} |")


def demo():
    """Self-check: the textbook synergy case must come out positive."""
    scores = {
        "merit_fx_noC_noV": 0.20,   # C=0 V=0
        "merit_no_coverage": 0.25,  # C=0 V=1
        "merit_no_victim_demotion": 0.20,  # C=1 V=0
        "merit_full": 0.60,         # C=1 V=1
    }
    ic, iv = FACTORS.index("C"), FACTORS.index("V")
    res = interactions(scores, ic, iv)
    assert len(res) == 1, res
    assert abs(res[0][3] - 0.35) < 1e-9, res
    # main effect of V, held at C=1 and C=0
    deltas = sorted(round(d, 3) for _, _, d in main_effects(scores, iv))
    assert deltas == [0.05, 0.40], deltas
    # a factor that is None in the cell itself must never be paired
    assert main_effects({"merit_no_trace": 0.1, "merit_full": 0.9}, ic) == []
    # ...but a None elsewhere stands for the default level, so B is comparable
    # the degenerate cell must never enter an effect estimate
    im = FACTORS.index("M")
    assert main_effects({"merit_fx_noM_noTT": 0.0, "merit_no_trace_topology": 0.9}, im) == []
    ib = FACTORS.index("B")
    pairs = main_effects({"merit_full": 0.9, "merit_evidence_service_trace": 0.4}, ib)
    assert [(h, l, round(d, 3)) for h, l, d in pairs] == [
        ("merit_full", "merit_evidence_service_trace", 0.5)], pairs
    print("demo OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+",
                    default=["evidence/history/legacy-benchmarks/benchmark_merit_factorial_2026-08-31.json",
                             "evidence/history/legacy-benchmarks/benchmark_merit_factorial_ss_2026-08-31.json",
                             "evidence/history/legacy-benchmarks/benchmark_merit_factorial_ss2_2026-08-31.json"])
    ap.add_argument("--output", default="output/generated-docs/merit-factorial-ablation.md")
    ap.add_argument("--metrics", nargs="+", default=["AC@1", "Avg@5"])
    ap.add_argument("--demo", action="store_true", help="run the self-check and exit")
    args = ap.parse_args()
    if args.demo:
        demo()
        raise SystemExit(0)
    demo()
    data = load(args.input)
    datasets = list(dict.fromkeys(d for d, _ in data))
    out = ["# MERIT 全因子消融實驗", "",
           f"由 `script/analyze_merit_factorial.py` 從 "
           + ", ".join(f"`{os.path.basename(p)}`" for p in args.input) + " 產生。",
           "", FACTOR_DOC]
    summarise(data, datasets, args.metrics, out)
    out.append("\n---\n\n# 完整格表\n")
    for metric in args.metrics:
        render(data, datasets, metric, out)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"[完成] {args.output}")
