"""Generate paper/generated/re1-robustness-table-zh.tex from curated RE1 aggregate evidence.

Usage: ./py310/bin/python script/generate_re1_robustness_table.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAULTS = ["CPU", "MEM", "DISK", "DELAY", "LOSS"]
DATASETS = ["re1-ob", "re1-ss", "re1-tt"]
METHOD_LABEL = {"merit": r"\method{}", "baro": "BARO"}

rows = {(r["method"], r["dataset"]): r
        for r in json.loads((ROOT / "evidence/data/re1-merit-baro.json").read_text())}


def values(row):
    vals = [row[f"Avg@5-{f}"] for f in FAULTS]
    return vals + [sum(vals) / len(vals)]


lines = [
    "% 由 script/generate_re1_robustness_table.py 自動產生；資料來源 evidence/data/re1-merit-baro.json",
    "",
    r"\begin{table*}[!t]",
    r"\centering",
    r"\fontsize{10}{12}\selectfont",
    r"\setlength{\tabcolsep}{5pt}",
    r"\caption{\method{} 與 BARO 在 RCAEval RE1 各 fault type 的服務層級 AVG@5（\%）。"
    r"RE1 未提供 traces，\method{} 沿相同管線退化為 metric-only 組態。"
    r"每個資料集含 125 個案例（每種 fault type 25 個）；平均為五種 fault type 等權平均。"
    r"\textbf{粗體}表示同一資料集、同一 fault type 中兩方法較佳者。}",
    r"\label{tab:re1-robustness-zh}",
    r"\begin{tabular}{ll*{6}{r}}",
    r"\toprule",
    r"資料集 & 方法 & "
    + " & ".join(rf"\multicolumn{{1}}{{c}}{{{f}}}" for f in FAULTS + [r"平均"])
    + r" \\",
    r"\midrule",
]

for d, ds_label in zip(DATASETS, ["RE1-OB", "RE1-SS", "RE1-TT"]):
    pair = {m: values(rows[(m, d)]) for m in ("merit", "baro")}
    for k, method in enumerate(("merit", "baro")):
        cells = []
        for i, v in enumerate(pair[method]):
            s = f"{v * 100:.1f}"
            other = pair["merit" if method == "baro" else "baro"][i]
            if v >= other:
                s = rf"\textbf{{{s}}}"
            cells.append(s)
        head = ds_label if k == 0 else ""
        lines.append(f"{head} & {METHOD_LABEL[method]} & " + " & ".join(cells) + r" \\")
    if d != DATASETS[-1]:
        lines.append(r"\midrule")

lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]

out = ROOT / "paper" / "generated" / "re1-robustness-table-zh.tex"
out.write_text("\n".join(lines))
print(f"wrote {out}")
