#!/usr/bin/env python3
"""Turn run_sensitivity_sweep.sh output into the thesis sensitivity tables.

Reads output/sensitivity-results/<dataset>__<param>__<value>.txt and writes one
markdown table per parameter: rows are sweep values, columns are per-dataset
macro AC@1 and Avg@5 (the mean over fault types).

This reports sensitivity.  It does not pick values: the defaults were fixed
before the sweep and are only marked so the reader can see where they sit.

    ./py310/bin/python script/tabulate_sensitivity.py
"""

from __future__ import annotations

import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "output" / "sensitivity-results"
OUT = ROOT / "output" / "generated-docs" / "sensitivity-sweep-table.md"

# Defaults as of RCAEval/e2e/merit.py:_GabanzhafConfig (lines 33-80).
DEFAULTS = {
    "top_k_svcs": "15",
    "exact_candidate_k": "12",
    "coverage_conf": "0.7",
    "coverage_loud_thr": "0.4",
    "victim_floor": "0.5",
    "trace_topology_conf": "0.85",
}
PARAM_ORDER = list(DEFAULTS)
DATASET_ORDER = ["re2-tt", "re2-ob", "re2-ss"]

METRIC_RE = re.compile(r"^\s*(AC@1|Avg@5)-([A-Z]+):\s*([0-9.]+)\s*$")
WALL_RE = re.compile(r"^\s*\[wall:\s*(\d+)s\]\s*$")


def parse(path: Path) -> dict:
    """Macro-average each metric over fault types; also return wall seconds."""
    buckets: dict[str, list[float]] = {"AC@1": [], "Avg@5": []}
    wall = None
    for line in path.read_text().splitlines():
        m = METRIC_RE.match(line)
        if m:
            buckets[m.group(1)].append(float(m.group(3)))
            continue
        m = WALL_RE.match(line)
        if m:
            wall = int(m.group(1))
    return {
        "ac1": statistics.fmean(buckets["AC@1"]) if buckets["AC@1"] else None,
        "avg5": statistics.fmean(buckets["Avg@5"]) if buckets["Avg@5"] else None,
        "wall": wall,
        "n_faults": len(buckets["AC@1"]),
    }


def num(value: str) -> float:
    return float(value)


def main() -> None:
    runs: dict[tuple[str, str, str], dict] = {}
    for path in sorted(RESULTS.glob("*__*__*.txt")):
        dataset, param, value = path.stem.split("__")
        if param == "canary":
            continue
        runs[(dataset, param, value)] = parse(path)

    if not runs:
        raise SystemExit(f"no sweep results under {RESULTS}")

    lines = [
        "# MERIT 超參數敏感度掃描",
        "",
        "由 `run_sensitivity_sweep.sh` 產生，`script/tabulate_sensitivity.py` 製表。",
        "",
        "**這是事後敏感度分析，不是超參數選擇程序。**預設值在掃描前即已固定"
        "（`RCAEval/e2e/merit.py:33-80`），表中標記 **粗體** 只是讓讀者看出預設值"
        "落在區間何處，不表示它是依據本表選出的。",
        "",
        "AC@1 與 Avg@5 為 fault type 的 macro 平均。",
        "",
        "重跑指令：",
        "",
        "```bash",
        "JOBS=6 bash run_sensitivity_sweep.sh",
        "./py310/bin/python script/tabulate_sensitivity.py",
        "```",
        "",
    ]

    total_wall = sum(r["wall"] or 0 for r in runs.values())
    lines += [
        f"本次掃描共 {len(runs)} runs，累計 CPU wall-clock "
        f"{total_wall}s（約 {total_wall / 3600:.1f} 小時，6 路平行）。",
        "",
    ]

    for param in PARAM_ORDER:
        values = sorted(
            {v for (_d, p, v) in runs if p == param}, key=num
        )
        if not values:
            continue
        datasets = [d for d in DATASET_ORDER if any((d, param, v) in runs for v in values)]

        lines.append(f"## `{param}`（預設 {DEFAULTS[param]}）")
        lines.append("")
        header = "| 值 | " + " | ".join(
            f"{d} AC@1 | {d} Avg@5" for d in datasets
        ) + " |"
        sep = "|---|" + "---:|" * (2 * len(datasets))
        lines += [header, sep]

        for value in values:
            label = f"**{value}**" if value == DEFAULTS[param] else value
            cells = []
            for dataset in datasets:
                run = runs.get((dataset, param, value))
                if run is None or run["ac1"] is None:
                    cells += ["—", "—"]
                else:
                    cells += [f"{run['ac1']:.3f}", f"{run['avg5']:.3f}"]
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")

        # Spread per dataset drives the one-line verdict.
        spreads = []
        for dataset in datasets:
            got = [
                runs[(dataset, param, v)]["ac1"]
                for v in values
                if (dataset, param, v) in runs
                and runs[(dataset, param, v)]["ac1"] is not None
            ]
            if len(got) >= 2:
                spreads.append((dataset, (max(got) - min(got)) * 100))
        if spreads:
            worst_ds, worst = max(spreads, key=lambda item: item[1])
            detail = "、".join(f"{d} {s:.1f} pp" for d, s in spreads)
            verdict = (
                "結果對此參數不敏感"
                if worst < 2.0
                else "屬敏感參數，論文須明確報告此區間"
            )
            lines += [
                f"AC@1 在掃描區間內的變動幅度：{detail}"
                f"（最大為 {worst_ds} 的 {worst:.1f} pp）。{verdict}。",
                "",
            ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT} ({len(runs)} runs)")


if __name__ == "__main__":
    main()
