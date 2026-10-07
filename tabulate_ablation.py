#!/usr/bin/env python3
"""Turn output/ablation-runs/<dataset>__<mode>.txt files into comparison tables.

For each dataset, prints a Markdown table of fault (rows) x mode (columns) for
the chosen metric, and writes the same as CSV next to the inputs.

Usage:
  ./py310/bin/python tabulate_ablation.py                 # Avg@5 tables
  ./py310/bin/python tabulate_ablation.py --metric AC@1   # pick a metric
  ./py310/bin/python tabulate_ablation.py --metric all    # every metric
  ./py310/bin/python tabulate_ablation.py --dir output/ablation-runs
"""
import argparse
import csv
import glob
import os
import re
from collections import defaultdict

FAULTS = ["CPU", "MEM", "DISK", "SOCKET", "DELAY", "LOSS"]
METRICS = ["AC@1", "AC@3", "AC@5", "Avg@5"]
LINE = re.compile(r"(AC@1|AC@3|AC@5|Avg@5)-([A-Z]+):\s+([\d.]+)")


def parse(path):
    """Return {(metric, fault): value} from one result file."""
    out = {}
    with open(path) as f:
        for line in f:
            m = LINE.search(line)
            if m:
                out[(m.group(1), m.group(2))] = float(m.group(3))
    return out


def table(data, dataset, metric, modes):
    """Markdown rows (fault x mode) + matching list-of-rows for CSV."""
    head = ["fault"] + modes
    rows = [head]
    for fault in FAULTS:
        row = [fault]
        for mode in modes:
            v = data[mode].get((metric, fault))
            row.append("" if v is None else f"{v:g}")
        rows.append(row)
    md = [f"### {dataset} — {metric}", "",
          "| " + " | ".join(head) + " |",
          "| " + " | ".join("---" for _ in head) + " |"]
    for r in rows[1:]:
        md.append("| " + " | ".join(r) + " |")
    return "\n".join(md), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="output/ablation-runs")
    ap.add_argument("--metric", default="Avg@5",
                    help="AC@1 | AC@3 | AC@5 | Avg@5 | all")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*__*.txt")))
    if not files:
        raise SystemExit(f"no ablation files in {args.dir}/ (run run_ablation.sh first)")

    # data[dataset][mode] = {(metric, fault): value}
    data = defaultdict(dict)
    mode_order = defaultdict(list)
    for fp in files:
        base = os.path.basename(fp)[:-4]            # strip .txt
        dataset, mode = base.split("__", 1)
        data[dataset][mode] = parse(fp)
        if mode not in mode_order[dataset]:
            mode_order[dataset].append(mode)

    metrics = METRICS if args.metric == "all" else [args.metric]
    for dataset in sorted(data):
        modes = mode_order[dataset]
        for metric in metrics:
            md, rows = table(data[dataset], dataset, metric, modes)
            print(md + "\n")
            csv_path = os.path.join(args.dir, f"{dataset}__{metric.replace('@','')}.csv")
            with open(csv_path, "w", newline="") as f:
                csv.writer(f).writerows(rows)
    print(f"CSV written to {args.dir}/<dataset>__<metric>.csv")


if __name__ == "__main__":
    main()
