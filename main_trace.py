"""Trace-method benchmark harness (microrank / tracerca).

A copy of main.py that feeds each case's traces.csv (span dataframe) instead
of simple_metrics.csv — main.py itself is the shared paper baseline and must
not change. Spans are window-sliced to ±length minutes around inject_time
(same window metric methods get), and inject_time is converted to
microseconds to match span startTime units. The evaluation section is copied
verbatim from main.py so AC@K/Avg@5 are computed identically.

Usage:
    ./py310/bin/python main_trace.py --method microrank --dataset re2-tt
"""
import argparse
import glob
import json
import os
import warnings
from datetime import datetime
from os.path import basename, dirname, join

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm

from RCAEval.benchmark.evaluation import Evaluator
from RCAEval.classes.graph import Node
from RCAEval.utility import dump_json, load_json
from RCAEval.e2e.microrank import microrank
from RCAEval.e2e.tracerca import tracerca

METHODS = {"microrank": microrank, "tracerca": tracerca}


def parse_args():
    parser = argparse.ArgumentParser(description="RCAEval trace-method evaluation")
    parser.add_argument("--method", type=str, choices=sorted(METHODS), help="Choose a method.")
    parser.add_argument("--dataset", type=str, help="Choose a dataset.", choices=[
        "re2-ob", "re2-ss", "re2-tt"
    ])
    parser.add_argument("--length", type=int, default=20, help="Window length in minutes")
    parser.add_argument("--tdelta", type=int, default=0, help="Simulated anomaly-detection delay")
    parser.add_argument("--test", action="store_true", help="Smoke test on first 2 cases")
    parser.add_argument("--result-dir", type=str, default=None, help="Override result output directory")
    return parser.parse_args()


args = parse_args()

DATASET_MAP = {
    "re2-ob": "data/RE2/RE2-OB",
    "re2-ss": "data/RE2/RE2-SS",
    "re2-tt": "data/RE2/RE2-TT",
}
dataset = DATASET_MAP[args.dataset]

data_paths = sorted(glob.glob(os.path.join(dataset, "**/traces.csv"), recursive=True))
if not data_paths:
    raise SystemExit(f"no traces.csv found under {dataset}")
if args.test is True:
    data_paths = data_paths[:2]

output_path = "output"
result_path = args.result_dir if args.result_dir else join(output_path, "results")
os.makedirs(result_path, exist_ok=True)


def process(data_path):
    data_dir = dirname(data_path)
    service, metric = basename(dirname(data_dir)).split("_")
    case = basename(data_dir)
    rp = join(result_path, f"{service}_{metric}_{case}.json")

    with open(join(data_dir, "inject_time.txt")) as f:
        inject_time = int(f.readlines()[0].strip()) + args.tdelta

    spans = pd.read_csv(data_path)
    # spans use microsecond timestamps; slice the same ±length window metric
    # methods get, then hand inject_time to the method in microseconds
    inject_us = inject_time * 1_000_000
    window_us = args.length * 60 * 1_000_000
    spans = spans[
        (spans["startTime"] >= inject_us - window_us)
        & (spans["startTime"] < inject_us + window_us)
    ]

    func = METHODS[args.method]
    try:
        out = func(spans, inject_us, dataset=args.dataset)
        dump_json(filename=rp, data={0: out.get("ranks")})
    except Exception as e:
        print(f"{args.method} failed on {data_path}: {e}")
        with open(join(result_path, f"{service}_{metric}_{case}_failed.json"), "w") as f:
            json.dump({"error": str(e)}, f)


start_time = datetime.now()

for data_path in tqdm(data_paths):
    process(data_path)

end_time = datetime.now()
time_taken = end_time - start_time
avg_speed = round(time_taken.total_seconds() / len(data_paths), 2)


# ======== EVALUTION (copied verbatim from main.py) ===========
rps = glob.glob(join(result_path, "*.json"))
services = sorted(list(set([basename(x).split("_")[0] for x in rps])))
faults = sorted(list(set([basename(x).split("_")[1] for x in rps])))

s_evaluator_all = Evaluator()
f_evaluator_all = Evaluator()
s_evaluator_cpu = Evaluator()
f_evaluator_cpu = Evaluator()
s_evaluator_mem = Evaluator()
f_evaluator_mem = Evaluator()
s_evaluator_lat = Evaluator()
f_evaluator_lat = Evaluator()
s_evaluator_loss = Evaluator()
f_evaluator_loss = Evaluator()
s_evaluator_io = Evaluator()
f_evaluator_io = Evaluator()
s_evaluator_socket = Evaluator()
f_evaluator_socket = Evaluator()

for service in services:
    for fault in faults:
        for rp in rps:
            s, m = basename(rp).split("_")[:2]
            if s != service or m != fault:
                continue  # ignore

            data = load_json(rp)
            if "error" in data:
                continue  # ignore

            for i, ranks in data.items():
                s_ranks = [Node(x.split("_")[0].replace("-db", ""), "unknown") for x in ranks]
                # remove duplication
                old_s_ranks = s_ranks.copy()
                s_ranks = (
                    [old_s_ranks[0]]
                    + [
                        old_s_ranks[i]
                        for i in range(1, len(old_s_ranks))
                        if old_s_ranks[i] not in old_s_ranks[:i]
                    ]
                    if old_s_ranks
                    else []
                )

                f_ranks = [Node(x.split("_")[0], x.split("_")[1] if "_" in x else "unknown") for x in ranks]

                if fault == "cpu":
                    s_evaluator_cpu.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                    f_evaluator_cpu.add_case(ranks=f_ranks, answer=Node(service, fault))
                elif fault == "mem":
                    s_evaluator_mem.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                    f_evaluator_mem.add_case(ranks=f_ranks, answer=Node(service, fault))
                elif fault == "delay":
                    s_evaluator_lat.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                    f_evaluator_lat.add_case(ranks=f_ranks, answer=Node(service, "latency"))
                elif fault == "loss":
                    s_evaluator_loss.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                    f_evaluator_loss.add_case(ranks=f_ranks, answer=Node(service, "latency"))
                elif fault == "disk":
                    s_evaluator_io.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                    f_evaluator_io.add_case(ranks=f_ranks, answer=Node(service, "diskio"))
                elif fault == "socket":
                    s_evaluator_socket.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                    f_evaluator_socket.add_case(ranks=f_ranks, answer=Node(service, "socket"))


print("--- Evaluation results ---")
for name, s_evaluator, f_evaluator in [
    ("cpu", s_evaluator_cpu, f_evaluator_cpu),
    ("mem", s_evaluator_mem, f_evaluator_mem),
    ("io", s_evaluator_io, f_evaluator_io),
    ("socket", s_evaluator_socket, f_evaluator_socket),
    ("delay", s_evaluator_lat, f_evaluator_lat),
    ("loss", s_evaluator_loss, f_evaluator_loss),
]:
    if name == "io":
        name = "disk"

    if s_evaluator.average(5) is not None:
        print(f"AC@1-{name.upper()}:".ljust(12), round(s_evaluator.accuracy(1), 2))
        print(f"AC@3-{name.upper()}:".ljust(12), round(s_evaluator.accuracy(3), 2))
        print(f"AC@5-{name.upper()}:".ljust(12), round(s_evaluator.accuracy(5), 2))
        print(f"Avg@5-{name.upper()}:".ljust(12), round(s_evaluator.average(5), 2))

print("---")
print("Avg speed:", avg_speed)
