"""Feasibility probe: run circa on ONE case of a dataset with a hard time cap.

Prints per-case wall time, or TIMEOUT if the cap is hit. Used to honestly
report whether CIRCA's PC graph construction scales to the 370–396 column
datasets (re2-ss / re2-tt) instead of guessing.

Usage: ./py310/bin/python probe_circa.py re2-ss [cap_seconds=600]
"""
import glob
import signal
import sys
import time

import numpy as np
import pandas as pd

DATASET_MAP = {
    "re2-ob": "data/RE2/RE2-OB",
    "re2-ss": "data/RE2/RE2-SS",
    "re2-tt": "data/RE2/RE2-TT",
}

ds = sys.argv[1]
cap = int(sys.argv[2]) if len(sys.argv) > 2 else 600
case = sorted(glob.glob(DATASET_MAP[ds] + "/*_cpu/1"))[0]
print(f"probe: {ds} case={case} cap={cap}s")

inject = int(open(case + "/inject_time.txt").read().strip())
data = pd.read_csv(case + "/simple_metrics.csv")
# same preprocessing main.py applies before calling the method
data = data.loc[:, ~data.columns.str.endswith("_latency-50")]
data = data.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
normal = data[data["time"] < inject].tail(600)
anomal = data[data["time"] >= inject].head(600)
data = pd.concat([normal, anomal], ignore_index=True)
data = data.rename(columns={c: c.replace("_latency-90", "_latency") for c in data.columns})
print(f"shape: {data.shape}")

from RCAEval.e2e.circa import circa


def on_alarm(signum, frame):
    raise TimeoutError


signal.signal(signal.SIGALRM, on_alarm)
signal.alarm(cap)
t0 = time.time()
try:
    out = circa(data, inject_time=inject, dataset=ds)
    signal.alarm(0)
    print(f"OK: {time.time() - t0:.1f}s/case  top5={out['ranks'][:5]}")
except TimeoutError:
    print(f"TIMEOUT: >{cap}s for one case -> full dataset infeasible")
