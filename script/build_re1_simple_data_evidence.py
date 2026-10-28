"""Bundle a run123.py RE1 batch (``--prefer-simple-data``) with its provenance into one evidence file.

Usage (from the repository root, after the batch finished):

    PYTHONPATH=. python script/build_re1_simple_data_evidence.py \
        --run123-json output/benchmarks/re1-simple-data-20261028.json \
        --batch re1-simple-data-20261028 \
        --output evidence/data/re1-merit-baro-simple-data-20261028.json

The output keeps the aggregated run123 records untouched under ``records`` and adds:
``manifest`` (git commit / dirty state, dataset release, exact command, window,
environment, denominators) and ``cases`` (per case: input files with sha256, root
service, service-level rank of the root, top-5 services). Service-level scoring
follows main.py: metric names are mapped to ``<service>`` by ``split("_")[0]`` with
``-db`` stripped, duplicates removed in order. Written with LF line endings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from script.create_artifact_manifest import git_state

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIRS = {"re1-ob": "online-boutique", "re1-ss": "sock-shop-2", "re1-tt": "train-ticket"}
# Zenodo record 13305663 (RCAEval RE1 release), checksums as published by the Zenodo API.
ZENODO_RELEASE = {
    "record": "https://zenodo.org/records/13305663",
    "doi": "10.5281/zenodo.13305663",
    "archives_md5_declared_by_zenodo": {
        "online-boutique.zip": "ed1c9989b57365ea78b72eb93d17bff9",
        "sock-shop-2.zip": "75c8c577be9671692330eb9dc220a1b3",
        "train-ticket.zip": "d585407d184e8da50bf9d01cd0fc928b",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def service_of(metric: str) -> str:
    return metric.split("_")[0].replace("-db", "")


def dedup_services(ranks: list[str]) -> list[str]:
    seen, out = set(), []
    for s in (service_of(m) for m in ranks):
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def case_records(method: str, dataset: str, length: int) -> list[dict]:
    result_dir = REPO_ROOT / "output" / "runs" / method / dataset / "results"
    data_root = REPO_ROOT / "data" / DATASET_DIRS[dataset]
    out = []
    for case_dir in sorted(p for p in data_root.glob("*/*") if p.is_dir()):
        service_fault, rep = case_dir.parent.name, case_dir.name
        root_service, fault = service_fault.rsplit("_", 1)
        result = result_dir / f"{service_fault}_{rep}.json"
        input_file = case_dir / ("simple_data.csv" if (case_dir / "simple_data.csv").is_file() else "data.csv")
        ranks = json.loads(result.read_text(encoding="utf-8"))["0"]
        services = dedup_services(ranks)
        rank = services.index(root_service) + 1 if root_service in services else None
        frame = pd.read_csv(input_file, usecols=["time"])
        inject = int((case_dir / "inject_time.txt").read_text().split()[0])
        out.append({
            "case": f"{service_fault}/{rep}",
            "root_service": root_service,
            "fault": fault,
            "input_file": input_file.name,
            "input_sha256": {input_file.name: sha256(input_file), "inject_time.txt": sha256(case_dir / "inject_time.txt")},
            "inject_time": inject,
            "rows_before_after": [int((frame["time"] < inject).sum()), int((frame["time"] >= inject).sum())],
            "window_rows_each_side": length * 60 // 2,
            "root_service_rank": rank,
            "top5_services": services[:5],
            "top5_metrics": ranks[:5],
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run123-json", type=Path, required=True)
    ap.add_argument("--batch", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--length", type=int, default=20)
    ap.add_argument("--command", default="python run123.py --prefer-simple-data --methods merit baro --re1 --workers 3 --output output/benchmarks/<batch>.csv")
    ap.add_argument("--run-commit", default=None,
                    help="Commit the batch was executed at, when it differs from HEAD at build time (e.g. this script was committed afterwards)")
    args = ap.parse_args()

    records = json.loads(args.run123_json.read_text(encoding="utf-8"))
    cases = {}
    for r in records:
        cases.setdefault(r["method"], {})[r["dataset"]] = case_records(r["method"], r["dataset"], args.length)

    git = git_state(REPO_ROOT)
    manifest = {
        "batch": args.batch,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": "RE1 MERIT/BARO rerun on the curated simple_data.csv input (issue #12); does not replace evidence/data/re1-merit-baro.json",
        "git": git,
        "run_commit": args.run_commit or git["sha"],
        "command": args.command.replace("<batch>", args.batch),
        "harness": "run123.py -> algo_timer.py -> main.py --prefer-simple-data; scorer = main.py service-level Evaluator (RCAEval.benchmark.evaluation), AC@k / Avg@5 per fault",
        "input_selection": "simple_data.csv when present in the case directory, else data.csv (RE1-OB has no simple_data.csv)",
        "window": {"length_minutes": args.length, "rows_each_side": args.length * 60 // 2, "tdelta_seconds": 0},
        "dataset_release": ZENODO_RELEASE,
        "denominators": {ds: {"cases": len(cases[next(iter(cases))][ds]), "per_fault": 25} for ds in DATASET_DIRS},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "lock_file": "requirements_py310.lock",
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "line_endings": "LF",
    }
    payload = {"manifest": manifest, "records": records, "cases": cases}
    text = json.dumps(payload, ensure_ascii=False, indent=1) + "\n"
    args.output.write_bytes(text.encode("utf-8"))  # bytes: no platform newline translation
    print(json.dumps({"output": str(args.output), "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                      "git_sha": git["sha"], "dirty": git["dirty"], "records": len(records),
                      "cases": {m: {d: len(v) for d, v in dd.items()} for m, dd in cases.items()}}, indent=1))


if __name__ == "__main__":
    main()
