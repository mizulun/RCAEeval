"""
RCAEval 自動化測試腳本
對指定資料集執行指定方法的 RCA 測試，支援平行執行。

使用方式:
    python run123.py                              # 執行所有方法 x 所有資料集
    python run123.py --methods baro nsigma        # 只執行指定方法
    python run123.py --datasets re1-ob re2-tt     # 指定任意資料集組合
    python run123.py --re1                        # 只執行 RE1 資料集
    python run123.py --re1 --re2                  # 執行 RE1 + RE2 資料集
    python run123.py --re2 --re3                  # 執行 RE2 + RE3 資料集
    python run123.py --test                       # smoke test（每個資料集只跑前 2 筆）
    python run123.py --output results.csv         # 指定輸出摘要檔案
    python run123.py --workers 3                  # 同時執行 3 個任務（預設 3）
    python run123.py --workers 1                  # 關閉平行，循序執行
    python run123.py --methods merit_full merit_pure_banzhaf \
        --datasets re2-ob re2-ss re2-tt --workers 1 \
        --output benchmark_merit_ablation.csv     # 比較 closed-form(=full) 與純枚舉 Banzhaf
    python run123.py --methods merit_full merit_no_trace merit_no_coverage \
        --re2 --workers 1 --output benchmark_merit_ablation.csv
"""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from statistics import median
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

RE1_DATASETS = ["re1-ob", "re1-ss", "re1-tt"]
RE2_DATASETS = ["re2-ob", "re2-ss", "re2-tt"]
RE3_DATASETS = ["re3-ob", "re3-ss", "re3-tt"]
ALL_RE_DATASETS = RE1_DATASETS + RE2_DATASETS + RE3_DATASETS

DEFAULT_METHODS = [
    "baro", "nsigma", "e_diagnosis",
    "pc_pagerank", "fci_pagerank", "ges_pagerank",
    "lingam_pagerank", "granger_pagerank", "ntlr_pagerank",
    "circa", "microcause", "cloudranger", "easyrca",
    "microrank", "tracerca",
]

METHOD_VARIANTS = {
    "gabanzhaf_no_victim_demotion": {
        "base_method": "gabanzhaf",
        "env": {"GABANZHAF_DISABLE_VICTIM_DEMOTION": "1"},
        "metadata": {
            "base_method": "gabanzhaf",
            "ablation_role": "no_victim_demotion",
        },
    },
    "gabanzhaf_no_coverage": {
        "base_method": "gabanzhaf",
        "env": {"GABANZHAF_DISABLE_COVERAGE": "1"},
        "metadata": {
            "base_method": "gabanzhaf",
            "ablation_role": "no_coverage",
        },
    },
    "gabanzhaf_no_trace": {
        "base_method": "gabanzhaf",
        "env": {"GABANZHAF_DISABLE_TRACE": "1"},
        "metadata": {
            "base_method": "gabanzhaf",
            "ablation_role": "no_trace",
        },
    },
    "gabanzhaf_no_metric": {
        "base_method": "gabanzhaf",
        "env": {"GABANZHAF_DISABLE_METRICS": "1"},
        "metadata": {
            "base_method": "gabanzhaf",
            "ablation_role": "no_metric",
        },
    },
    "gabanzhaf_residual": {
        "base_method": "gabanzhaf",
        "env": {"GABANZHAF_TRACE_RESIDUAL": "1"},
        "metadata": {
            "base_method": "gabanzhaf",
            "ablation_role": "trace_residual",
        },
    },
    "gabanzhaf_evidence_service": {
        "base_method": "gabanzhaf",
        "env": {"GABANZHAF_RANK_MODE": "service_evidence"},
        "metadata": {
            "base_method": "gabanzhaf",
            "rank_mode": "service_evidence",
            "ablation_role": "evidence_service",
        },
    },
    "gabanzhaf_evidence_metric": {
        "base_method": "gabanzhaf",
        "env": {"GABANZHAF_RANK_MODE": "metric_evidence"},
        "metadata": {
            "base_method": "gabanzhaf",
            "rank_mode": "metric_evidence",
            "ablation_role": "evidence_metric",
        },
    },
    "gabanzhaf_evidence_trace": {
        "base_method": "gabanzhaf",
        "env": {"GABANZHAF_RANK_MODE": "trace_evidence"},
        "metadata": {
            "base_method": "gabanzhaf",
            "rank_mode": "trace_evidence",
            "ablation_role": "evidence_trace",
        },
    },
    "gabanzhaf_evidence_service_trace": {
        "base_method": "gabanzhaf",
        "env": {"GABANZHAF_RANK_MODE": "service_evidence_trace"},
        "metadata": {
            "base_method": "gabanzhaf",
            "rank_mode": "service_evidence_trace",
            "ablation_role": "evidence_service_trace",
        },
    },
    "gabanzhaf_pure_banzhaf": {
        "base_method": "gabanzhaf",
        "env": {"GABANZHAF_BANZHAF_MODE": "bruteforce"},
        "metadata": {
            "base_method": "gabanzhaf",
            "banzhaf_mode": "bruteforce",
            "ablation_role": "pure_banzhaf",
        },
    },
    "gabanzhaf_full": {
        "base_method": "gabanzhaf",
        "env": {},
        "metadata": {
            "base_method": "gabanzhaf",
            "trace_topology": "on",
            "ablation_role": "trace_frontier_full",
        },
    },
    "gabanzhaf_no_trace_topology": {
        "base_method": "gabanzhaf",
        "env": {"GABANZHAF_DISABLE_TRACE_TOPOLOGY": "1"},
        "metadata": {
            "base_method": "gabanzhaf",
            "trace_topology": "off",
            "ablation_role": "no_trace_topology",
        },
    },
    "gabanzhaf_k1": {
        "base_method": "gabanzhaf",
        "env": {"GABANZHAF_EXACT_CANDIDATE_K": "1"},
        "metadata": {
            "base_method": "gabanzhaf",
            "exact_candidate_k": 1,
            "ablation_role": "exact_candidate_k=1",
        },
    },
    "gabanzhaf_k2": {
        "base_method": "gabanzhaf",
        "env": {"GABANZHAF_EXACT_CANDIDATE_K": "2"},
        "metadata": {
            "base_method": "gabanzhaf",
            "exact_candidate_k": 2,
            "ablation_role": "exact_candidate_k=2",
        },
    },
    "gabanzhaf_k3": {
        "base_method": "gabanzhaf",
        "env": {"GABANZHAF_EXACT_CANDIDATE_K": "3"},
        "metadata": {
            "base_method": "gabanzhaf",
            "exact_candidate_k": 3,
            "ablation_role": "exact_candidate_k=3",
        },
    },
    "gabanzhaf_k4": {
        "base_method": "gabanzhaf",
        "env": {"GABANZHAF_EXACT_CANDIDATE_K": "4"},
        "metadata": {
            "base_method": "gabanzhaf",
            "exact_candidate_k": 4,
            "ablation_role": "exact_candidate_k=4",
        },
    },
    "gabanzhaf_k5": {
        "base_method": "gabanzhaf",
        "env": {"GABANZHAF_EXACT_CANDIDATE_K": "5"},
        "metadata": {
            "base_method": "gabanzhaf",
            "exact_candidate_k": 5,
            "ablation_role": "exact_candidate_k=5",
        },
    },
    "gabanzhaf_k6": {
        "base_method": "gabanzhaf",
        "env": {"GABANZHAF_EXACT_CANDIDATE_K": "6"},
        "metadata": {
            "base_method": "gabanzhaf",
            "exact_candidate_k": 6,
            "ablation_role": "exact_candidate_k=6",
        },
    },
    "gabanzhaf_k8": {
        "base_method": "gabanzhaf",
        "env": {"GABANZHAF_EXACT_CANDIDATE_K": "8"},
        "metadata": {
            "base_method": "gabanzhaf",
            "exact_candidate_k": 8,
            "ablation_role": "exact_candidate_k=8",
        },
    },
    "gabanzhaf_k10": {
        "base_method": "gabanzhaf",
        "env": {"GABANZHAF_EXACT_CANDIDATE_K": "10"},
        "metadata": {
            "base_method": "gabanzhaf",
            "exact_candidate_k": 10,
            "ablation_role": "exact_candidate_k=10",
        },
    },
    "gabanzhaf_k12": {
        "base_method": "gabanzhaf",
        "env": {"GABANZHAF_EXACT_CANDIDATE_K": "12"},
        "metadata": {
            "base_method": "gabanzhaf",
            "exact_candidate_k": 12,
            "ablation_role": "exact_candidate_k=12",
        },
    },
    "gabanzhaf_k16": {
        "base_method": "gabanzhaf",
        "env": {"GABANZHAF_EXACT_CANDIDATE_K": "16"},
        "metadata": {
            "base_method": "gabanzhaf",
            "exact_candidate_k": 16,
            "ablation_role": "exact_candidate_k=16",
        },
    },
    # Trace-vs-metric rebalance sweep: adjust the off-diagonal trace
    # probability cap (trace_topology_conf, base 0.85) so trace can
    # reorder the ranking.
    **{
        f"gabanzhaf_tconf_{tag}": {
            "base_method": "gabanzhaf",
            "env": {"GABANZHAF_TRACE_TOPOLOGY_CONF": val},
            "metadata": {
                "base_method": "gabanzhaf",
                "trace_topology_conf": float(val),
                "ablation_role": f"trace_topology_conf={val}",
            },
        }
        for tag, val in [("07", "0.7"), ("085", "0.85"), ("095", "0.95")]
    },
    # Victim demotion sweep: shrink an abnormal service's self-coverage toward
    # victim_floor when its causal upstream is also abnormal (propagation victim).
    **{
        f"gabanzhaf_victim_{tag}": {
            "base_method": "gabanzhaf",
            "env": {"GABANZHAF_USE_VICTIM_DEMOTION": "1", "GABANZHAF_VICTIM_FLOOR": val},
            "metadata": {
                "base_method": "gabanzhaf",
                "victim_floor": float(val),
                "ablation_role": f"victim_floor={val}",
            },
        }
        for tag, val in [("03", "0.3"), ("05", "0.5"), ("07", "0.7")]
    },
    # Factorial ablation: multi-flag cells that leave-one-out variants cannot
    # reach. Factor levels are M(etric)/T(race)/C(overage)/V(ictim)/B(anzhaf);
    # "-" marks a factor that is a structural no-op in that cell.
    # Historical aggregate inputs live under evidence/history/legacy-benchmarks/.
    **{
        f"gabanzhaf_fx_{tag}": {
            "base_method": "gabanzhaf",
            "env": {f"GABANZHAF_DISABLE_{flag}": "1" for flag in flags},
            "metadata": {
                "base_method": "gabanzhaf",
                "ablation_role": "factorial",
                "factor_levels": levels,
            },
        }
        for tag, flags, levels in [
            ("noC_noV", ["COVERAGE", "VICTIM_DEMOTION"], "M1/Tfull/C0/V0/B1"),
            ("noTT_noV", ["TRACE_TOPOLOGY", "VICTIM_DEMOTION"], "M1/Tevid/C1/V0/B1"),
            ("noTT_noC", ["TRACE_TOPOLOGY", "COVERAGE"], "M1/Tevid/C0/V1/B1"),
            ("noTT_noC_noV", ["TRACE_TOPOLOGY", "COVERAGE", "VICTIM_DEMOTION"],
             "M1/Tevid/C0/V0/B1"),
            ("noM_noTT", ["METRICS", "TRACE_TOPOLOGY"], "M0/Tevid/C-/V-/B1"),
            ("noM_noV", ["METRICS", "VICTIM_DEMOTION"], "M0/Tfull/C-/V0/B1"),
        ]
    },
    # Evidence-blend weight sweep. These weights sat hard-coded inside the
    # metric evidence stage until 2026-08-31; the factorial ablation showed that
    # stage dominates, so they need the same sensitivity treatment as the other
    # knobs. Each axis is swept low/high around its default (the default point
    # is merit_full, not repeated here).
    **{
        f"gabanzhaf_w_{axis}_{tag}": {
            "base_method": "gabanzhaf",
            "env": {f"GABANZHAF_{axis.upper()}": val},
            "metadata": {
                "base_method": "gabanzhaf",
                "ablation_role": f"{axis}={val}",
            },
        }
        for axis, values in [
            ("ev_w_mag", [("035", "0.35"), ("075", "0.75")]),
            ("ev_w_persistence", [("015", "0.15"), ("045", "0.45")]),
            ("early_w_mag", [("05", "0.5"), ("09", "0.9")]),
            ("fusion_w_early", [("04", "0.4"), ("08", "0.8")]),
            ("agg_w_p95", [("03", "0.3"), ("07", "0.7")]),
            ("svc_w_dominance", [("015", "0.15"), ("050", "0.50")]),
        ]
        for tag, val in values
    },
    # Victim demotion minus its coverage-parent exemption. Isolates how much of
    # victim demotion's value comes from the demotion itself versus from the
    # exemption that stops trace edges demoting a coverage-inferred root.
    "gabanzhaf_no_coverage_parent_protection": {
        "base_method": "gabanzhaf",
        "env": {"GABANZHAF_DISABLE_COVERAGE_PARENT_PROTECTION": "1"},
        "metadata": {
            "base_method": "gabanzhaf",
            "ablation_role": "victim_demotion_without_exemption",
        },
    },
}


def _register_merit_variants():
    """Expose merit_* names while keeping gabanzhaf_* aliases reproducible."""
    for name, variant in list(METHOD_VARIANTS.items()):
        prefix = "gabanzhaf_"
        if not name.startswith(prefix):
            continue
        merit_name = "merit_" + name[len(prefix):]
        metadata = dict(variant.get("metadata", {}))
        metadata["base_method"] = "merit"
        METHOD_VARIANTS.setdefault(
            merit_name,
            {
                "base_method": "merit",
                "env": dict(variant.get("env", {})),
                "metadata": metadata,
            },
        )


_register_merit_variants()

# 保存每個 (method, dataset) 的詳細結果（用於後續 print）
_results_store = {}
_results_lock = threading.Lock()

FAULT_TYPES = ["CPU", "MEM", "DISK", "SOCKET", "DELAY", "LOSS"]
METRICS = ["AC@1", "AC@3", "AC@5", "Avg@5"]


def parse_output(stdout):
    """解析 main.py 的輸出，提取各 fault type 的 AC@1/AC@3/AC@5/Avg@5。"""
    results = {}
    for metric in ["AC@1", "AC@3", "AC@5", "Avg@5"]:
        for fault in FAULT_TYPES:
            pattern = rf"{metric}-{fault}:\s+([\d.]+)"
            m = re.search(pattern, stdout)
            if m:
                results[f"{metric}-{fault}"] = float(m.group(1))
    m = re.search(r"Avg speed:\s+([\d.]+)", stdout)
    if m:
        results["Avg speed"] = float(m.group(1))
    # 純演算法均速（每 case 秒數，已扣除 pandas 讀檔時間；由 algo_timer.py 印出）
    m = re.search(r"Algo speed:\s+([\d.]+)", stdout)
    if m:
        results["Algo speed"] = float(m.group(1))
    return results


# 保護 print 在平行模式下不互相交錯
_print_lock = threading.Lock()


def _safe_print(text):
    with _print_lock:
        print(text, flush=True)


def result_dir_for(method, dataset):
    """每個 (method, dataset) 獨立的結果目錄。"""
    return os.path.join("output", "runs", method, dataset, "results")


def clear_result_dir(method, dataset):
    """刪除上一次同一 (method, dataset) 的結果目錄。"""
    rdir = result_dir_for(method, dataset)
    if os.path.exists(rdir):
        shutil.rmtree(rdir)


def resolve_method_variant(method):
    variant = METHOD_VARIANTS.get(method)
    if variant is None:
        return method, {}, {}
    return (
        variant["base_method"],
        dict(variant.get("env", {})),
        dict(variant.get("metadata", {})),
    )


def run_single(
    method,
    dataset,
    length=20,
    tdelta=0,
    smoke_test=False,
    diagnostics=False,
):
    base_method, extra_env, metadata = resolve_method_variant(method)
    rdir = result_dir_for(method, dataset)
    # 刪除上一次同資料集的結果
    clear_result_dir(method, dataset)

    cmd = [sys.executable, "algo_timer.py",
           "--method", base_method, "--dataset", dataset,
           "--length", str(length), "--tdelta", str(tdelta),
           "--result-dir", rdir]
    if smoke_test:
        cmd.append("--test")

    env = os.environ.copy()
    env.update(extra_env)
    env.setdefault("MPLCONFIGDIR", "/private/tmp/rcaeval-mpl-cache")
    env.setdefault("XDG_CACHE_HOME", "/private/tmp/rcaeval-cache")
    if diagnostics:
        env["GABANZHAF_WRITE_DIAGNOSTICS"] = "1"

    start = datetime.now()
    proc = subprocess.run(cmd, text=True, capture_output=True,
                          cwd=os.path.dirname(os.path.abspath(__file__)),
                          env=env)
    elapsed = round((datetime.now() - start).total_seconds(), 1)

    lines = [
        f"\n{'='*60}",
        f"  方法: {method:<20} 資料集: {dataset}  [{elapsed}s]",
        f"  指令: {' '.join(cmd)}",
        "=" * 60,
    ]
    if proc.stdout:
        lines.append(proc.stdout.rstrip())
    if proc.stderr:
        lines.append(f"[STDERR] {proc.stderr[-500:].rstrip()}")
    _safe_print("\n".join(lines))

    record = {
        "method": method, "dataset": dataset,
        "status": "OK" if proc.returncode == 0 else "FAIL",
        "wall_time_sec": elapsed, "returncode": proc.returncode,
    }
    record.update(metadata)
    if proc.returncode == 0:
        parsed = parse_output(proc.stdout)
        record.update(parsed)
        # 儲存詳細結果供後續 print
        with _results_lock:
            _results_store[(method, dataset)] = parsed

    return record


def aggregate_repetitions(records):
    """Aggregate repeated runs of one method/dataset using medians."""
    if not records:
        raise ValueError("records must not be empty")
    first = dict(records[0])
    first["repetitions"] = len(records)
    first["status"] = "OK" if all(r.get("status") == "OK" for r in records) else "FAIL"
    first["returncode"] = 0 if first["status"] == "OK" else next(
        (r.get("returncode", -1) for r in records if r.get("status") != "OK"),
        -1,
    )

    walls = [float(r.get("wall_time_sec", 0.0)) for r in records]
    first["wall_time_sec"] = float(median(walls))
    first["wall_time_min_sec"] = float(min(walls))
    first["wall_time_max_sec"] = float(max(walls))

    numeric_keys = {
        key
        for record in records
        for key, value in record.items()
        if (
            key in ("Avg speed", "Algo speed")
            or any(key.startswith(f"{metric}-") for metric in METRICS)
        )
        and isinstance(value, (int, float))
    }
    for key in numeric_keys:
        values = [float(r[key]) for r in records if key in r]
        first[key] = float(median(values))
        if key == "Avg speed":
            first["Avg speed min"] = float(min(values))
            first["Avg speed max"] = float(max(values))
    return first


def run_repeated(
    method,
    dataset,
    repetitions=1,
    length=20,
    tdelta=0,
    smoke_test=False,
    diagnostics=False,
):
    runs = [
        run_single(
            method,
            dataset,
            length=length,
            tdelta=tdelta,
            smoke_test=smoke_test,
            diagnostics=diagnostics,
        )
        for _ in range(max(1, int(repetitions)))
    ]
    return aggregate_repetitions(runs)


def save_csv(records, path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if not records:
        return
    metadata_keys = [
        key
        for key in [
            "base_method",
            "banzhaf_mode",
            "rank_mode",
            "exact_candidate_k",
            "trace_topology",
            "trace_topology_conf",
            "victim_floor",
            "ablation_role",
        ]
        if any(key in record for record in records)
    ]
    keys = ["method"] + metadata_keys + [
        "dataset",
        "status",
        "repetitions",
        "wall_time_sec",
        "wall_time_min_sec",
        "wall_time_max_sec",
    ]
    metric_keys = sorted({
        k for r in records for k in r
        if any(k.startswith(f"{m}-") for m in METRICS) or k in ("Avg speed", "Algo speed")
    })
    keys += metric_keys + ["Avg speed min", "Avg speed max", "returncode"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)
    print(f"[完成] 結果已儲存至 {path}")


def json_path(csv_path):
    return os.path.splitext(csv_path)[0] + ".json"


def save_json(records, csv_path):
    path = json_path(csv_path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"[完成] 結果已儲存至 {path}")


def load_and_show(path):
    """從 JSON 檔載入結果並印出。"""
    jpath = json_path(path) if path.endswith(".csv") else path
    if not os.path.exists(jpath):
        print(f"[錯誤] 找不到結果檔：{jpath}")
        print("  請先執行測試，或指定正確的 --output 路徑。")
        return
    with open(jpath, encoding="utf-8") as f:
        records = json.load(f)
    methods = list(dict.fromkeys(r["method"] for r in records))
    datasets = list(dict.fromkeys(r["dataset"] for r in records))
    print(f"\n[載入] {jpath}  （共 {len(records)} 筆）")
    print_table(records)
    print_per_dataset_results(records, methods, datasets)


def print_block(method, dataset, rec, label=None):
    """
    印出單一 (method, dataset) 的結果區塊。
    直排：Fault Type（CPU/MEM/IO/SOCKET/DELAY/LOSS）
    橫排：AC@1  AC@3  AC@5  Avg@5
    """
    title = label or f"{method}  ×  {dataset}"
    col_w = 8
    sep = "-" * (14 + col_w * len(METRICS))

    print(f"\n  {title}")
    print(f"  {sep}")
    hdr = f"  {'Fault':<12}"
    for m in METRICS:
        hdr += f" {m:>{col_w}}"
    print(hdr)
    print(f"  {sep}")

    for fault in FAULT_TYPES:
        row = f"  {fault:<12}"
        for metric in METRICS:
            val = rec.get(f"{metric}-{fault}", "-")
            row += f" {str(val):>{col_w}}"
        print(row)

    print(f"  {sep}")
    status = rec.get("status", "?")
    elapsed = rec.get("wall_time_sec", "-")
    algo = rec.get("Algo speed")
    algo_note = f"  純演算法均速: {algo}s/case" if algo is not None else ""
    print(f"  狀態: {status}  耗時: {elapsed}s{algo_note}")


def print_table(records):
    if not records:
        return

    ok = sum(1 for r in records if r["status"] == "OK")
    print("\n" + "=" * 60)
    print("測試摘要")
    print(f"共 {len(records)} 個任務，成功 {ok}，失敗 {len(records)-ok}")
    print("=" * 60)

    for r in records:
        print_block(r["method"], r["dataset"], r)

    print("\n" + "=" * 60)


def print_per_dataset_results(records, methods, datasets):
    """依資料集分組，每個資料集底下列出各方法的結果區塊。"""
    print("\n" + "=" * 60)
    print("各資料集詳細結果")

    for dataset in datasets:
        print(f"\n{'━'*60}")
        print(f"  資料集: {dataset}")
        print(f"{'━'*60}")
        for method in methods:
            rec = next((r for r in records
                        if r["method"] == method and r["dataset"] == dataset), None)
            if rec is None:
                continue
            print_block(method, dataset, rec, label=f"方法: {method}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="RCAEval 自動化測試")
    parser.add_argument("--methods", nargs="+", default=None,
                        help="指定方法（預設執行所有方法）")
    parser.add_argument("--datasets", nargs="+", default=None,
                        choices=ALL_RE_DATASETS, metavar="DATASET",
                        help="指定任意資料集組合（可多選），例如: re1-ob re2-ss re3-tt")
    parser.add_argument("--re1", action="store_true", help="加入 RE1 資料集")
    parser.add_argument("--re2", action="store_true", help="加入 RE2 資料集")
    parser.add_argument("--re3", action="store_true", help="加入 RE3 資料集")
    parser.add_argument("--length", type=int, default=20, help="時間序列長度（分鐘）")
    parser.add_argument("--tdelta", type=int, default=0, help="異常偵測延遲秒數")
    parser.add_argument("--test", action="store_true",
                        help="Smoke test：每個資料集只跑前 2 筆")
    parser.add_argument("--output", default="output/benchmarks/benchmark_results.csv",
                        help="輸出 CSV 路徑")
    parser.add_argument("--skip-failed", action="store_true",
                        help="遇到失敗時繼續執行")
    parser.add_argument("--workers", type=int, default=3,
                        help="同時執行的任務數（預設 3，設為 1 即循序執行）")
    parser.add_argument("--repetitions", type=int, default=1,
                        help="每個 method × dataset 重跑次數，時間以中位數彙整")
    parser.add_argument("--diagnostics", action="store_true",
                        help="將每個 case 的演算法 diagnostics 寫入巢狀 sidecar 目錄")
    parser.add_argument("--show", action="store_true",
                        help="不執行測試，只載入已有結果並印出（需先跑過測試）")
    args = parser.parse_args()

    if args.show:
        load_and_show(args.output)
        return

    # 組合資料集：--datasets 優先；否則根據 --re1/--re2/--re3 組合；都沒指定則全部
    if args.datasets:
        datasets = list(args.datasets)
    else:
        datasets = []
        if args.re1:
            datasets += RE1_DATASETS
        if args.re2:
            datasets += RE2_DATASETS
        if args.re3:
            datasets += RE3_DATASETS
        if not datasets:
            datasets = ALL_RE_DATASETS

    methods = args.methods or DEFAULT_METHODS
    total = len(methods) * len(datasets)

    workers = max(1, args.workers)
    print("\nRCAEval 自動化測試")
    print(f"  方法: {methods}")
    print(f"  資料集: {datasets}")
    print(f"  總任務: {total}  平行數: {workers}  smoke_test={args.test}")

    tasks = [(m, d) for m in methods for d in datasets]

    records = []
    failed = False

    def _task(method, dataset):
        return run_repeated(
            method,
            dataset,
            repetitions=args.repetitions,
            length=args.length,
            tdelta=args.tdelta,
            smoke_test=args.test,
            diagnostics=args.diagnostics,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_task = {
            pool.submit(_task, m, d): (i + 1, m, d)
            for i, (m, d) in enumerate(tasks)
        }

        completed = 0
        for future in as_completed(future_to_task):
            completed += 1
            idx, method, dataset = future_to_task[future]
            try:
                rec = future.result()
            except Exception as exc:
                rec = {
                    "method": method, "dataset": dataset,
                    "status": "FAIL", "wall_time_sec": 0,
                    "returncode": -1, "error": str(exc),
                }
                _safe_print(f"[例外] {method} x {dataset}: {exc}")

            records.append(rec)
            _safe_print(
                f"[進度 {completed}/{total}] {method} x {dataset} → {rec['status']}"
            )

            if rec["status"] == "FAIL" and not args.skip_failed:
                _safe_print(
                    f"\n[錯誤] {method} x {dataset} 執行失敗，取消剩餘任務。\n"
                    "  加上 --skip-failed 可跳過失敗繼續執行。"
                )
                for f in future_to_task:
                    f.cancel()
                failed = True
                break

    # 依原始任務順序排列
    task_order = {(m, d): i for i, (m, d) in enumerate(tasks)}
    records.sort(key=lambda r: task_order.get((r["method"], r["dataset"]), 9999))

    print_table(records)
    print_per_dataset_results(records, methods, datasets)
    save_csv(records, args.output)
    save_json(records, args.output)


if __name__ == "__main__":
    main()
