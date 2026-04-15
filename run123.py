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
"""

import argparse
import csv
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

DATASET_DIR_MAP = {
    "re1-ob": "data/online-boutique",
    "re1-ss": "data/sock-shop-2",
    "re1-tt": "data/train-ticket",
    "re2-ob": "data/RE2/RE2-OB",
    "re2-ss": "data/RE2/RE2-SS",
    "re2-tt": "data/RE2/RE2-TT",
    "re3-ob": "data/RE3/RE3-OB",
    "re3-ss": "data/RE3/RE3-SS",
    "re3-tt": "data/RE3/RE3-TT",
}

RE1_DATASETS = ["re1-ob", "re1-ss", "re1-tt"]
RE2_DATASETS = ["re2-ob", "re2-ss", "re2-tt"]
RE3_DATASETS = ["re3-ob", "re3-ss", "re3-tt"]
ALL_RE_DATASETS = RE1_DATASETS + RE2_DATASETS + RE3_DATASETS

DEFAULT_METHODS = [
    "baro", "nsigma", "e_diagnosis",
    "pc_pagerank", "fci_pagerank", "ges_pagerank",
    "lingam_pagerank", "granger_pagerank", "ntlr_pagerank",
    "circa", "microcause", "cloudranger", "easyrca",
    "microrank", "tracerca", "crfd",
]

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


def run_single(method, dataset, length=20, tdelta=0, smoke_test=False):
    rdir = result_dir_for(method, dataset)
    # 刪除上一次同資料集的結果
    clear_result_dir(method, dataset)

    cmd = [sys.executable, "main.py",
           "--method", method, "--dataset", dataset,
           "--length", str(length), "--tdelta", str(tdelta),
           "--result-dir", rdir]
    if smoke_test:
        cmd.append("--test")

    start = datetime.now()
    proc = subprocess.run(cmd, text=True, capture_output=True,
                          cwd=os.path.dirname(os.path.abspath(__file__)))
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
    if proc.returncode == 0:
        parsed = parse_output(proc.stdout)
        record.update(parsed)
        # 儲存詳細結果供後續 print
        with _results_lock:
            _results_store[(method, dataset)] = parsed

    return record


def save_csv(records, path):
    if not records:
        return
    keys = ["method", "dataset", "status", "wall_time_sec"]
    metric_keys = sorted({
        k for r in records for k in r
        if any(k.startswith(f"{m}-") for m in METRICS) or k == "Avg speed"
    })
    keys += metric_keys + ["returncode"]
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
    print(f"  狀態: {status}  耗時: {elapsed}s")


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


def collect_inject_times(datasets):
    """
    掃描各資料集目錄下所有 inject_time.txt，回傳:
    { dataset: [ {service, fault, case, timestamp, datetime_str}, ... ] }
    """
    root = os.path.dirname(os.path.abspath(__file__))
    result = {}
    for ds in datasets:
        data_dir = DATASET_DIR_MAP.get(ds)
        if not data_dir:
            continue
        abs_dir = os.path.join(root, data_dir)
        if not os.path.isdir(abs_dir):
            continue
        entries = []
        for path in sorted(glob.glob(
                os.path.join(abs_dir, "*", "*", "inject_time.txt"))):
            try:
                with open(path) as f:
                    ts = int(f.read().strip())
                # path: abs_dir / service_fault / case / inject_time.txt
                parts = path.replace(abs_dir + os.sep, "").split(os.sep)
                sf = parts[0]          # e.g. "adservice_cpu"
                case = parts[1]        # e.g. "1"
                idx = sf.rfind("_")
                service = sf[:idx] if idx != -1 else sf
                fault   = sf[idx+1:].upper() if idx != -1 else "?"
                dt_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
                entries.append({
                    "service": service, "fault": fault,
                    "case": case, "timestamp": ts, "datetime": dt_str,
                })
            except Exception:
                continue
        if entries:
            result[ds] = entries
    return result


def print_inject_times(datasets):
    """印出各資料集所有 case 的異常注入時間。"""
    data = collect_inject_times(datasets)
    if not data:
        return

    print("\n" + "=" * 60)
    print("各 Case 異常注入時間 (inject_time)")
    for ds in datasets:
        entries = data.get(ds)
        if not entries:
            continue
        print(f"\n{'━'*60}")
        print(f"  資料集: {ds}")
        print(f"  {'服務':<30} {'故障':<8} {'Case':<6} {'注入時間'}")
        print(f"  {'-'*30} {'-'*8} {'-'*6} {'-'*19}")
        for e in entries:
            print(f"  {e['service']:<30} {e['fault']:<8} {e['case']:<6} {e['datetime']}")
    print("=" * 60)


def compare_anomaly_detection(methods, datasets):
    """
    比對各方法自行偵測的異常時間 vs 資料集的真實 inject_time。
    只顯示結果 JSON 中含有 detected_time 與 inject_time 的 case（即方法自行偵測者）。
    """
    root = os.path.dirname(os.path.abspath(__file__))
    rows = []

    for method in methods:
        for dataset in datasets:
            rdir = result_dir_for(method, dataset)
            if not os.path.isdir(rdir):
                continue
            for jpath in sorted(glob.glob(os.path.join(rdir, "*.json"))):
                try:
                    with open(jpath, encoding="utf-8") as f:
                        d = json.load(f)
                except Exception:
                    continue
                if "detected_time" not in d or "inject_time" not in d:
                    continue
                detected = d["detected_time"]
                inject   = d["inject_time"]
                if detected is None:
                    continue
                err_sec = detected - inject
                fname = os.path.basename(jpath).replace(".json", "")
                parts = fname.split("_")
                case  = parts[-1] if parts else "?"
                fault = parts[-2].upper() if len(parts) >= 2 else "?"
                svc   = "_".join(parts[:-2]) if len(parts) >= 3 else fname
                rows.append({
                    "method": method, "dataset": dataset,
                    "service": svc, "fault": fault, "case": case,
                    "inject_time": inject, "detected_time": detected,
                    "error_sec": err_sec,
                    "split_source": d.get("split_source", "?"),
                })

    if not rows:
        return

    def _stats(errs_signed):
        """回傳 (count, mean_abs, median_abs, max_abs, p90_abs) 或 None。"""
        if not errs_signed:
            return None
        abs_e = [abs(e) for e in errs_signed]
        abs_e_sorted = sorted(abs_e)
        n = len(abs_e_sorted)
        p90 = abs_e_sorted[int(n * 0.9)] if n >= 2 else abs_e_sorted[-1]
        return {
            "n":      n,
            "mean":   sum(abs_e) / n,
            "median": abs_e_sorted[n // 2],
            "max":    abs_e_sorted[-1],
            "p90":    p90,
        }

    def _print_stats_block(label, errs_signed, indent="    "):
        st = _stats(errs_signed)
        if st is None:
            return
        print(f"{indent}{label}  n={st['n']}")
        print(f"{indent}  平均絕對誤差={st['mean']:.1f}s  "
              f"中位數={st['median']:.0f}s  "
              f"P90={st['p90']:.0f}s  "
              f"最大={st['max']:.0f}s")

    print("\n" + "=" * 70)
    print("異常時間偵測比對（detected_time vs inject_time）")
    print(f"  共 {len(rows)} 個 case 有自行偵測結果")
    print("=" * 70)

    # ── 依 dataset × method 分組，逐 case 列表 ────────────────────────────
    seen_datasets = list(dict.fromkeys(r["dataset"] for r in rows))
    for ds in seen_datasets:
        ds_rows = [r for r in rows if r["dataset"] == ds]
        seen_methods = list(dict.fromkeys(r["method"] for r in ds_rows))
        for m in seen_methods:
            m_rows = [r for r in ds_rows if r["method"] == m]

            early = [r for r in m_rows if r["error_sec"] < 0]
            late  = [r for r in m_rows if r["error_sec"] > 0]
            exact = [r for r in m_rows if r["error_sec"] == 0]

            print(f"\n{'━'*70}")
            print(f"  方法={m}  資料集={ds}  ({len(m_rows)} cases: "
                  f"提早={len(early)}  延後={len(late)}  精確={len(exact)})")
            print(f"{'━'*70}")

            hdr = (f"  {'服務':<28} {'故障':<7} {'Case':<5}"
                   f" {'inject':>8} {'detected':>8} {'誤差(s)':>8}  方向")
            sep = "  " + "-" * 68
            print(hdr)
            print(sep)

            for r in sorted(m_rows, key=lambda x: (x["fault"], x["service"], x["case"])):
                inj_str = datetime.fromtimestamp(r["inject_time"]).strftime("%H:%M:%S")
                det_str = datetime.fromtimestamp(r["detected_time"]).strftime("%H:%M:%S")
                err     = r["error_sec"]
                if err < 0:
                    direction = f"↑提早 {abs(err):>4}s"
                elif err > 0:
                    direction = f"↓延後 {err:>4}s"
                else:
                    direction = "  精確    "
                print(f"  {r['service']:<28} {r['fault']:<7} {r['case']:<5}"
                      f" {inj_str:>8} {det_str:>8} {err:>+8}  {direction}")

            # ── 分類統計 ──────────────────────────────────────────────────
            print(sep)
            _print_stats_block("【提早偵測】", [r["error_sec"] for r in early])
            _print_stats_block("【延後偵測】", [r["error_sec"] for r in late])
            _print_stats_block("【全體    】", [r["error_sec"] for r in m_rows])

            # ── 依 fault type 細分 ────────────────────────────────────────
            fault_types = sorted(set(r["fault"] for r in m_rows))
            if len(fault_types) > 1:
                print(f"    ── 依故障類型 ──")
                for ft in fault_types:
                    ft_rows = [r for r in m_rows if r["fault"] == ft]
                    ft_early = sum(1 for r in ft_rows if r["error_sec"] < 0)
                    ft_late  = sum(1 for r in ft_rows if r["error_sec"] > 0)
                    ft_errs  = [abs(r["error_sec"]) for r in ft_rows]
                    mean_e   = sum(ft_errs) / len(ft_errs) if ft_errs else 0
                    print(f"    {ft:<8}  n={len(ft_rows)}  "
                          f"提早={ft_early}  延後={ft_late}  "
                          f"平均|誤差|={mean_e:.1f}s")

    # ── 全體跨 dataset/method 統計 ────────────────────────────────────────
    all_early = [r for r in rows if r["error_sec"] < 0]
    all_late  = [r for r in rows if r["error_sec"] > 0]
    all_abs   = [abs(r["error_sec"]) for r in rows]
    within_30  = sum(1 for e in all_abs if e <= 30)
    within_60  = sum(1 for e in all_abs if e <= 60)
    within_120 = sum(1 for e in all_abs if e <= 120)
    N = len(rows)

    print(f"\n{'═'*70}")
    print(f"  全體總計  N={N}  "
          f"提早={len(all_early)} ({100*len(all_early)/N:.0f}%)  "
          f"延後={len(all_late)} ({100*len(all_late)/N:.0f}%)")
    _print_stats_block("", [r["error_sec"] for r in rows], indent="  ")
    print(f"  誤差區間分佈：")
    print(f"    ≤ 30s : {within_30:>4}/{N}  ({100*within_30/N:.0f}%)")
    print(f"    ≤ 60s : {within_60:>4}/{N}  ({100*within_60/N:.0f}%)")
    print(f"    ≤120s : {within_120:>4}/{N}  ({100*within_120/N:.0f}%)")
    print(f"    >120s : {N-within_120:>4}/{N}  ({100*(N-within_120)/N:.0f}%)")
    print("═" * 70)


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
    parser.add_argument("--output", default="benchmark_results.csv",
                        help="輸出 CSV 路徑")
    parser.add_argument("--skip-failed", action="store_true",
                        help="遇到失敗時繼續執行")
    parser.add_argument("--workers", type=int, default=3,
                        help="同時執行的任務數（預設 3，設為 1 即循序執行）")
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
        return run_single(method, dataset, args.length, args.tdelta, args.test)

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
    print_inject_times(datasets)
    compare_anomaly_detection(methods, datasets)
    save_csv(records, args.output)
    save_json(records, args.output)


if __name__ == "__main__":
    main()
