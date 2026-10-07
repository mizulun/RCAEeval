"""Constant-justification sweep for MERIT (E1-E6, B, R). Research-only.

Never changes RCAEval/e2e/merit.py or main.py. Uses the same cached-runner
pattern as script/merit_design_audit.py: run the public m.merit() once per
case to capture the preprocessed inputs (metric matrices, call graph, trace
evidence) that _merit_core consumes, then re-invoke _merit_core / rebuild
trace edges directly for each swept configuration, so each case's expensive
preprocessing (metric preprocessing, trace parsing) is paid once.

Protocol matches artifacts/formal-rerun-20260930: full_record_boundary +
graph_source=per_case_pre_fault (see script/run_paper_dataset.py,
script/merit_design_audit.py).

Run:
  PYTHONPATH=. ./py310/bin/python script/merit_param_sweep.py --output <dir>
"""
from dataclasses import replace
import argparse, copy, hashlib, importlib, inspect, json, os, platform, time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from script.run_paper_dataset import discover_data_paths, prepare_case
from RCAEval.benchmark.input_boundaries import full_record_boundary
from script.merit_source_provenance import merit_source_hashes, snapshot_merit_sources

m = importlib.import_module("RCAEval.e2e.merit")


def transformed(fn, replacements):
    source = inspect.getsource(fn)
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError(f"Intervention source drift: {old!r}")
        source = source.replace(old, new)
    ns = dict(fn.__globals__)
    exec(compile(source, "<research-intervention>", "exec"), ns)
    return ns[fn.__name__]


# ---------------------------------------------------------------------------
# E1: IQR=0 denominator (merit_evidence._metric_evidence). Baseline ("full") uses the
# production rule (denominator 1). Two alternatives.
# ---------------------------------------------------------------------------
_E1_SIGMA_NORMAL = transformed(m._metric_evidence, [
    ("iqr = np.where((q75 - q25) > 0, q75 - q25, 1.)",
     "iqr = np.where((q75 - q25) > 0, q75 - q25, np.where(std > 0, std, 1.))"),
])
_E1_EXCLUDE_COL = transformed(m._metric_evidence, [
    ("rz = np.nan_to_num(rz, nan=0., posinf=0., neginf=0.)",
     "rz = np.nan_to_num(rz, nan=0., posinf=0., neginf=0.)\n"
     "    rz = np.where((q75 - q25) <= 0, 0., rz)"),
])

# ---------------------------------------------------------------------------
# E2: near-constant truncation cap (merit_evidence._metric_evidence). Baseline cap=10.
# ---------------------------------------------------------------------------
_E2_CAPS = {}
for _cap in (5, 20, 50):
    _E2_CAPS[_cap] = transformed(m._metric_evidence, [
        ("np.clip(rz, 0., 10.)", f"np.clip(rz, 0., {_cap}.)"),
    ])
_E2_UNCAPPED = transformed(m._metric_evidence, [
    ("rz = np.where(near_zero_baseline, np.clip(rz, 0., 10.), rz)", "rz = rz"),
])

# ---------------------------------------------------------------------------
# B: persistence threshold in merit_evidence._metric_evidence. Baseline threshold=3.
# ---------------------------------------------------------------------------
_B_PERSIST = {}
for _thr in (2, 4, 5):
    _B_PERSIST[_thr] = transformed(m._metric_evidence, [
        ("rz > 3.0", f"rz > {_thr}.0"),
    ])

# ---------------------------------------------------------------------------
# E5 decoupled: only the trace-edge latency filter set to 0 (coverage gate
# stays at default). The production hook makes this intervention explicit and
# avoids copying/recompiling the whole orchestration function.
# ---------------------------------------------------------------------------
def _E5_TRACEFILTER_ONLY_0(edges, latency_family, minimum):
    return m._filter_trace_topology_edges(edges, latency_family, 0.0)


def run_core(data, inject, dataset, capture, overrides=None, edges_override=None,
             patch_target=None, patch_fn=None, core_fn=None):
    kw = copy.deepcopy(capture)
    kw.pop("args", None)
    if overrides:
        kw.update(overrides)
    if edges_override is not None:
        kw["trace_topology_edges"] = edges_override
    fn = core_fn or m._merit_core
    if patch_target:
        with patch.object(m, patch_target, patch_fn):
            result = fn(data, inject, dataset, None, kw)
    else:
        result = fn(data, inject, dataset, None, kw)
    d = result["diagnostics"]
    return {"service_ranks": d["service_ranks"],
            "game_candidates": d.get("game_candidates", [])}


def rebuild_edges(case_dir, call_graph, inject, boundary, **cfg_kwargs):
    cfg = m._GabanzhafConfig.from_kwargs(cfg_kwargs)
    return m._trace_topology_from_case_dir(str(case_dir), call_graph, cfg, inject, boundary, True)


def timed(fn, *a, **kw):
    t = time.perf_counter()
    out = fn(*a, **kw)
    return out, time.perf_counter() - t


def build_records(data, inject, dataset, capture, has_traces, case_dir, boundary):
    """Returns list of {'group','name','status', 'seconds', ...diag}."""
    records = []

    def add(group, name, fn, *a, **kw):
        try:
            diag, secs = timed(fn, *a, **kw)
            records.append({"group": group, "name": name, "status": "ok",
                             "seconds": secs, **diag})
        except Exception as exc:
            records.append({"group": group, "name": name, "status": "error",
                             "error": repr(exc)})

    call_graph = capture.get("call_graph", {})

    # ---- E1 ----
    add("E1", "sigma_normal", lambda: run_core(data, inject, dataset, capture,
        patch_target="_metric_evidence", patch_fn=_E1_SIGMA_NORMAL))
    add("E1", "exclude_col", lambda: run_core(data, inject, dataset, capture,
        patch_target="_metric_evidence", patch_fn=_E1_EXCLUDE_COL))

    # ---- E2 ----
    for cap in (5, 20, 50):
        add("E2", f"cap_{cap}", lambda cap=cap: run_core(data, inject, dataset, capture,
            patch_target="_metric_evidence", patch_fn=_E2_CAPS[cap]))
    add("E2", "uncapped", lambda: run_core(data, inject, dataset, capture,
        patch_target="_metric_evidence", patch_fn=_E2_UNCAPPED))

    # ---- E3 ----
    for ratio in (0.10, 0.15, 0.35, 0.50):
        add("E3", f"ratio_{ratio}", lambda ratio=ratio: run_core(
            data, inject, dataset, capture, overrides={"early_window_ratio": ratio}))

    # ---- E4 (trace_topology_min_conf) : SS has no traces.csv -> N/A ----
    if has_traces:
        all_edges_feat, _ = timed(rebuild_edges, case_dir, call_graph, inject, boundary,
                                   trace_topology_min_conf=0.0)
        all_edges = all_edges_feat.edges
        n_all = len(all_edges)
        for thr in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6):
            filtered = [e for e in all_edges if e[2] >= thr]
            retention = (len(filtered) / n_all) if n_all else None

            def _run(filtered=filtered):
                return run_core(data, inject, dataset, capture, edges_override=filtered)
            try:
                diag, secs = timed(_run)
                records.append({"group": "E4", "name": f"min_conf_{thr}", "status": "ok",
                                 "seconds": secs, "retention": retention,
                                 "n_edges_total": n_all, "n_edges_kept": len(filtered), **diag})
            except Exception as exc:
                records.append({"group": "E4", "name": f"min_conf_{thr}", "status": "error",
                                 "error": repr(exc)})
    else:
        for thr in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6):
            records.append({"group": "E4", "name": f"min_conf_{thr}", "status": "n_a_no_trace"})

    # ---- E5 (coverage_lat_floor) ----
    for floor in (0.0, 0.05, 0.2, 0.3):
        add("E5", f"floor_both_{floor}", lambda floor=floor: run_core(
            data, inject, dataset, capture, overrides={"coverage_lat_floor": floor}))

    def _covgate_only_0(config):
        return replace(config, coverage_lat_floor=0.0)
    original_coverage_parents = m._coverage_parents

    def _cov_patch(call_graph, svc_anom_dict, svc_latency, config):
        return original_coverage_parents(call_graph, svc_anom_dict, svc_latency,
                                          _covgate_only_0(config))
    add("E5", "covgate_only_0", lambda: run_core(data, inject, dataset, capture,
        patch_target="_coverage_parents", patch_fn=_cov_patch))
    add("E5", "tracefilter_only_0", lambda: run_core(data, inject, dataset, capture,
        patch_target="_filter_trace_topology_edges", patch_fn=_E5_TRACEFILTER_ONLY_0))

    # ---- E6: victim thresholds ----
    for up in (0.35, 0.45, 0.55):
        for norm in (0.10, 0.20, 0.30):
            if up == 0.45 and norm == 0.20:
                continue  # default, already recorded as 'full'
            add("E6", f"up_{up}_norm_{norm}", lambda up=up, norm=norm: run_core(
                data, inject, dataset, capture,
                overrides={"frontier_abnormal_threshold": up, "frontier_normal_threshold": norm}))
    for factor in (0.0, 0.25, 0.75, 1.0):
        add("E6", f"uncertain_factor_{factor}", lambda factor=factor: run_core(
            data, inject, dataset, capture,
            overrides={"frontier_uncertain_stop_factor": factor}))

    # ---- B: confirmatory checks ----
    for thr in (2, 4, 5):
        add("B", f"persist_thr_{thr}", lambda thr=thr: run_core(data, inject, dataset, capture,
            patch_target="_metric_evidence", patch_fn=_B_PERSIST[thr]))
    for n_loud in (1, 3):
        add("B", f"min_loud_{n_loud}", lambda n_loud=n_loud: run_core(
            data, inject, dataset, capture, overrides={"coverage_min_loud": n_loud}))
    if has_traces:
        for lag in (1, 2, 5, 8):
            def _run_lag(lag=lag):
                feat, _ = timed(rebuild_edges, case_dir, call_graph, inject, boundary,
                                 trace_topology_max_lag=lag)
                return run_core(data, inject, dataset, capture, edges_override=feat.edges)
            add("B", f"lag_{lag}", _run_lag)
    else:
        for lag in (1, 2, 5, 8):
            records.append({"group": "B", "name": f"lag_{lag}", "status": "n_a_no_trace"})

    # ---- R: existing (a)-class sweeps rerun on the formal protocol ----
    for k in (10, 12, 20, 25):
        add("R", f"top_k_svcs_{k}", lambda k=k: run_core(data, inject, dataset, capture,
            overrides={"top_k_svcs": k}))
    for k in (6, 8, 10, 15):
        add("R", f"exact_candidate_k_{k}", lambda k=k: run_core(data, inject, dataset, capture,
            overrides={"exact_candidate_k": k}))
    for c in (0.5, 0.6, 0.8, 0.9):
        add("R", f"coverage_conf_{c}", lambda c=c: run_core(data, inject, dataset, capture,
            overrides={"coverage_conf": c}))
    for thr in (0.3, 0.5, 0.6):
        add("R", f"coverage_loud_thr_{thr}", lambda thr=thr: run_core(data, inject, dataset, capture,
            overrides={"coverage_loud_thr": thr}))
    for floor in (0.3, 0.4, 0.6, 0.7):
        add("R", f"victim_floor_{floor}", lambda floor=floor: run_core(data, inject, dataset, capture,
            overrides={"victim_floor": floor}))
    if has_traces:
        for conf in (0.7, 0.8, 0.9, 0.95):
            def _run_conf(conf=conf):
                feat, _ = timed(rebuild_edges, case_dir, call_graph, inject, boundary,
                                 trace_topology_conf=conf)
                return run_core(data, inject, dataset, capture, edges_override=feat.edges)
            add("R", f"trace_topology_conf_{conf}", _run_conf)
    else:
        for conf in (0.7, 0.8, 0.9, 0.95):
            records.append({"group": "R", "name": f"trace_topology_conf_{conf}", "status": "n_a_no_trace"})
    for param, values in [
        ("ev_w_mag", (0.35, 0.75)), ("ev_w_persistence", (0.15, 0.45)),
        ("early_w_mag", (0.5, 0.9)), ("fusion_w_early", (0.4, 0.8)),
        ("agg_w_p95", (0.3, 0.7)), ("svc_w_dominance", (0.15, 0.50)),
    ]:
        for v in values:
            add("R", f"{param}_{v}", lambda param=param, v=v: run_core(
                data, inject, dataset, capture, overrides={param: v}))

    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--datasets", nargs="+", default=["re2-ss", "re2-ob", "re2-tt"])
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    if any(k.startswith(("GABANZHAF_", "MERIT_")) for k in os.environ):
        raise RuntimeError("Unset method environment overrides for frozen defaults")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_sha256": hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest(),
        "method_source_sha256": merit_source_hashes(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(), "numpy": np.__version__,
        "protocol": "full_record_retrospective; prepare_case from paper harness; current defaults except swept param",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    snapshot_merit_sources(args.output / "source")
    for dataset in args.datasets:
        paths = discover_data_paths(Path.cwd(), dataset)
        if not paths:
            raise RuntimeError(f"No cases: {dataset}")
        if args.limit:
            paths = paths[: args.limit]
        for n, path in enumerate(paths):
            case = path.parent.parent.name + "/" + path.parent.name
            out = args.output / dataset / (case.replace("/", "__") + ".json")
            out.parent.mkdir(exist_ok=True, parents=True)
            case_dir = path.parent
            has_traces = (case_dir / "traces.csv").exists()
            try:
                data, inject = prepare_case(path)
                boundary = full_record_boundary(data["time"], inject).to_dict()
                ns = SimpleNamespace(data_path=str(path), input_boundary=boundary,
                                      graph_source="per_case_pre_fault")
                capture = {}
                original = m._merit_core

                def capturing(data, inject, dataset, anomalies, kwargs):
                    capture.update(copy.deepcopy(kwargs))
                    return original(data, inject, dataset, anomalies, kwargs)

                t0 = time.perf_counter()
                with patch.object(m, "_merit_core", capturing):
                    public = m.merit(data, inject_time=inject, dataset=dataset, args=ns,
                                      return_diagnostics=True)
                public_seconds = time.perf_counter() - t0
                full_record = {"group": "baseline", "name": "full", "status": "ok",
                                "seconds": public_seconds,
                                "service_ranks": public["diagnostics"]["service_ranks"],
                                "game_candidates": public["diagnostics"].get("game_candidates", [])}
                t1 = time.perf_counter()
                records = [full_record] + build_records(
                    data, inject, dataset, capture, has_traces, case_dir, boundary)
                sweep_seconds = time.perf_counter() - t1
                result = {"dataset": dataset, "case": case, "has_traces": has_traces,
                          "public_seconds": public_seconds, "sweep_seconds": sweep_seconds,
                          "graph_edges": sum(map(len, capture.get("call_graph", {}).values())),
                          "records": records}
            except Exception as exc:
                result = {"dataset": dataset, "case": case, "error": repr(exc), "records": []}
            out.write_text(json.dumps(result, ensure_ascii=False))
            errs = [r["name"] for r in result.get("records", []) if r.get("status") == "error"]
            print(dataset, n + 1, len(paths), case, "errors:", errs, flush=True)


if __name__ == "__main__":
    main()
