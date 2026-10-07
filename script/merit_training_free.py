"""Training-free research adapters. Production MERIT and prior artifacts are read-only."""
import argparse
import copy
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from script.merit_design_audit import m, transformed, run_variant
from script.run_paper_dataset import discover_data_paths, prepare_case
from RCAEval.benchmark.input_boundaries import full_record_boundary
from RCAEval.e2e import merit_trace as mt

C = 1.3489795003921634  # 2 * Phi^{-1}(.75), no fitted parameter
NEW = ['mean_iqr', 'median_std', 'normal_iqr', 'normal_iqr_matched_threshold',
       'iqr_z_threshold', 'trace_raw_cosine', 'trace_centered_cosine']
REUSE = ['full', 'standard_z', 'singleton', 'shapley']
VARIANTS = REUSE + NEW
LABELS = ['原版 median/IQR', 'mean/std（重用）', 'Singleton（重用）', 'Shapley（重用）',
          'mean/IQR', 'median/std', 'median/(IQR/1.349)',
          '校準尺度＋等效原門檻', '僅改 IQR 持續性門檻', 'Trace 原值 cosine', 'Trace 中心化 cosine']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metric_adapter(variant):
    changes = []
    if variant == 'mean_iqr':
        changes.append(('med = np.nanmedian(normal_mat, axis=0)',
                        'med = np.nanmean(normal_mat, axis=0)'))
    if variant == 'median_std':
        changes.append(('iqr = np.where((q75 - q25) > 0, q75 - q25, 1.)',
                        'iqr = np.where(std > 0, std, 1.)'))
    if variant.startswith('normal_iqr'):
        changes.append(('iqr = np.where((q75 - q25) > 0, q75 - q25, 1.)',
                        f'iqr = np.where((q75 - q25) > 0, (q75 - q25) / {C!r}, 1.)'))
    if variant == 'normal_iqr_matched_threshold':
        # Keep the original >3 indicator, including unit-scale fallback columns.
        changes.append(('rz > 3.0', f'rz > np.where((q75 - q25) > 0, {3*C!r}, 3.0)'))
    if variant == 'iqr_z_threshold':
        changes.append(('rz > 3.0', f'rz > np.where((q75 - q25) > 0, {3/C!r}, 3.0)'))
    return transformed(m._metric_evidence, changes)


def metric_diagnostics(normal, anomal, cols):
    q = np.percentile(normal, 75, axis=0) - np.percentile(normal, 25, axis=0)
    std = np.std(normal, axis=0)
    return [dict(metric=str(col), median=float(np.median(normal[:, i])),
                 mean=float(np.mean(normal[:, i])), iqr=float(q[i]), std=float(std[i]),
                 zero_iqr=bool(q[i] <= 0), near_constant=bool(std[i] <= 1e-6),
                 tiny_positive_iqr=bool(0 < q[i] <= 1e-6),
                 original_clip=bool(q[i] <= 0 and std[i] <= 1e-6))
            for i, col in enumerate(cols)]


def max_lag_score(x, y, max_lag=3):
    values = [m._lagged_cosine_score(x, y, lag) for lag in range(max_lag + 1)]
    lag = int(np.argmax(values))
    return float(values[lag]), lag


def shift_null(x, y, max_lag=3):
    """Exact cyclic group orbit, identity included; diagnostic, never an edge gate.

    Valid randomization requires conditional invariance of y under cyclic shifts
    and independence from x, NOT merely time-series stationarity.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) != len(y) or len(x) <= max_lag:
        return None
    # Each lag uses the same truncation rule as the observed statistic.
    rotated = y[(np.arange(len(y))[None, :] - np.arange(len(y))[:, None]) % len(y)]
    null = np.zeros(len(y))
    for lag in range(max_lag + 1):
        a = x[:len(x)-lag] if lag else x
        b = rotated[:, lag:]
        denom = np.linalg.norm(a) * np.linalg.norm(b, axis=1)
        scores = np.divide(np.sum(b*a, axis=1), denom,
                           out=np.zeros(len(y)), where=denom > 1e-12)
        null = np.maximum(null, np.clip(scores, 0, 1))
    observed = null[0]
    return dict(statistic=float(observed), p=float(np.mean(null >= observed - 1e-12)),
                shifts=len(y), null_q95=float(np.quantile(null, .95)))


def trace_alternatives(path, graph, inject, boundary):
    cfg = m._GabanzhafConfig.from_kwargs({})
    groups = {'trace_raw_cosine': [], 'trace_centered_cosine': []}
    diagnostics = []
    for kind, name in [('latency', 'tracets_lat.csv'), ('error', 'tracets_err.csv')]:
        file = path.parent / name
        if not file.exists():
            diagnostics.append(dict(kind=kind, status='missing_file'))
            continue
        frame = m.filter_timestamp_frame(pd.read_csv(file), m.ObservationBoundary.from_dict(boundary))
        times = pd.to_numeric(frame['time'], errors='coerce').to_numpy(float)
        if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
            raise ValueError(f'Invalid or non-increasing trace time: {file}')
        series = m._service_trace_series_from_frame(frame)
        pre = times < inject
        if not pre.any():
            raise ValueError(f'No normal trace baseline: {file}')
        centered = {s: v - v[pre].mean() for s, v in series.items()}
        for variant, values in [('trace_raw_cosine', series), ('trace_centered_cosine', centered)]:
            # Min-max would undo centering. Keep signed values in the centered arm.
            with patch.object(mt, '_norm_trace_arr', lambda x: np.asarray(x, float)):
                groups[variant].append(m._trace_multilag_edges(graph, values, cfg, kind))
        for caller, callees in graph.items():
            for callee in callees:
                if callee not in series or caller not in series:
                    continue
                x, y = centered[callee], centered[caller]
                observed, lag = max_lag_score(x, y)
                # Null uses ONLY the normal period; fault-containing records are
                # nonstationary and cannot justify random-shift calibration.
                null = shift_null(x[pre], y[pre])
                diagnostics.append(dict(kind=kind, source=callee, target=caller,
                    n=len(times), n_normal=int(pre.sum()), centered_score=observed, lag=lag,
                    equally_spaced=bool(len(times)<3 or np.allclose(np.diff(times), np.diff(times)[0])),
                    normal_shift_null=null, null_status='diagnostic_only_no_cyclic_invariance_guarantee'))
    return {v: m._merge_trace_topology_edges(g) for v, g in groups.items()}, diagnostics


def run_new(data, inject, dataset, base, variant, edges=None):
    if variant.startswith('trace_'):
        kw = copy.deepcopy(base); kw['trace_topology_edges'] = edges
        return run_variant(data, inject, dataset, kw, 'full')
    with patch.object(m, '_metric_evidence', metric_adapter(variant)):
        return run_variant(data, inject, dataset, base, 'full')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prior', type=Path, default=Path('artifacts/merit-design-audit-20260908'))
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--limit', type=int)
    args = ap.parse_args()
    if args.output.exists():
        raise ValueError('Use a fresh output directory; never overwrite prior research')
    if any(k.startswith(('GABANZHAF_', 'MERIT_')) for k in os.environ):
        raise ValueError('Unset method overrides')
    prior_manifest = json.loads((args.prior/'manifest.json').read_text())
    from script.merit_source_provenance import (
        MERIT_SOURCE_FILES, merit_source_hashes, require_merit_sources,
    )
    require_merit_sources(prior_manifest)
    if digest(m.__file__) != prior_manifest['source_sha256']:
        raise ValueError('Production source differs from prior run; reuse forbidden')
    inventory = {(r['dataset'], r['case']): r['files'] for r in
                 json.loads((args.prior/'input_inventory.json').read_text())}
    args.output.mkdir(parents=True)
    sources = [Path(__file__), Path(m.__file__), Path('script/merit_design_audit.py'),
               Path('script/run_paper_dataset.py'), Path('script/summarize_merit_design_audit.py'),
               Path('script/summarize_merit_training_free.py'), Path('script/merit_training_free_synthetic.py'),
               Path('tests/test_merit_training_free.py')]
    (args.output/'source').mkdir()
    sources.extend(Path(p) for p in MERIT_SOURCE_FILES if Path(p) not in sources)
    for file in sources:
        (args.output/'source'/file.name).write_bytes(file.read_bytes())
    manifest = dict(status='interim', variants=VARIANTS, new_variants=NEW, reused=REUSE,
        source_sha256=digest(m.__file__), sources={str(p):digest(p) for p in sources},
        method_source_sha256=merit_source_hashes(),
        prior=str(args.prior), prior_manifest_sha256=digest(args.prior/'manifest.json'),
        python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
        config=vars(m._GabanzhafConfig.from_kwargs({})), limit=args.limit,
        protocol=prior_manifest['protocol'], threshold_constant=C,
        null='all cyclic shifts incl identity, normal window only; diagnostic, never gates edges',
        inference='development cases; no label-based selection; no independent validation',
        created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    for dataset in ['re2-ss', 're2-ob', 're2-tt']:
        paths = discover_data_paths(Path.cwd(), dataset)
        if args.limit: paths = paths[:args.limit]
        for n, path in enumerate(paths):
            case = path.parent.parent.name+'/'+path.parent.name
            oldfile = args.prior/dataset/(case.replace('/', '__')+'.json')
            old = json.loads(oldfile.read_text())
            hashes = {}
            for name, info in inventory[dataset, case].items():
                file = path.parent/name
                if file.exists() != info['exists']:
                    raise ValueError(f'Input existence drift: {file}')
                if file.exists():
                    hashes[name] = digest(file)
                    if hashes[name] != info['sha256']: raise ValueError(f'Input drift: {file}')
            result = dict(dataset=dataset, case=case, input_sha256=hashes,
                          prior_case_sha256=digest(oldfile), records=[])
            try:
                data, inject = prepare_case(path)
                boundary = full_record_boundary(data['time'], inject).to_dict()
                capture = {}; original = m._merit_core
                def capturing(data, inject, dataset, anomalies, kwargs):
                    capture.update(copy.deepcopy(kwargs))
                    return original(data, inject, dataset, anomalies, kwargs)
                ns = SimpleNamespace(data_path=str(path), input_boundary=boundary, graph_source='per_case_pre_fault')
                with patch.object(m, '_merit_core', capturing):
                    public = m.merit(data, inject_time=inject, dataset=dataset, args=ns, return_diagnostics=True)
                old_records = {r['variant']: r for r in old['records']}
                for key in ['service_ranks', 'game_candidates', 'coverage_strength_matrix',
                            'metric_service_evidence', 'game_services', 'service_banzhaf']:
                    if public['diagnostics'][key] != old_records['full'][key]:
                        raise ValueError(f'Baseline diagnostic drift: {key}')
                for v in REUSE:
                    result['records'].append(dict(copy.deepcopy(old_records[v]), reused_from=str(oldfile)))
                normal_raw, anomal_raw = m._split_raw_windows(data, inject, None)
                normal, anomal, no_time = m._preprocess_windows(normal_raw, anomal_raw, dataset,
                    m._GabanzhafConfig.from_kwargs(capture))
                result['metric_baselines'] = metric_diagnostics(normal.drop(columns='time', errors='ignore').values,
                    anomal.drop(columns='time', errors='ignore').values, list(no_time.columns))
                edges, td = trace_alternatives(path, capture['call_graph'], inject, boundary)
                result.update(trace_diagnostics=td, trace_edges=len(capture['trace_topology_edges']),
                    graph_edges=sum(map(len, capture['call_graph'].values())), new_trace_edges=edges)
                for v in NEW:
                    tick = time.perf_counter()
                    try:
                        d = run_new(data, inject, dataset, capture, v, edges.get(v))
                        result['records'].append(dict(variant=v, status='ok',
                            core_seconds=time.perf_counter()-tick, **d))
                    except Exception as exc:
                        result['records'].append(dict(variant=v, status='error', error=repr(exc)))
            except Exception as exc:
                result['error'] = repr(exc)
                existing = {r['variant'] for r in result['records']}
                result['records'] += [dict(variant=v, status='error', error=repr(exc)) for v in VARIANTS if v not in existing]
            # Check input immutability at the end as well as before reuse.
            if any(digest(path.parent/k) != h for k,h in hashes.items()):
                raise ValueError(f'Input changed during run: {case}')
            out = args.output/dataset/(case.replace('/', '__')+'.json')
            out.parent.mkdir(exist_ok=True); out.write_text(json.dumps(result, ensure_ascii=False))
            print(dataset, n+1, len(paths), case,
                  [r for r in result['records'] if r['status']=='error'], flush=True)
    assert digest(m.__file__) == manifest['source_sha256']
    require_merit_sources(manifest)


if __name__ == '__main__': main()
