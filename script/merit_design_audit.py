"""Research-only interventions; never changes MERIT defaults or production source.
Run: PYTHONPATH=. py310/bin/python script/merit_design_audit.py --output artifacts/merit-design-audit
"""
from dataclasses import replace
import argparse, copy, hashlib, importlib, inspect, itertools, json, os, platform, time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from script.run_paper_dataset import discover_data_paths, prepare_case
from RCAEval.benchmark.input_boundaries import full_record_boundary
from script.merit_source_provenance import merit_source_hashes, snapshot_merit_sources
m = importlib.import_module('RCAEval.e2e.merit')
VARIANTS = ['full','no_metric','standard_z','magnitude_only','no_early','no_dominance',
 'aggregate_mean','aggregate_max','no_trace','no_trace_evidence','no_dynamic_edges',
 'lag0','time_shuffle','no_coverage','rescue_only','edges_only','ungated_coverage',
 'all_candidates','singleton','shapley','direct_evidence','no_victim','no_C_no_V']


def transformed(fn, replacements):
    source = inspect.getsource(fn)
    for old,new in replacements:
        if source.count(old) != 1:
            raise ValueError(f'Intervention source drift: {old}')
        source=source.replace(old,new)
    ns=dict(fn.__globals__)
    exec(compile(source, '<research-intervention>', 'exec'),ns)
    return ns[fn.__name__]


def shapley(A,w):
    """Exact polynomial quadrature up to floating-point error, O(K^2 N)."""
    A=np.asarray(A,float); w=np.asarray(w,float)
    if len(A)==0: return np.zeros(0)
    x,q=np.polynomial.legendre.leggauss((len(A)+1)//2)
    result=np.zeros(len(A))
    for t,weight in zip((x+1)/2,q/2):
        factors=1-t*A
        result+=weight*np.sum(w[None,:]*A*np.prod(factors,axis=0)/factors,axis=1)
    return result


def run_variant(data, inject, dataset, base, variant, special=None):
    kw=copy.deepcopy(base); kw.pop('args',None)
    overrides={
      'no_metric':{'use_metrics':False},'no_early':{'use_early_full_fusion':False},
      'no_dominance':{'use_metric_type_dominance':False},
      'magnitude_only':{'ev_w_mag':1.,'ev_w_persistence':0.,'ev_w_peak':0.,'early_w_mag':1.,'early_w_peak':0.},
      'no_trace':{'use_trace':False},'no_dynamic_edges':{'trace_topology_edges':[]},
      'no_trace_evidence':{'trace_service_evidence':{},'trace_latency_evidence':{},'trace_error_evidence':{}},
      'no_coverage':{'use_coverage':False}, 'no_victim':{'use_victim_demotion':False},
      'no_C_no_V':{'use_coverage':False,'use_victim_demotion':False},
      'ungated_coverage':{'coverage_min_loud':1,'coverage_loud_thr':0.,'coverage_lat_floor':0.},
      'all_candidates':{'top_k_svcs':100000,'exact_candidate_k':100000},
      'singleton':{'banzhaf_mode':'individual'},'direct_evidence':{'rank_mode':'service_evidence_trace'}}
    kw.update(overrides.get(variant,{}))
    # Coverage gating intervention must not change the downstream trace latency filter.
    if variant=='ungated_coverage':
        kw.pop('coverage_lat_floor')
    if special is not None:
        kw['trace_topology_edges']=special.edges
        kw['trace_service_evidence']=special.service_evidence
        kw['trace_latency_evidence']=special.latency_evidence
        kw['trace_error_evidence']=special.error_evidence
    fn=m._merit_core; target=None; replacement=None
    if variant=='standard_z':
        target='_metric_evidence'; replacement=transformed(m._metric_evidence,[
          ('med = np.nanmedian(normal_mat, axis=0)','med = np.nanmean(normal_mat, axis=0)'),
          ('iqr = np.where((q75 - q25) > 0, q75 - q25, 1.)','iqr = np.where(std > 0, std, 1.)')])
    elif variant=='aggregate_mean':
        target='_service_evidence';replacement=transformed(m._service_evidence,[
          ('w_agg[0] * np.percentile(arr, 95)\n            + w_agg[1] * np.mean(top)\n            + w_agg[2] * np.max(arr)','np.mean(arr)')])
    elif variant=='aggregate_max': kw.update(agg_w_p95=0.,agg_w_top3=0.,agg_w_max=1.)
    elif variant=='rescue_only':
        target='_apply_coverage_edges';replacement=lambda matrix,*args: matrix
    elif variant=='edges_only':
        fn=transformed(m._merit_core,[('for parent in coverage_parents:\n','for parent in []:\n')])
    elif variant=='ungated_coverage':
        target='_coverage_parents'
        def replacement(graph, weights, latency, config):
            cfg=replace(config, coverage_lat_floor=0.)
            return original_parents(graph,weights,latency,cfg)
        original_parents=m._coverage_parents
    elif variant=='shapley':
        target='_banzhaf_power';replacement=lambda A,w,c:shapley(A,w)
    if target:
        with patch.object(m,target,replacement):result=fn(data,inject,dataset,None,kw)
    else: result=fn(data,inject,dataset,None,kw)
    d=result['diagnostics']
    # Production diagnostics label Banzhaf decomposition; do not report it for Shapley.
    return {k:d[k] for k in ['service_ranks','game_candidates','coverage_rescued_k',
      'coverage_strength_matrix','metric_service_evidence','game_services','service_banzhaf']}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--datasets',nargs='+',default=['re2-ss','re2-ob','re2-tt'])
    ap.add_argument('--limit',type=int);args=ap.parse_args()
    if any(k.startswith(('GABANZHAF_','MERIT_')) for k in os.environ):
        raise RuntimeError('Unset method environment overrides for frozen defaults')
    args.output.mkdir(parents=True,exist_ok=True)
    manifest={'status':'interim','source_sha256':hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest(),
      'method_source_sha256': merit_source_hashes(),
      'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'python':platform.python_version(),'numpy':np.__version__,'variants':VARIANTS,
      'protocol':'full_record_retrospective; prepare_case from paper harness; current defaults; victim on',
      'bootstrap_cluster':'dataset x root_service x fault_type; exploratory intervals',
      'runtime':'core including repeated preprocessing; trace extraction separately timed, not end-to-end comparison',
      'coverage_isolation':'rescue_only/edges_only keep coverage-derived victim edges and parent protection fixed',
      'shuffle':'production deterministic salt mechanism; no task/label-dependent random seed',
      'config':vars(m._GabanzhafConfig.from_kwargs({}))}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    snapshot_merit_sources(args.output / 'source')
    for dataset in args.datasets:
      paths=discover_data_paths(Path.cwd(),dataset)
      if not paths: raise RuntimeError(f'No cases: {dataset}')
      if args.limit:paths=paths[:args.limit]
      for n,path in enumerate(paths):
        case=path.parent.parent.name+'/'+path.parent.name
        out=args.output/dataset/(case.replace('/','__')+'.json');out.parent.mkdir(exist_ok=True)
        records=[]
        try:
          data,inject=prepare_case(path)
          boundary=full_record_boundary(data['time'],inject).to_dict()
          ns=SimpleNamespace(data_path=str(path),input_boundary=boundary,graph_source='per_case_pre_fault')
          capture={};original=m._merit_core
          def capturing(data,inject,dataset,anomalies,kwargs):
            capture.update(copy.deepcopy(kwargs));return original(data,inject,dataset,anomalies,kwargs)
          t=time.perf_counter()
          with patch.object(m,'_merit_core',capturing):
            public=m.merit(data,inject_time=inject,dataset=dataset,args=ns,return_diagnostics=True)
          public_seconds=time.perf_counter()-t
          special={}
          for v,options in [('lag0',{'trace_topology_max_lag':0}),('time_shuffle',{'trace_control':'time_shuffle'})]:
            cfg=m._GabanzhafConfig.from_kwargs(options)
            special[v]=m._trace_topology_from_case_dir(str(path.parent),capture['call_graph'],cfg,inject,boundary,True)
          for variant in VARIANTS:
            t=time.perf_counter()
            try:
              d=run_variant(data,inject,dataset,capture,variant,special.get(variant))
              if variant=='full':
                assert d['service_ranks']==public['diagnostics']['service_ranks'],'public/core drift'
              record={'variant':variant,'status':'ok','core_seconds':time.perf_counter()-t,**d}
              if variant=='no_metric' and not any(capture['trace_service_evidence'].values()):
                record['status']='no_evidence'
              records.append(record)
            except Exception as exc:records.append({'variant':variant,'status':'error','error':repr(exc)})
          result={'dataset':dataset,'case':case,'input_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [path,path.parent/'inject_time.txt',path.parent/'traces.csv'] if f.exists()},
             'trace_edges':len(capture['trace_topology_edges']),'graph_edges':sum(map(len,capture['call_graph'].values())),
             'public_full_seconds':public_seconds,'records':records}
        except Exception as exc:
          result={'dataset':dataset,'case':case,'error':repr(exc),'records':[{'variant':v,'status':'error','error':repr(exc)} for v in VARIANTS]}
        out.write_text(json.dumps(result,ensure_ascii=False))
        print(dataset,n+1,len(paths),case,[(r['variant'],r['status']) for r in result['records'] if r['status']=='error'],flush=True)

if __name__=='__main__':main()
