"""Frozen follow-up experiments; production and previous artifacts untouched."""
import argparse, copy, json, time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from script.merit_training_free import m, metric_adapter, digest
from script.merit_design_audit import transformed
from script.run_paper_dataset import prepare_case, discover_data_paths
from RCAEval.benchmark.input_boundaries import full_record_boundary, ObservationBoundary
from script.merit_source_provenance import merit_source_hashes, snapshot_merit_sources, require_merit_sources

ARMS=['original','median_std','fallback_only']
ROOTS={'re1-ob':'online-boutique','re1-ss':'sock-shop-2','re1-tt':'train-ticket'}

def scorer(arm):
    if arm=='median_std': return metric_adapter('median_std')
    if arm in ['fallback_only','positive_only']:
        expr=('np.where((q75-q25)>0, q75-q25, np.where(std>0,std,1.))'
              if arm=='fallback_only' else 'np.where(((q75-q25)>0)&(std>0),std,1.)')
        return transformed(m._metric_evidence,[('iqr = np.where((q75 - q25) > 0, q75 - q25, 1.)','iqr = '+expr)])
    return m._metric_evidence

def compact(d):
    return {k:d[k] for k in ['service_ranks','game_candidates','game_services','coverage_strength_matrix',
        'metric_service_evidence','service_banzhaf','coverage_rescued_k','metric_evidence']}

def core(data,inject,ds,base,arm,pollution=0.):
    fn=scorer(arm)
    def evidence(normal,anomal,cols,config=None):
        k=int(len(normal)*pollution)
        if k>len(anomal): raise ValueError('Insufficient post samples for contamination')
        normal=np.concatenate([normal,anomal[:k]],axis=0) if k else normal
        return fn(normal,anomal,cols,config)
    with patch.object(m,'_metric_evidence',evidence):
        return compact(m._merit_core(data,inject,ds,None,copy.deepcopy(base))['diagnostics'])

def capture(data,inject,ds,path,boundary):
    base={};original=m._merit_core
    def wrapped(d,t,s,a,kw):
        base.update(copy.deepcopy(kw));return original(d,t,s,a,kw)
    ns=SimpleNamespace(data_path=str(path),input_boundary=boundary.to_dict(),graph_source='per_case_pre_fault')
    with patch.object(m,'_merit_core',wrapped):
        d=m.merit(data,inject_time=inject,dataset=ds,args=ns,return_diagnostics=True)['diagnostics']
    return base,compact(d)

def run_case(path,ds):
    started=time.perf_counter();data,inject=prepare_case(path)
    boundary=full_record_boundary(data.time,inject)
    base,full=capture(data,inject,ds,path,boundary)
    records=[]
    def add(group,condition,arm,fn):
        tick=time.perf_counter()
        try: r=dict(status='ok',**fn())
        except Exception as e:r=dict(status='error',error=repr(e))
        records.append(dict(group=group,condition=str(condition),arm=arm,seconds=time.perf_counter()-tick,**r))
    originals={}
    for arm in ARMS+['positive_only']:
        add('scale','full',arm,lambda arm=arm:full if arm=='original' else core(data,inject,ds,base,arm))
        if records[-1]['status']=='ok':originals[arm]=records[-1]
    if ds.startswith('re2'):
        for amount in [.05,.1,.2]:
            for arm in ARMS:
                add('pollution',amount,arm,lambda arm=arm,amount=amount:core(data,inject,ds,base,arm,amount))
        for group,values in [('delay',[60,120,180]),('horizon',[60,180,300])]:
            for seconds in values:
                split=inject+seconds if group=='delay' else inject
                end=boundary.end if group=='delay' else inject+seconds
                b=ObservationBoundary(boundary.start,end,split,group)
                frame=data[(data.time>=b.start)&(data.time<b.end)].copy()
                if end>boundary.end or not (frame.time<split).any() or sum(frame.time>=split)<5:
                    records.extend(dict(group=group,condition=str(seconds),arm=a,status='unsupported',
                        reason='insufficient observation duration or fewer than 5 post samples') for a in ARMS)
                    continue
                try: cb,cd=capture(frame,split,ds,path,b)
                except Exception as exc:
                    records.extend(dict(group=group,condition=str(seconds),arm=a,status='error',error=repr(exc)) for a in ARMS)
                    continue
                for arm in ARMS:
                    add(group,seconds,arm,lambda arm=arm:cd if arm=='original' else core(frame,split,ds,cb,arm))
        for arm in ['original','median_std']:
            for backend in ['direct','singleton']:
                def ranking(arm=arm,backend=backend):
                    d=copy.deepcopy(originals[arm]);d={k:v for k,v in d.items() if k not in ['group','condition','arm','seconds','status']}
                    if backend=='singleton':
                        A=np.asarray(d['coverage_strength_matrix']);w=np.array([d['metric_service_evidence'][s] for s in d['game_services']])
                        scores=dict(zip(d['game_candidates'],(A@w).tolist()))
                    else:scores=d['metric_service_evidence']
                    metric_ranks=m._rank_metrics(list(d['metric_evidence']),scores,d['metric_evidence'])
                    d['service_ranks']=list(dict.fromkeys(m._parse_svc(c) for c in metric_ranks))
                    d['service_banzhaf']=scores
                    if backend=='direct':d['game_candidates']=list(d['metric_service_evidence'])
                    return d
                add('backend',backend,arm,ranking)
    return dict(dataset=ds,case=path.parent.parent.name+'/'+path.parent.name,input=str(path),
        inject=inject,boundary=boundary.to_dict(),normal_rows=int(sum(data.time<inject)),
        post_rows=int(sum(data.time>=inject)),seconds=time.perf_counter()-started,records=records)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--limit',type=int)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    import os
    if any(k.startswith(('MERIT_','GABANZHAF_')) for k in os.environ):raise ValueError('Unset environment overrides')
    manifest=dict(status='development',source=digest(m.__file__),runner=digest(__file__),
        method_source_sha256=merit_source_hashes(),
        arms=ARMS,extra_scale_arm='positive_only',pollution_fractions=[.05,.1,.2],delay_seconds=[60,120,180],
        horizon_seconds=[60,180,300],re1_paths=ROOTS,limit=args.limit,
        pollution='append floor(fraction*n_normal) initial post samples to metric baseline only; observation unchanged; diagnostic overlap intentional',
        delay='move algorithm split, including graph baseline; retain full record',
        horizon='keep all normal rows, truncate all modalities with common boundary',
        validation='RE1 cross-dataset, not asserted previously unseen; no tuning',
        config=vars(m._GabanzhafConfig.from_kwargs({})))
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (args.output/'source').mkdir()
    snapshot_merit_sources(args.output / 'source')
    for f in [Path(__file__),Path(m.__file__),Path('script/merit_training_free.py'),Path('script/run_paper_dataset.py')]:
        (args.output/'source'/f.name).write_bytes(f.read_bytes())
    for ds in ['re2-ss','re2-ob','re2-tt']+list(ROOTS):
        paths=discover_data_paths(Path.cwd(),ds) if ds.startswith('re2') else sorted((Path('data')/ROOTS[ds]).glob('*/*/data.csv'))
        if args.limit:paths=paths[:args.limit]
        if not paths:raise ValueError(f'No cases {ds}')
        for i,path in enumerate(paths):
            inputs={f.name:digest(f) for f in path.parent.iterdir() if f.name in [path.name,'inject_time.txt','traces.csv','tracets_lat.csv','tracets_err.csv']}
            try:result=run_case(path,ds)
            except Exception as e:result=dict(dataset=ds,case=path.parent.parent.name+'/'+path.parent.name,error=repr(e),records=[])
            result['input_sha256']=inputs
            assert all(digest(path.parent/k)==v for k,v in inputs.items())
            out=args.output/ds/(result['case'].replace('/','__')+'.json');out.parent.mkdir(exist_ok=True)
            out.write_text(json.dumps(result,ensure_ascii=False))
            print(ds,i+1,len(paths),result.get('error',''),sum(r['status']=='error' for r in result['records']),flush=True)
    assert digest(m.__file__)==manifest['source']
    require_merit_sources(manifest)

if __name__=='__main__':main()
