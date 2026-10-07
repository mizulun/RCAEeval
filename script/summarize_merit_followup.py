"""Paired descriptive analysis for frozen follow-up arms and scenarios."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from script.merit_training_free import digest,m

METRICS=['AC@1','AC@3','AC@5','AVG@5']

def values(r,root):
    if r['status']=='unsupported':return None
    ranks=r.get('service_ranks',[]) if r['status']=='ok' else []
    rank=ranks.index(root)+1 if root in ranks else 10**6
    return np.array([rank<=1,rank<=3,rank<=5,sum(rank<=k for k in range(1,6))/5],float)

def estimate(items):
    """items: (fault, root-service, four-vector); equal-fault macro, cluster bootstrap."""
    clusters={}
    for fault,service,val in items:clusters.setdefault((fault,service),[]).append(val)
    if not clusters:return None
    rng=np.random.default_rng(20260909);point=[];boot=[]
    for fault in sorted({k[0] for k in clusters}):
        keys=sorted(k for k in clusters if k[0]==fault)
        group=np.array([np.mean(clusters[k],axis=0) for k in keys])
        # Point estimand averages cases within fault, bootstrap resamples clusters
        # carrying all their repeated cases; use weighted sums for unequal sizes.
        sums=np.array([np.sum(clusters[k],axis=0) for k in keys]);sizes=np.array([len(clusters[k]) for k in keys])
        idx=rng.integers(0,len(keys),(5000,len(keys)))
        point.append(sums.sum(0)/sizes.sum())
        boot.append(sums[idx].sum(1)/sizes[idx].sum(1)[:,None])
    p=np.mean(point,axis=0);b=np.mean(boot,axis=0)
    p[abs(p)<1e-14]=0;b[abs(b)<1e-14]=0
    return dict(zip(METRICS,[dict(value=float(p[i]),ci=np.quantile(b[:,i],[.025,.975]).tolist()) for i in range(4)]))

def collect(directory, expected_counts=None):
    allcases=[];hashes={};counts={};overlap=[]
    for ds,expected in (expected_counts or [('re2-ss',90),('re2-ob',90),('re2-tt',90),('re1-ob',125),('re1-ss',125),('re1-tt',125)]):
        files=sorted((directory/ds).glob('*.json'))
        assert len(files)==expected,(ds,len(files))
        counts[ds]=len(files)
        for f in files:
            c=json.loads(f.read_text());assert not c.get('error'),c
            service,fault=c['case'].split('/')[0].rsplit('_',1)
            records={(r['group'],r['condition'],r['arm']):r for r in c['records']}
            # Shared clean/full baselines appear in each scenario as references;
            # these are aliases, not additional executions or independent cases.
            expanded=list(c['records'])
            if ds.startswith('re2'):
                for a in ['original','median_std','fallback_only']:
                    base=records['scale','full',a]
                    for group,condition in [('pollution','0'),('delay','0'),('horizon','full')]:
                        expanded.append(dict(base,group=group,condition=condition,alias=True))
                    if a!='fallback_only':expanded.append(dict(base,group='backend',condition='full',alias=True))
            for r in expanded:
                if r['group']=='backend':ref=records['scale','full',r['arm']]
                elif r['group']=='scale' or r.get('alias'):ref=records['scale','full','original']
                else:ref=records[r['group'],r['condition'],'original']
                clean=records['scale','full',r['arm']]
                v=values(r,service);rv=values(ref,service);cv=values(clean,service)
                ranks=r.get('service_ranks',[])
                allcases.append(dict(dataset=ds,case=c['case'],root=service,fault=fault,
                    group=r['group'],condition=r['condition'],arm=r['arm'],status=r['status'],
                    alias=r.get('alias',False),metrics=None if v is None else v.tolist(),
                    delta=None if v is None or rv is None else (v-rv).tolist(),
                    clean_delta=None if v is None or cv is None else (v-cv).tolist(),
                    root_rank=ranks.index(service)+1 if service in ranks else None,
                    top5=ranks[:5],candidate=(None if r['group']=='backend' and r['condition']=='direct' else service in r.get('game_candidates',[])),
                    seconds=(None if r['arm']=='original' and r['group'] in ['scale','delay','horizon'] else r.get('seconds')),
                    error=r.get('error',r.get('reason',''))))
            metric_hash=next(v for k,v in c['input_sha256'].items() if k in ['data.csv','simple_metrics.csv'])
            if metric_hash in hashes:overlap.append([hashes[metric_hash],ds+'/'+c['case']])
            hashes[metric_hash]=ds+'/'+c['case']
    return allcases,counts,overlap

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    directory=args.input;manifest=json.loads((directory/'manifest.json').read_text())
    assert digest(m.__file__)==manifest['source']
    from script.merit_source_provenance import require_merit_sources
    require_merit_sources(manifest)
    expected=[(ds,1) for ds in ['re2-ss','re2-ob','re2-tt','re1-ob','re1-ss','re1-tt']] if args.smoke else None
    cases,counts,overlap=collect(directory,expected);summaries=[]
    keys=sorted({(c['dataset'],c['group'],c['condition'],c['arm']) for c in cases})
    for ds,group,condition,arm in keys:
        groupcases=[c for c in cases if (c['dataset'],c['group'],c['condition'],c['arm'])==(ds,group,condition,arm)]
        for fault in ['ALL']+sorted({c['fault'] for c in groupcases}):
            sub=[c for c in groupcases if fault=='ALL' or c['fault']==fault]
            estimates={k:estimate([(c['fault'],c['root'],np.array(c[k])) for c in sub if c[k] is not None])
                       for k in ['metrics','delta','clean_delta']}
            summaries.append(dict(dataset=ds,group=group,condition=condition,arm=arm,fault=fault,
                n=len(sub),valid=sum(c['metrics'] is not None for c in sub),errors=sum(c['status']=='error' for c in sub),
                unsupported=sum(c['status']=='unsupported' for c in sub),
                fixed=sum(c['delta'] is not None and c['delta'][0]>0 for c in sub),
                broken=sum(c['delta'] is not None and c['delta'][0]<0 for c in sub),
                candidate_recall=float(np.mean([c['candidate'] for c in sub if c['candidate'] is not None and c['metrics'] is not None])) if any(c['candidate'] is not None and c['metrics'] is not None for c in sub) else None,
                seconds_median=float(np.median([c['seconds'] for c in sub if c['seconds'] is not None])) if any(c['seconds'] is not None for c in sub) else None,
                **estimates))
    info=dict(smoke=args.smoke,datasets=counts,real_records=sum(not c['alias'] for c in cases),
        errors=sum(c['status']=='error' and not c['alias'] for c in cases),
        unsupported=sum(c['status']=='unsupported' and not c['alias'] for c in cases),
        duplicate_metric_files=overlap,production_sha256=digest(m.__file__),
        prior_re1_exposure='evidence/data/re1-merit-baro.json exists; not an untouched test set',
        ci='5000 fault-stratified root-service×fault cluster bootstrap, seed20260909, unadjusted descriptive')
    payload=dict(info=info,summary=summaries,cases=cases)
    (directory/'analysis.json').write_text(json.dumps(payload,ensure_ascii=False))
    with (directory/'cases.csv').open('w') as f:
        fields=list(cases[0]);writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(cases)
    (directory/'verification.json').write_text(json.dumps(info,ensure_ascii=False,indent=2))
    print(json.dumps(info,ensure_ascii=False))

if __name__=='__main__':main()
