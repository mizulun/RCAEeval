"""Generate auditable tables and descriptive, stratified paired cluster intervals."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from script.merit_design_audit import VARIANTS
METRICS=['AC@1','AC@3','AC@5','AVG@5']
LABELS=['完整組態','無 metric','mean/std','僅幅度','無 early fusion','無 dominance',
'服務 mean','服務 max','無 trace','無 trace 分數','無動態邊','僅 lag-0','時間打亂',
'無 coverage','僅救援候選','僅覆蓋既有候選','取消覆蓋門檻','全部候選','Singleton','Shapley','直接證據和','無 victim','無 C／V']

def score(record,root):
    if record['status']=='no_evidence':return None
    ranks=record.get('service_ranks',[]) if record['status']=='ok' else []
    rank=ranks.index(root)+1 if root in ranks else 10**6
    return np.array([rank<=1,rank<=3,rank<=5,sum(rank<=k for k in range(1,6))/5],float)


def summarize(root):
    rows=[];pairs=[];rng=np.random.default_rng(20260908)
    for dataset in ['re2-ss','re2-ob','re2-tt']:
        cases=[json.loads(p.read_text()) for p in sorted((root/dataset).glob('*.json'))]
        if len(cases)!=90:raise ValueError(f'{dataset}: expected 90 cases, got {len(cases)}')
        faults=sorted({c['case'].split('/')[0].rsplit('_',1)[1] for c in cases})
        for variant,label in zip(VARIANTS,LABELS):
            info=[]
            for c in cases:
                service,fault=c['case'].split('/')[0].rsplit('_',1)
                rec={r['variant']:r for r in c['records']};r=rec[variant];base=rec['full']
                val=score(r,service);b=score(base,service)
                info.append((c,service,fault,r,val,b))
            for fault in ['ALL']+faults:
                subset=[x for x in info if fault=='ALL' or x[2]==fault]
                vals=[x[4] for x in subset if x[4] is not None]
                result=dict(dataset=dataset,variant=variant,label=label,fault=fault,n=len(subset),
                  n_valid=len(vals),errors=sum(x[3]['status']=='error' for x in subset),
                  no_evidence=sum(x[3]['status']=='no_evidence' for x in subset))
                if vals:
                    # Six-type equal macro. All groups have 15 cases; keep definition explicit.
                    groups=[np.mean([x[4] for x in subset if x[2]==f and x[4] is not None],axis=0)
                      for f in faults if any(x[2]==f and x[4] is not None for x in subset)]
                    result.update(zip(METRICS,np.mean(groups,axis=0).tolist()))
                rows.append(result)
            valid=[x for x in info if x[4] is not None and x[5] is not None]
            if not valid:continue
            clusters={}
            for _,service,fault,r,val,b in valid:clusters.setdefault((fault,service),[]).append(val-b)
            # Stratify by fault type; cluster root-service repeated injections together.
            distributions=[]
            for fault in faults:
                group=np.array([np.mean(v,axis=0) for (f,s),v in clusters.items() if f==fault])
                idx=rng.integers(0,len(group),size=(5000,len(group)))
                distributions.append(group[idx].mean(axis=1))
            boot=np.mean(distributions,axis=0)
            p=dict(dataset=dataset,variant=variant,n=len(valid),clusters=len(clusters),
             fixed=sum(x[4][0]>x[5][0] for x in valid),broken=sum(x[4][0]<x[5][0] for x in valid),
             rank_changed=sum(x[3].get('service_ranks',[])!=next(r for r in x[0]['records'] if r['variant']=='full').get('service_ranks',[]) for x in valid),
             candidate_recall=float(np.mean([x[1] in x[3].get('game_candidates',[]) for x in valid])) if variant!='direct_evidence' else None,
             coverage_rescued_cases=sum(x[3].get('coverage_rescued_k',0)>0 for x in valid),
             core_seconds_median=float(np.median([x[3]['core_seconds'] for x in valid if 'core_seconds' in x[3]])),
             no_graph_cases=sum(x[0].get('graph_edges',0)==0 for x in valid),
             no_dynamic_edge_cases=sum(x[0].get('trace_edges',0)==0 for x in valid))
            demoted=[]; offdiag=[]
            for x in valid:
                r=x[3]; A=np.asarray(r.get('coverage_strength_matrix',[]))
                if A.size==0:demoted.append(0);offdiag.append(0);continue
                diagonal={(i,r['game_services'].index(c)) for i,c in enumerate(r['game_candidates']) if c in r['game_services']}
                demoted.append(sum(A[i,j]<1 for i,j in diagonal))
                offdiag.append(sum(A[i,j]>0 for i in range(A.shape[0]) for j in range(A.shape[1]) if (i,j) not in diagonal))
            p['demotion_cases']=sum(n>0 for n in demoted)
            p['demoted_candidates_total']=sum(demoted)
            p['nonself_coverage_cases']=sum(n>0 for n in offdiag)
            p['nonself_coverage_edges_total']=sum(offdiag)
            point=np.mean([np.mean([x[4]-x[5] for x in valid if x[2]==f],axis=0) for f in faults],axis=0)
            point[np.abs(point)<1e-14]=0.0
            boot[np.abs(boot)<1e-14]=0.0
            for i,key in enumerate(METRICS):
                p[key]={'delta':float(point[i]),'ci95':np.quantile(boot[:,i],[.025,.975]).tolist()}
            pairs.append(p)
    return rows,pairs


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);args=ap.parse_args()
    rows,pairs=summarize(args.input)
    (args.input/'paired.json').write_text(json.dumps(pairs,ensure_ascii=False,indent=2,default=lambda x: x.item()))
    with (args.input/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    tex=['% Generated by script/summarize_merit_design_audit.py; do not edit numbers.']
    md=['# MERIT 設計稽核實驗（interim）','差值 = 變體 − 完整版；區間為 5000 次 fault 分層、root-service×fault 配對 cluster bootstrap；未多重比較校正。']
    for ds in ['re2-ss','re2-ob','re2-tt']:
        tex += [r'\begin{table}[htbp]\centering\small\setlength{\tabcolsep}{3pt}',r'\caption{'+ds.upper()+r' 全流程受控對照（interim，每組 90 案例）。數值為六類故障等權平均；差值為變體減完整組態的 AC@1 百分點，括號為描述性 95\% 配對 cluster 區間；N/A 表示無有效模態證據；最佳值粗體、次佳不同值底線。}',r'\label{tab:audit-'+ds+r'}',r'\begin{tabular}{lrrrrl}\toprule',r'變體 & AC@1 & AC@3 & AC@5 & AVG@5 & $\Delta$AC@1（pp）\\\midrule']
        md += ['','## '+ds,'','| 變體 | AC@1 | AC@3 | AC@5 | AVG@5 | ΔAC@1 [95% CI] pp | 改對/改錯 |','|---|---:|---:|---:|---:|---|---|']
        for r in [r for r in rows if r['dataset']==ds and r['fault']=='ALL']:
            p=next((p for p in pairs if p['dataset']==ds and p['variant']==r['variant']),None)
            numbers=[f'{r[k]:.4f}' if k in r else 'N/A' for k in METRICS]
            delta='--' if p is None else f"{p['AC@1']['delta']*100:+.2f} [{p['AC@1']['ci95'][0]*100:+.2f}, {p['AC@1']['ci95'][1]*100:+.2f}]"
            formatted=[]
            for key,num in zip(METRICS,numbers):
                values=sorted({x[key] for x in rows if x['dataset']==ds and x['fault']=='ALL' and key in x},reverse=True)
                if key in r and r[key]==values[0]:formatted.append(r'\textbf{'+num+'}')
                elif key in r and len(values)>1 and r[key]==values[1]:formatted.append(r'\underline{'+num+'}')
                else:formatted.append(num)
            tex += [r['label']+' & '+' & '.join(formatted)+' & '+delta+r'\\']
            md += ['| '+' | '.join([r['label']]+numbers+[delta,'N/A' if p is None else f"{p['fixed']}/{p['broken']}"])+' |']
        tex += [r'\bottomrule\end{tabular}\end{table}']
    Path('paper/generated/design-audit-tables-zh.tex').write_text('\n'.join(tex)+'\n')
    out = Path('output/generated-docs/merit-design-audit-20260908.md')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(md)+'\n')

if __name__=='__main__':main()
