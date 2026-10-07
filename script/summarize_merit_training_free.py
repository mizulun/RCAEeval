"""Separate outputs: never writes prior thesis tables or prior experiment artifacts."""
import argparse
import csv
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from script import summarize_merit_design_audit as old
from script.merit_training_free import VARIANTS, LABELS


def contrast(cases, coefficients):
    """Common resampling for every contrast, invariant to variant iteration order."""
    clusters={}
    for c in cases:
        service,fault=c['case'].split('/')[0].rsplit('_',1)
        records={r['variant']:r for r in c['records']}
        delta=sum(weight*old.score(records[v],service) for v,weight in coefficients.items())
        clusters.setdefault((fault,service),[]).append(delta)
    rng=np.random.default_rng(20260908);boot=[];point=[]
    for fault in sorted({key[0] for key in clusters}):
        group=np.array([np.mean(clusters[key],axis=0) for key in sorted(clusters) if key[0]==fault])
        index=rng.integers(0,len(group),size=(5000,len(group)))
        boot.append(group[index].mean(axis=1));point.append(group.mean(axis=0))
    boot=np.mean(boot,axis=0);point=np.mean(point,axis=0)
    boot[abs(boot)<1e-14]=0.;point[abs(point)<1e-14]=0.
    return {key:dict(delta=float(point[i]),ci95=np.quantile(boot[:,i],[.025,.975]).tolist())
            for i,key in enumerate(old.METRICS)}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);args=ap.parse_args()
    with patch.object(old,'VARIANTS',VARIANTS), patch.object(old,'LABELS',LABELS):
        rows,pairs=old.summarize(args.input)
    for ds in ['re2-ss','re2-ob','re2-tt']:
        cases=[json.loads(p.read_text()) for p in sorted((args.input/ds).glob('*.json'))]
        for p in pairs:
            if p['dataset']==ds:
                coefs={'full':0.} if p['variant']=='full' else {p['variant']:1.,'full':-1.}
                p.update(contrast(cases,coefs))
    (args.input/'paired.json').write_text(json.dumps(pairs,ensure_ascii=False,indent=2,default=lambda x:x.item()))
    with (args.input/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    md=['# MERIT 免訓練改善受控結果（interim）','',
        '每系統 90 個已查看的開發案例，六類故障等權。CI：5000 次 fault 分層、root-service×fault cluster bootstrap；未做多重比較校正。',
        '原版、mean/std、singleton、Shapley 重用經來源與輸入雜湊檢查的舊結果；其執行時間來自舊批次，不做速度勝負比較。',
        '本輪 CI 使用同一組重抽樣索引比較所有變體，避免迭代順序影響區間；相較前輪 CI 可能有 Monte Carlo 差异，逐案例結果未改。','']
    diagnostics=[]; case_rows=[]
    for ds in ['re2-ss','re2-ob','re2-tt']:
        md += ['## '+ds,'','| 變體 | AC@1 | AC@3 | AC@5 | AVG@5 | ΔAC@1 [95% CI] pp | 改對/改錯 | 候選召回 |',
               '|---|---:|---:|---:|---:|---|---|---:|']
        for r in [r for r in rows if r['dataset']==ds and r['fault']=='ALL']:
            p=next(p for p in pairs if p['dataset']==ds and p['variant']==r['variant'])
            d=p['AC@1'];delta=f"{100*d['delta']:+.2f} [{100*d['ci95'][0]:+.2f}, {100*d['ci95'][1]:+.2f}]"
            md.append('| '+' | '.join([r['label']]+[f'{r[k]:.4f}' for k in old.METRICS]+[delta,f"{p['fixed']}/{p['broken']}",f"{p['candidate_recall']:.4f}"])+' |')
        cases=[json.loads(p.read_text()) for p in sorted((args.input/ds).glob('*.json'))]
        metrics=[x for c in cases for x in c.get('metric_baselines',[])]
        trace=[x for c in cases for x in c.get('trace_diagnostics',[]) if 'normal_shift_null' in x]
        d=dict(dataset=ds,case_metric_columns=len(metrics),
            zero_iqr=sum(x['zero_iqr'] for x in metrics),near_constant=sum(x['near_constant'] for x in metrics),
            tiny_positive_iqr=sum(x['tiny_positive_iqr'] for x in metrics),
            original_clip=sum(x['original_clip'] for x in metrics),trace_pairs=len(trace),
            normal_null_p_le_05=sum(x['normal_shift_null'] is not None and x['normal_shift_null']['p']<=.05 for x in trace),
            irregular_pairs=sum(not x['equally_spaced'] for x in trace))
        diagnostics.append(d)
        md += ['',f"基線診斷（案例×指標，非獨立樣本）：{d['case_metric_columns']} 欄；IQR=0：{d['zero_iqr']}；std≤1e-6：{d['near_constant']}；0<IQR≤1e-6：{d['tiny_positive_iqr']}。",
               f"Trace 診斷：{d['trace_pairs']} 個案例×模態×邊，正常期 cyclic-shift p≤.05 有 {d['normal_null_p_le_05']} 個。這不是 FPR 估計，也不是因果發現；真實正常期依賴未知，未用於篩邊。",'']
        for c in cases:
            root,fault=c['case'].split('/')[0].rsplit('_',1)
            full=next(r for r in c['records'] if r['variant']=='full')
            for r in c['records']:
                a=old.score(r,root);b=old.score(full,root)
                case_rows.append(dict(dataset=ds,case=c['case'],root=root,fault=fault,variant=r['variant'],
                    status=r['status'],root_rank=(r.get('service_ranks',[]).index(root)+1 if root in r.get('service_ranks',[]) else None),
                    candidate=root in r.get('game_candidates',[]),fixed=bool(a[0]>b[0]),broken=bool(a[0]<b[0]),
                    core_seconds=r.get('core_seconds'),reused='reused_from' in r))
    with (args.input/'case_comparisons.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(case_rows[0]));writer.writeheader();writer.writerows(case_rows)
    (args.input/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    (args.input/'results-zh.md').write_text('\n'.join(md)+'\n')
    counts={s:sum(r['status']==s for r in case_rows) for s in ['ok','error','no_evidence']}
    (args.input/'completion.json').write_text(json.dumps(dict(records=len(case_rows),counts=counts),indent=2))


if __name__=='__main__':main()
