"""Final independent checks on completed follow-up records; no experiment reruns."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import pandas as pd
from script.merit_training_free import digest,m
from script.summarize_merit_followup import values
from script.merit_source_provenance import require_merit_sources

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);args=ap.parse_args()
    root=args.input;payload=json.loads((root/'analysis.json').read_text())
    manifest=json.loads((root/'manifest.json').read_text());assert digest(m.__file__)==manifest['source']
    require_merit_sources(manifest)
    assert digest(root/'source'/'merit_followup.py')==manifest['runner']
    checked=0;oldchecked=0;windows=[];filehashes=[]
    for ds,n in payload['info']['datasets'].items():
        paths=sorted((root/ds).glob('*.json'));assert len(paths)==n
        for path in paths:
            c=json.loads(path.read_text());assert not c.get('error')
            inputs=Path(c['input']);times=pd.read_csv(inputs,usecols=['time'])['time'].to_numpy(float)
            for filename,h in c['input_sha256'].items():
                assert digest(inputs.parent/filename)==h,(path,filename)
                filehashes.append(dict(path=str(inputs.parent/filename),sha256=h))
            rr={(r['group'],r['condition'],r['arm']):r for r in c['records']}
            assert len(rr)==(35 if ds.startswith('re2') else 4)
            service=c['case'].split('/')[0].rsplit('_',1)[0]
            for key,r in rr.items():
                if r['status']=='unsupported':continue
                assert r['status']=='ok',(path,key,r.get('error'))
                a=np.array(r['coverage_strength_matrix']);assert np.isfinite(a).all() and ((a>=0)&(a<=1)).all()
                assert len(r['service_ranks'])==len(set(r['service_ranks']))
                if key[0]=='backend':
                    base=rr['scale','full',key[2]]
                    assert r['coverage_strength_matrix']==base['coverage_strength_matrix']
                    assert r['metric_service_evidence']==base['metric_service_evidence']
                    if key[1]=='singleton':
                        assert r['game_candidates']==base['game_candidates']
                        w=np.array([base['metric_service_evidence'][s] for s in base['game_services']])
                        np.testing.assert_allclose([r['service_banzhaf'][s] for s in base['game_candidates']],a@w)
                    checked+=1
                val=values(r,service)
                saved=next(x for x in payload['cases'] if x['dataset']==ds and x['case']==c['case'] and
                    (x['group'],x['condition'],x['arm'])==key)
                np.testing.assert_array_equal(val,saved['metrics'])
            if ds.startswith('re2'):
                prior=Path('artifacts/merit-training-free-20260909')/ds/path.name
                old={r['variant']:r for r in json.loads(prior.read_text())['records']}
                for arm,v in [('original','full'),('median_std','median_std')]:
                    for k in ['service_ranks','game_candidates','game_services','coverage_strength_matrix','metric_service_evidence','service_banzhaf']:
                        assert rr['scale','full',arm][k]==old[v][k],(path,arm,k)
                    oldchecked+=1
                for group,conds in [('horizon',[60,180,300]),('delay',[60,120,180])]:
                    for condition in conds:
                        split=c['inject']+(condition if group=='delay' else 0)
                        end=c['inject']+condition if group=='horizon' else c['boundary']['end']
                        npost=int(((times>=split)&(times<end)).sum())
                        nearly= max(round(npost*manifest['config']['early_window_ratio']),1)
                        windows.append(dict(dataset=ds,case=c['case'],group=group,condition=condition,
                            pre_rows=int((times<split).sum()),post_rows=npost,
                            early_full_fallback=nearly<manifest['config']['early_window_min_len'],
                            status=rr[group,str(condition),'original']['status']))
    for r in payload['summary']:
        if r['metrics']:
            assert all(0<=r['metrics'][k]['value']<=1 for k in ['AC@1','AC@3','AC@5','AVG@5'])
    with (root/'window-diagnostics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(windows[0]));w.writeheader();w.writerows(windows)
    report=dict(status='verified',cases=sum(payload['info']['datasets'].values()),
        real_records=payload['info']['real_records'],backend_checks=checked,prior_baseline_checks=oldchecked,
        input_files_checked=len(filehashes),errors=payload['info']['errors'],unsupported=payload['info']['unsupported'],
        production_unchanged=True,tests='see test-verification.json',browser='verify separately before completion')
    (root/'final-audit.json').write_text(json.dumps(report,indent=2))
    (root/'verified-inputs.json').write_text(json.dumps(filehashes,indent=2))
    lines=['# 這次實驗做了什麼？','',
        '我們沒有训练模型，也沒有更換正式 MERIT。先把可能的改善原因拆開，再故意加入不理想的條件，最後換到 RE1 檢查。',
        '', '## 先看完整紀錄下的尺度拆解','',
        '| 資料集 | 方法 | AC@1 | AVG@5 | 改對／改錯 |','|---|---|---:|---:|---|']
    names={'original':'原版','median_std':'全部改 std','fallback_only':'只修 IQR=0','positive_only':'只改正 IQR'}
    for r in payload['summary']:
        if r['group']=='scale' and r['fault']=='ALL':
            lines.append(f"| {r['dataset']} | {names[r['arm']]} | {100*r['metrics']['AC@1']['value']:.2f}% | {100*r['metrics']['AVG@5']['value']:.2f}% | {r['fixed']}/{r['broken']} |")
    lines+=['','## 怎麼讀其他組別？','',
        '- 污染：把少量故障初期資料放進正常基線，看看方法是否因此判斷失準。異常窗口固定，才能隔離基線污染的影響。',
        '- 延遲：方法晚60、120、180秒收到切分點，連呼叫圖基線也一起變動。這更接近偵測晚到的狀況。',
        '- 觀察時間：只給故障後1、3、5分鐘的資料，確認改善是否需要等待很久。所有模態同步截止；不夠長的紀錄明列不適用。',
        '- 後段：使用同一份metric證據，比較完整方法、直接排序、singleton，看看圖與重疊折減還帶來多少價值。',
        '- RE1：固定公式後換資料集檢查。因為過去已做過RE1實驗，這不叫從未查看過的獨立測試集。',
        '', '在 [互動網頁](index.html) 選擇組別和條件，即可看所有指標、配對區間與改錯案例。區間只是目前資料上的描述；任何一次提升都不等於普遍定位保證。',
        '', f"已檢查 {report['cases']} 個不同案例、{report['real_records']} 筆實際記錄、{checked} 組後段比較及 {oldchecked} 組前輪基準一致性。"]
    (root/'explanation-zh.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report))

if __name__=='__main__':main()
