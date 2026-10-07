"""Verify reuse, summarize factorial contrasts and write the research conclusions."""
import argparse
import json
from pathlib import Path
import numpy as np
from script.merit_training_free import digest, m, REUSE, VARIANTS
from script.summarize_merit_training_free import contrast
from script.merit_source_provenance import require_merit_sources


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);args=ap.parse_args()
    root=args.input;manifest=json.loads((root/'manifest.json').read_text());prior=Path(manifest['prior'])
    assert digest(m.__file__)==manifest['source_sha256']
    require_merit_sources(manifest)
    require_merit_sources(json.loads((prior / 'manifest.json').read_text()))
    # Core/harness drift must invalidate reuse even when production file matches.
    source_checks={}
    for path in ['script/merit_design_audit.py','script/run_paper_dataset.py','RCAEval/benchmark/input_boundaries.py']:
        snapshot=prior/'source'/Path(path).name
        assert snapshot.exists() and digest(path)==digest(snapshot),path
        source_checks[path]=digest(path)
    comparisons={
        '中心替換（IQR固定）':{'mean_iqr':1,'full':-1},
        '尺度替換（median固定）':{'median_std':1,'full':-1},
        '中心替換（std固定）':{'standard_z':1,'median_std':-1},
        '中心×尺度交互作用':{'standard_z':1,'median_std':-1,'mean_iqr':-1,'full':1},
        '中心化淨作用（都取消min-max）':{'trace_centered_cosine':1,'trace_raw_cosine':-1},
        '校準尺度與等效原門檻－原版':{'normal_iqr_matched_threshold':1,'full':-1}}
    output=[];checked=0;reused=0
    md=['# 本輪發現與受控對比（interim）','',
        '這些對比解釋目前開發案例上的演算法行為，不是獨立驗證，也不是資料生成機制的因果估計。','',
        '## 關鍵結論','',
        '1. **std 分母值得優先驗證，mean 中心不是主要收益來源。** SS/OB 的 median/std 已達到與 mean/std 相同 AC@1；只換 mean、保留 IQR 反而下降。這支持尺度估計及其下游效應值得研究，不能直接證明正常資料為高斯。',
        '2. **TT 呈現指標取捨。** median/std 第一名略降、AVG@5 上升；mean/IQR 也提高前五名指標但降低 AC@1，不能用單一指標宣布淘汰。',
        '3. **單純乘上1.349不能重現 std 的大幅收益。** 等效門檻對照在 SS/OB 仍有少數變化；IQR=0 fallback 欄與正 IQR 欄沒有共同縮放，須把這個機制與理想尺度不變性分開。',
        '4. **trace 中心化沒有帶來 AC@1 提升。** TT 的前五名有小幅變化；SS 無 trace，是結構性無作用。原值 cosine 是取消 min-max 的必要對照。',
        '5. **不新增 victim 扣除或 p 值篩邊。** 合成反例與可識別性條件尚不支持把 cosine 當傳播係數，或把位移顯著性當因果邊。','',
        '## 隔離對比（差值，百分點）','',
        '| 系統 | 對比 | ΔAC@1 [95% CI] | ΔAC@3 | ΔAC@5 | ΔAVG@5 |',
        '|---|---|---|---:|---:|---:|']
    mechanisms=[]
    for ds in ['re2-ss','re2-ob','re2-tt']:
        files=sorted((root/ds).glob('*.json'));assert len(files)==90
        cases=[json.loads(f.read_text()) for f in files]
        for c in cases:
            assert not c.get('error'),c.get('error')
            rr={r['variant']:r for r in c['records']};assert set(rr)==set(VARIANTS)
            assert all(r['status']=='ok' for r in rr.values())
            oldfile=prior/ds/(c['case'].replace('/','__')+'.json')
            assert digest(oldfile)==c['prior_case_sha256']
            old={r['variant']:r for r in json.loads(oldfile.read_text())['records']}
            for v in REUSE:
                assert {k:v0 for k,v0 in rr[v].items() if k!='reused_from'}==old[v]
                reused+=1
            for v in ['singleton','shapley']:
                for key in ['game_candidates','game_services','metric_service_evidence','coverage_strength_matrix']:
                    assert rr[v][key]==rr['full'][key]
                checked+=1
            for r in rr.values():
                A=np.asarray(r['coverage_strength_matrix']);assert np.all(np.isfinite(A)) and np.all((A>=0)&(A<=1))
                ranks=r['service_ranks'];assert len(ranks)==len(set(ranks))
            mechanisms.append(dict(dataset=ds,case=c['case'],
                zero_iqr_columns=sum(x['zero_iqr'] for x in c['metric_baselines']),
                calibrated_matched_rank_changed=rr['normal_iqr_matched_threshold']['service_ranks']!=rr['full']['service_ranks'],
                raw_trace_rank_changed=rr['trace_raw_cosine']['service_ranks']!=rr['full']['service_ranks'],
                centered_vs_raw_rank_changed=rr['trace_centered_cosine']['service_ranks']!=rr['trace_raw_cosine']['service_ranks']))
        for label,coefs in comparisons.items():
            value=contrast(cases,coefs);output.append(dict(dataset=ds,contrast=label,coefficients=coefs,metrics=value))
            p=value['AC@1'];ci=p['ci95']
            md.append(f"| {ds} | {label} | {p['delta']*100:+.2f} [{ci[0]*100:+.2f}, {ci[1]*100:+.2f}] | "+
                ' | '.join(f"{value[k]['delta']*100:+.2f}" for k in ['AC@3','AC@5','AVG@5'])+' |')
    synth=json.loads((root/'synthetic.json').read_text())
    md+=['','## 虛無對照的合成驗證','','| 情境 | p≤.05 次數／400 | 比例 [Wilson 95% CI] |','|---|---|---|']
    for label,s in synth['null_simulations'].items():
        md.append(f"| {label} | {s['rejected']}/{s['n']} | {s['rate']:.4f} [{s['wilson95'][0]:.4f}, {s['wilson95'][1]:.4f}] |")
    md+=['','第一列滿足循環群不變性；第二列是一般 AR(1)，此模擬接近名目水準不構成普遍保證；第三列違反群不變性，不能把它的高拒絕率解讀為直接因果關係。',
         '', '## 版本與驗證','',f'{reused} 筆舊結果逐欄相等；{checked} 組 singleton/Shapley 同矩陣、候選、權重檢查通過。270 案例共2970筆記錄，七種新介入1890筆、重用1080筆；沒有錯誤。',
         '', '完整各故障類型數值在 summary.csv；全部指標的 CI 在 paired.json／contrasts.json，負面案例在 case_comparisons.csv。沒有獨立留出資料，暫不更換正式預設。']
    (root/'findings-zh.md').write_text('\n'.join(md)+'\n')
    (root/'contrasts.json').write_text(json.dumps(output,ensure_ascii=False,indent=2))
    (root/'mechanism-cases.json').write_text(json.dumps(mechanisms,indent=2))
    verification=dict(cases=270,new_records=1890,reused_records=reused,errors=0,
        same_matrix_checks=checked,source_sha256=digest(m.__file__),reuse_source_checks=source_checks)
    (root/'verification.json').write_text(json.dumps(verification,indent=2))
    (root/'source'/'analysis').mkdir(exist_ok=True)
    for f in ['script/summarize_merit_training_free.py','script/finalize_merit_training_free.py']:
        (root/'source'/'analysis'/Path(f).name).write_bytes(Path(f).read_bytes())
    (root/'analysis-sources.json').write_text(json.dumps({f:digest(f) for f in
        ['script/summarize_merit_training_free.py','script/finalize_merit_training_free.py']},indent=2))


if __name__=='__main__': main()
