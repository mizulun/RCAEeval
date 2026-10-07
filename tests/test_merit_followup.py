import numpy as np
from script.merit_followup import scorer, m, core


def test_fallback_factorial_changes_only_selected_scale_columns():
    normal=np.array([[0,0,0],[0,1,0],[0,2,0],[0,3,0],[1,4,0]],float)
    anomaly=np.array([[2,8,2],[3,9,3]],float)
    q=np.percentile(normal,75,axis=0)-np.percentile(normal,25,axis=0);std=normal.std(0)
    scales={'original':np.where(q>0,q,1),
            'fallback_only':np.where(q>0,q,np.where(std>0,std,1)),
            'positive_only':np.where((q>0)&(std>0),std,1),
            'median_std':np.where(std>0,std,1)}
    cfg=m._GabanzhafConfig.from_kwargs(dict(use_early_full_fusion=False,ev_w_mag=1,ev_w_persistence=0,ev_w_peak=0))
    for arm,scale in scales.items():
        z=abs((anomaly-np.median(normal,axis=0))/scale)
        z[:,2]=np.clip(z[:,2],0,10)
        expected=m._norm01_arr(m._norm01_arr(np.percentile(z,95,axis=0)))
        np.testing.assert_allclose(list(scorer(arm)(normal,anomaly,list('abc'),cfg).values()),expected)


def test_pollution_changes_baseline_only(monkeypatch):
    import script.merit_followup as f
    normal=np.zeros((20,2));abnormal=np.ones((10,2));seen={}
    def score(n,a,c,config=None):seen.update(normal=n.copy(),anomal=a.copy());return {}
    monkeypatch.setattr(f,'scorer',lambda arm:score)
    def fake_core(*args):
        m._metric_evidence(normal,abnormal,['a','b'])
        return {'diagnostics':{}}
    monkeypatch.setattr(m,'_merit_core',fake_core);monkeypatch.setattr(f,'compact',lambda d:d)
    core(None,None,None,{},'original',.1)
    assert seen['normal'].shape==(22,2)
    np.testing.assert_array_equal(seen['normal'][-2:],abnormal[:2])
    np.testing.assert_array_equal(seen['anomal'],abnormal)


def test_delay_and_horizon_exposure_contract(monkeypatch,tmp_path):
    import pandas as pd
    import script.merit_followup as f
    data=pd.DataFrame({'time':np.arange(0,1000,10),'a_cpu':np.arange(100)})
    monkeypatch.setattr(f,'prepare_case',lambda p:(data,400.))
    captures=[]
    def capture(frame,split,ds,path,b):
        captures.append((split,b.start,b.end,frame.time.min(),frame.time.max()))
        return {},{'service_ranks':['a'],'game_candidates':['a'],'game_services':['a'],
            'coverage_strength_matrix':[[1.]],'metric_service_evidence':{'a':1.},
            'service_banzhaf':{'a':1.},'coverage_rescued_k':0,'metric_evidence':{'a_cpu':1.}}
    monkeypatch.setattr(f,'capture',capture)
    monkeypatch.setattr(f,'core',lambda *a,**kw:capture(data,400,None,None,f.full_record_boundary(data.time,400))[1])
    result=f.run_case(tmp_path/'a_cpu'/'1'/'data.csv','re2-test')
    assert len(result['records'])==35
    for horizon in [60,180,300]:
        assert (400.,0.,400.+horizon,0,400+horizon-10) in captures
    for delay in [60,120,180]:
        assert (400.+delay,0.,1000.,0,990) in captures


def test_followup_estimand_groups_repetitions_and_faults():
    from script.summarize_merit_followup import estimate,values
    # Fault types have unequal case counts; estimand is macro across faults.
    items=[('cpu','a',np.ones(4))]*3+[('mem','b',np.zeros(4))]
    result=estimate(items)
    assert all(v['value']==.5 for v in result.values())
    assert values({'status':'unsupported'},'a') is None
    np.testing.assert_array_equal(values({'status':'error'},'a'),np.zeros(4))
