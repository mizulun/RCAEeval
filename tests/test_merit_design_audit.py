import itertools, math
import numpy as np
from script.merit_design_audit import shapley, m, run_variant


def test_shapley_quadrature_matches_enumeration_and_efficiency():
    rng=np.random.default_rng(20260908)
    for k in range(1,7):
        A=rng.random((k,5));A[0,0]=1;w=rng.random(5)
        exact=np.zeros(k)
        for i in range(k):
            others=[x for x in range(k) if x!=i]
            for n in range(k):
                for S in itertools.combinations(others,n):
                    exact[i]+=math.factorial(n)*math.factorial(k-n-1)/math.factorial(k)*np.sum(w*A[i]*np.prod(1-A[list(S)],axis=0))
        np.testing.assert_allclose(shapley(A,w),exact,atol=1e-12)
        np.testing.assert_allclose(sum(shapley(A,w)),np.sum(w*(1-np.prod(1-A,axis=0))),atol=1e-12)


def test_diagonal_overlap_and_wrong_edge_counterexample():
    w=np.array([1.,.21]);A=np.eye(2)
    np.testing.assert_allclose(m._probabilistic_banzhaf(A,w),w)
    A[1,0]=.7
    np.testing.assert_allclose(m._probabilistic_banzhaf(A,w),[.65,.56])
    A[0,0]=.75
    np.testing.assert_allclose(m._probabilistic_banzhaf(A,w),[.4875,.6475])


def test_trace_evidence_is_noop_in_metric_banzhaf_path_and_ties_stable():
    import pandas as pd
    data=pd.DataFrame({'time':range(40),'a_cpu':([0,1]*10)+[4]*20,'b_cpu':([0,1]*10)+[2]*20})
    base={'return_diagnostics':True,'trace_service_evidence':{'a':0.,'b':1.}}
    a=run_variant(data,20,None,base,'full');b=run_variant(data,20,None,base,'no_trace_evidence')
    assert a['service_ranks']==b['service_ranks']
    assert a['coverage_strength_matrix']==b['coverage_strength_matrix']
    assert m._rank_metrics(['b_cpu','a_cpu'],{'a':1,'b':1},{'a_cpu':1,'b_cpu':1})==['b_cpu','a_cpu']


def test_metric_shapes_and_constant_baseline_finite():
    normal=np.tile([0.,1.,0.,1.],25)[:,None]
    normal=np.repeat(normal,3,axis=1)
    anomaly=np.zeros((100,3));anomaly[:,0]=4;anomaly[-1,1]=100;anomaly[:,2]=.5
    cfg=m._GabanzhafConfig.from_kwargs({'use_early_full_fusion':False})
    ev=m._metric_evidence(normal,anomaly,['s_a','s_b','s_c'],cfg)
    assert ev['s_a']>ev['s_b']>ev['s_c']
    ev=m._metric_evidence(np.zeros_like(normal),anomaly,['s_a','s_b','s_c'],cfg)
    assert all(np.isfinite(list(ev.values())))


def test_coverage_interventions_keep_victim_controls_and_root_rescue():
    import pandas as pd
    # Root with no metric anomaly; two strongly anomalous latency children.
    data=pd.DataFrame({'time':range(40),'root_cpu':[0,1]*20,
       'a_latency':([0,1]*10)+[10]*20,'b_latency':([0,1]*10)+[10]*20})
    base={'return_diagnostics':True,'call_graph':{'root':['a','b']},
          'top_k_svcs':1,'exact_candidate_k':1}
    full=run_variant(data,20,None,base,'full')
    rescue=run_variant(data,20,None,base,'rescue_only')
    edges=run_variant(data,20,None,base,'edges_only')
    assert 'root' in full['game_candidates'] and 'root' in rescue['game_candidates']
    assert 'root' not in edges['game_candidates']
    ri=full['game_candidates'].index('root');j=full['game_services'].index('a')
    assert full['coverage_strength_matrix'][ri][j] == .7
    assert rescue['coverage_strength_matrix'][ri][j] == 0
    assert run_variant(data,20,None,base,'ungated_coverage')['service_ranks']


def test_scale_and_normalization_limits():
    x=np.array([0,1,2,3,100.]);assert np.median(x)==2
    assert np.percentile(x,75)-np.percentile(x,25)==2
    scores=np.array([.1,.3,.8])
    np.testing.assert_array_equal(np.argsort(scores/scores.max()),np.argsort(scores/scores.sum()))
    A=np.ones((2,1));w=np.ones(1)
    np.testing.assert_allclose(m._probabilistic_banzhaf(A,w),[.5,.5])


def test_submodular_marginals_and_dummy_invariance():
    rng=np.random.default_rng(8);A=rng.random((4,3));w=rng.random(3)
    for S,T in [([], [1]),([1],[1,2]),([1,2],[1,2,3])]:
        left=np.sum(w*A[0]*np.prod(1-A[S],axis=0))
        right=np.sum(w*A[0]*np.prod(1-A[T],axis=0))
        assert left>=right
    np.testing.assert_allclose(shapley(np.vstack([A,np.zeros(3)]),w)[:4],shapley(A,w),atol=1e-12)
    np.testing.assert_allclose(m._probabilistic_banzhaf(np.vstack([A,np.zeros(3)]),w)[:4],m._probabilistic_banzhaf(A,w))


def test_standard_z_changes_only_location_and_scale():
    from script.merit_design_audit import transformed
    cfg=m._GabanzhafConfig.from_kwargs({'use_early_full_fusion':False,'ev_w_mag':1.,'ev_w_persistence':0.,'ev_w_peak':0.})
    normal=np.array([[0,0,1],[1,2,2],[2,3,4],[3,4,5],[100,5,6]],float)
    anomaly=np.array([[4,7,8],[5,8,9],[6,9,10]],float)
    fn=transformed(m._metric_evidence,[('med = np.nanmedian(normal_mat, axis=0)','med = np.nanmean(normal_mat, axis=0)'),('iqr = np.where((q75 - q25) > 0, q75 - q25, 1.)','iqr = np.where(std > 0, std, 1.)')])
    score=fn(normal,anomaly,['a_cpu','b_cpu','c_cpu'],cfg)
    z=np.abs((anomaly-normal.mean(axis=0))/normal.std(axis=0))
    expected=m._norm01_arr(m._norm01_arr(np.percentile(z,95,axis=0)))
    np.testing.assert_allclose(list(score.values()),expected)


def test_evaluation_denominators_and_avg5():
    from script.summarize_merit_design_audit import score
    np.testing.assert_allclose(score({'status':'ok','service_ranks':['a','b','root']},'root'),[0,1,1,.6])
    np.testing.assert_array_equal(score({'status':'error'},'root'),np.zeros(4))
    assert score({'status':'no_evidence','service_ranks':['root']},'root') is None
