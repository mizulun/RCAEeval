import numpy as np
from script.merit_training_free import m, metric_adapter, C, shift_null, max_lag_score, run_new


def test_calibrated_scale_and_matched_threshold_equivalence_positive_iqr():
    rng=np.random.default_rng(5); normal=rng.normal(size=(80,7));anomaly=rng.normal(3,2,(40,7))
    cols=[f's{i}_cpu' for i in range(7)]
    original=m._metric_evidence(normal,anomaly,cols)
    matched=metric_adapter('normal_iqr_matched_threshold')(normal,anomaly,cols)
    np.testing.assert_allclose(list(original.values()),list(matched.values()),atol=1e-8)
    a=metric_adapter('normal_iqr')(normal,anomaly,cols)
    b=metric_adapter('iqr_z_threshold')(normal,anomaly,cols)
    np.testing.assert_allclose(list(a.values()),list(b.values()),atol=1e-8)


def test_location_scale_factorial_against_direct_arithmetic():
    normal=np.array([[0,0,2],[1,1,3],[2,2,4],[3,3,5],[100,4,8]],float)
    anomaly=np.array([[5,7,9],[6,8,10]],float);cols=['a','b','c']
    cfg=m._GabanzhafConfig.from_kwargs(dict(use_early_full_fusion=False,ev_w_mag=1.,ev_w_persistence=0.,ev_w_peak=0.))
    for variant,center,scale in [('mean_iqr',normal.mean(0),np.percentile(normal,75,axis=0)-np.percentile(normal,25,axis=0)),
                                ('median_std',np.median(normal,axis=0),normal.std(0))]:
        actual=metric_adapter(variant)(normal,anomaly,cols,cfg)
        expected=m._norm01_arr(m._norm01_arr(np.percentile(abs((anomaly-center)/scale),95,axis=0)))
        np.testing.assert_allclose(list(actual.values()),expected)


def test_centering_cancellation_and_signed_cosine_difference():
    x=np.array([10.,11.,9.,12.]);y=np.array([20.,19.,21.,18.])
    np.testing.assert_allclose(m._norm_trace_arr(x),m._norm_trace_arr(x-x[:2].mean()))
    assert m._lagged_cosine_score(x,y,0)>.9
    assert m._lagged_cosine_score(x-x.mean(),y-y.mean(),0)==0


def test_cyclic_null_matches_enumeration_including_identity_and_ties():
    rng=np.random.default_rng(8);x=rng.normal(size=16);y=rng.normal(size=16)
    null=[max_lag_score(x,np.roll(y,k))[0] for k in range(len(y))]
    actual=shift_null(x,y)
    assert actual['p']==np.mean(np.array(null)>=null[0]-1e-12)
    assert shift_null(np.zeros(16),y)['p']==1
    assert shift_null(x[:3],y[:3]) is None


def test_residual_error_bound_and_wrong_edge_reverses_root():
    rng=np.random.default_rng(9);u=rng.random(100);p=rng.random(100);eps=rng.normal(size=100)
    estimated=np.maximum((u+p)-(p+eps),0)
    assert np.all(abs(estimated-u)<=abs(eps)+1e-12)
    A=np.array([[1.,0.],[.7,1.]]);w=np.array([1.,.21])
    assert np.argmax(m._probabilistic_banzhaf(A,w))==0
    A[0,0]=.75
    assert np.argmax(m._probabilistic_banzhaf(A,w))==1


def test_constant_and_near_constant_are_finite_all_metric_variants():
    normal=np.column_stack([np.zeros(40),np.arange(40)*1e-12,np.tile([0.,1.],20)])
    anomaly=np.tile([2.,1e-9,4.],(20,1))
    for v in ['mean_iqr','median_std','normal_iqr','normal_iqr_matched_threshold','iqr_z_threshold']:
        assert np.all(np.isfinite(list(metric_adapter(v)(normal,anomaly,['a','b','c']).values())))


def test_positive_scale_invariance_and_zero_scale_counterexample():
    rng=np.random.default_rng(42);normal=rng.normal(size=(100,4));abnormal=rng.normal(4,2,(40,4))
    scales=np.array([2.,10.,100.,.5]);offset=np.array([5.,-10.,0.,200.]);cols=list('abcd')
    for variant in ['mean_iqr','median_std','normal_iqr']:
        fn=metric_adapter(variant)
        np.testing.assert_allclose(list(fn(normal,abnormal,cols).values()),
            list(fn(normal*scales+offset,abnormal*scales+offset,cols).values()),atol=1e-8)
    x=np.array([0.,0.,0.,0.,1.]);assert np.percentile(x,75)-np.percentile(x,25)==0
    assert abs(2-np.median(x))/1 != abs(200-np.median(100*x))/1


def test_contrast_resampling_is_order_independent_and_zero_for_identical_results():
    from script.summarize_merit_training_free import contrast
    cases=[]
    for f in ['cpu','mem']:
        for s in ['a','b']:
            for rep in range(3):
                ranks=[s,'z'] if rep else ['z',s]
                cases.append(dict(case=f'{s}_{f}/{rep}',records=[
                    dict(variant=v,status='ok',service_ranks=ranks) for v in ['full','same']]))
    a=contrast(cases,{'same':1,'full':-1});b=contrast(list(reversed(cases)),{'same':1,'full':-1})
    assert a==b
    assert all(v['delta']==0 and v['ci95']==[0,0] for v in a.values())
