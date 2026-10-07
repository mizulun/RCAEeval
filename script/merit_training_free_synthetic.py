"""Mechanism validation; no incident labels or fitted model."""
import argparse
import json
from pathlib import Path
import numpy as np
from script.merit_training_free import m, shift_null, C


def experiments(repetitions=400):
    rng = np.random.default_rng(20260909)
    nulls = {}; n = 64
    for mode in ['independent_circular', 'independent_ar1', 'independent_shared_trend']:
        ps = []
        for _ in range(repetitions):
            if mode == 'independent_circular':
                # Circular convolution and independent uniform rotations: exact
                # cyclic-invariance null, with strong serial dependence.
                seqs = []
                for j in range(2):
                    z = rng.normal(size=n)
                    seqs.append(sum(.8**k*np.roll(z, k) for k in range(12)))
                x, y = seqs
            elif mode == 'independent_ar1':
                z = rng.normal(size=(2,n+256)); seqs = np.zeros_like(z)
                for t in range(1, z.shape[1]): seqs[:,t] = .8*seqs[:,t-1]+z[:,t]
                x,y = seqs[:,-n:]
            else:
                trend = np.linspace(-3,3,n)
                x,y = trend+rng.normal(0,.25,(2,n))
            ps.append(shift_null(x,y)['p'])
        hits = sum(p <= .05 for p in ps)
        # Wilson interval for independent synthetic draws, not real case bootstrap.
        ph = hits/repetitions; z = 1.959963984540054
        den=1+z*z/repetitions; center=(ph+z*z/(2*repetitions))/den
        half=z*np.sqrt(ph*(1-ph)/repetitions+z*z/(4*repetitions**2))/den
        nulls[mode]=dict(n=repetitions, rejected=hits, rate=ph,
                        wilson95=[center-half,center+half], p_values=ps)
    A=np.array([[1.,0.],[.7,1.]]); w=np.array([1.,.21])
    changed=A.copy();changed[0,0]=.75
    u=np.array([1.,0.]);p=np.array([0.,2.]);r=u+p
    epsilon=np.array([1.2,-.5])
    normal=np.array([0.,0.,0.,0.,1.])
    # The unit fallback is not invariant to measurement-unit changes.
    iqr=lambda a: np.percentile(a,75)-np.percentile(a,25)
    out=dict(seed=20260909, null_simulations=nulls,
        victim=dict(local=u.tolist(), propagated=p.tolist(), observed=r.tolist(),
            exact_residual=np.maximum(r-p,0).tolist(),
            misspecified_residual=np.maximum(r-(p+epsilon),0).tolist(),
            original_banzhaf=m._probabilistic_banzhaf(A,w).tolist(),
            demoted_banzhaf=m._probabilistic_banzhaf(changed,w).tolist()),
        zero_scale=dict(normal=normal.tolist(),iqr=iqr(normal),
            anomaly=2., original_deviation=2., scaled_by_100_deviation=200.),
        outlier=[dict(L=L,median=float(np.median([0,1,2,3,L])),
             iqr=float(iqr([0,1,2,3,L])),mean=float(np.mean([0,1,2,3,L])),
             std=float(np.std([0,1,2,3,L]))) for L in [4,10,100,10000]],
        temporal=dict(sustained=dict(p95=4.,persistence=1.,peak=4.),
                      spike=dict(p95=0.,persistence=.01,peak=100.)),
        normal_iqr_constant=C)
    return out


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    args.output.write_text(json.dumps(experiments(),indent=2))
