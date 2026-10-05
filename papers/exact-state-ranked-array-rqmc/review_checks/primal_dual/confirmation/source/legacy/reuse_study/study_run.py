"""Setup-inclusive complete trajectory execution and study cells."""
import time
import numpy as np
import reuse_kernel as rk
import run as previous

METHODS=('reuse','rebuild','rank_stream','full_basis')


def execute(spec, seed, method, audit=False):
    m,w,T,K=spec['m'],spec['w'],spec['horizon'],spec['K']
    if not (1<=m<=30 and m<=w<=52 and T*(1<<m)*K<2**63):
        raise ValueError('outside the measured integer domain')
    start=time.perf_counter()
    C=previous.base_rows(m,w)
    cuts,dest=previous.problem(spec)
    setup=time.perf_counter()-start
    result=rk.trajectory(C,cuts,dest,m,T,int(seed),METHODS.index(method),bool(audit))
    seconds=time.perf_counter()-start
    return seconds,setup,result


def cells(development=False):
    if development:
        return [('scaling',dict(model='repair',K=K,w=w,m=m,horizon=T),METHODS)
                for K,w,m,T in ((63,52,12,1),(63,52,12,100),
                                (63,52,20,100),(255,52,20,100))]
    out=[('scaling',dict(model='repair',K=63,w=w,m=m,horizon=100),METHODS)
         for w in (30,52) for m in (8,12,16,20)]
    out += [('scaling',dict(model='repair',K=255,w=52,m=m,horizon=100),METHODS)
            for m in (12,16,20)]
    out += [('amortization',dict(model='repair',K=K,w=52,m=20,horizon=T),METHODS[:2])
            for K in (63,255) for T in (1,4,16,64,100,400)]
    return out


def warmup():
    spec=dict(model='repair',K=7,w=12,m=3,horizon=2)
    for method in METHODS:
        for audit in (False,True):
            execute(spec,9600001,method,audit)


def validate(result, spec):
    hist,readout=result[:2]
    assert hist.shape==(spec['horizon'],spec['K']+1)
    assert np.all(hist>=0) and np.all(hist.sum(axis=1)==1<<spec['m'])
    assert np.array_equal(readout[:,0],hist@np.arange(spec['K']+1,dtype=np.int64))
    assert np.array_equal(readout[:,1],hist[:,(spec['K']+1)//2:].sum(axis=1))
