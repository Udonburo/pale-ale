"""Fully costed trajectories under the declared inverse-LMS tape."""
import time
import numpy as np
from scipy.stats import qmc
import kernel as k
import model


METHODS=('rank_stream','eager_basis','eager_constraints','lazy_constraints')
STAT_NAMES=('U_rng_rows','V_rng_rows','shift_rng_words','transformed_constraints',
            'block_cdf_queries','cdf_bit_steps','independent_equations',
            'deterministic_cdf_queries','max_V_row_requested','rank_blocks','independent_runs')


def base_rows(m,w):
    engine=qmc.Sobol(d=2,scramble=False,bits=max(m,w))
    columns,b=model.rank_map(engine,m,w)
    assert b==0
    return np.asarray([sum(((int(g)>>(w-1-i))&1)<<(m-1-j)
                            for j,g in enumerate(columns)) for i in range(w)],dtype=np.int64)


def problem(spec):
    K,w=spec['K'],spec['w']
    if spec['model']=='repair':
        cuts,dest,_=model.model(K,w)
    elif spec['model']=='dyadic':
        assert w>=4
        cuts=np.asarray([[(1+x%5)<<(w-4),(2+x%5+x%3)<<(w-4)]
                         for x in range(K+1)],dtype=np.int64)
        dest=np.asarray([[min(x+1,K),max(x-1,0),x] for x in range(K+1)],dtype=np.int64)
    else:
        raise ValueError(spec['model'])
    return cuts,dest


def workspace(m,w,constraints=True):
    return (np.full(m,-1,dtype=np.int64),np.full(w,-1,dtype=np.int64),
            np.full(1,-1,dtype=np.int64),
            np.full((m+1,w),-1,dtype=np.int64) if constraints else None,
            np.empty((m+1,w),dtype=np.int64) if constraints else None,
            np.empty((m+1,w),dtype=np.int64) if constraints else None,
            np.zeros(11,dtype=np.int64))


def trajectory(spec,seed,method):
    if method not in METHODS:
        raise ValueError(method)
    m,w,T,K=spec['m'],spec['w'],spec['horizon'],spec['K']
    if not (1<=m<=30 and m<=w<=52 and T*(1<<m)*K<2**63):
        raise ValueError('trajectory outside tested packed domain')
    beginning=time.perf_counter()
    at=beginning
    C=base_rows(m,w)
    cuts,dest=problem(spec)
    xs,ns=np.array([0],dtype=np.int64),np.array([1<<m],dtype=np.int64)
    phase={'setup':time.perf_counter()-at,'template':0.,'source_and_map':0.,
           'update_including_deferred_work':0.,'readout':0.}
    at=time.perf_counter()
    constraints=method.endswith('constraints')
    deps,heads,ranks,jumps=k.prepare(C,m) if constraints else (None,None,None,None)
    phase['template']=time.perf_counter()-at
    histories,traces,activity=[],[],[]
    audit=0.
    for tick in range(T):
        at=time.perf_counter()
        U,V,shift,cv,cu,cc,stats=workspace(m,w,constraints)
        if method!='lazy_constraints':
            k.eager_source(U,V,shift,seed,tick,stats)
        if not constraints:
            cols,b=k.affine_map(C,U,V,int(shift[0]))
        phase['source_and_map']+=time.perf_counter()-at
        at=time.perf_counter()
        if method=='rank_stream':
            counts=k.stream_update(xs,ns,cols,b,cuts,dest)
        elif method=='eager_basis':
            counts=k.basis_update(xs,ns,cols,b,w,cuts,dest)
        else:
            counts=k.update(xs,ns,cuts,dest,deps,heads,ranks,jumps,U,V,shift,cv,cu,cc,seed,tick,stats)
        xs,ns=k.compact(counts)
        phase['update_including_deferred_work']+=time.perf_counter()-at
        at=time.perf_counter()
        first,tail=k.readout(counts)
        phase['readout']+=time.perf_counter()-at
        at=time.perf_counter()
        assert np.all(counts>=0) and int(sum(ns))==1<<m
        histories.append([[int(x),int(n)] for x,n in zip(xs,ns)])
        traces.append([int(first),int(tail)])
        activity.append([int(v) for v in stats])
        audit+=time.perf_counter()-at
    seconds=time.perf_counter()-beginning-audit
    return dict(seconds=seconds,phases=phase,histories=histories,traces=traces,
                activity=activity,activity_names=STAT_NAMES,
                numerator=sum(t[0] for t in traces),tail_numerator=sum(t[1] for t in traces),
                max_states=max(len(h) for h in histories))


def cells():
    result=[dict(model='repair',K=63,w=w,m=m,horizon=100)
            for w in (30,52) for m in (12,20)]
    result.append(dict(model='repair',K=255,w=52,m=20,horizon=100))
    result.extend(dict(model='dyadic',K=31,w=52,m=m,horizon=100) for m in (12,20))
    return result


def warmup():
    for name in ('repair','dyadic'):
        for method in METHODS:
            trajectory(dict(model=name,K=7,m=3,w=12,horizon=2),123,method)
