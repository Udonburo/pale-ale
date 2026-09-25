"""Direct binary projection and full-sort reference execution.

Ideal randomization law equality does not imply same-integer-seed equality
with SciPy's point generator. See Proposition 1 of the manuscript.
"""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
from time import perf_counter
import numpy as np
from numba import njit
from scipy.stats import qmc
import reference as legacy

BITS = 30

def log2_size(n):
    if isinstance(n,(bool,np.bool_)) or not isinstance(n,(int,np.integer)):
        raise ValueError('N must be an integer power of two')
    n=int(n)
    if n<1 or n>(1<<BITS) or n&(n-1):
        raise ValueError('N must be 2^m with 0 <= m <= 30')
    return n.bit_length()-1


@njit(cache=True)
def _fill_signs(out,mask,shift):
    for r in range(len(out)):
        x=np.uint32(r)&np.uint32(mask)
        x^=x>>np.uint32(16)
        x^=x>>np.uint32(8)
        x^=x>>np.uint32(4)
        bit=(np.uint32(0x6996)>>(x&np.uint32(15)))&np.uint32(1)
        out[r]=1. if (bit^np.uint32(shift)) else -1.


def signs_from_parameters(n,mask,shift,out=None):
    """Deterministic full rank sequence; validates the proven support."""
    log2_size(n)
    if not isinstance(mask,(int,np.integer)) or not isinstance(shift,(int,np.integer)):
        raise ValueError('Mask and shift must be integers')
    if shift not in (0,1) or mask<0 or mask>=n or (n>1 and mask%2!=1) or (n==1 and mask!=0):
        raise ValueError('Outside the odd-affine projection support')
    if out is None:out=np.empty(n,dtype=np.float64)
    if out.shape!=(n,) or out.dtype!=np.float64 or not out.flags.c_contiguous or not out.flags.writeable:
        raise ValueError('Output must be a writable contiguous float64 vector of length N')
    _fill_signs(out,int(mask),int(shift))
    return out


class BinaryProjection:
    """One uniform m-bit word selects (odd mask, independent fair shift).

    N=1 separately consumes a fair bit. The output buffer is reused; consumers
    must copy it if they retain old steps. All array updates finish before reuse.
    """
    def __init__(self,n):
        self.m=log2_size(n);self.n=int(n);self.out=np.empty(n,dtype=np.float64)
        self.last_parameters=None

    def draw(self,rng):
        word=int(rng.integers(0,max(self.n,2)))
        mask=0 if self.n==1 else 2*(word>>1)+1
        shift=word&1
        _fill_signs(self.out,mask,shift)
        self.last_parameters=(mask,shift)
        return self.out


def scipy_signs(n,rng):
    m=log2_size(n)
    points=qmc.Sobol(2,scramble=True,bits=BITS,seed=int(rng.integers(2**32))).random_base2(m)
    return np.where(points[np.argsort(points[:,0]),1]>=.5,1.,-1.)


def fit_parameters(epsilon):
    """Grader/replay extraction; never used by the direct production generator."""
    n=len(epsilon);m=log2_size(n)
    z=(np.asarray(epsilon)>0).astype(np.uint8)
    b=int(z[0]);a=sum(int(z[1<<j]^b)<<j for j in range(m))
    reconstructed=signs_from_parameters(n,a,b)
    if not np.array_equal(epsilon,reconstructed):raise ValueError('Non-affine rank sequence')
    return a,b


def array_kernel(c,n,rng,rng_noise,mode='BINARY',profile=False,record_trace=False,provider=None):
    """Original state ordering, arithmetic, absorption, payoff and readout noise.

    Only the source of rank-ordered epsilon changes. State sorting remains.
    profile/trace are audit modes; cost claims use unprofiled end-to-end execution.
    provider is an explicit fixed-sequence replay hook, excluded from production.
    """
    m=log2_size(n)
    if mode not in ('BINARY','SCIPY'):raise ValueError(mode)
    times={k:0. for k in ('setup','net_generate','point_sort','projection','state_sort','transition','payoff','noise')}
    start=perf_counter()
    generator=BinaryProjection(n) if mode=='BINARY' and provider is None else None
    x=np.zeros((3,n));alive=np.ones((3,n),dtype=bool);transitions=0
    if profile:times['setup']=perf_counter()-start
    history=[]
    for t in range(c.horizon):
        if record_trace:history.append((x.copy(),alive.copy()))
        if profile:start=perf_counter()
        orders=[np.lexsort((x[i],~alive[i])) for i in range(3)]
        if profile:times['state_sort']+=perf_counter()-start
        if provider is not None:
            epsilon=provider(t)
            if np.shape(epsilon)!=(n,) or not np.all((epsilon==-1)|(epsilon==1)):
                raise ValueError('Replay provider must return N signs')
        elif mode=='BINARY':
            if profile:start=perf_counter()
            epsilon=generator.draw(rng)
            if profile:times['projection']+=perf_counter()-start
        else:
            if profile:start=perf_counter()
            points=qmc.Sobol(2,scramble=True,bits=BITS,seed=int(rng.integers(2**32))).random_base2(m)
            if profile:times['net_generate']+=perf_counter()-start;start=perf_counter()
            point_order=np.argsort(points[:,0])
            if profile:times['point_sort']+=perf_counter()-start;start=perf_counter()
            epsilon=np.where(points[point_order,1]>=.5,1.,-1.)
            if profile:times['projection']+=perf_counter()-start
        if profile:start=perf_counter()
        for i in range(3):
            labels=orders[i]
            active=alive[i,labels]
            target=labels[active]
            transitions+=len(target)
            x[i,target]=c.rho*x[i,target]+c.drift+c.scales[i]*epsilon[active]
            alive[i,target]=np.abs(x[i,target])<c.barrier
        if profile:times['transition']+=perf_counter()-start
    if record_trace:history.append((x.copy(),alive.copy()))
    if profile:start=perf_counter()
    signals=(x*x if c.payoff=='even' else np.maximum(x-c.strike,0.)).T
    if profile:times['payoff']+=perf_counter()-start;start=perf_counter()
    y=signals+legacy.noise_draw(rng_noise,signals.shape)
    if profile:times['noise']+=perf_counter()-start
    stats=dict(transitions=transitions,generated_float_coordinates=2*n*c.horizon if mode=='SCIPY' else 0,
               generated_rank_signs=n*c.horizon,randomization_words=c.horizon,
               point_sorts=c.horizon if mode=='SCIPY' else 0,state_sorts=3*c.horizon,
               times=times)
    if record_trace:stats.update(history=history,x=x,alive=alive)
    return y,signals,stats
