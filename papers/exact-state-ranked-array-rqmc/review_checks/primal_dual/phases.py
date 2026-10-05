"""Staged fixed-workload diagnostics, distinct from fused trajectory timings."""
from pathlib import Path
import json
import random
import statistics
import time
import numpy as np
from numba import njit
import executors as ex
import workload

METHODS = ('reuse','rebuild','basis_direct','primal_reuse')


@njit(cache=True)
def source(m,w,seed,tick):
    U,V = np.full(m,-1,dtype=np.int64),np.full(w,-1,dtype=np.int64)
    c,stats = np.full(1,-1,dtype=np.int64),np.zeros(11,dtype=np.int64)
    ex.base.eager_source(U,V,c,seed,tick,stats)
    return U,V,c[0]


@njit(cache=True)
def prime(mode,C,deps,heads,ranks,jumps,U,V,c,needed):
    m,w = len(U),len(V)
    spaces = ex.rk.workspaces(m,w,mode==1)
    CUr,processed,bt,bv,bu,bc,cv,cu,cc = spaces
    work,by_k = np.empty(0,dtype=np.int64),np.empty((0,0),dtype=np.int64)
    for k in range(m+1):
        remaining = needed[k]
        if not remaining:
            continue
        for j in range(w):
            if jumps[k,j] != j:
                continue
            if mode == 0:
                ex.rk.ensure_reuse(k,j,deps,heads,U,V,c,cv,cu,cc,work,by_k,False)
            else:
                ex.rk.ensure_rebuild(k,j,C,U,V,c,CUr,processed,bt,bv,bu,bc,cv,cu,cc,work,by_k,False)
            remaining -= 1
            if not remaining:
                break
    return spaces


@njit(cache=True)
def cached_queries(mode,C,deps,heads,ranks,jumps,U,V,c,spaces,starts,ks,thresholds):
    CUr,processed,bt,bv,bu,bc,cv,cu,cc = spaces
    work,by_k,maxima = np.empty(0,dtype=np.int64),np.empty((0,0),dtype=np.int64),np.empty(0,dtype=np.int64)
    answers = np.empty(len(starts),dtype=np.int64)
    for i in range(len(starts)):
        answers[i] = ex.rk.block_below(starts[i],ks[i],thresholds[i],mode,C,deps,heads,ranks,jumps,
                                      U,V,c,CUr,processed,bt,bv,bu,bc,cv,cu,cc,work,by_k,maxima,False)
    return answers


@njit(cache=True)
def basis_queries(cols,b,suffix,bases,pivots,starts,ks,thresholds,q):
    answers=np.empty(len(starts),dtype=np.int64)
    offset=0
    for i in range(len(starts)):
        if i % q == 0:
            offset,mask=b,starts[i]
            while mask:
                p=ex.cr.msb(mask)
                offset ^= cols[len(cols)-1-p]
                mask ^= 1 << p
        rho=suffix[len(cols)-ks[i]]
        answers[i]=ex.cr.coset_below(rho,offset,thresholds[i],bases,pivots) << (ks[i]-rho)
    return answers


@njit(cache=True)
def aggregate(answers,blocks,dest,weights):
    q=dest.shape[1]-1
    counts=np.zeros(len(dest),dtype=np.int64)
    for i in range(len(blocks)):
        _,k,x=blocks[i]
        previous=0
        for j in range(q):
            current=answers[q*i+j]
            counts[dest[x,j]]+=current-previous
            previous=current
        counts[dest[x,q]]+=(1 << k)-previous
    xs,ns=ex.base.compact(counts)
    readout=ex.weighted_readout(counts,weights)
    return counts,xs,ns,readout


def prepare(data,method):
    C,m=data['C'],len(data['U'])
    if method=='reuse':
        return ex.base.prepare(C,m)
    if method=='rebuild':
        ranks,jumps=ex.rk.profile(C,m)
        dummy=np.empty((0,0),dtype=np.int64)
        return dummy,dummy,ranks,jumps
    if method=='primal_reuse':
        return ex.prepare_primal(C,m)
    return ()


def construct(data,method,prepared,U,V,c,needed):
    if method in ('reuse','rebuild'):
        return prime(int(method=='rebuild'),data['C'],*prepared,U,V,c,needed)
    if method=='basis_direct':
        cols,b=ex.base.affine_map(data['C'],U,V,c)
        suffix,bases,pivots=ex.cr.basis_cache(cols,len(V))
    else:
        suffix,pivots,ids,B_rows,T_rows=prepared
        cols,b,bases=ex.transform_primal(U,V,c,pivots,ids,B_rows,T_rows)
    return cols,b,suffix,bases,pivots


def query(data,method,prepared,built,U,V,c):
    args=(data['starts'],data['ks'],data['thresholds'])
    if method in ('reuse','rebuild'):
        return cached_queries(int(method=='rebuild'),data['C'],*prepared,U,V,c,built,*args)
    return basis_queries(*built,*args,data['cuts'].shape[1])


def fused(data,method,prepared,U,V,c):
    if method in ('reuse','rebuild'):
        dummy=np.empty(0,dtype=np.int64)
        counts=ex.rk.constraint_step(data['xs'],data['ns'],data['cuts'],data['dest'],int(method=='rebuild'),
                                    data['C'],*prepared,U,V,c,dummy,np.empty((0,0),dtype=np.int64),dummy,False)
    else:
        built=construct(data,method,prepared,U,V,c,None)
        counts=ex.basis_direct_update(data['xs'],data['ns'],*built,data['cuts'],data['dest'])
    xs,ns=ex.base.compact(counts)
    return counts,xs,ns,ex.weighted_readout(counts,data['weights'])


def main():
    out=Path(__file__).resolve().parent/'workloads'
    records=[]
    preparations=[]
    snapshots=[]
    for i,spec in enumerate(workload.SPECS):
        with np.load(out/f'{i}.npz') as file:
            data={k:file[k] for k in file.files}
        snapshots.append(data)
        batch=ex.rk.batch(data['C'],len(data['U']),data['U'],data['V'],int(data['c'][0]),
                          data['starts'],data['ks'],data['thresholds'],0)
        needed=batch[3]
        expected=batch[0]
        common=aggregate(expected,data['blocks'],data['dest'],data['weights'])
        plans={method:prepare(data,method) for method in METHODS}
        # Compile all diagnostic variants before measuring.
        for method in METHODS:
            U,V,c=source(spec['m'],spec['w'],102000000,spec['tick'])
            built=construct(data,method,plans[method],U,V,c,needed)
            answers=query(data,method,plans[method],built,U,V,c)
            assert np.array_equal(answers,expected)
            result=fused(data,method,plans[method],U,V,c)
            assert np.array_equal(result[0],common[0])
        for round_no in range(21):
            order=list(METHODS)
            random.Random(105000000+100*i+round_no).shuffle(order)
            for method in order:
                # Batches suppress timer quantization; raw per-call times retained.
                for repeat in range(5):
                    begin=time.perf_counter()
                    U,V,c=source(spec['m'],spec['w'],102000000,spec['tick'])
                    a=time.perf_counter()
                    built=construct(data,method,plans[method],U,V,c,needed)
                    b=time.perf_counter()
                    answers=query(data,method,plans[method],built,U,V,c)
                    d=time.perf_counter()
                    result=aggregate(answers,data['blocks'],data['dest'],data['weights'])
                    e=time.perf_counter()
                    assert np.array_equal(result[0],common[0])
                    f=time.perf_counter()
                    Uf,Vf,cf=source(spec['m'],spec['w'],102000000,spec['tick'])
                    result_f=fused(data,method,plans[method],Uf,Vf,cf)
                    g=time.perf_counter()
                    assert np.array_equal(result_f[0],common[0])
                    records.append(dict(workload=i,method=method,round=round_no,repeat=repeat,
                                        source=a-begin,construction=b-a,query=d-b,aggregation=e-d,
                                        staged_total=e-begin,fused_total=g-f))
        for method in METHODS:
            timings=[]
            for batch_no in range(9):
                a=time.perf_counter()
                for _ in range(200):prepare(data,method)
                timings.append((time.perf_counter()-a)/200)
            preparations.append(dict(workload=i,method=method,seconds=timings,
                                     median_seconds=statistics.median(timings)))
        print('phase diagnostic complete for workload '+str(i),flush=True)
    amort=[]
    # Preselected repair m=20 and tandem m=20 snapshots. The query set stays fixed.
    for i in (1,4):
        data=snapshots[i];spec=workload.SPECS[i]
        for T in (1,4,16,64,256):
            for round_no in range(9):
                order=list(METHODS)
                random.Random(105100000+i*1000+T*10+round_no).shuffle(order)
                for method in order:
                    begin=time.perf_counter()
                    plan=prepare(data,method)
                    for _ in range(T):
                        U,V,c=source(spec['m'],spec['w'],102000000,spec['tick'])
                        result=fused(data,method,plan,U,V,c)
                    elapsed=time.perf_counter()-begin
                    amort.append(dict(workload=i,repeats=T,round=round_no,method=method,
                                      seconds=elapsed,seconds_per_step=elapsed/T))
    summaries=[]
    for i in range(len(workload.SPECS)):
        for method in METHODS:
            rows=[r for r in records if r['workload']==i and r['method']==method]
            summaries.append(dict(workload=i,method=method,
                                  **{k:statistics.median(r[k] for r in rows)
                                     for k in ('source','construction','query','aggregation','staged_total','fused_total')}))
    report=dict(status='PASS',summaries=summaries,records=records,preparations=preparations,
                amortization=amort,
                caveat='Staged diagnostic with preidentified demanded relations, materialized query/flow arrays and Python dispatch. The oracle demand pass is untimed. These are not additive phases of the production fused trajectory; fused totals are measured separately. Marginal phase medians need not sum to the total median.')
    (out/'phases.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
