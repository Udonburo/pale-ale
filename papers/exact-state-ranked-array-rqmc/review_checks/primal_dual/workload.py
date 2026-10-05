"""Fixed incoming states and query sets: exact prediction and resampled shifts."""
from pathlib import Path
import json
import math
from fractions import Fraction
import numpy as np
from numba import njit
import executors as ex

HERE = Path(__file__).resolve().parent
SPECS = [dict(model='repair', K=63, w=52, m=m, horizon=100, tick=50) for m in (12,20)]
SPECS += [dict(model='repair', K=255, w=52, m=20, horizon=100, tick=50)]
SPECS += [dict(model='tandem', w=52, m=m, horizon=200, tick=100) for m in (12,20)]


def queries(xs, ns, cuts):
    starts, widths, thresholds, states, branches = [], [], [], [], []
    end, blocks = 0, []
    for x, n in zip(xs, ns):
        start = end
        end += int(n)
        while start < end:
            k = (end-start).bit_length()-1
            if start:
                k = min(k, (start & -start).bit_length()-1)
            blocks.append((start, k, int(x)))
            for j, tau in enumerate(cuts[x]):
                starts.append(start); widths.append(k); thresholds.append(int(tau))
                states.append(int(x)); branches.append(j)
            start += 1 << k
    return tuple(np.asarray(v, dtype=np.int64) for v in (starts,widths,thresholds,states,branches)), np.asarray(blocks,dtype=np.int64)


def extract(spec, seed=102000000):
    spec = dict(spec)
    tick = spec.pop('tick')
    _, _, result = ex.execute(spec, seed, 'reuse', False)
    hist = result[0][tick-1]
    xs = np.flatnonzero(hist).astype(np.int64)
    ns = hist[xs]
    C = ex.original.base_rows(spec['m'], spec['w'])
    cuts, dest, weights = ex.problem(spec)
    U = np.full(spec['m'], -1, dtype=np.int64)
    V = np.full(spec['w'], -1, dtype=np.int64)
    c, stats = np.full(1,-1,dtype=np.int64), np.zeros(11,dtype=np.int64)
    ex.base.eager_source(U,V,c,seed,tick,stats)
    query, blocks = queries(xs,ns,cuts)
    return dict(C=C,U=U,V=V,c=c,xs=xs,ns=ns,cuts=cuts,dest=dest,weights=weights,
                starts=query[0],ks=query[1],thresholds=query[2],states=query[3],branches=query[4],blocks=blocks)


def prediction(C,U,V,starts,ks,thresholds):
    m,w = len(U),len(V)
    deps,heads,ranks,jumps = ex.base.prepare(C,m)
    result = []
    for k in range(m+1):
        indices = np.flatnonzero(ks == k)
        positions = np.flatnonzero(deps[k])
        d = len(positions)
        sets = [set() for _ in range(d)]
        v = [int(ex.rk.xor_rows(deps[k,j],V)[0]) for j in positions]
        u = [int(ex.rk.xor_rows(heads[k,j],U)[0]) for j in positions]
        weights = [int(deps[k,j]).bit_count()+int(heads[k,j]).bit_count() for j in positions]
        for idx in indices:
            tau,start = int(thresholds[idx]),int(starts[idx])
            if tau in (0,1 << w):
                continue
            last = w-(tau & -tau).bit_length()+1
            prefix = 0
            for s,j in enumerate(positions):
                if j >= last:
                    break
                sets[s].add(prefix)
                bit = ((v[s] & tau).bit_count() ^ (u[s] & start).bit_count()) & 1
                prefix = (prefix << 1) | bit
        sizes = [len(v) for v in sets]
        probabilities = [Fraction(n, 1 << s) for s,n in enumerate(sizes)]
        mean = sum(probabilities, Fraction(0))
        second = sum(((2*s+1)*v for s,v in enumerate(probabilities)), Fraction(0))
        row_xors = sum((a*b for a,b in zip(weights,probabilities)), Fraction(0))
        psi = sum((min(Fraction(1),Fraction(len(indices),1 << s)) for s in range(d)), Fraction(0))
        exact = dict(mean=mean,variance=second-mean*mean,row_xors=row_xors,psi=psi)
        assert exact['variance'] >= 0
        result.append(dict(k=k, queries=len(indices), dependencies=d, distinct_prefixes=sizes,
                           probabilities=[float(v) for v in probabilities],
                           source_row_weights=weights,
                           exact_fractions={name:[value.numerator,value.denominator]
                                            for name,value in exact.items()},
                           exact_mean=float(mean),exact_variance=float(exact['variance']),
                           exact_row_xors=float(row_xors),psi=float(psi)))
    return result


@njit(cache=True)
def sample_shifts(C,U,V,starts,ks,thresholds,shifts):
    m,w = len(U),len(V)
    deps,heads,ranks,jumps = ex.base.prepare(C,m)
    J = np.zeros((len(shifts),m+1),dtype=np.int64)
    row_xors = np.zeros(len(shifts),dtype=np.int64)
    for rep in range(len(shifts)):
        CUr,processed,bt,bv,bu,bc,cv,cu,cc = ex.rk.workspaces(m,w,False)
        work,by_k = np.zeros(11,dtype=np.int64),np.zeros((3,m+1),dtype=np.int64)
        for i in range(len(starts)):
            ex.rk.block_below(starts[i],ks[i],thresholds[i],0,C,deps,heads,ranks,jumps,
                             U,V,shifts[rep],CUr,processed,bt,bv,bu,bc,cv,cu,cc,
                             work,by_k,J[rep],True)
        assert np.array_equal(J[rep],by_k[2])
        row_xors[rep] = work[3]
    return J,row_xors


def main():
    out = HERE/'workloads'
    out.mkdir(exist_ok=True)
    summaries = []
    for i,spec in enumerate(SPECS):
        data = extract(spec)
        shifts = np.random.default_rng(104000000+i).integers(0,1 << spec['w'],4096,dtype=np.int64)
        pred = prediction(data['C'],data['U'],data['V'],data['starts'],data['ks'],data['thresholds'])
        J,xors = sample_shifts(data['C'],data['U'],data['V'],data['starts'],data['ks'],data['thresholds'],shifts)
        for item in pred:
            k = item['k']
            item['sample_mean'] = float(J[:,k].mean())
            item['sample_sd'] = float(J[:,k].std(ddof=1))
            item['exact_standard_error'] = math.sqrt(item['exact_variance']/len(shifts))
        # At the unaltered tape, both forward-basis methods give the same update.
        deps,heads,ranks,jumps = ex.base.prepare(data['C'],spec['m'])
        cols,b = ex.base.affine_map(data['C'],data['U'],data['V'],int(data['c'][0]))
        suffix,bases,pivots = ex.cr.basis_cache(cols,spec['w'])
        counted = ex.basis_direct_update(data['xs'],data['ns'],cols,b,suffix,bases,pivots,data['cuts'],data['dest'])
        work,by_k,maxima = np.zeros(11,dtype=np.int64),np.zeros((3,spec['m']+1),dtype=np.int64),np.zeros(spec['m']+1,dtype=np.int64)
        dual = ex.rk.constraint_step(data['xs'],data['ns'],data['cuts'],data['dest'],0,data['C'],deps,heads,ranks,jumps,
                                    data['U'],data['V'],int(data['c'][0]),work,by_k,maxima,True)
        assert np.array_equal(dual,counted)
        np.savez_compressed(out/f'{i}.npz',**data,shifts=shifts,J=J,row_xors=xors)
        summary = dict(index=i,spec=spec,seed=102000000,shift_seed=104000000+i,shifts=len(shifts),
                       occupied=len(data['xs']),blocks=len(data['blocks']),queries=len(data['starts']),
                       widths=pred,exact_total_mean=float(sum((Fraction(*p['exact_fractions']['mean']) for p in pred), Fraction(0))),
                       sampled_total_mean=float(J.sum(axis=1).mean()),
                       sampled_total_se=float(J.sum(axis=1).std(ddof=1)/math.sqrt(len(shifts))),
                       psi_total=float(sum((Fraction(*p['exact_fractions']['psi']) for p in pred), Fraction(0))),
                       exact_row_xors=float(sum((Fraction(*p['exact_fractions']['row_xors']) for p in pred), Fraction(0))),sample_row_xors=float(xors.mean()))
        summaries.append(summary)
        print(json.dumps({k:summary[k] for k in ('index','occupied','queries','exact_total_mean','sampled_total_mean','psi_total')}),flush=True)
    # Same Q=16 and same k=0, w=10; repeated prefix vs maximal prefix diversity.
    synthetic=[]
    C=np.zeros(10,dtype=np.int64);U=np.array([1],dtype=np.int64)
    V=np.array([1 << (9-j) for j in range(10)],dtype=np.int64)
    starts=np.zeros(16,dtype=np.int64);ks=starts.copy()
    for name,thresholds in [('repeated',np.ones(16,dtype=np.int64)),
                            ('diverse',np.arange(16,dtype=np.int64)*64+1)]:
        pred=prediction(C,U,V,starts,ks,thresholds)
        J,xors=sample_shifts(C,U,V,starts,ks,thresholds,np.arange(1024,dtype=np.int64))
        assert float(J[:,0].mean())==pred[0]['exact_mean']
        synthetic.append(dict(name=name,Q=16,exact_mean=pred[0]['exact_mean'],
                              enumerated_mean=float(J[:,0].mean()),psi=pred[0]['psi']))
    report=dict(status='PASS',workloads=summaries,synthetic=synthetic,
                interpretation='Conditional on fixed incoming states, U,V and queries. Shift replicas are not new trajectory-performance replications.')
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
