"""Exact independent enumeration and shared-work checks for the successor."""
from fractions import Fraction
from itertools import product
from pathlib import Path
import argparse
import json
import random
import numpy as np
import reuse_kernel as rk
import run as previous
import model


def triangles(n):
    for values in product(*(range(1 << i) for i in range(n))):
        yield np.asarray([(1 << (n-1-i)) | (v << (n-i))
                          for i,v in enumerate(values)],dtype=np.int64)


def triangle(n, rng):
    return np.asarray([(1 << (n-1-i)) | (rng.randrange(1 << i) << (n-i))
                       for i in range(n)],dtype=np.int64)


def direct_words(C,U,V,c):
    """Forward substitution in plain Python; no dependency preparation/counting."""
    m,w=len(U),len(V)
    out=[]
    for r in range(1 << m):
        ur=sum(((int(row)&r).bit_count()%2) << (m-1-i)
               for i,row in enumerate(U))
        rhs=sum(((int(row)&ur).bit_count()%2) << (w-1-i)
                for i,row in enumerate(C)) ^ int(c)
        y=0
        for i,row in enumerate(V):
            bit=((rhs >> (w-1-i))&1) ^ ((int(row)&y).bit_count()%2)
            y |= bit << (w-1-i)
        out.append(y)
    return out


def all_queries(m,w):
    values=[(start,k,t) for k in range(m+1)
            for start in range(0,1 << m,1 << k) for t in range((1 << w)+1)]
    return [np.asarray([row[j] for row in values],dtype=np.int64) for j in range(3)]


def verify_batch(C,m,U,V,c,queries):
    a=rk.batch(C,m,U,V,c,*queries,0)
    b=rk.batch(C,m,U,V,c,*queries,1)
    words=direct_words(C,U,V,c)
    expected=np.asarray([sum(y<int(t) for y in words[int(s):int(s)+(1 << int(k))])
                         for s,k,t in zip(*queries)],dtype=np.int64)
    assert np.array_equal(a[0],expected)
    assert np.array_equal(b[0],expected)
    assert np.array_equal(a[2],b[2])
    assert np.array_equal(a[3],a[2][2])
    assert np.array_equal(b[3],b[2][2])
    seen=a[4]!=-1
    assert np.array_equal(seen,b[4]!=-1)
    for j in (4,5,6):
        assert np.array_equal(a[j][seen],b[j][seen])
    assert int(a[1][3]) <= (m+1)*int(a[1][2])
    return len(expected)


def checks():
    rng=random.Random(9620301)
    batches=queries=0
    # Full matrices, triangular transforms and shifts in these small dimensions.
    for m,w in ((1,1),(1,2),(2,1),(2,2),(2,3)):
        qq=all_queries(m,w)
        for values in product(range(1 << m),repeat=w):
            C=np.asarray(values,dtype=np.int64)
            p=rk.profile(C,m)
            dep,head,rank,jump=rk.base.prepare(C,m)
            assert np.array_equal(p[0],rank) and np.array_equal(p[1],jump)
            for U in triangles(m):
                for V in triangles(w):
                    for c in range(1 << w):
                        queries += verify_batch(C,m,U,V,c,qq)
                        batches += 1
    # Includes non-monotone query order and rank-deficient maps beyond enumeration.
    for case in range(80):
        m,w=rng.randrange(1,9),rng.randrange(1,13)
        C=np.asarray([rng.randrange(1 << m) for _ in range(w)],dtype=np.int64)
        if case%5==0:C[:]=0
        U,V=triangle(m,rng),triangle(w,rng)
        triples=[]
        for _ in range(50):
            k=rng.randrange(m+1)
            triples.append((rng.randrange(1 << (m-k)) << k,k,rng.randrange((1 << w)+1)))
        qq=[np.asarray([x[j] for x in triples],dtype=np.int64) for j in range(3)]
        queries+=verify_batch(C,m,U,V,rng.randrange(1 << w),qq);batches+=1

    sharp=[]
    for w in (4,7,10):
        m=3
        C=np.asarray([rng.randrange(1 << m) for _ in range(w)],dtype=np.int64)
        U,V=triangle(m,rng),triangle(w,rng)
        for q in range(w):
            Q=1 << q
            starts=np.zeros(Q,dtype=np.int64)
            ks=np.zeros(Q,dtype=np.int64)
            ts=np.asarray([(j << (w-q)) | 1 for j in range(Q)],dtype=np.int64)
            total=0
            dist=[0]*(w+1)
            for c in range(1 << w):
                a=rk.batch(C,m,U,V,c,starts,ks,ts,0)
                J=int(a[2][2,0]);total+=J;dist[J]+=1
                assert J==int(a[3][0])
            mean=Fraction(total,1 << w)
            bound=sum((min(Fraction(1),Fraction(Q,1 << (s-1)))
                       for s in range(1,w+1)),Fraction(0))
            predicted=Fraction(q+2)-Fraction(1,1 << (w-q-1))
            assert mean==bound==predicted
            sharp.append(dict(w=w,m=m,Q=Q,q=q,mean=str(mean),finite_bound=str(bound),
                              coarse_bound=min(w,q+2),distribution=dist))

    # A predictable batch is essential: using the fresh answer selects a long prefix.
    w,m=7,2
    C=np.zeros(w,dtype=np.int64)
    U=next(triangles(m));V=next(triangles(w))
    adaptive=[]
    for c in range(1 << w):
        qq=[np.asarray([v],dtype=np.int64) for v in (0,0,c | 1)]
        a=rk.batch(C,m,U,V,c,*qq,0)
        adaptive.append(int(a[2][2,0]))
        assert a[0][0]==int(c<(c | 1))
    assert min(adaptive)==max(adaptive)==w

    trajectory_count=steps=0
    for kind in ('repair','dyadic'):
        for m,w,K in ((3,12,7),(8,30,31),(12,52,63),(12,52,255)):
            spec=dict(model=kind,m=m,w=w,K=K,horizon=12)
            C=previous.base_rows(m,w);cuts,dest=previous.problem(spec)
            for seed in (9602001,9602002):
                outputs=[]
                for method in range(4):
                    value=rk.trajectory(C,cuts,dest,m,12,seed,method,True)
                    assert np.all(value[0]>=0) and np.all(value[0].sum(axis=1)==1 << m)
                    outputs.append(value)
                    plain=rk.trajectory(C,cuts,dest,m,12,seed,method,False)
                    assert np.array_equal(value[0],plain[0]) and np.array_equal(value[1],plain[1])
                    trajectory_count+=1;steps+=12
                for value in outputs[1:]:
                    assert np.array_equal(value[0],outputs[0][0])
                    assert np.array_equal(value[1],outputs[0][1])
                assert np.array_equal(outputs[0][3],outputs[1][3])
                assert np.array_equal(outputs[0][3][:,2,:],outputs[0][4])
    return dict(passed=True,exact_batches=batches,exact_queries=queries,
                sharp_shift_cases=sharp,adaptive_counterexample=dict(w=7,
                    Q=1,mean_J=7,incorrect_unconditional_bound=2),
                trajectories=trajectory_count,trajectory_steps=steps,environment=model.environment())


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=checks()
    if args.output:args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('sharp_shift_cases','environment')}),flush=True)
