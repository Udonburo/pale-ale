"""Check manuscript Algorithm 3 against enumeration and the measured kernel."""
from itertools import product
from pathlib import Path
import json
import random
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'confirmation/source/legacy'))
import counter_reference as measured
assert Path(measured.__file__).resolve()==HERE/'confirmation/source/legacy/counter_reference.py'


def primal_below(k,w,v,basis,pivots,tau):
    """Literal Algorithm 3; basis and offset preparation are not part of it."""
    rho=len(basis)
    assert 0 <= tau <= 1 << w and rho <= k
    if tau==0:return 0
    if tau==1 << w:return 1 << k
    a=0;mu=1 << (k-rho)
    for i,p in enumerate(pivots):
        above_v=v >> (p+1)
        above_tau=tau >> (p+1)
        if above_v < above_tau:return mu*(a+(1 << (rho-i)))
        if above_v > above_tau:return mu*a
        bit=(tau >> p) & 1
        if bit:a+=1 << (rho-i-1)
        if ((v >> p) & 1)!=bit:v^=basis[i]
    return mu*(a+int(v<tau))


def image(basis):
    values=[0]
    for vector in basis:
        values += [x ^ vector for x in values]
    assert len(values)==len(set(values))
    return values


def check_basis(w,basis,offsets,ks,all_thresholds):
    rho=len(basis)
    pivots=[e.bit_length()-1 for e in basis]
    assert pivots==sorted(set(pivots),reverse=True)
    bases=np.zeros((rho+1,rho),dtype=np.int64)
    packed_pivots=np.zeros_like(bases)
    bases[rho,:]=basis;packed_pivots[rho,:]=pivots
    values=image(basis)
    checks=0
    for v in offsets:
        outputs=[v ^ x for x in values]
        thresholds=(range((1 << w)+1) if all_thresholds else
                    sorted({0,1 << w,*outputs,*(x+1 for x in outputs)}))
        for tau in thresholds:
            expected=sum(x<tau for x in outputs)
            compiled=int(measured.coset_below(rho,v,tau,bases,packed_pivots))
            assert compiled==expected,(w,basis,v,tau,compiled,expected)
            for k in ks:
                got=primal_below(k,w,v,basis,pivots,tau)
                assert got==expected*(1 << (k-rho)),(w,k,basis,v,tau,got,expected)
                checks+=1
    return checks


def main():
    small_bases=small_queries=non_reduced=0
    # All echelon vector lists through width 4, including every choice of
    # lower bits, then every offset and threshold. k=rho+2 tests deficiency.
    for w in range(1,5):
        for mask in range(1 << w):
            pivots=[p for p in range(w-1,-1,-1) if (mask >> p) & 1]
            for tails in product(*(range(1 << p) for p in pivots)):
                basis=[(1 << p)|tail for p,tail in zip(pivots,tails)]
                non_reduced+=int(any((basis[i] >> p) & 1
                                    for i in range(len(basis)) for p in pivots[i+1:]))
                small_queries+=check_basis(w,basis,range(1 << w),(len(basis),len(basis)+2),True)
                small_bases+=1
    rng=random.Random(20261004)
    wide_queries=0
    for trial in range(64):
        w=(30,52,62)[trial%3]
        rho=trial%6
        pivots=sorted(rng.sample(range(w),rho),reverse=True)
        basis=[(1 << p)|rng.randrange(1 << p) for p in pivots]
        offsets=(0,(1 << w)-1,rng.randrange(1 << w))
        wide_queries+=check_basis(w,basis,offsets,(rho,62),False)
    report=dict(status='PASS',algorithm='Algorithm 3 / Proposition 3',
                exhaustive_word_widths=[1,2,3,4],exhaustive_echelon_bases=small_bases,
                non_reduced_bases=non_reduced,exhaustive_queries=small_queries,
                wider_cases=64,wider_queries=wide_queries,
                largest_rank_width=62,largest_word_width=62,
                comparisons=['explicit distinct-output enumeration','saved compiled coset_below','literal Algorithm 3'],
                performance_timings_repeated=False)
    (HERE/'primal_counter_checks.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':main()
