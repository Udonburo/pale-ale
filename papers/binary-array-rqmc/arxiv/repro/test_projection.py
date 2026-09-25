"""Independent finite-group, real-SciPy and actual stopped-executable checks."""
from collections import Counter
from itertools import product
import inspect
import json
from pathlib import Path
import sys
import unittest
import numpy as np
import scipy
from scipy.stats import qmc
from scipy.stats._sobol import _cscramble
from projection import (BITS,BinaryProjection,log2_size,signs_from_parameters,fit_parameters,
                        scipy_signs,array_kernel,legacy)

AUDIT={'scipy':scipy.__version__,'exact_groups':[],'scipy_vectors':0,'common_randomizations':0,'replays':[]}


def gf_inverse(matrix):
    m=len(matrix);a=np.column_stack((matrix.copy(),np.eye(m,dtype=np.uint8)))
    for col in range(m):
        pivot=next(i for i in range(col,m) if a[i,col])
        a[[col,pivot]]=a[[pivot,col]]
        for i in range(m):
            if i!=col and a[i,col]:a[i]^=a[col]
    return a[:,m:]


def capture_original(c,n,seed,noise_seed):
    source,start=inspect.getsourcelines(legacy.array_run)
    order_line=start+next(i for i,s in enumerate(source) if 'orders = ' in s)
    history=[];original_trace=sys.gettrace()
    def tracer(frame,event,arg):
        if frame.f_code is legacy.array_run.__code__:
            if (event=='line' and frame.f_lineno==order_line) or event=='return':
                history.append((frame.f_locals['x'].copy(),frame.f_locals['alive'].copy()))
            return tracer
        return None
    try:
        sys.settrace(tracer)
        result=legacy.array_run(c,n,np.random.default_rng(seed),np.random.default_rng(noise_seed))
    finally:sys.settrace(original_trace)
    return result,history


class ProjectionChecks(unittest.TestCase):
    def test_source_convention_and_bounds(self):
        self.assertEqual(scipy.__version__,'1.15.3')
        s=qmc.Sobol(2,scramble=False,bits=30)
        np.testing.assert_array_equal(s._sv[0],2**np.arange(29,-1,-1,dtype=np.uint32))
        np.testing.assert_array_equal(s._sv[1]>>29,np.ones(30))
        for n in (0,-1,3,513,2**31,2.0,True):
            with self.assertRaises(ValueError):BinaryProjection(n)
        self.assertEqual(log2_size(2**30),30) # Validate limit without allocating 8 GiB.
        for m in range(8):
            n=1<<m
            for b in (0,1):
                for a in (range(1,n,2) if m else [0]):
                    expected=np.array([2*(b^((a&r).bit_count()%2))-1 for r in range(n)],dtype=float)
                    np.testing.assert_array_equal(signs_from_parameters(n,a,b),expected)
        for a,b in [(0,0),(2,1),(9,0),(1,2)]:
            with self.assertRaises(ValueError):signs_from_parameters(8,a,b)
        np.testing.assert_array_equal(signs_from_parameters(1,0,0),[-1.])
        np.testing.assert_array_equal(signs_from_parameters(1,0,1),[1.])

    def test_exact_joint_law_from_scramble_group(self):
        for m in range(1,6):
            n=1<<m
            qbits=((np.arange(n)[:,None]>>np.arange(m-1,-1,-1))&1).astype(np.uint8)
            weights=1<<np.arange(m-1,-1,-1)
            lower=[(i,j) for i in range(m) for j in range(i)]
            count=Counter()
            for entries in product((0,1),repeat=len(lower)):
                l=np.eye(m,dtype=np.uint8)
                for (i,j),bit in zip(lower,entries):l[i,j]=bit
                ranks=((qbits@l.T)%2)@weights
                raw_z=qbits.sum(axis=1)%2
                for shift in range(n):
                    for second_shift in (0,1):
                        z=np.zeros(n,dtype=np.uint8)
                        z[ranks^shift]=raw_z^second_shift
                        count[z.tobytes()]+=1
            wanted={((signs_from_parameters(n,a,b)+1)/2).astype(np.uint8).tobytes()
                    for a in range(1,n,2) for b in (0,1)}
            self.assertEqual(set(count),wanted)
            expected=2**(len(lower)+1)
            self.assertEqual(set(count.values()),{expected})
            AUDIT['exact_groups'].append(dict(m=m,group_size=2**len(lower),full_randomizations=sum(count.values()),
                distinct_vectors=len(count),multiplicity_per_vector=expected))

    def test_real_scipy_all_ranks_and_common_randomization(self):
        rng=np.random.default_rng(56003911)
        for m in (0,1,2,3,4,6,9,12,16,20):
            n=1<<m
            for _ in range(4 if m==20 else 16 if m==16 else 64):
                eps=scipy_signs(n,rng)
                a,b=fit_parameters(eps)
                self.assertTrue((n==1 and a==0) or a%2==1)
                np.testing.assert_array_equal(eps,signs_from_parameters(n,a,b))
                AUDIT['scipy_vectors']+=1
        # Inject the SAME independent LMS matrices/shifts into real SciPy and
        # derive a,b from those matrices, independently of the output vector.
        for m in (0,1,3,6,9,12):
            for _ in range(16):
                ltm=np.tril(rng.integers(0,2,(2,30,30),dtype=np.uint32))
                for d in range(2):np.fill_diagonal(ltm[d],1)
                shifts=rng.integers(0,2**30,2,dtype=np.uint32)
                class ControlledSobol(qmc.Sobol):
                    def _scramble(self):
                        self._shift=shifts.copy()
                        _cscramble(dim=2,bits=30,ltm=ltm.copy(),sv=self._sv)
                points=ControlledSobol(2,scramble=True,bits=30,seed=0).random_base2(m)
                epsilon=np.where(points[np.argsort(points[:,0]),1]>=.5,1.,-1.)
                c=(np.ones(m,dtype=np.uint8)@gf_inverse(ltm[0,:m,:m].astype(np.uint8)))%2 if m else []
                a=sum(int(v)<<(m-1-j) for j,v in enumerate(c))
                s=int(shifts[0])>>(30-m) if m else 0
                b=(int(shifts[1])>>29)^((a&s).bit_count()%2)
                np.testing.assert_array_equal(epsilon,signs_from_parameters(1<<m,a,b))
                AUDIT['common_randomizations']+=1

    def test_joint_dependency_and_buffer_reuse(self):
        support=[signs_from_parameters(8,a,b) for a in range(1,8,2) for b in (0,1)]
        self.assertEqual({int(np.prod(e[[0,2,4,6]])) for e in support},{1})
        wrong=np.array([-1,1,-1,1,-1,1,1,-1],dtype=float)
        with self.assertRaises(ValueError):fit_parameters(wrong)
        class Words:
            def __init__(self):self.i=0
            def integers(self,lo,hi):
                self.i+=1;return (self.i-1)%hi
        p=BinaryProjection(8);r=Words();samples=[p.draw(r).copy() for _ in range(8)]
        self.assertEqual({e.tobytes() for e in samples},{e.tobytes() for e in support})
        self.assertEqual(r.i,8)

    def test_actual_legacy_execution_replay_and_states(self):
        cases=[legacy.Condition('one_step',1,.9,.1,'call',2,'recent',0.,0.,21),
               legacy.Condition('mixed_stop',9,.91,2.4,'call',3,'recent',.03,.3,22),
               legacy.Condition('even_stop',17,.93,3.2,'even',2,'recent',0.,0.,23),
               legacy.Condition('all_absorbed',12,.9,.1,'even',2,'recent',0.,0.,24),
               legacy.Condition('ties_forever',11,.9,2.,'call',2,'recent',0.,0.,25,scales=(0.,0.,0.)),
               legacy.Condition('long',60,.963,5.1,'call',5,'distant',.037,.65,26)]
        for ci,c in enumerate(cases):
            for n in (1,8,512):
                seed=99710+ci*7+n;noise_seed=88319+n
                rng=np.random.default_rng(seed)
                signs=[scipy_signs(n,rng) for _ in range(c.horizon)]
                (old_y,old_signal,old_stats),old_history=capture_original(c,n,seed,noise_seed)
                new_y,new_signal,new_stats=array_kernel(c,n,np.random.default_rng(18),
                    np.random.default_rng(noise_seed),provider=lambda t:signs[t],record_trace=True)
                np.testing.assert_array_equal(old_y,new_y);np.testing.assert_array_equal(old_signal,new_signal)
                self.assertEqual(len(old_history),c.horizon+1)
                for old,new in zip(old_history,new_stats['history']):
                    np.testing.assert_array_equal(old[0],new[0]);np.testing.assert_array_equal(old[1],new[1])
                old_tau=np.sum([h[1] for h in old_history[:-1]],axis=0)
                new_tau=np.sum([h[1] for h in new_stats['history'][:-1]],axis=0)
                np.testing.assert_array_equal(old_tau,new_tau)
                self.assertEqual(old_stats['transitions'],new_stats['transitions'])
                if c.name=='all_absorbed':np.testing.assert_array_equal(new_tau,np.ones((3,n)))
                AUDIT['replays'].append(dict(condition=c.name,n=n,steps=c.horizon,output_equal=True,
                    all_states_equal=True,stopping_times_equal=True,transitions=old_stats['transitions']))



if __name__ == "__main__":
    unittest.main(verbosity=2)
