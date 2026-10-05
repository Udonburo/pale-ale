"""Matched fixed-template and incremental inverse-equation executors."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from numba import njit
import kernel as base
import counter_reference as cr

# Work counters (only collected in untimed replay):
# 0 queries; 1 dependent tests; 2 new relations; 3 source-row XORs for reuse;
# 4 realized C U rows; 5 C U row XORs; 6 rebuilt equation rows;
# 7 Gaussian reductions; 8 rank blocks; 9 independent runs; 10 independent bits.
WORK_NAMES = ('queries', 'dependent_tests', 'new_relations', 'reuse_row_xors',
              'CU_rows', 'CU_row_xors', 'rebuilt_rows', 'elimination_reductions',
              'rank_blocks', 'independent_runs', 'independent_bits')


@njit(cache=True)
def profile(C, m):
    w = len(C)
    ranks = np.zeros((m+1, w), dtype=np.int64)
    jumps = np.full((m+1, w), w, dtype=np.int64)
    for k in range(m+1):
        basis = np.zeros(k, dtype=np.int64)
        rank = 0
        for j in range(w):
            value = C[j] & ((1 << k)-1)
            while value:
                p = cr.msb(value)
                if basis[p]:
                    value ^= basis[p]
                else:
                    basis[p] = value
                    rank += 1
                    break
            ranks[k, j] = rank
        upcoming = w
        for j in range(w-1, -1, -1):
            previous = ranks[k, j-1] if j else 0
            if ranks[k, j] == previous:
                upcoming = j
            jumps[k, j] = upcoming
    return ranks, jumps


@njit(cache=True)
def xor_rows(mask, rows):
    value, visits = 0, 0
    while mask:
        p = cr.msb(mask)
        value ^= rows[len(rows)-1-p]
        mask ^= 1 << p
        visits += 1
    return value, visits


@njit(cache=True)
def ensure_reuse(k, j, deps, heads, U, V, c, cv, cu, cc, work, by_k, audit):
    if cv[k, j] != -1:
        return
    v, nv = xor_rows(deps[k, j], V)
    u, nu = xor_rows(heads[k, j], U)
    cv[k, j], cu[k, j] = v, u
    cc[k, j] = cr.parity(deps[k, j] & c)
    if audit:
        work[2] += 1
        work[3] += nv+nu
        by_k[2, k] += 1


@njit(cache=True)
def ensure_rebuild(k, j, C, U, V, c, CUr, processed,
                   bt, bv, bu, bc, cv, cu, cc, work, by_k, audit):
    """Eliminate only the required prefix of the realized inverse equations.

    C U rows are shared over all suffix widths; pivots and dependent equations
    are shared over all queries of one width. The common static rank profile
    allows independent-run batching without requiring unused equation rows.
    """
    if cv[k, j] != -1:
        return
    w = len(C)
    for i in range(processed[k], j+1):
        if CUr[i] == -1:
            value, visits = xor_rows(C[i], U)
            CUr[i] = value
            if audit:
                work[4] += 1
                work[5] += visits
        h = CUr[i]
        v = h & ((1 << k)-1)
        e, g = V[i], (c >> (w-1-i)) & 1
        if audit:
            work[6] += 1
        while v:
            p = cr.msb(v)
            if bt[k, p]:
                v ^= bt[k, p]
                e ^= bv[k, p]
                h ^= bu[k, p]
                g ^= bc[k, p]
                if audit:
                    work[7] += 1
            else:
                bt[k, p], bv[k, p], bu[k, p], bc[k, p] = v, e, h, g
                break
        if v == 0:
            cv[k, i], cu[k, i], cc[k, i] = e, h, g
            if audit:
                work[2] += 1
                by_k[2, k] += 1
    processed[k] = j+1


@njit(cache=True)
def block_below(start, k, threshold, mode, C, deps, heads, ranks, jumps,
                U, V, c, CUr, processed, bt, bv, bu, bc, cv, cu, cc,
                work, by_k, maxima, audit):
    w = len(C)
    if audit:
        work[0] += 1
        by_k[0, k] += 1
    if threshold == 0:
        return 0
    if threshold == 1 << w:
        return 1 << k
    count, j, tests = 0, 0, 0
    last = w-cr.msb(threshold & -threshold)
    while j < last:
        if jumps[k, j] != j:
            end = min(jumps[k, j], last)
            length = end-j
            segment = (threshold >> (w-end)) & ((1 << length)-1)
            count += segment << (k-ranks[k, end-1])
            if audit:
                work[9] += 1
                work[10] += length
            j = end
        else:
            if mode == 0:
                ensure_reuse(k,j,deps,heads,U,V,c,cv,cu,cc,work,by_k,audit)
            else:
                ensure_rebuild(k,j,C,U,V,c,CUr,processed,bt,bv,bu,bc,
                               cv,cu,cc,work,by_k,audit)
            failed = (cr.parity(cv[k,j] & threshold) ^
                      cr.parity(cu[k,j] & start) ^ cc[k,j])
            tests += 1
            if ((threshold >> (w-1-j)) & 1) and failed:
                count += 1 << (k-ranks[k,j])
            if failed:
                break
            j += 1
    if audit:
        work[1] += tests
        by_k[1,k] += tests
        maxima[k] = max(maxima[k], tests)
    return count


@njit(cache=True)
def workspaces(m, w, rebuild):
    cv = np.full((m+1,w), -1, dtype=np.int64)
    cu, cc = np.empty_like(cv), np.empty_like(cv)
    if rebuild:
        CUr = np.full(w, -1, dtype=np.int64)
        processed = np.zeros(m+1, dtype=np.int64)
        bt = np.zeros((m+1,m), dtype=np.int64)
        bv, bu, bc = np.empty_like(bt), np.empty_like(bt), np.empty_like(bt)
    else:
        CUr, processed = np.empty(0,dtype=np.int64), np.empty(0,dtype=np.int64)
        bt = np.empty((0,0),dtype=np.int64)
        bv,bu,bc = bt,bt,bt
    return CUr, processed, bt, bv, bu, bc, cv, cu, cc


@njit(cache=True)
def constraint_step(xs, ns, cuts, dest, mode, C, deps, heads, ranks, jumps,
                    U,V,c,work,by_k,maxima,audit):
    m, w = len(U), len(V)
    CUr,processed,bt,bv,bu,bc,cv,cu,cc = workspaces(m,w,mode == 1)
    counts = np.zeros(len(cuts), dtype=np.int64)
    end = 0
    for i in range(len(xs)):
        x, start = xs[i], end
        end += ns[i]
        while start < end:
            k = cr.msb(end-start)
            if start:
                k = min(k, cr.msb(start & -start))
            if audit:
                work[8] += 1
            previous = 0
            for q in range(cuts.shape[1]):
                current = block_below(start,k,cuts[x,q],mode,C,deps,heads,ranks,jumps,
                                      U,V,c,CUr,processed,bt,bv,bu,bc,cv,cu,cc,
                                      work,by_k,maxima,audit)
                counts[dest[x,q]] += current-previous
                previous = current
            counts[dest[x,cuts.shape[1]]] += (1 << k)-previous
            start += 1 << k
    return counts


@njit(cache=True)
def trajectory(C, cuts, dest, m, horizon, seed, method, audit):
    # 0 reuse; 1 rebuild; 2 rank streamer; 3 full-map basis.
    w, D = len(C), len(cuts)
    if method == 0:
        deps,heads,ranks,jumps = base.prepare(C,m)
    elif method == 1:
        ranks,jumps = profile(C,m)
        deps,heads = np.empty((0,0),dtype=np.int64),np.empty((0,0),dtype=np.int64)
    else:
        deps = np.empty((0,0),dtype=np.int64)
        heads,ranks,jumps = deps,deps,deps
    histories = np.empty((horizon,D),dtype=np.int64)
    readouts = np.empty((horizon,2),dtype=np.int64)
    activity = np.zeros((horizon,11),dtype=np.int64) if audit else np.empty((0,11),dtype=np.int64)
    widths = np.zeros((horizon,3,m+1),dtype=np.int64) if audit else np.empty((0,3,m+1),dtype=np.int64)
    max_tests = np.zeros((horizon,m+1),dtype=np.int64) if audit else np.empty((0,m+1),dtype=np.int64)
    occupied = np.zeros(horizon,dtype=np.int64) if audit else np.empty(0,dtype=np.int64)
    xs,ns = np.array([0],dtype=np.int64),np.array([1 << m],dtype=np.int64)
    for tick in range(horizon):
        U = np.full(m,-1,dtype=np.int64)
        V = np.full(w,-1,dtype=np.int64)
        shift = np.full(1,-1,dtype=np.int64)
        source_stats = np.zeros(11,dtype=np.int64)
        base.eager_source(U,V,shift,seed,tick,source_stats)
        if audit:
            work,by_k,maxima = activity[tick],widths[tick],max_tests[tick]
            occupied[tick] = len(xs)
        else:
            work = np.empty(0,dtype=np.int64)
            by_k = np.empty((0,0),dtype=np.int64)
            maxima = np.empty(0,dtype=np.int64)
        if method < 2:
            counts = constraint_step(xs,ns,cuts,dest,method,C,deps,heads,ranks,jumps,
                                     U,V,shift[0],work,by_k,maxima,audit)
        else:
            cols,b = base.affine_map(C,U,V,shift[0])
            if method == 2:
                counts = base.stream_update(xs,ns,cols,b,cuts,dest)
            else:
                counts = base.basis_update(xs,ns,cols,b,w,cuts,dest)
        xs,ns = base.compact(counts)
        histories[tick] = counts
        readouts[tick,0],readouts[tick,1] = base.readout(counts)
    return histories,readouts,activity,widths,max_tests,occupied


@njit(cache=True)
def batch(C,m,U,V,c,starts,ks,thresholds,mode):
    # Untimed mathematical/work audit with one shared cache per randomization.
    deps,heads,ranks,jumps = base.prepare(C,m)
    CUr,processed,bt,bv,bu,bc,cv,cu,cc = workspaces(m,len(C),mode == 1)
    work = np.zeros(11,dtype=np.int64)
    by_k = np.zeros((3,m+1),dtype=np.int64)
    maxima = np.zeros(m+1,dtype=np.int64)
    output = np.empty(len(starts),dtype=np.int64)
    for i in range(len(starts)):
        output[i] = block_below(starts[i],ks[i],thresholds[i],mode,C,deps,heads,ranks,jumps,
                               U,V,c,CUr,processed,bt,bv,bu,bc,cv,cu,cc,
                               work,by_k,maxima,True)
    return output,work,by_k,maxima,cv,cu,cc
