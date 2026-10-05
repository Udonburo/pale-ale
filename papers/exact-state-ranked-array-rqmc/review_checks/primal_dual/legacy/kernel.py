"""Exact inverse-constraint execution on a shared, counter-addressed tape."""
import numpy as np
from numba import njit
import counter_reference as cr


@njit(cache=True)
def philox4x32(c0, c1, c2, c3, k0, k1):
    mask = np.uint64(0xffffffff)
    c0, c1, c2, c3 = np.uint64(c0), np.uint64(c1), np.uint64(c2), np.uint64(c3)
    k0, k1 = np.uint64(k0), np.uint64(k1)
    for _ in range(10):
        p0 = np.uint64(0xd2511f53)*c0
        p1 = np.uint64(0xcd9e8d57)*c2
        c0, c1, c2, c3 = ((p1 >> np.uint64(32)) ^ c1 ^ k0,
                          p1 & mask, (p0 >> np.uint64(32)) ^ c3 ^ k1, p0 & mask)
        k0 = (k0 + np.uint64(0x9e3779b9)) & mask
        k1 = (k1 + np.uint64(0xbb67ae85)) & mask
    return c0, c1, c2, c3


@njit(cache=True)
def word(seed, tick, domain, index, bits):
    key = np.uint64(seed)
    a, b, _, _ = philox4x32(index, domain, tick, 0,
                            key & np.uint64(0xffffffff), key >> np.uint64(32))
    value = a | (b << np.uint64(32))
    return np.int64(value & np.uint64((1 << bits)-1))


@njit(cache=True)
def row(rows, i, domain, seed, tick, stats):
    if rows[i] == -1:
        n = len(rows)
        v = 1 << (n-1-i)
        if i:
            v |= word(seed, tick, domain, i, i) << (n-i)
            stats[domain-1] += 1
        rows[i] = v
    return rows[i]


@njit(cache=True)
def shift_value(shift, width, seed, tick, stats):
    if shift[0] == -1:
        shift[0] = word(seed, tick, 3, 0, width)
        stats[2] += 1
    return shift[0]


@njit(cache=True)
def eager_source(U, V, shift, seed, tick, stats):
    for i in range(len(U)):
        row(U, i, 1, seed, tick, stats)
    for i in range(len(V)):
        row(V, i, 2, seed, tick, stats)
    shift_value(shift, len(V), seed, tick, stats)


@njit(cache=True)
def prepare(C, m):
    """Fixed annihilator rows for each suffix of rank coordinates."""
    w = len(C)
    dependencies = np.zeros((m+1, w), dtype=np.int64)
    heads = np.zeros_like(dependencies)
    ranks = np.zeros_like(dependencies)
    for k in range(m+1):
        bs = np.zeros(k, dtype=np.int64)
        es = np.zeros(k, dtype=np.int64)
        hs = np.zeros(k, dtype=np.int64)
        rank = 0
        for j in range(w):
            v = C[j] & ((1 << k)-1)
            e, h = 1 << (w-1-j), C[j]
            for i in range(k-1, -1, -1):
                if (v >> i) & 1:
                    if bs[i]:
                        v ^= bs[i]
                        e ^= es[i]
                        h ^= hs[i]
                    else:
                        bs[i], es[i], hs[i] = v, e, h
                        rank += 1
                        break
            if v == 0:
                dependencies[k, j], heads[k, j] = e, h
            ranks[k, j] = rank
    jumps = np.full((m+1,w),w,dtype=np.int64)
    for k in range(m+1):
        upcoming=w
        for j in range(w-1,-1,-1):
            if dependencies[k,j]:upcoming=j
            jumps[k,j]=upcoming
    return dependencies, heads, ranks, jumps


@njit(cache=True)
def transform(mask, rows, domain, seed, tick, stats):
    value = 0
    while mask:
        bit = cr.msb(mask)
        i = len(rows)-1-bit
        value ^= row(rows, i, domain, seed, tick, stats)
        mask ^= 1 << bit
        if domain == 2:
            stats[8] = max(stats[8], i+1)
    return value


@njit(cache=True)
def inconsistent(k, j, start, threshold, deps, heads, U, V, shift,
                 cache_v, cache_u, cache_c, seed, tick, stats):
    if cache_v[k,j] == -1:
        d, h = deps[k,j], heads[k,j]
        cache_v[k,j] = transform(d, V, 2, seed, tick, stats)
        cache_u[k,j] = transform(h, U, 1, seed, tick, stats)
        cache_c[k,j] = cr.parity(d & shift_value(shift,len(V),seed,tick,stats))
        stats[3] += 1
    return (cr.parity(cache_v[k,j] & threshold) ^
            cr.parity(cache_u[k,j] & start) ^ cache_c[k,j])


@njit(cache=True)
def block_below(start, k, threshold, deps, heads, ranks, jumps, U, V, shift,
                cache_v, cache_u, cache_c, seed, tick, stats):
    w = len(V)
    stats[4] += 1
    if threshold == 0:
        stats[7] += 1
        return 0
    if threshold == 1 << w:
        stats[7] += 1
        return 1 << k
    count, deterministic, j = 0, True, 0
    last=w-cr.msb(threshold & -threshold)
    while j < last:
        if deps[k,j] == 0:
            end=min(jumps[k,j],last)
            length=end-j
            segment=(threshold >> (w-end)) & ((1 << length)-1)
            count += segment << (k-ranks[k,end-1])
            stats[5] += length
            stats[6] += length
            stats[10] += 1
            j=end
        else:
            stats[5] += 1
            bit = (threshold >> (w-1-j)) & 1
            deterministic = False
            failed = inconsistent(k,j,start,threshold,deps,heads,U,V,shift,
                                    cache_v,cache_u,cache_c,seed,tick,stats)
            if bit and failed:
                count += 1 << (k-ranks[k,j])
            if failed:
                break
            j+=1
    if deterministic:
        stats[7] += 1
    return count


@njit(cache=True)
def update(states, ns, cuts, dest, deps, heads, ranks, jumps, U, V, shift,
           cache_v, cache_u, cache_c, seed, tick, stats):
    counts = np.zeros(len(cuts), dtype=np.int64)
    end = 0
    for i in range(len(states)):
        x, start = states[i], end
        end += ns[i]
        while start < end:
            k = cr.msb(end-start)
            if start:
                lowbit = start & -start
                k = min(k, cr.msb(lowbit))
            stats[9] += 1
            previous = 0
            for q in range(cuts.shape[1]):
                current = block_below(start,k,cuts[x,q],deps,heads,ranks,jumps,U,V,shift,
                                      cache_v,cache_u,cache_c,seed,tick,stats)
                counts[dest[x,q]] += current-previous
                previous = current
            counts[dest[x,cuts.shape[1]]] += (1 << k)-previous
            start += 1 << k
    return counts


@njit(cache=True)
def affine_map(C, U, V, c):
    """Independent full forward solve; never uses prefix constraints."""
    m, w = len(U), len(V)
    rows = np.zeros(w,dtype=np.int64)
    b = 0
    for i in range(w):
        value = 0
        for j in range(m):
            if (C[i] >> (m-1-j)) & 1:
                value ^= U[j]
        for j in range(i):
            if (V[i] >> (w-1-j)) & 1:
                value ^= rows[j]
        rows[i] = value
        bi = ((c >> (w-1-i)) & 1) ^ cr.parity(V[i] & b)
        b |= bi << (w-1-i)
    cols = np.zeros(m,dtype=np.int64)
    for j in range(m):
        for i in range(w):
            cols[j] |= ((rows[i] >> (m-1-j)) & 1) << (w-1-i)
    return cols,b


@njit(cache=True)
def stream_update(states, ns, cols, b, cuts, dest):
    m = len(cols)
    delta = np.empty(m,dtype=np.int64)
    acc = 0
    for j in range(m):
        acc ^= cols[m-1-j]
        delta[j] = acc
    counts = np.zeros(len(cuts),dtype=np.int64)
    y, end = b, 0
    for i in range(len(states)):
        x, start = states[i], end
        end += ns[i]
        for r in range(start,end):
            if r:
                q, j = r, 0
                while not (q & 1):
                    q >>= 1
                    j += 1
                y ^= delta[j]
            branch = 0
            while branch < cuts.shape[1] and y >= cuts[x,branch]:
                branch += 1
            counts[dest[x,branch]] += 1
    return counts


@njit(cache=True)
def basis_update(states, ns, cols, b, width, cuts, dest):
    suffix,bases,pivots = cr.basis_cache(cols,width)
    dummy = np.zeros((1,1),dtype=np.int64)
    m = len(cols)
    ids0,v0,weights0 = np.empty(m+1,dtype=np.int64),np.empty(m+1,dtype=np.int64),np.empty(m+1,dtype=np.int64)
    ids1,v1,weights1 = np.empty(m+1,dtype=np.int64),np.empty(m+1,dtype=np.int64),np.empty(m+1,dtype=np.int64)
    counts = np.zeros(len(cuts),dtype=np.int64)
    before,end,used0 = 0,0,0
    for i in range(len(states)):
        x=states[i]
        end += ns[i]
        used1 = cr.prefix_terms(end,cols,b,suffix,False,ids1,v1,weights1)
        previous=0
        for j in range(cuts.shape[1]):
            t=cuts[x,j]
            hi=cr.terms_below(end,t,width,used1,ids1,v1,weights1,bases,pivots,dummy,dummy,False)
            lo=cr.terms_below(before,t,width,used0,ids0,v0,weights0,bases,pivots,dummy,dummy,False)
            counts[dest[x,j]] += hi-lo-previous
            previous=hi-lo
        counts[dest[x,cuts.shape[1]]] += ns[i]-previous
        ids0,ids1=ids1,ids0
        v0,v1=v1,v0
        weights0,weights1=weights1,weights0
        before,used0=end,used1
    return counts


@njit(cache=True)
def compact(counts):
    S=0
    for n in counts:
        S += n != 0
    xs,ns=np.empty(S,dtype=np.int64),np.empty(S,dtype=np.int64)
    j=0
    for x in range(len(counts)):
        if counts[x]:
            xs[j],ns[j]=x,counts[x]
            j+=1
    return xs,ns


@njit(cache=True)
def readout(counts):
    first,tail=0,0
    for x in range(len(counts)):
        first+=x*counts[x]
        if x >= len(counts)//2:
            tail+=counts[x]
    return first,tail
