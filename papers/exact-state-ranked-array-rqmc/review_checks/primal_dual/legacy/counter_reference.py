"""Packed-word and Python-int exact two-prefix counts; no alphabet expansion."""
from operator import index
import numpy as np
from numba import njit


def checked(columns, offset, width):
    cols = tuple(index(c) for c in columns)
    offset, width = index(offset), index(width)
    if not 0 <= len(cols) <= 62 or not 1 <= width <= 62:
        raise ValueError('compiled domain requires 0<=m<=62, 1<=w<=62')
    if not 0 <= offset < 1 << width or any(c < 0 or c >= 1 << width for c in cols):
        raise ValueError('word outside declared width')
    return np.asarray(cols, dtype=np.int64), offset


@njit(cache=True)
def msb(x):
    p = 0
    for s in (32, 16, 8, 4, 2, 1):
        if x >> s:
            x >>= s
            p += s
    return p


@njit(cache=True)
def parity(x):
    x ^= x >> 32
    x ^= x >> 16
    x ^= x >> 8
    x ^= x >> 4
    return (0x6996 >> (x & 15)) & 1


@njit(cache=True)
def basis_cache(cols, width):
    """Two passes: actual rho allocation, not an m-by-m deficient-map cache."""
    m = len(cols)
    scratch = np.zeros(min(m, width), dtype=np.int64)
    sp = np.zeros(len(scratch), dtype=np.int64)
    rho = 0
    for j in range(m-1, -1, -1):
        v = cols[j]
        for i in range(rho):
            if (v >> sp[i]) & 1:
                v ^= scratch[i]
        if v:
            p = msb(v)
            i = rho
            while i > 0 and sp[i-1] < p:
                scratch[i], sp[i] = scratch[i-1], sp[i-1]
                i -= 1
            scratch[i], sp[i] = v, p
            rho += 1
    bases = np.zeros((rho+1, rho), dtype=np.int64)
    pivots = np.zeros((rho+1, rho), dtype=np.int64)
    suffix = np.zeros(m+1, dtype=np.int64)
    k = 0
    for j in range(m-1, -1, -1):
        v = cols[j]
        for i in range(k):
            if (v >> pivots[k, i]) & 1:
                v ^= bases[k, i]
        if v:
            p = msb(v)
            loc = k
            for i in range(k):
                if pivots[k, i] < p:
                    loc = i
                    break
            for i in range(loc):
                bases[k+1, i], pivots[k+1, i] = bases[k, i], pivots[k, i]
            bases[k+1, loc], pivots[k+1, loc] = v, p
            for i in range(loc, k):
                bases[k+1, i+1], pivots[k+1, i+1] = bases[k, i], pivots[k, i]
            k += 1
        suffix[j] = k
    return suffix, bases, pivots


@njit(cache=True)
def prefix_terms(a, cols, b, suffix, coalesced, ids, offsets, weights):
    m = len(cols)
    if a == 1 << m:
        ids[0], offsets[0], weights[0] = suffix[0], b, 1 << (m-suffix[0])
        return 1
    if a == 0:
        return 0
    used, fixed = 0, b
    if not coalesced:
        for j in range(m):
            free, k = m-j-1, suffix[j+1]
            if (a >> free) & 1:
                ids[used], offsets[used], weights[used] = k, fixed, 1 << (free-k)
                used += 1
                fixed ^= cols[j]
        return used
    active, weight = suffix[1], 0
    for j in range(m):
        free, k = m-j-1, suffix[j+1]
        if k != active:
            if weight:
                ids[used], offsets[used], weights[used] = active, fixed, weight
                used += 1
            active, weight = k, 0
        if (a >> free) & 1:
            weight += 1 << (free-k)
            if suffix[j] != k:
                ids[used], offsets[used], weights[used] = k, fixed, weight
                used += 1
                weight = 0
            fixed ^= cols[j]
    if weight:
        ids[used], offsets[used], weights[used] = active, fixed, weight
        used += 1
    return used


@njit(cache=True)
def coset_below(k, v, t, bases, pivots):
    count = 0
    for i in range(k):
        p = pivots[k, i]
        above_v, above_t = v >> (p+1), t >> (p+1)
        if above_v < above_t:
            return count + (1 << (k-i))
        if above_v > above_t:
            return count
        bit = (t >> p) & 1
        if bit:
            count += 1 << (k-i-1)
        if ((v >> p) & 1) != bit:
            v ^= bases[k, i]
    return count + int(v < t)


@njit(cache=True)
def rectangle_cache(bases, width):
    """Prefix equation dependencies reused across offsets and all queries.

    For each suffix image, eliminate output-prefix rows once. A dependent
    equation is a parity check on the output RHS; independent rows are free.
    No query-time Gaussian elimination, and no one-system-per-rectangle cache.
    """
    rho = len(bases)-1
    dependencies = np.zeros((rho+1, width), dtype=np.int64)
    ranks = np.zeros((rho+1, width), dtype=np.int64)
    for k in range(rho+1):
        rows = np.zeros(k, dtype=np.int64)
        expressions = np.zeros(k, dtype=np.int64)
        rank = 0
        for j in range(width):
            p = width-1-j
            row = 0
            for i in range(k):
                row |= ((bases[k, i] >> p) & 1) << i
            expression = 1 << p
            for i in range(k-1, -1, -1):
                if (row >> i) & 1:
                    if rows[i]:
                        row ^= rows[i]
                        expression ^= expressions[i]
                    else:
                        rows[i], expressions[i] = row, expression
                        rank += 1
                        break
            if row == 0:
                dependencies[k, j] = expression
            ranks[k, j] = rank
    return dependencies, ranks


@njit(cache=True)
def rectangle_below(k, v, t, width, dependencies, ranks):
    if t == 1 << width:
        return 1 << k
    count, rhs = 0, t ^ v
    for j in range(width):
        relation = dependencies[k, j]
        inconsistent = 0 if relation == 0 else parity(relation & rhs)
        if ((t >> (width-1-j)) & 1) and (relation == 0 or inconsistent):
            count += 1 << (k-ranks[k, j])
        if inconsistent:
            return count
    return count


@njit(cache=True)
def terms_below(a, t, width, used, ids, offsets, weights,
                bases, pivots, dependencies, ranks, rectangle):
    if t == 0:
        return 0
    if t == 1 << width:
        return a
    total = 0
    for i in range(used):
        k, v = ids[i], offsets[i]
        if rectangle:
            n = rectangle_below(k, v, t, width, dependencies, ranks)
        else:
            n = coset_below(k, v, t, bases, pivots)
        total += weights[i]*n
    return total


class Counter:
    """Checked entry point for isolated compiled queries and verification."""
    def __init__(self, columns, offset, width, method='coalesced_basis'):
        if method not in ('dyadic_basis', 'coalesced_basis', 'rectangle_shared'):
            raise ValueError(method)
        self.cols, self.b = checked(columns, offset, width)
        self.width, self.method = width, method
        self.suffix, self.bases, self.pivots = basis_cache(self.cols, width)
        self.dependencies, self.ranks = (rectangle_cache(self.bases, width)
                                        if method == 'rectangle_shared' else
                                        (np.zeros((1,1), dtype=np.int64),)*2)
        self.work = [np.empty(len(self.cols)+1, dtype=np.int64) for _ in range(3)]

    def prefix(self, a, t):
        a, t = index(a), index(t)
        if not 0 <= a <= 1 << len(self.cols) or not 0 <= t <= 1 << self.width:
            raise ValueError('prefix outside declared domain')
        used = prefix_terms(int(a), self.cols, self.b, self.suffix,
                            self.method == 'coalesced_basis', *self.work)
        return int(terms_below(int(a), int(t), self.width, used, *self.work,
                              self.bases, self.pivots, self.dependencies, self.ranks,
                              self.method == 'rectangle_shared'))


def python_suffixes(columns):
    spaces = [None]*(len(columns)+1)
    spaces[-1] = ()
    for j in range(len(columns)-1, -1, -1):
        previous, v = spaces[j+1], int(columns[j])
        for g in previous:
            if v & (1 << (g.bit_length()-1)):
                v ^= g
        spaces[j] = (tuple(sorted(previous+(v,), reverse=True)) if v else previous)
    return spaces


def python_below(basis, v, t):
    count, k = 0, len(basis)
    for i, g in enumerate(basis):
        p = g.bit_length()-1
        if (v >> (p+1)) < (t >> (p+1)):
            return count+(1 << (k-i))
        if (v >> (p+1)) > (t >> (p+1)):
            return count
        bit = (t >> p)&1
        if bit:
            count += 1 << (k-i-1)
        if ((v >> p)&1) != bit:
            v ^= g
    return count+int(v<t)


def python_prefix(columns, b, a, t):
    m = len(columns)
    spaces = python_suffixes(columns)
    if a == 1 << m:
        return (1 << (m-len(spaces[0]))) * python_below(spaces[0], b, t)
    total, v = 0, b
    for j, c in enumerate(columns):
        free, V = m-j-1, spaces[j+1]
        if (a >> free)&1:
            total += (1 << (free-len(V))) * python_below(V, v, t)
            v ^= c
    return total
