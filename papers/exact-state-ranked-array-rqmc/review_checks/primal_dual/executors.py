"""Six exact executors on the same tape; retained kernels are unmodified."""
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent / 'legacy'))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'legacy/reuse_study'))
import numpy as np
from numba import njit
import kernel as base
import counter_reference as cr
import reuse_kernel as rk
import run as original
import model as original_model

METHODS = ('reuse', 'rebuild', 'rank_stream', 'basis_prefix',
           'basis_direct', 'primal_reuse')


@njit(cache=True)
def columns_of(rows, m):
    w = len(rows)
    columns = np.zeros(m, dtype=np.int64)
    for j in range(m):
        for i in range(w):
            columns[j] |= ((rows[i] >> (m-1-j)) & 1) << (w-1-i)
    return columns


@njit(cache=True)
def prepare_primal(C, m):
    """A nested echelon frame and C=B T; no duplicate suffix-vector solve."""
    w = len(C)
    cols = columns_of(C, m)
    suffix, bases, pivots = cr.basis_cache(cols, w)
    r = int(suffix[0])
    ids = np.zeros_like(bases)
    for k in range(1, r+1):
        for i in range(k):
            for j in range(r):
                if bases[k, i] == bases[r, j]:
                    ids[k, i] = j
                    break
    B_rows = columns_of(bases[r], w)  # transpose r packed w-bit columns
    # columns_of expects r input rows of width w; result has w r-bit rows.
    T_rows = np.zeros(r, dtype=np.int64)
    for j in range(m):
        v = cols[j]
        for i in range(r):
            if (v >> pivots[r, i]) & 1:
                v ^= bases[r, i]
                T_rows[i] |= 1 << (m-1-j)
        assert v == 0
    return suffix, pivots, ids, B_rows, T_rows


@njit(cache=True)
def transform_primal(U, V, c, pivots, ids, B_rows, T_rows):
    """Solve V Z=B in packed rows, form M=Z(TU), and retain echelon pivots."""
    m, w, r = len(U), len(V), len(T_rows)
    TU = np.empty(r, dtype=np.int64)
    for j in range(r):
        TU[j], _ = rk.xor_rows(T_rows[j], U)
    Z = np.empty(w, dtype=np.int64)
    M_rows = np.empty(w, dtype=np.int64)
    b = 0
    for i in range(w):
        value = B_rows[i]
        mask = V[i] ^ (1 << (w-1-i))
        while mask:
            p = cr.msb(mask)
            value ^= Z[w-1-p]
            mask ^= 1 << p
        Z[i] = value
        M_rows[i], _ = rk.xor_rows(value, TU)
        bi = ((c >> (w-1-i)) & 1) ^ cr.parity(V[i] & b)
        b |= bi << (w-1-i)
    Z_columns = columns_of(Z, r)
    bases = np.zeros_like(ids)
    for k in range(1, r+1):
        for j in range(k):
            bases[k, j] = Z_columns[ids[k, j]]
    return columns_of(M_rows, m), b, bases


@njit(cache=True)
def basis_direct_update(xs, ns, cols, b, suffix, bases, pivots, cuts, dest):
    """The same maximal aligned blocks as the constraint executor."""
    counts = np.zeros(len(cuts), dtype=np.int64)
    m, end = len(cols), 0
    for i in range(len(xs)):
        x, start = xs[i], end
        end += ns[i]
        while start < end:
            k = cr.msb(end-start)
            if start:
                k = min(k, cr.msb(start & -start))
            offset, mask = b, start
            while mask:
                p = cr.msb(mask)
                offset ^= cols[m-1-p]
                mask ^= 1 << p
            rho = suffix[m-k]
            previous = 0
            for j in range(cuts.shape[1]):
                current = cr.coset_below(rho, offset, cuts[x, j], bases, pivots) << (k-rho)
                counts[dest[x, j]] += current-previous
                previous = current
            counts[dest[x, cuts.shape[1]]] += (1 << k)-previous
            start += 1 << k
    return counts


@njit(cache=True)
def weighted_readout(counts, weights):
    a, b = 0, 0
    for i in range(len(counts)):
        a += counts[i]*weights[i, 0]
        b += counts[i]*weights[i, 1]
    return a, b


@njit(cache=True)
def trajectory(C, cuts, dest, weights, m, horizon, seed, method, audit):
    w, D = len(C), len(cuts)
    deps = np.empty((0, 0), dtype=np.int64)
    heads, ranks, jumps = deps, deps, deps
    suffix = np.empty(0, dtype=np.int64)
    pivots, ids = deps, deps
    B_rows, T_rows = suffix, suffix
    if method == 0:
        deps, heads, ranks, jumps = base.prepare(C, m)
    elif method == 1:
        ranks, jumps = rk.profile(C, m)
    elif method == 5:
        suffix, pivots, ids, B_rows, T_rows = prepare_primal(C, m)
    histories = np.empty((horizon, D), dtype=np.int64)
    readouts = np.empty((horizon, 2), dtype=np.int64)
    activity = np.zeros((horizon, 11), dtype=np.int64) if audit else np.empty((0, 11), dtype=np.int64)
    widths = np.zeros((horizon, 3, m+1), dtype=np.int64) if audit else np.empty((0, 3, m+1), dtype=np.int64)
    max_tests = np.zeros((horizon, m+1), dtype=np.int64) if audit else np.empty((0, m+1), dtype=np.int64)
    occupied = np.zeros(horizon, dtype=np.int64) if audit else np.empty(0, dtype=np.int64)
    xs, ns = np.array([0], dtype=np.int64), np.array([1 << m], dtype=np.int64)
    for tick in range(horizon):
        U, V = np.full(m, -1, dtype=np.int64), np.full(w, -1, dtype=np.int64)
        shift = np.full(1, -1, dtype=np.int64)
        source_stats = np.zeros(11, dtype=np.int64)
        base.eager_source(U, V, shift, seed, tick, source_stats)
        if audit:
            work, by_k, maxima = activity[tick], widths[tick], max_tests[tick]
            occupied[tick] = len(xs)
        else:
            work = np.empty(0, dtype=np.int64)
            by_k = np.empty((0, 0), dtype=np.int64)
            maxima = np.empty(0, dtype=np.int64)
        if method < 2:
            counts = rk.constraint_step(xs, ns, cuts, dest, method, C, deps, heads, ranks, jumps,
                                        U, V, shift[0], work, by_k, maxima, audit)
        elif method == 5:
            cols, b, bases = transform_primal(U, V, shift[0], pivots, ids, B_rows, T_rows)
            counts = basis_direct_update(xs, ns, cols, b, suffix, bases, pivots, cuts, dest)
        else:
            cols, b = base.affine_map(C, U, V, shift[0])
            if method == 2:
                counts = base.stream_update(xs, ns, cols, b, cuts, dest)
            elif method == 3:
                counts = base.basis_update(xs, ns, cols, b, w, cuts, dest)
            else:
                suffix, bases, pivots = cr.basis_cache(cols, w)
                counts = basis_direct_update(xs, ns, cols, b, suffix, bases, pivots, cuts, dest)
        xs, ns = base.compact(counts)
        histories[tick] = counts
        readouts[tick, 0], readouts[tick, 1] = weighted_readout(counts, weights)
    return histories, readouts, activity, widths, max_tests, occupied


@njit(cache=True)
def tandem_tables(width, horizon):
    if width < 4:
        raise ValueError('tandem benchmark requires word width >= 4')
    # One extra level; no state arriving at this guard is used within horizon.
    level = horizon+1
    D = (level+1)*(level+2)//2
    cuts = np.empty((D, 2), dtype=np.int64)
    dest = np.empty((D, 3), dtype=np.int64)
    weights = np.empty((D, 2), dtype=np.int64)
    for total in range(level+1):
        for q2 in range(total+1):
            q1, x = total-q2, total*(total+1)//2+q2
            cuts[x, 0] = 7*(1 << (width-4))+1
            cuts[x, 1] = 12*(1 << (width-4))+1
            dest[x, 0] = x+1 if q1 else x
            dest[x, 1] = (total-1)*total//2+q2-1 if q2 else x
            dest[x, 2] = (total+1)*(total+2)//2+q2 if total < level else x
            weights[x, 0], weights[x, 1] = total, int(q2 > 4)
    return cuts, dest, weights


def problem(spec):
    if spec['model'] == 'repair':
        cuts, dest, _ = original_model.model(spec['K'], spec['w'])
        weights = np.column_stack((np.arange(spec['K']+1, dtype=np.int64),
                                   (np.arange(spec['K']+1) >= (spec['K']+2)//2).astype(np.int64)))
        return cuts, dest, weights
    if spec['model'] == 'tandem':
        return tandem_tables(spec['w'], spec['horizon'])
    raise ValueError(spec['model'])


def execute(spec, seed, method, audit=False):
    m, w, T = spec['m'], spec['w'], spec['horizon']
    scale = spec.get('K', T)
    if not (1 <= m <= 30 and m <= w <= 52 and T*(1 << m)*scale < 2**63):
        raise ValueError('outside the declared packed-integer domain')
    start = time.perf_counter()
    C = original.base_rows(m, w)
    cuts, dest, weights = problem(spec)
    setup = time.perf_counter()-start
    result = trajectory(C, cuts, dest, weights, m, T, int(seed), METHODS.index(method), bool(audit))
    return time.perf_counter()-start, setup, result


def cells():
    out = [dict(model='repair', K=63, w=w, m=m, horizon=100)
           for w in (30, 52) for m in (8, 12, 16, 20)]
    out += [dict(model='repair', K=255, w=52, m=m, horizon=100) for m in (12, 16, 20)]
    out += [dict(model='tandem', w=w, m=m, horizon=200)
            for w in (30, 52) for m in (12, 16, 20)]
    return out


def validate(result, spec):
    hist, readouts = result[:2]
    _, _, weights = problem(spec)
    assert hist.shape == (spec['horizon'], len(weights))
    assert np.all(hist >= 0) and np.all(hist.sum(axis=1) == 1 << spec['m'])
    assert np.array_equal(readouts, hist @ weights)
    if spec['model'] == 'tandem':
        for tick, counts in enumerate(hist):
            assert not np.any(counts[weights[:, 0] > tick+1])


def warmup():
    for model in ('repair', 'tandem'):
        spec = dict(model=model, m=3, w=8, horizon=3)
        if model == 'repair':
            spec['K'] = 7
        for method in METHODS:
            for audit in (False, True):
                _, _, result = execute(spec, 102000099, method, audit)
                validate(result, spec)
