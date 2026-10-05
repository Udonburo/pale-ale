"""Independent scalar GF(2) oracle and checks of all six executors."""
from pathlib import Path
import json
import numpy as np
from numba import njit
import executors as ex


@njit(cache=True)
def scalar_outputs(C, U, V, c):
    m, w = len(U), len(V)
    out = np.zeros(1 << m, dtype=np.int64)
    for rank in range(1 << m):
        u = np.zeros(m, dtype=np.int64)
        for i in range(m):
            for j in range(m):
                u[i] ^= ((U[i] >> (m-1-j)) & 1)*((rank >> (m-1-j)) & 1)
        y = np.zeros(w, dtype=np.int64)
        for i in range(w):
            y[i] = (c >> (w-1-i)) & 1
            for j in range(m):
                y[i] ^= ((C[i] >> (m-1-j)) & 1)*u[j]
            for j in range(i):
                y[i] ^= ((V[i] >> (w-1-j)) & 1)*y[j]
            out[rank] |= y[i] << (w-1-i)
    return out


@njit(cache=True)
def unit_lower(n, code):
    out = np.zeros(n, dtype=np.int64)
    at = 0
    for i in range(n):
        out[i] = 1 << (n-1-i)
        for j in range(i):
            out[i] |= ((code >> at) & 1) << (n-1-j)
            at += 1
    return out


@njit(cache=True)
def tape_check(C, U, V, c):
    m, w = len(U), len(V)
    expected = scalar_outputs(C, U, V, c)
    suffix, pivots, ids, B_rows, T_rows = ex.prepare_primal(C, m)
    columns, b, bases = ex.transform_primal(U, V, c, pivots, ids, B_rows, T_rows)
    other_cols, other_b = ex.base.affine_map(C, U, V, c)
    assert b == other_b and np.array_equal(columns, other_cols)
    nqueries = ((1 << (m+1))-1)*((1 << w)+1)
    starts = np.empty(nqueries, dtype=np.int64)
    ks, thresholds = np.empty_like(starts), np.empty_like(starts)
    want, at = np.empty_like(starts), 0
    for k in range(m+1):
        rho = suffix[m-k]
        for j in range(rho):
            assert ex.cr.msb(bases[rho, j]) == pivots[rho, j]
        for start in range(0, 1 << m, 1 << k):
            image = np.empty(1 << rho, dtype=np.int64)
            for coord in range(1 << rho):
                v = expected[start]
                for i in range(rho):
                    if (coord >> i) & 1:
                        v ^= bases[rho, i]
                image[coord] = v
            assert np.array_equal(np.sort(image), np.unique(expected[start:start+(1 << k)]))
            for threshold in range((1 << w)+1):
                answer = 0
                for r in range(start, start+(1 << k)):
                    answer += expected[r] < threshold
                got = ex.cr.coset_below(rho, expected[start], threshold, bases, pivots) << (k-rho)
                assert got == answer
                starts[at], ks[at], thresholds[at], want[at] = start, k, threshold, answer
                at += 1
    for mode in (0, 1):
        result = ex.rk.batch(C, m, U, V, c, starts, ks, thresholds, mode)
        assert np.array_equal(want, result[0])
        assert np.array_equal(result[2][2], result[3])
    return at


@njit(cache=True)
def exhaustive():
    queries, tapes = 0, 0
    for code in range(64):
        C = np.array([code & 3, (code >> 2) & 3, (code >> 4) & 3], dtype=np.int64)
        for u in range(2):
            U = unit_lower(2, u)
            for v in range(8):
                V = unit_lower(3, v)
                for c in range(8):
                    queries += tape_check(C, U, V, c)
                    tapes += 1
    return tapes, queries


def main():
    ex.warmup()
    kats = [([0]*4, [0]*2, [0x6627e8d5,0xe169c58d,0xbc57ac4c,0x9b00dbd8]),
            ([0xffffffff]*4, [0xffffffff]*2, [0x408f276d,0x41c83b0e,0xa20bc7c6,0x6d5451fd]),
            ([0x243f6a88,0x85a308d3,0x13198a2e,0x03707344], [0xa4093822,0x299f31d0],
             [0xd16cfe09,0x94fdcceb,0x5001e420,0x24126ea1])]
    for counter, key, expected_kat in kats:
        assert [int(v) for v in ex.base.philox4x32(*counter, *key)] == expected_kat
    tapes, queries = exhaustive()
    rng = np.random.default_rng(102000088)
    additional = 0
    for n in range(160):
        m, w = int(rng.integers(1, 7)), int(rng.integers(1, 8))
        C = rng.integers(0, 1 << m, w, dtype=np.int64)
        if n % 8 == 0:
            C[:] = 0
        elif n % 8 == 1:
            C[:] = C[0]
        U = unit_lower(m, int(rng.integers(0, 1 << (m*(m-1)//2))))
        V = unit_lower(w, int(rng.integers(0, 1 << (w*(w-1)//2))))
        additional += tape_check(C, U, V, int(rng.integers(0, 1 << w)))
    trajectories, steps = 0, 0
    for model in ('repair', 'tandem'):
        for m in (3, 7, 10):
            spec = dict(model=model, m=m, w=12, horizon=20)
            if model == 'repair':
                spec['K'] = 15
            for seed in (102000000, 102000001):
                expected = None
                for method in ex.METHODS:
                    _, _, result = ex.execute(spec, seed, method, True)
                    ex.validate(result, spec)
                    if expected is None:
                        expected = result
                    assert all(np.array_equal(a, b) for a, b in zip(expected[:2], result[:2]))
                    trajectories += 1
                    steps += spec['horizon']
    # Endpoint atoms and transition-dependent rewards that share destinations.
    width = 8
    cuts, dest, _ = ex.tandem_tables(width, 2)
    for y in (0, 7*16, 7*16+1, 12*16, 12*16+1, 255):
        branch = int(y > 7*16) + int(y > 12*16)
        assert int(np.searchsorted(cuts[0], y, side='right')) == branch
    C = np.array([3, 1, 2], dtype=np.int64)
    U, V = unit_lower(2, 1), unit_lower(3, 5)
    outputs = scalar_outputs(C, U, V, 3)
    flow = [int(np.sum(outputs < 2)), int(np.sum((outputs >= 2) & (outputs < 6))),
            int(np.sum(outputs >= 6))]
    rewards = (7, -2, 11)
    assert sum(f*g for f, g in zip(flow, rewards)) == sum(rewards[0 if y < 2 else 1 if y < 6 else 2] for y in outputs)
    report = dict(status='PASS', philox_upstream_known_answers=len(kats), exhaustive_tapes=int(tapes), exhaustive_queries=int(queries),
                  additional_tapes=160, additional_queries=int(additional),
                  method_trajectories=trajectories, histogram_steps=steps,
                  checks=['scalar GF2 oracle', 'suffix images', 'echelon pivots',
                          'full affine offsets', 'dual reuse and rebuild',
                          'all six dynamic executors', 'tandem atoms', 'transition rewards'])
    path = Path(__file__).with_name('checks.json')
    path.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
