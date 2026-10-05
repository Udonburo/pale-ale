"""Independent direction-column -> explicit points -> sort -> rank-map bridge.

This author regression was written from the audit's stated coverage. It does
not incorporate the reviewer's unavailable implementation or its test counts.
"""
from pathlib import Path
import json
import numpy as np
from scipy.stats import qmc
import executors as ex

HERE = Path(__file__).resolve().parent


def apply(rows, word):
    out = 0
    for row in rows:
        out = (out << 1) | ((int(row) & int(word)).bit_count() & 1)
    return out


def solve_lower(rows, rhs):
    n, out = len(rows), 0
    for i, row in enumerate(rows):
        bit = ((rhs >> (n-1-i)) & 1) ^ ((int(row) & out).bit_count() & 1)
        out |= bit << (n-1-i)
    assert apply(rows, out) == rhs
    return out


def invert(rows):
    n = len(rows)
    left, right = list(map(int, rows)), [1 << (n-1-i) for i in range(n)]
    for j in range(n):
        pivot = next(i for i in range(j, n) if (left[i] >> (n-1-j)) & 1)
        left[j], left[pivot] = left[pivot], left[j]
        right[j], right[pivot] = right[pivot], right[j]
        for i in range(n):
            if i != j and (left[i] >> (n-1-j)) & 1:
                left[i] ^= left[j]
                right[i] ^= right[j]
    assert left == [1 << (n-1-i) for i in range(n)]
    return right


def matrix_rows(columns, width):
    m = len(columns)
    return [sum(((int(col) >> (width-1-i)) & 1) << (m-1-j)
                for j, col in enumerate(columns)) for i in range(width)]


def point(columns, index):
    out = 0
    for j, col in enumerate(columns):
        if (index >> j) & 1:
            out ^= int(col)
    return out


def directions(m, w):
    engine = qmc.Sobol(d=2, scramble=False, bits=max(m, w))
    raw = engine._sv[:, :m].astype(np.int64)
    cs = [int(v) >> (engine.bits-m) for v in raw[0]]
    cy = [int(v) >> (engine.bits-w) for v in raw[1]]
    inverse = invert(matrix_rows(cs, m))
    cy_rows = matrix_rows(cy, w)
    derived = matrix_rows([apply(cy_rows, apply(inverse, 1 << (m-1-j)))
                           for j in range(m)], w)
    assert derived == ex.original.base_rows(m, w).tolist()
    return engine, raw, cs, cy, derived


def main():
    rng = np.random.default_rng(105000017)
    saved = json.loads((HERE/'confirmation/data/matrices.json').read_text())
    for record in saved.values():
        _, raw, _, _, C = directions(record['m'], record['w'])
        assert raw.tolist() == record['direction_columns']
        assert C == record['C_rows']
    full_sorts = compared = scipy_points = sampled = 0
    cases = sorted({(m, w) for m in range(1, 9) for w in (max(m, 4), 30, 52)})
    for m, w in cases:
        engine, _, cs, cy, C = directions(m, w)
        actual = engine.random_base2(m)
        for index, pair in enumerate(actual):
            gray = index ^ (index >> 1)
            assert int(pair[0]*(1 << m)) == point(cs, gray)
            assert int(pair[1]*(1 << w)) == point(cy, gray)
        scipy_points += 1 << m
        for tape in range(2):
            U, V = np.full(m, -1, dtype=np.int64), np.full(w, -1, dtype=np.int64)
            shift, stats = np.full(1, -1, dtype=np.int64), np.zeros(11, dtype=np.int64)
            ex.base.eager_source(U, V, shift, 105000000+tape, m, stats)
            for sorting_shift in (0, int(rng.integers(1, 1 << m))):
                output_shift = int(rng.integers(0, 1 << w))
                c = apply(C, apply(U, sorting_shift)) ^ apply(V, output_shift)
                cols, b = ex.base.affine_map(np.asarray(C, dtype=np.int64), U, V, c)
                for gray_order in (False, True):
                    points = []
                    for index in range(1 << m):
                        code = index ^ (index >> 1) if gray_order else index
                        s = solve_lower(U, point(cs, code)) ^ sorting_shift
                        y = solve_lower(V, point(cy, code)) ^ output_shift
                        points.append((s, y))
                    points.sort()
                    assert [s for s, _ in points] == list(range(1 << m))
                    for rank, (_, y) in enumerate(points):
                        want = int(b)
                        for j, column in enumerate(cols):
                            if (rank >> (m-1-j)) & 1:
                                want ^= int(column)
                        assert y == want, (m, w, tape, rank)
                    compared += len(points)
                    full_sorts += 1
    # All saved generator sizes, including m=20: pointwise rank assignments.
    for record in saved.values():
        m, w = record['m'], record['w']
        _, _, cs, cy, C = directions(m, w)
        U, V = np.full(m, -1, dtype=np.int64), np.full(w, -1, dtype=np.int64)
        shift, stats = np.full(1, -1, dtype=np.int64), np.zeros(11, dtype=np.int64)
        ex.base.eager_source(U, V, shift, 105000001, 31, stats)
        sorting_shift, output_shift = (1 << m)-1, int(rng.integers(0, 1 << w))
        c = apply(C, apply(U, sorting_shift)) ^ apply(V, output_shift)
        cols, b = ex.base.affine_map(np.asarray(C, dtype=np.int64), U, V, c)
        for index in [0, (1 << m)-1, *map(int, rng.integers(0, 1 << m, 256))]:
            rank = solve_lower(U, point(cs, index)) ^ sorting_shift
            y = solve_lower(V, point(cy, index)) ^ output_shift
            assert y == int(b) ^ apply(matrix_rows(cols, w), rank)
            sampled += 1
    report = dict(status='PASS', saved_generators=len(saved), full_sorts=full_sorts,
                  sorted_point_comparisons=compared, scipy_unscrambled_points=scipy_points,
                  sampled_saved_generator_points=sampled, nonzero_sorting_shifts=True,
                  binary_and_gray_index_orders=True)
    (HERE/'point-sorting-checks.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
