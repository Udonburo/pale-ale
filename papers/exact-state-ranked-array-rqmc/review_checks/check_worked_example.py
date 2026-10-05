"""Independent bit-matrix check of the manuscript's four-rank example."""
import json
from pathlib import Path


def mv(a, v):
    return [sum(x * y for x, y in zip(row, v)) % 2 for row in a]


def left(v, a):
    return [sum(v[i] * a[i][j] for i in range(len(v))) % 2
            for j in range(len(a[0]))]


def bits(n, width):
    return [(n >> (width - 1 - i)) & 1 for i in range(width)]


def integer(v):
    return sum(x << (len(v) - 1 - i) for i, x in enumerate(v))


def solve_lower(a, rhs):
    out = []
    for j, row in enumerate(a):
        assert row[j] == 1 and not any(row[j + 1:])
        out.append((rhs[j] + sum(row[i] * out[i] for i in range(j))) % 2)
    return out


def main():
    cmat = [[1, 0, 1], [0, 1, 1], [0, 1, 0], [1, 0, 0]]
    u = [[1, 0, 0], [1, 1, 0], [0, 1, 1]]
    v = [[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 1, 1]]
    shift = [0, 1, 0, 1]
    deps = [[1, 1, 1, 0], [0, 0, 0, 1]]
    hs = [left(d, cmat) for d in deps]
    assert hs == [[1, 0, 0], [1, 0, 0]]
    assert [left(d, v) for d in deps] == [[0, 0, 1, 0], [1, 0, 1, 1]]
    assert [left(h, u) for h in hs] == hs
    tapes = [(u, v, shift),
             ([[int(i == j) for j in range(3)] for i in range(3)],
              [[int(i == j) for j in range(4)] for i in range(4)], [0] * 4)]
    outputs = []
    query_checks = 0
    for uu, vv, cc in tapes:
        direct = []
        for r in range(4, 8):
            rhs = [(x + y) % 2 for x, y in zip(mv(cmat, mv(uu, bits(r, 3))), cc)]
            direct.append(integer(solve_lower(vv, rhs)))
        outputs.append(direct)
        r0 = bits(4, 3)
        for tau in range(17):
            if tau in (0, 16):
                counted = 0 if tau == 0 else 4
            else:
                tb = bits(tau, 4)
                counted = 2 * tb[0] + tb[1]  # Two independent rows.
                for j, (d, h) in enumerate(zip(deps, hs), start=2):
                    if not any(tb[j:]):
                        break
                    eps = (sum(x * y for x, y in zip(left(d, vv), tb))
                           + sum(x * y for x, y in zip(left(h, uu), r0))
                           + sum(x * y for x, y in zip(d, cc))) % 2
                    if eps:
                        counted += tb[j]
                        break
            assert counted == sum(y < tau for y in direct), (direct, tau, counted)
            query_checks += 1
    assert outputs == [[13, 4, 0, 9], [9, 5, 15, 3]], outputs
    report = dict(outputs=outputs, thresholds_checked=query_checks,
                  support_costs=[sum(d) + sum(h) for d, h in zip(deps, hs)],
                  example_queries={'F(10)': 3, 'F(9)': 2}, status='PASS')
    Path(__file__).with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
