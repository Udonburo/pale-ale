"""Exact finite-law consequences of Proposition 1; no simulation or timing."""
from fractions import Fraction
from itertools import product
import unittest


def projected_support(m):
    """Standalone integer enumeration, independent of the production sampler."""
    n = 1 << m
    return [tuple(2 * (offset ^ ((mask & rank).bit_count() % 2)) - 1
                  for rank in range(n))
            for mask in range(1, n, 2) for offset in (0, 1)]


def pair_support(n):
    return [tuple(sign for first in choices for sign in (first, -first))
            for choices in product((-1, 1), repeat=n // 2)]


def moments(support, observable):
    values = [observable(row) for row in support]
    mean = Fraction(sum(values), len(values))
    variance = Fraction(sum(x * x for x in values), len(values)) - mean * mean
    return mean, variance


class ProjectionStructureChecks(unittest.TestCase):
    def test_support_size_and_parameter_recovery(self):
        for m in range(1, 8):
            n = 1 << m
            support = projected_support(m)
            self.assertEqual(len(support), n)
            self.assertEqual(len(set(support)), n)
            recovered = set()
            for row in support:
                bits = [(x + 1) // 2 for x in row]
                offset = bits[0]
                mask = sum((bits[1 << j] ^ offset) << j for j in range(m))
                self.assertEqual(mask % 2, 1)
                recovered.add((mask, offset))
            self.assertEqual(recovered, {(a, b) for a in range(1, n, 2) for b in (0, 1)})

    def test_exact_covariance_and_balance(self):
        for m in range(1, 8):
            n = 1 << m
            support = projected_support(m)
            for row in support:
                self.assertEqual(sum(row), 0)
                self.assertTrue(all(row[r] == -row[r + 1] for r in range(0, n, 2)))
            for r in range(n):
                self.assertEqual(sum(row[r] for row in support), 0)
                for s in range(n):
                    expected = 1 if r == s else -1 if (r ^ s) == 1 else 0
                    self.assertEqual(sum(row[r] * row[s] for row in support), n * expected)

    def test_linear_readout_variance(self):
        for m in range(1, 7):
            n = 1 << m
            support = projected_support(m)
            weights = [[1] * n, list(range(n)), [r // 2 for r in range(n)],
                       [(-1) ** r for r in range(n)], [(r * r + 3 * r) % 11 - 5 for r in range(n)]]
            for w in weights:
                mean, variance = moments(support, lambda row: sum(a * b for a, b in zip(w, row)))
                expected = sum((w[r] - w[r + 1]) ** 2 for r in range(0, n, 2))
                self.assertEqual(mean, 0)
                self.assertEqual(variance / (n * n), Fraction(expected, n * n))

    def test_equal_covariance_different_nonlinear_variance(self):
        projected, pairs = projected_support(3), pair_support(8)
        self.assertEqual(len(projected), 8)
        self.assertEqual(len(pairs), 16)
        self.assertNotEqual(set(projected), set(pairs))
        for r in range(8):
            for s in range(8):
                self.assertEqual(Fraction(sum(row[r] * row[s] for row in projected), 8),
                                 Fraction(sum(row[r] * row[s] for row in pairs), 16))
        for coefficient, expected in [(-1, 0), (1, 4)]:
            observable = lambda row: row[0] * row[2] + coefficient * row[4] * row[6]
            self.assertEqual(moments(projected, observable), (0, expected))
            self.assertEqual(moments(pairs, observable), (0, 2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
