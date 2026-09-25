"""Exact stopped-chain example: joint rank law versus its covariance alone.

This mathematical example was added during the September 2026 manuscript
revision. It is separate from the eight measured performance cases.
Two implementations check the complete array-average payoff distribution:
rational state propagation and fixed-label, integer suffix enumeration.
No simulation kernel or saved benchmark result is used.
"""
from collections import Counter
from fractions import Fraction as F
from itertools import product
import json


def rank_laws():
    projected = []
    for c1, c2, offset in product((0, 1), repeat=3):
        projected.append(tuple(2*((offset + c1*(r//4) + c2*((r//2)%2) + r%2)%2)-1
                               for r in range(8)))
    pairs = [tuple(s for e in orientation for s in (e, -e))
             for orientation in product((-1, 1), repeat=4)]
    return {"projected": tuple(projected), "independent_rank_pairs": tuple(pairs)}


def rational_propagation(law):
    # A sorted multiset suffices for the symmetric array-average payoff.
    # Sorting is active-first, then by value. Equal states are interchangeable.
    states = {tuple((False, F(0)) for _ in range(8)): F(1)}
    sizes = []
    for _ in range(6):
        following = Counter()
        for state, probability in states.items():
            for signs in law:
                updated = []
                for (dead, x), sign in zip(state, signs):
                    y = x if dead else F(3, 4)*x + sign
                    updated.append((dead or abs(y) >= 3, y))
                following[tuple(sorted(updated))] += probability/len(law)
        states = following
        sizes.append(len(states))
    payoffs = Counter()
    for state, probability in states.items():
        payoffs[sum(x*x for _, x in state)/8] += probability
    return dict(payoffs), sizes


def integer_suffix_enumeration(law):
    def step(values, dead, scale, signs):
        order = sorted(range(8), key=lambda i: (dead[i], values[i], i))
        result = [4*x for x in values]  # frozen values rescaled too
        stopped = list(dead)
        for rank, label in enumerate(order):
            if not dead[label]:
                result[label] = 3*values[label] + 4*scale*signs[rank]
                stopped[label] = abs(result[label]) >= 12*scale
        return tuple(result), tuple(stopped), 4*scale

    prefix_states = set()
    for prefix in product(law, repeat=3):
        values, dead, scale = (0,)*8, (False,)*8, 1
        for signs in prefix:
            values, dead, scale = step(values, dead, scale, signs)
        assert not any(dead)
        prefix_states.add(tuple(sorted(values)))
    assert prefix_states == {(-148, -76, -52, -20, 20, 52, 76, 148)}
    # The future law is assigned by current rank. At equal states labels do
    # not affect the payoff multiset, so this representative covers all prefixes.
    initial = next(iter(prefix_states))
    payoffs = Counter()
    for suffix in product(law, repeat=3):
        values, dead, scale = initial, (False,)*8, 64
        for signs in suffix:
            values, dead, scale = step(values, dead, scale, signs)
        payoffs[F(sum(x*x for x in values), 8*scale*scale)] += F(1, len(law)**3)
    return dict(payoffs)


def moments(distribution):
    assert sum(distribution.values()) == 1
    mean = sum(x*p for x, p in distribution.items())
    return mean, sum((x-mean)**2*p for x, p in distribution.items())


def single_path_mean():
    total = F(0)
    for tape in product((-1, 1), repeat=6):
        x = F(0)
        for sign in tape:
            x = F(3, 4)*x + sign
            if abs(x) >= 3:
                break
        total += x*x
    return total/64


def verify():
    report = {}
    means, variances = [], []
    for name, law in rank_laws().items():
        distribution, sizes = rational_propagation(law)
        assert distribution == integer_suffix_enumeration(law)
        mean, variance = moments(distribution)
        assert mean == single_path_mean() == F(40354351, 16777216)
        means.append(mean)
        variances.append(variance)
        report[name] = {"mean": str(mean), "variance": str(variance),
                        "variance_decimal": float(variance),
                        "state_counts": sizes, "payoff_values": len(distribution),
                        "suffixes": len(law)**3}
    difference = variances[1]-variances[0]
    assert variances == [F(150156208090179, 562949953421312),
                         F(150252523990083, 562949953421312)]
    assert difference == F(45927, 268435456)
    report["variance_difference"] = str(difference)
    return report


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
