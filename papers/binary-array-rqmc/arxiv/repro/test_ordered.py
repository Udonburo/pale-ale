"""Independent lexsort reference, adversarial ties and whole-path replay."""
from dataclasses import replace
from itertools import product
import json
from pathlib import Path
import unittest
import numpy as np
from ordered import advance_order, array_kernel, legacy, projection


def reference_step(x, alive, epsilon, rho, drift, scale, barrier):
    previous = np.lexsort((x, ~alive))
    k = int(alive.sum())
    labels = previous[:k]
    new_x = x.copy()
    new_alive = alive.copy()
    new_x[labels] = rho*x[labels] + drift + scale*epsilon[:k]
    new_alive[labels] = np.abs(new_x[labels]) < barrier
    output = np.empty_like(previous)
    stats = advance_order(previous, k, epsilon, new_x, new_alive, output, np.empty((3, len(x)), dtype=np.int64))
    return output, np.lexsort((new_x, ~new_alive)), stats, new_x, new_alive


class OrderingChecks(unittest.TestCase):
    def test_exhaustive_small_orders_and_absorption(self):
        count = 0
        for values in product((-1., 0., 1.), repeat=4):
            for statuses in product((False, True), repeat=4):
                for signs in product((-1., 1.), repeat=4):
                    actual, expected, stats, _, alive = reference_step(
                        np.array(values), np.array(statuses), np.array(signs), .75, .125, .5, 1.)
                    np.testing.assert_array_equal(actual, expected)
                    self.assertEqual(stats[0], int(alive.sum()))
                    count += 1
        self.assertEqual(count, 20736)

    def test_roundoff_ties_signed_zero_and_infinities(self):
        tiny = np.nextafter(0., 1.)
        cases = [
            (np.array([2., 1., 0., -1.]), np.ones(4), 1., 2.**54, 0., 2.**60),
            (np.array([3., 2., 1., 0.]), np.ones(4), 0., 1., 0., 5.),
            (np.array([2*tiny, tiny, -0., 0.]), np.ones(4), .125, 0., 0., 1.),
            (np.array([0., -0., 0., -0.]), np.array([1., -1., 1., -1.]), 1., 0., -0., 1.),
            (np.array([8., 4., 2., 1.]), np.ones(4), 1.e308, 0., 0., 1.e308),
        ]
        groups = 0
        with np.errstate(over="ignore", under="ignore"):
            for x, epsilon, rho, drift, scale, barrier in cases:
                actual, expected, stats, _, _ = reference_step(x, np.ones(4, bool), epsilon, rho, drift, scale, barrier)
                np.testing.assert_array_equal(actual, expected)
                groups += stats[1]
        self.assertGreater(groups, 0)

    def test_random_order_fuzz(self):
        rng = np.random.default_rng(1209361)
        for rep in range(400):
            n = int(rng.integers(1, 300))
            x = rng.normal(size=n) if rep % 2 else rng.integers(-8, 9, n).astype(float)
            alive = rng.random(n) < .7
            epsilon = 2*rng.integers(0, 2, n)-1.
            actual, expected, stats, _, next_alive = reference_step(
                x, alive, epsilon, float(rng.choice([0., .01, .9, 1., 3.])),
                float(rng.choice([0., .1, 1.e18])), float(rng.choice([-1., 0., 1.])),
                float(rng.choice([.1, 1., 2., 1.e20])))
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(stats[0], int(next_alive.sum()))

    def compare_path(self, c, n, seed):
        old_rng = np.random.default_rng(seed)
        new_rng = np.random.default_rng(seed)
        old_noise = np.random.default_rng(seed+1)
        new_noise = np.random.default_rng(seed+1)
        old = projection.array_kernel(c, n, old_rng, old_noise, record_trace=True)
        new = array_kernel(c, n, new_rng, new_noise, record_trace=True)
        for i in (0, 1):
            self.assertEqual(old[i].tobytes(), new[i].tobytes())
        self.assertEqual(old[2]["transitions"], new[2]["transitions"])
        self.assertEqual(old_rng.bit_generator.state, new_rng.bit_generator.state)
        self.assertEqual(old_noise.bit_generator.state, new_noise.bit_generator.state)
        for t, ((ox, oa), (nx, na)) in enumerate(zip(old[2]["history"], new[2]["history"])):
            self.assertEqual(ox.tobytes(), nx.tobytes())
            self.assertEqual(oa.tobytes(), na.tobytes())
            if t < c.horizon:
                for i in range(3):
                    np.testing.assert_array_equal(new[2]["orders"][t][i], np.lexsort((ox[i], ~oa[i])))
        self.assertEqual(new[2]["randomization_words"], c.horizon)
        return new[2]

    def test_complete_joint_paths_and_prng_states(self):
        config = json.loads((Path(__file__).resolve().parent / "data/cases.json").read_text(encoding="utf-8"))
        for ci, c0 in enumerate(config["conditions"]):
            c = legacy.Condition(**c0)
            for n in (1, 8, 512, 4096):
                self.compare_path(c, n, 317700 + ci*7 + n)

    def test_edge_paths_and_no_early_randomization_stop(self):
        base = legacy.Condition("edge", 8, .9, 2., "even", 2, "recent", 0., 0., 2)
        cases = [replace(base, horizon=1), replace(base, scales=(0., 0., 0.)),
                 replace(base, rho=0.), replace(base, barrier=.01),
                 replace(base, scales=(-1., 0., 2.)), replace(base, drift=2.**54, barrier=2.**60)]
        for ci, c in enumerate(cases):
            for n in (1, 8, 512):
                self.compare_path(c, n, 517700+ci+n)

    def test_supplied_scipy_rank_vectors(self):
        c = legacy.Condition("scipy_replay", 12, .91, 2.4, "call", 3, "recent", .03, .3, 22)
        for n in (1, 8, 512):
            rng = np.random.default_rng(741+n)
            signs = [projection.scipy_signs(n, rng) for _ in range(c.horizon)]
            old = projection.array_kernel(c, n, None, np.random.default_rng(47), provider=lambda t: signs[t], record_trace=True)
            new = array_kernel(c, n, None, np.random.default_rng(47), provider=lambda t: signs[t], record_trace=True)
            self.assertEqual(old[0].tobytes(), new[0].tobytes())
            for a, b in zip(old[2]["history"], new[2]["history"]):
                self.assertEqual(a[0].tobytes(), b[0].tobytes())
                self.assertEqual(a[1].tobytes(), b[1].tobytes())

    def test_public_entry_and_invalid_conditions(self):
        c = legacy.Condition("public", 8, .9, 2., "call", 2, "recent", .1, .1, 2)
        for bad in [replace(c, rho=-.1), replace(c, drift=float("nan")),
                    replace(c, scales=(1., 2.)), replace(c, barrier=0.),
                    replace(c, horizon=0), replace(c, horizon=True), replace(c, payoff="unknown")]:
            with self.assertRaises(ValueError):
                array_kernel(bad, 8, np.random.default_rng(1), np.random.default_rng(2))
        for n in (0, 3, True):
            with self.assertRaises(ValueError):
                array_kernel(c, n, np.random.default_rng(1), np.random.default_rng(2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
