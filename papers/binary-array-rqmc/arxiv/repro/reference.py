"""Stopped-chain reference and independent readout noise.

Extracted computational definitions; see data/provenance.json.
"""
from dataclasses import dataclass, asdict
from time import perf_counter
import numpy as np
from scipy.stats import qmc

@dataclass(frozen=True)
class Condition:
    name: str
    horizon: int
    rho: float
    barrier: float
    payoff: str
    degree: int
    location: str
    drift: float
    strike: float
    wire_seed: int
    scales: tuple = (.85, 1., 1.15)

    def to_dict(self):
        return asdict(self)


def noise_draw(rng, shape):
    # Exact discrete probability denominator; never Gaussian.
    z = rng.integers(0, 20, size=shape, dtype=np.int8)
    return (z == 19).astype(float) - (z == 0).astype(float)


def array_run(c, n, rng, rng_noise):
    """Established array-RQMC, scalar-state sort, independent net each step.

    Labeled paths are retained. Absorbed paths receive but ignore suffix bits.
    State ordering precedes the fresh randomized transition net at each time.
    Independent per-coordinate LMS+shift makes the assigned transition bit fair.
    """
    x = np.zeros((3, n))
    alive = np.ones((3, n), dtype=bool)
    transitions = 0
    rng_sec, run_sec = 0., 0.
    for t in range(c.horizon):
        start = perf_counter()
        orders = [np.lexsort((x[i], ~alive[i])) for i in range(3)]
        run_sec += perf_counter()-start
        start = perf_counter()
        points = qmc.Sobol(2, scramble=True, bits=30, seed=int(rng.integers(2**32))).random_base2(n.bit_length()-1)
        epsilon = np.where(points[np.argsort(points[:, 0]), 1] >= .5, 1., -1.)
        rng_sec += perf_counter()-start
        start = perf_counter()
        for i in range(3):
            labels = orders[i]
            active = alive[i, labels]
            target = labels[active]
            transitions += len(target)
            x[i, target] = c.rho*x[i, target] + c.drift + c.scales[i]*epsilon[active]
            alive[i, target] = np.abs(x[i, target]) < c.barrier
        run_sec += perf_counter()-start
    signals = (x*x if c.payoff == "even" else np.maximum(x-c.strike, 0.)).T
    start = perf_counter()
    y = signals + noise_draw(rng_noise, signals.shape)
    return y, signals, dict(transitions=transitions, primitive_uses=transitions,
                           transformed_coordinates=0, rng_wall=rng_sec,
                           program_wall=run_sec, noise_wall=perf_counter()-start,
                           transform_wall=0., generated_coordinates=2*n*c.horizon)
