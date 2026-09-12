"""Exact order maintenance for scalar monotone binary branches.

Computational function bodies are unchanged from the measured implementation.
"""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
from time import perf_counter
import numpy as np
from numba import njit
import projection

legacy = projection.legacy

@njit(cache=True)
def _repair_ties(labels, size, x):
    """Only roundoff-created, label-inverted equal-value groups need sorting."""
    groups = 0
    count = 0
    start = 0
    while start < size:
        end = start + 1
        inversion = False
        while end < size and x[labels[end]] == x[labels[start]]:
            if labels[end] < labels[end - 1]:
                inversion = True
            end += 1
        if inversion:
            labels[start:end] = np.sort(labels[start:end])
            groups += 1
            count += end - start
        start = end
    return groups, count


@njit(cache=True, inline="always")
def _before(left, right, x):
    return x[left] < x[right] or (x[left] == x[right] and left < right)


@njit(cache=True)
def advance_order(previous, active_count, epsilon, x, alive, output, workspace):
    """Reproduce lexsort((x, ~alive)), including original-label stable ties.

Input order was correct BEFORE the already-completed monotone binary update.
All old active labels occupy its prefix. Workspace has three N-sized int rows.
The output must not alias previous/workspace. Finite parameters and nonnegative
rho ensure each sign branch has nondecreasing updated x (infinities may absorb).
"""
    if active_count == 0:
        output[:] = previous
        return 0, 0, 0
    minus, plus, newly_absorbed = workspace[0], workspace[1], workspace[2]
    nm = 0
    np_ = 0
    repair_minus = False
    repair_plus = False
    for rank in range(active_count):
        label = previous[rank]
        if epsilon[rank] < 0:
            if nm and x[label] == x[minus[nm - 1]] and label < minus[nm - 1]:
                repair_minus = True
            minus[nm] = label
            nm += 1
        else:
            if np_ and x[label] == x[plus[np_ - 1]] and label < plus[np_ - 1]:
                repair_plus = True
            plus[np_] = label
            np_ += 1
    groups = 0
    tie_labels = 0
    if repair_minus:
        g, k = _repair_ties(minus, nm, x)
        groups += g
        tie_labels += k
    if repair_plus:
        g, k = _repair_ties(plus, np_, x)
        groups += g
        tie_labels += k
    i = 0
    j = 0
    survivors = 0
    newly_dead = 0
    while i < nm or j < np_:
        if j == np_ or (i < nm and _before(minus[i], plus[j], x)):
            label = minus[i]
            i += 1
        else:
            label = plus[j]
            j += 1
        if alive[label]:
            output[survivors] = label
            survivors += 1
        else:
            newly_absorbed[newly_dead] = label
            newly_dead += 1
    i = 0
    j = active_count
    rank = survivors
    while i < newly_dead or j < len(previous):
        if j == len(previous) or (i < newly_dead and _before(newly_absorbed[i], previous[j], x)):
            output[rank] = newly_absorbed[i]
            i += 1
        else:
            output[rank] = previous[j]
            j += 1
        rank += 1
    return survivors, groups, tie_labels


def validate_condition(c):
    if isinstance(c.horizon, (bool, np.bool_)) or not isinstance(c.horizon, (int, np.integer)) or c.horizon < 1:
        raise ValueError("Positive integer horizon required")
    if len(c.scales) != 3 or not np.isfinite([c.rho, c.drift, c.barrier, c.strike, *c.scales]).all():
        raise ValueError("Three finite scales and finite transition parameters required")
    if c.rho < 0 or c.barrier <= 0 or c.payoff not in ("call", "even"):
        raise ValueError("Nonnegative rho, positive barrier and a declared payoff required")


def array_kernel(c, n, rng, rng_noise, profile=False, record_trace=False, provider=None):
    projection.log2_size(n)
    validate_condition(c)
    times = {k: 0. for k in ("setup", "projection", "state_order", "transition", "payoff", "noise")}
    start = perf_counter()
    generator = projection.BinaryProjection(n) if provider is None else None
    x = np.zeros((3, n))
    alive = np.ones((3, n), dtype=bool)
    orders = np.tile(np.arange(n, dtype=np.int64), (3, 1))
    next_orders = np.empty_like(orders)
    workspace = np.empty((3, n), dtype=np.int64)
    counts = np.full(3, n, dtype=np.int64)
    transitions = 0
    tie_groups = 0
    tie_labels = 0
    if profile:
        times["setup"] = perf_counter() - start
    history = []
    order_history = []
    for t in range(c.horizon):
        if record_trace:
            history.append((x.copy(), alive.copy()))
            order_history.append(orders.copy())
        if provider is None:
            if profile:
                start = perf_counter()
            epsilon = generator.draw(rng)
            if profile:
                times["projection"] += perf_counter() - start
        else:
            epsilon = provider(t)
            if np.shape(epsilon) != (n,) or not np.all((epsilon == -1) | (epsilon == 1)):
                raise ValueError("Replay provider must return N signs")
        if profile:
            start = perf_counter()
        # Retain the predecessor's actual vector expressions, evaluation order
        # and active-mask indexing. This is not a fused-arithmetic speed claim.
        for i in range(3):
            labels = orders[i]
            active = alive[i, labels]
            target = labels[active]
            transitions += len(target)
            x[i, target] = c.rho*x[i, target] + c.drift + c.scales[i]*epsilon[active]
            alive[i, target] = np.abs(x[i, target]) < c.barrier
        if profile:
            times["transition"] += perf_counter() - start
        if t + 1 < c.horizon:
            if profile:
                start = perf_counter()
            for i in range(3):
                counts[i], g, k = advance_order(orders[i], counts[i], epsilon, x[i], alive[i], next_orders[i], workspace)
                tie_groups += g
                tie_labels += k
            orders, next_orders = next_orders, orders
            if profile:
                times["state_order"] += perf_counter() - start
    if record_trace:
        history.append((x.copy(), alive.copy()))
    if profile:
        start = perf_counter()
    signals = (x*x if c.payoff == "even" else np.maximum(x - c.strike, 0.)).T
    if profile:
        times["payoff"] += perf_counter() - start
        start = perf_counter()
    y = signals + legacy.noise_draw(rng_noise, signals.shape)
    if profile:
        times["noise"] += perf_counter() - start
    stats = dict(transitions=transitions, generated_rank_signs=n*c.horizon,
                 randomization_words=c.horizon, state_sorts=0, order_updates=3*(c.horizon-1),
                 tie_groups=tie_groups, tie_labels=tie_labels, times=times)
    if record_trace:
        stats.update(history=history, orders=order_history, x=x, alive=alive)
    return y, signals, stats
