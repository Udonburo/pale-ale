"""Specified finite-word reference; SciPy adapter and machine-repair chain."""
import platform
import numpy as np
import scipy
from scipy.stats import qmc


def rank_map(engine, m, width):
    bits = int(engine.bits)
    if not 1 <= m <= bits or not 1 <= width <= bits <= 62:
        raise ValueError('invalid finite-word dimensions')
    rows = [sum(((int(engine._sv[0,j]) >> (bits-1-i)) & 1) << j
                for j in range(m)) for i in range(m)]
    inv = [1 << i for i in range(m)]
    for j in range(m):
        pivot = next((i for i in range(j,m) if (rows[i] >> j)&1), None)
        if pivot is None:
            raise ValueError('sorting prefix is not invertible')
        rows[j], rows[pivot] = rows[pivot], rows[j]
        inv[j], inv[pivot] = inv[pivot], inv[j]
        for i in range(m):
            if i != j and ((rows[i] >> j)&1):
                rows[i] ^= rows[j]
                inv[i] ^= inv[j]
    words = [int(engine._sv[1,j]) >> (bits-width) for j in range(m)]
    cols = []
    for k in range(m):
        v = 0
        for j in range(m):
            if (inv[j] >> k)&1:
                v ^= words[j]
        cols.append(v)
    s = sum(((int(engine._shift[0]) >> (bits-1-i))&1) << i for i in range(m))
    b = int(engine._shift[1]) >> (bits-width)
    for j in range(m):
        if (inv[j]&s).bit_count() & 1:
            b ^= words[j]
    return np.asarray(cols, dtype=np.int64), b


def make_map(seed, m, width):
    engine = qmc.Sobol(d=2, scramble=True, bits=max(m,width), seed=int(seed))
    return rank_map(engine, m, width)


def strict_cut(p, width):
    """U=y/2^w < the actual binary64 p, translated without decimal rounding."""
    num, den = float(p).as_integer_ratio()
    return min(1 << width, max(0, ((num << width)+den-1)//den))


def model(K, width):
    if not 3 <= K <= 4095 or not 1 <= width <= 62:
        raise ValueError('model outside tested domain')
    lam, mu, servers = 0.07, 1.13, 3
    clock = lam*K + mu*servers
    cuts = np.empty((K+1, 2), dtype=np.int64)
    dest = np.empty((K+1, 3), dtype=np.int64)
    floats = np.empty((K+1, 2), dtype=np.float64)
    for x in range(K+1):
        birth, death = lam*(K-x), mu*min(x, servers)
        p1, p2 = birth/clock, (birth+death)/clock
        floats[x] = p1, p2
        cuts[x] = strict_cut(p1,width), strict_cut(p2,width)
        dest[x] = min(x+1,K), max(x-1,0), x
    return cuts, dest, floats


def symbol(cols,b,r):
    value=b
    for j,c in enumerate(cols):
        if (r >> (len(cols)-1-j))&1:
            value ^= int(c)
    return value


def environment():
    import numba
    return dict(python=platform.python_version(), numpy=np.__version__,
                scipy=scipy.__version__, numba=numba.__version__,
                platform=platform.platform(), processor=platform.processor())
