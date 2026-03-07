import time
import numpy as np
from numba import njit, prange

# following code doesn't use atomics
@njit(parallel=True)
def lorentzian_histogram_numba(n, bins=100, nchunks=4, xmin=-10.0, xmax=10.0):
    """
    Parallel Lorentzian histogram without atomics:
    each chunk builds its own histogram, then we reduce.
    """
    dx = (xmax - xmin) / bins
    partial = np.zeros((nchunks, bins), dtype=np.int64)

    for c in prange(nchunks):
        start = (n * c) // nchunks
        end = (n * (c + 1)) // nchunks

        for _ in range(start, end):
            u = np.random.random()
            x = 1.0 / np.tan(np.pi * u)

            if xmin <= x < xmax:
                b = int((x - xmin) / dx)
                if 0 <= b < bins:
                    partial[c, b] += 1

    counts = np.zeros(bins, dtype=np.int64)
    for c in range(nchunks):
        counts += partial[c]

    return counts


def run_numba(n, n_counts, bins, xmin=-10.0, xmax=10.0, **kwargs):
    nchunks = n_counts
    t0 = time.perf_counter()
    counts = lorentzian_histogram_numba(n, bins, nchunks, xmin, xmax)
    dt = time.perf_counter() - t0
    print(f"Numba parallel Lorentzian histogram: {dt:.2f} seconds")
    return counts

'''
if __name__ == '__main__':
    n = 1*10**8
    bins = 100
    start = time.perf_counter()
    counts = lorentzian_histogram_numba(n, bins=bins)
    end = time.perf_counter() - start
    print(f"Numba Lorentzian histogram: {end:.2f} seconds")
'''