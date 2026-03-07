
from mpire import WorkerPool
import time
import numpy as np
from lorentz_func import lorentzian_histogram

def run_mpire(n, n_counts=4, bins=100, xmin=-10, xmax=10):
    """
    Run the Lorentzian sampling in parallel using mpire.
    """
    n_jobs = n_counts
    # Split n samples among jobs
    chunks = (n // n_jobs) * np.ones(n_jobs, dtype=int)
    chunks[:n % n_jobs] += 1 # Distribute remainder
    with WorkerPool(n_jobs=n_jobs) as pool:
        # See mpire docs for argument passing; alternatively use starmap
        results = pool.map(lorentzian_histogram, chunks)
    return np.sum(results, axis=0) # Aggregate results

'''
if __name__ == '__main__':
    n = 2*10**8
    bins = 200
    start = time.perf_counter()
    counts = run_mpire(n, bins=bins)
    end = time.perf_counter() - start
    print(f"MPIRE Lorentzian histogram: {end:.2f} seconds")
'''
