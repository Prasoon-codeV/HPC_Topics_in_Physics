from joblib import Parallel, delayed
import numpy as np
import time
from lorentz_func import lorentzian_histogram

def run_joblib(n, n_counts=4, bins=100, xmin=-10, xmax=10):
    """
    Run the Lorentzian sampling in parallel using joblib.
    """
    n_jobs = n_counts
    # Split n samples among jobs
    chunks = (n // n_jobs) * np.ones(n_jobs, dtype=int)
    chunks[:n % n_jobs] += 1 # Distribute remainder
    results = Parallel(n_jobs=n_jobs)(
        delayed(lorentzian_histogram)(chunk, bins, xmin, xmax)
        for chunk in chunks
    )
    return np.sum(results, axis=0) # Aggregate results

'''
if __name__ == '__main__':
    n = 2*10**8
    bins = 200
    start = time.perf_counter()
    counts = run_joblib(n, bins=bins)
    end = time.perf_counter() - start
    print(f"Joblib Lorentzian histogram: {end:.2f} seconds")
'''