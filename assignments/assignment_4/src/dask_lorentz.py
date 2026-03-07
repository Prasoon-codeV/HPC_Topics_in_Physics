import time
import numpy as np
from lorentz_func import lorentzian_histogram
import dask
from dask import delayed

@delayed
def delayed_lorentzian_histogram(n, bins=1000, xmin=-10, xmax=10):
    """
    Delayed function for lorentzian_histogram.
    """
    return lorentzian_histogram(n, bins, xmin, xmax)

def run_dask(n, n_counts=4, bins = 1000):
    """
    Run the Lorentzian sampling in parallel using Dask.
    """
    n_tasks = n_counts
    # Split n samples among tasks
    chunks = (n // n_tasks) * np.ones(n_tasks, dtype=int)
    chunks[:n % n_tasks] += 1 # Distribute remainder
    tasks = [delayed_lorentzian_histogram(chunk) for chunk in chunks]
    results = dask.compute(*tasks) # Compute all tasks
    return np.sum(results, axis=0) # Aggregate results

'''
if __name__ == '__main__':
    n = 2*10**8
    bins = 200
    start = time.perf_counter()
    counts = run_dask(n)
    end = time.perf_counter() - start
    print(f"Dask Lorentzian histogram: {end:.2f} seconds")
 
'''