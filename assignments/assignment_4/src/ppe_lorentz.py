from concurrent.futures import ProcessPoolExecutor
from lorentz_func import lorentzian_histogram
import numpy as np
import time


def run_ppe(n, n_counts=4, bins=100, xmin=-10, xmax=10):
    """
    Run the Lorentzian sampling in parallel using ProcessPoolExecutor.
    """
    max_workers = n_counts
    chunks = (n // max_workers) * np.ones(max_workers, dtype=int) # Split n samples among workers
    chunks[:n % max_workers] += 1 # Distribute remainder
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(lorentzian_histogram, chunk, bins, xmin, xmax) for chunk in chunks]
    results = [f.result() for f in futures] # Collect results
    
    return np.sum(results, axis=0) # Aggregate results

'''
if __name__ == '__main__':
    n = 2*10**8
    bins = 200
    start = time.perf_counter()
    counts = run_ppe(n, bins=bins)
    end = time.perf_counter() - start
    print(f"Parallel Processing Lorentzian histogram: {end:.2f} seconds")
'''