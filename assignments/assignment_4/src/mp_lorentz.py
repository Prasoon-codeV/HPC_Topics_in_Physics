
import multiprocessing
import numpy as np
from functools import partial
import lorentz_func

def run_multiproc(n, n_counts=4, bins=100, xmin=-10, xmax=10):
    """
    Run the Lorentzian sampling in parallel using processes.
    """
    n_cores = n_counts
    # Split n samples among processes
    chunks = (n // n_cores) * np.ones(n_cores, dtype=int)
    chunks[:n % n_cores] += 1 # Distribute remainder
    
    # Use partial function to reset default arguments (bins, xmin, xmax)
    lorentzian_hist_func = partial(lorentz_func.lorentzian_histogram, 
                                   bins=bins, xmin=xmin, xmax=xmax)
    
    # Use Pool to distribute chunks to processes
    with multiprocessing.Pool(n_cores) as pool:
        results = pool.map(lorentzian_hist_func, chunks)
    
    return np.sum(results, axis=0) # Aggregate results