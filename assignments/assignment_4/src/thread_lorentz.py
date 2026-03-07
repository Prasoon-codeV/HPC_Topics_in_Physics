
import threading
import numpy as np
import lorentz_func
import time
from lorentz_func import lorentzian_histogram
import importlib as il

lorentzian_histogram = il.reload(lorentz_func).lorentzian_histogram

def add_chunk(n, counts, lock, bins=100, xmin=-10, xmax=10):
    """
    Generate n samples and add to global counts.
    """
    local_counts = lorentzian_histogram(n, bins, xmin, xmax)
    
    # Acquire lock to merge partial counts into global
    with lock:
        counts += local_counts


def run_threaded(n, n_counts=4, bins=100, xmin=-10, xmax=10):
    """
    Run the Lorentzian sampling in parallel using threads.
    """
    n_threads = n_counts
    # Split n samples among processes
    chunks = (n // n_threads) * np.ones(n_threads, dtype=int)
    chunks[:n % n_threads] += 1 # Distribute remainder
    threads = [None] * n_threads # Thread list
    counts = np.zeros(bins) # Global counts
    lock = threading.Lock() # Lock for global data
    
    #thread_count = 0
    
    for i in range(n_threads):
        t = threading.Thread(target=add_chunk, args=(chunks[i], counts, lock, bins, xmin, xmax))
        t.start() # Start thread
        threads[i] = t
        #thread_count += 1
    
    for t in threads:
        t.join() # Wait for all threads to finish
    return counts
