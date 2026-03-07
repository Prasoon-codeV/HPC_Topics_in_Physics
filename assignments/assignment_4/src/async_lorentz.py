import asyncio
import numpy as np
from lorentz_func import lorentzian_histogram
import time
import nest_asyncio
nest_asyncio.apply() # Allow nested event loops in Jupyter

async def async_lorentzian_histogram(n, bins=100, xmin=-10, xmax=10):
    """
    Async wrapper for lorentzian_histogram. Since lorentzian_histogram
    is CPU-bound and synchronous, we call it directly.
    """
    return lorentzian_histogram(n, bins, xmin, xmax)

async def add_chunk(n, counts, bins=100, xmin=-10, xmax=10, n_subchunks=10):
    """
    Generate n samples in subchunks and add to global counts.
    """
    # Split n samples among sub-chunks
    sub_chunks = (n // n_subchunks) * np.ones(n_subchunks, dtype=int)
    sub_chunks[:n % n_subchunks] += 1 # Distribute remainder

    # Gather results from subchunks
    local_counts = await asyncio.gather(*[
        async_lorentzian_histogram(chunk, bins, xmin, xmax)
        for chunk in sub_chunks
    ])
    
    counts += np.sum(local_counts, axis=0) # Merge partial counts
    
async def get_counts(n, n_tasks=4, bins=100, xmin=-10, xmax=10, n_subchunks=10):
    """
    Async function to run the Lorentzian sampling in parallel using asyncio.
    """
    # Split n samples among tasks
    chunks = (n // n_tasks) * np.ones(n_tasks, dtype=int)
    chunks[:n % n_tasks] += 1 # Distribute remainder
    counts = np.zeros(bins) # Global counts
    tasks = [
        asyncio.create_task(add_chunk(chunk, counts, bins, xmin, xmax, n_subchunks))
        for chunk in chunks
    ]
    await asyncio.gather(*tasks) # Wait for all tasks to finish
    return counts

def run_async(n, n_counts, bins, xmin=0, xmax=1, n_subchunks=10):
    n_tasks = n_counts
    asyncio.run(get_counts(n, n_tasks, bins, xmin, xmax, n_subchunks))

'''
if __name__ == '__main__':
    n = 2*10**8
    bins = 200
    start = time.perf_counter()
    counts = run_async(n, bins=bins)
    end = time.perf_counter() - start
    print(f"Asyncio Lorentzian histogram: {end:.2f} seconds")
'''