import numpy as np
from mpi4py import MPI
import time
from lorentz_func import lorentzian_histogram, strong_scaling, weak_scaling, speedup, efficiency
import plotly.graph_objects as go
from tqdm import tqdm

def lorentzian_histogram(n, bins=100, xmin=-10, xmax=10, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    u = rng.random(n)
    x = 1. / np.tan(np.pi * u)
    counts, _ = np.histogram(x, bins=bins, range=(xmin, xmax))
    return counts.astype(np.int64)


def run_mpi(n, n_counts=1, bins=100, xmin=-10, xmax=10, seed=42):
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if size < n_counts:
        if rank == 0:
            print(f"Warning: Requested n_counts={n_counts} exceeds available MPI ranks={size}. Using n_counts={size}.")
        n_counts = size
        return None
    
    # Using only first n_counts ranks for work, others idle

    active = rank < n_counts
    subcomm = comm.Split(color=0 if active else MPI.UNDEFINED, key=rank)

    global_counts = None
    if active:
        sub_rank = subcomm.Get_rank()
        sub_size = subcomm.Get_size()

        # Independent RNG stream per active rank
        ss = np.random.SeedSequence(seed)
        child_seed = ss.spawn(sub_size)[sub_rank]
        rng = np.random.default_rng(child_seed)

        chunks = np.full(sub_size, n // sub_size, dtype=np.int64)
        chunks[: n % sub_size] += 1

        local_counts = lorentzian_histogram(
            int(chunks[sub_rank]), bins=bins, xmin=xmin, xmax=xmax, rng=rng
        )
        global_counts = np.zeros_like(local_counts)
        subcomm.Allreduce(local_counts, global_counts, op=MPI.SUM)

        subcomm.Free()

    # Keep all ranks synchronized before returning control to caller
    comm.Barrier()

    global_counts = comm.bcast(global_counts if rank == 0 else None, root=0)
    return global_counts


if __name__ == '__main__':

    n = 10**9
    bins = 1000

    t = [1,2,4,8,16,32]


    mpi_strong_times = strong_scaling(run_mpi, n=n, bins=bins, technique='MPI')
    mpi_weak_times = weak_scaling(run_mpi, n_per_div=n, bins=bins, technique='MPI')

    np.savetxt("mpi_strong.txt", mpi_strong_times)
    np.savetxt("mpi_weak.txt", mpi_weak_times)

    speedups = speedup(mpi_strong_times)
    efficiencies = efficiency(speedups,t)




    
