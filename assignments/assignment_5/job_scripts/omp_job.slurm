#!/bin/bash
#SBATCH --job-name=nbody_omp
#SBATCH --output=nbody_%j.txt
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks-per-node=1     # MPI ranks per node
#SBATCH --cpus-per-task=32       # OpenMP threads per rank
#SBATCH --time=00:15:00         # Max runtime (HH:MM:SS)
#SBATCH --partition=vdi     # Partition/queue

# Load required modules
module purge
module load gnu12/12.4.0
module load openmpi4/4.1.6
module load ohpc

# Set OpenMP threads
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

g++ -fopenmp -O3 -std=c++17 nbody_omp.cc -o nbody_omp

for thread in 1 2 4 8 16 32; do
    export OMP_NUM_THREADS=$thread
    echo "Running nbody with N=4096 and threads=$thread"
    srun --cpu-bind=none ./nbody_omp 4096
done

