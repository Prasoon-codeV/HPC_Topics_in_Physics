#!/bin/bash
#SBATCH --job-name=nbody_hybrid
#SBATCH --output=nbody_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --partition=cpu-preempt

module purge
module load gnu12/12.4.0
module load openmpi4/4.1.6
module load ohpc

# OpenMP settings
export OMP_NUM_THREADS=16
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

# MPI info
N_RANKS=$SLURM_NTASKS
THREADS=$OMP_NUM_THREADS

BASE_N=2048
N=$(awk "BEGIN {print int($BASE_N*sqrt($N_RANKS*$THREADS))}")

echo "MPI ranks = $N_RANKS, OpenMP threads per rank = $THREADS"
echo "Running with N = $N bodies"

# Compile
mpicxx -fopenmp -O3 -std=c++17 nbody_mpi_omp.cc -o nbody_mpi_omp

# Use mpirun instead of srun
mpirun -np $N_RANKS \
       --bind-to core \
       --map-by socket \
       ./nbody_mpi_omp $N
