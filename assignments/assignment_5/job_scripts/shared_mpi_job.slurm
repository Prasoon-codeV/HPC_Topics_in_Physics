#!/bin/bash
#SBATCH -J nbody-mpi-shared      # Job name
#SBATCH -p cpu-preempt            # Partition
#SBATCH --nodes=1         # Number of nodes
#SBATCH --ntasks-per-node=32  # Max MPI ranks per node
#SBATCH -c 1              # CPUs per rank
#SBATCH -t 00:30:00       # Walltime
#SBATCH --exclusive       # Avoid interference
#SBATCH --output=outputs/parallel-output.txt
#SBATCH --mail-user=pxv220016@utdallas.edu

# Load necessary modules
module purge
module load gnu12/12.4.0
module load openmpi4/4.1.6

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_PROC_BIND=TRUE
export OMP_PLACES=cores

# Compile the MPI C++ code
echo "Compiling nbody_shared_mpi.cc..."
mpicxx -O3 -std=c++17 nbody_shared_mpi.cc -o nbody_shared_mpi

# Run the executable with different ranks
for ranks in 1 2 4 8 16 32; do
    echo "----------------------------------"
    echo "Running nbody with N=4096 and ranks=$ranks"
    mpiexec -n $ranks ./nbody_mpi 4096
done
