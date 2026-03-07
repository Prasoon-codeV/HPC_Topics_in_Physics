#!/bin/bash
#SBATCH -J hw4-mpi_lorentz
#SBATCH -p vdi
#SBATCH --nodes=1 # nodes
#SBATCH --ntasks-per-node=64 # total ranks
#SBATCH -c 1 # CPUs per rank
#SBATCH -t 00:30:00 # walltime
#SBATCH --exclusive # avoid interference
#SBATCH --output=parallel-output.txt
#SBATCH --mail-user=pxv220016@utdallas.edu


# Activate your environment
# source ~/miniforge3/etc/profile.d/conda.sh
# conda activate jlab
 
module purge
# module load gcc
# module load python
module load openmpi4/4.1.6

# Activate conda properly
# source ~/miniforge3/etc/profile.d/conda.sh
# conda activate jlab

# Avoid oversubscription by BLAS/NumPy threads
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# srun --cpu-bind=none python mpi_lorentz.py --n 10000000 --bins 100
# srun --cpu-bind=none python  mpi_lorentz.py
mpiexec -n 64 python mpi_lorentz.py
