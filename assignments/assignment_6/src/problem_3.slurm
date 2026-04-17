#!/bin/bash
#SBATCH --job-name=ode_cpu_gpu
#SBATCH --output=job_outputs/prob3_ode.out
#SBATCH --error=job_outputs/ode_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=00:30:00
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1


module load purge
module load cuda/11.8
module load python/3.10

# source activate jlab

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python problem_3.py 

