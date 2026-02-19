#!/bin/bash
#SBATCH --job-name=eigen-sweep
#SBATCH --output=eigen-%a_%A.out
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --array=0-11
#SBATCH --partition=cpu-preempt
#SBATCH --mail-user=pxv220016@utdallas.edu


# Optional: Load modules
# module unload python
source ~/miniforge3/bin/activate
# module load python
conda activate jlab

set -euo pipefail

# Sweep values
Ns=(10 20 40 80 160 320)
potentials=(well harmonic)

nN=${#Ns[@]}
np=${#potentials[@]}
ntasks=$((nN*np))

if [ "$SLURM_ARRAY_TASK_ID" -ge "$ntasks" ]; then
	echo "Invalid TASK_ID: $SLURM_ARRAY_TASK_ID exceeds total tasks $ntasks"
	exit 1
fi

iN=$(( SLURM_ARRAY_TASK_ID / np ))
ip=$(( SLURM_ARRAY_TASK_ID % np ))

N=${Ns[$iN]}
pot=${potentials[$ip]}
mkdir -p results

time python src/eigen.py --N "$N" --potential "$pot" --n-eigs 5


