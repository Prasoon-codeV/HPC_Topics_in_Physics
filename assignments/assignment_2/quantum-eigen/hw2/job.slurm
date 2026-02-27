#!/bin/bash
#SBATCH --job-name=monte-carlo
#SBATCH --output=hello-output.txt
#SBATCH --time=07:00:00
#SBATCH --ntasks=1
#SBATCH --partition=cpu-preempt
#SBATCH --mail-user=pxv220016@utdallas.edu
echo "Running all methods for quantum-eigen!"



prun python plot.py

