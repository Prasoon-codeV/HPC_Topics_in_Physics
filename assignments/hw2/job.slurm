#!/bin/bash
#SBATCH --job-name=hello-test
#SBATCH --output=hello-output.txt
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --partition=cpu-preempt
#SBATCH --mail-user=pxv220016@utdallas.edu
echo "Hello from Ganymede2 HPC cluster!"

prun python plot.py

