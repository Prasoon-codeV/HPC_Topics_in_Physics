#!/bin/bash
#SBATCH --job-name=monte-carlo
#SBATCH --output=monte-carlo-output.txt
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --partition=vdi
#SBATCH --mail-user=pxv220016@utdallas.edu
echo "Monte Carlo run for different methods!"

prun python monte_carlo_all.py

