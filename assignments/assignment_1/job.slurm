#!/bin/bash
#SBATCH --job-name=hello-test
#SBATCH --output=hello-output.txt
#SBATCH --time=0:05:00
#SBATCH --ntasks=1
#SBATCH --partition=cpu-preempt
echo "Hello from Ganymede2 HPC cluster!"
