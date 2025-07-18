#!/bin/bash

#SBATCH --partition cpu
#SBATCH --job-name=injection_afgpy_gaussian
#SBATCH -o ./log_nmma_x_fiesta
#SBATCH -e ./err_nmma_x_fiesta

#SBATCH --nodes=1
#SBATCH --ntasks-per-node 30
#SBATCH --cpus-per-task 1

#SBATCH --time=24:00:00

eval "$(conda shell.bash hook)"  # Initialize Conda in the script
conda activate nmma_x_fiesta

mpirun -n 30 python inference_nmma_x_fiesta.py
