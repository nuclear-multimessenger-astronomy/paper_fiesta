#!/bin/bash

#SBATCH --partition gpu
#SBATCH --job-name=injection_afgpy_gaussian
#SBATCH -o ./log_fiesta
#SBATCH -e ./err_fiesta

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=80G

#SBATCH --time=24:00:00

module load miniforge3
eval "$(conda shell.bash hook)"  # Initialize Conda in the script
conda activate fiesta

python inference_fiesta.py
