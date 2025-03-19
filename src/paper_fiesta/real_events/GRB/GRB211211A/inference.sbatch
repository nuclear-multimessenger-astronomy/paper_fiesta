#!/bin/bash

#SBATCH --partition gpu
#SBATCH --job-name=inference_GRB211211A
#SBATCH -o ./log
#SBATCH -e ./err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=80G

#SBATCH --time=24:00:00

conda activate fiesta

python inference_afgpy.py
