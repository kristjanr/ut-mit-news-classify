#!/bin/bash

# Job Details
#SBATCH --partition=gpu
#SBATCH -J gpt2_emb

# Resources
#SBATCH -t 06:45:00
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:tesla:1
#SBATCH -o ./slurm/log-%j.out # STDOUT

# Actual job command(s)
srun python vectorize-gpt2-cutoff.py "$@"


