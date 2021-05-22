#!/bin/bash

# Job Details
#SBATCH --partition=gpu
#SBATCH -J gpt-t-4

# Resources
#SBATCH -t 06:45:00
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:tesla:1
#SBATCH -o /gpfs/space/home/roosild/.slurm/log-%j.out # STDOUT

# Actual job command(s)
srun python tokenize-cutoff.py


