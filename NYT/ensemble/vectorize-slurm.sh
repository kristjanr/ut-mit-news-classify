#!/bin/bash

# Job Details
#SBATCH -J tfidf

# Resources
#SBATCH -t 07:59:00
#SBATCH --ntasks=1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=1

# Actual job command(s)
srun python vectorize.py


