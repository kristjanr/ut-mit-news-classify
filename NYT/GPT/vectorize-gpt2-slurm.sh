#!/bin/bash

# Job Details
#SBATCH --partition=gpu
#SBATCH -J gpt2_emb

# Resources
#SBATCH -t 07:59:00
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:tesla:1

# Actual job command(s)
model=$1
chunk_size=$2
srun python vectorize-gpt2-single.py $model $chunk_size


