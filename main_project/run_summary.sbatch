#!/bin/bash
#SBATCH --job-name=single_summary
#SBATCH --output=./log/summary_output_%j.log   # Log file for output
#SBATCH --error=./log/summary_error_%j.log    # Log file for errors
#SBATCH --partition=gpu                       # Specify GPU partition
#SBATCH --gres=gpu:a100:1                     # Request 1 P100 GPU
#SBATCH --cpus-per-task=16                    # Number of CPU cores per task
#SBATCH --mem=22G                             # Memory per task
#SBATCH --time=00:30:00                       # Time limit

# Create log directory
mkdir -p ./log

# Load required modules
module --ignore_cache load CUDA/12.1
source /data/$USER/conda/etc/profile.d/conda.sh && source /data/$USER/conda/etc/profile.d/mamba.sh

# Activate conda environment
conda activate nltk_envs

# Run the Python script
time python bw_summary_1.py
