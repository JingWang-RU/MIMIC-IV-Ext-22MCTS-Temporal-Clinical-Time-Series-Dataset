#!/bin/bash
#SBATCH --job-name=single_annotate
#SBATCH --output=./log/annotate_output122324_%j.log   # Log file for output
#SBATCH --error=./log/annotate_error122324_%j.log    # Log file for errors
#SBATCH --partition=gpu                       # Specify GPU partition
#SBATCH --gres=gpu:a100:1                     # Request 1 P100 GPU
#SBATCH --cpus-per-task=14                     # Number of CPU cores per task
#SBATCH --mem=23G                             # Memory per task
#SBATCH --time=36:00:00                       # Time limit
#SBATCH --array=0-10

# Create log directory
mkdir -p ./log

# Load required modules
module --ignore_cache load CUDA/12.1
source /data/$USER/conda/etc/profile.d/conda.sh

# Activate conda environment
conda activate nltk_envs

# Run the Python script
time python run_annotate_12232024.py
