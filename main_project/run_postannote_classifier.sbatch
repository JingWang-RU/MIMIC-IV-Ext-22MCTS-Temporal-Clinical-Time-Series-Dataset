#!/bin/bash
#SBATCH --job-name=bert_ft
#SBATCH --output=./log/bert_ft_output_%j.log   # Log file for output
#SBATCH --error=./log/bert_ft_error_%j.log    # Log file for errors
#SBATCH --partition=gpu                       # Specify GPU partition
#SBATCH --gres=gpu:a100:1                     # Request 1 P100 GPU
#SBATCH --cpus-per-task=2                   # Number of CPU cores per task
#SBATCH --mem=12G                             # Memory per task
#SBATCH --time=10:00:00                       # Time limit

# Create log directory
mkdir -p ./log

# Load required modules
module --ignore_cache load CUDA/12.1
source /data/$USER/conda/etc/profile.d/conda.sh

# Activate conda environment
conda activate nltk_envs

# Run the Python script
#time python postannote_classifier_bert.py
time python postannote_classifier_ftbert.py

