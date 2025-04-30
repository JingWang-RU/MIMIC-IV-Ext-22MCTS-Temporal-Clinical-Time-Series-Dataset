#!/bin/bash
#SBATCH --job-name=post_pretrain
#SBATCH --output=./log/pretrain_output_%j.log   # Log file for output
#SBATCH --error=./log/pretrain_error_%j.log    # Log file for errors
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=25                   # Number of CPU cores per task
#SBATCH --mem=50G                             # Memory per task
#SBATCH --time=380:00:00                       # Time limit

# Create log directory                      # Request 1 P100 GPU              # Specify GPU partition  #SBATCH --partition=gpu #SBATCH --gres=gpu:a100:1
mkdir -p ./log
# Load required modules 
module --ignore_cache load CUDA/12.1
source /data/$USER/conda/etc/profile.d/conda.sh

# Activate conda environment
conda activate nltk_envs

# Run the Python script
#time python postannote_test.py
#time python postannote_dataprocessing.py
time python postannote_pretrainfinetuneinfe.py
