#!/bin/bash
#SBATCH --job-name=vllm_context_gpu           # Job name
#SBATCH --output=./log/vllm_output_%j.log    # Standard output log
#SBATCH --error=./log/vllm_error_%j.log      # Standard error log
#SBATCH --partition=gpu                      # GPU partition
#SBATCH --gres=gpu:a100:4                    # Request 4 GPUs
#SBATCH --cpus-per-task=8                    # Number of CPU cores per task
#SBATCH --nodes=2                          # Number of nodes
#SBATCH --cpus-per-task=16                 # Number of CPU cores per task
#SBATCH --mem=128G                         # Memory per node
#SBATCH --time=48:00:00                    # Time limit
# Load CUDA module
module load CUDA/12.1

# Activate Conda environment
source /data/$USER/conda/etc/profile.d/conda.sh
conda activate nltk_envs

# Print debug information
echo "Job running on $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Allocated GPUs: $(nvidia-smi)"

# Run the Python script
python vllm_context.py

