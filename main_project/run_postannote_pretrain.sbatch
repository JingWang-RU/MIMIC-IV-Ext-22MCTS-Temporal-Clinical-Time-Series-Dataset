#!/bin/bash
#SBATCH --job-name=pretrain_multi_singlegpu_test
#SBATCH --output=./log/pretrain_multi_test_output_%j.log
#SBATCH --error=./log/pretrain_multi_test_error_%j.log
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=25
#SBATCH --mem=125G
#SBATCH --time=100:00:00

# Create log directory if it doesn't exist
mkdir -p ./log

# Load required modules 
module --ignore_cache load CUDA/12.1

# Activate conda environment
source /data/$USER/conda/etc/profile.d/conda.sh
conda activate nltk_envs

# Set NCCL environment variables for better performance and debugging
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NET
export NCCL_P2P_DISABLE=1
# Set OMP_NUM_THREADS
export OMP_NUM_THREADS=4

# Determine the master node's address (only one node here)
MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
MASTER_PORT=12345  # Choose an appropriate port that is free

# Launch the Python test script using torchrun for distributed training 
torchrun --nnodes=1 \
         --nproc_per_node=4 \
         --node_rank=0 \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         postannote_pretrainfinetuneinfe_mgpu.py
