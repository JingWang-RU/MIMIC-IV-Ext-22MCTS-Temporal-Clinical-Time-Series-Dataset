#!/bin/bash
#SBATCH --job-name=pretrain_gpt256
#SBATCH --output=./log/pretrain_gpt_256output_%j.log
#SBATCH --error=./log/pretrain_gpt_256%j.log
#SBATCH --partition=gpu
#SBATCH --time=1-12
#SBATCH --gres=lscratch:10,gpu:a100:1
#SBATCH --mem=12g
#SBATCH --cpus-per-task=8

# Create log directory if it doesn't exist
mkdir -p ./log

# Load required modules 
module --ignore_cache load CUDA/12.1
export HF_DATASETS_DISABLE_PROGRESS_BARS=1

# Activate conda environment
source /data/$USER/conda/etc/profile.d/conda.sh
conda activate nltk_envs
python postannote_gpt0131.py
