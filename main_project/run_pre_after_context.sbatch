#!/bin/bash
#SBATCH --job-name=context_cpu_processing   # Job name
#SBATCH --output=./log/job_output_%j.log  # Log file for standard output
#SBATCH --error=./log/job_error_%j.log    # Log file for standard error
#SBATCH --cpus-per-task=16                 # Request 8 CPUs
#SBATCH --mem=16G                         # Request 32 GB memory
#SBATCH --time=02:00:00                   # Set a time limit (2 hours)
#SBATCH --nodes=1                         # Run on a single node

# Create the log directory
mkdir -p ./log

# Load any necessary modules (if required)
module load python/3.8
source /data/$USER/conda/etc/profile.d/conda.sh && source /data/$USER/conda/etc/profile.d/mamba.sh
conda activate nltk_envs

# Run your Python script
python pre_after_threadpool.py 

