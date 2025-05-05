#!/bin/bash
#SBATCH --job-name=train_moe
#SBATCH --output=logs/train_moe_%j.log
#SBATCH --error=logs/train_moe_%j.err
#SBATCH --partition=gpu-v100              # <-- Fixed partition name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3               # 3 tasks for 3 GPUs
#SBATCH --gres=gpu:3                      # Request 3 GPUs
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=vinaysanjay.chaudhari@slu.edu

# Load conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cv_3.9.7

# Optional: reduce CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

# Make sure logs directory exists
mkdir -p logs

echo "Job started at: $(date)"
start_time=$(date +%s)

# Run training
srun python train.py

end_time=$(date +%s)
echo "Job ended at: $(date)"
echo "Total runtime: $((end_time - start_time)) seconds"