#!/bin/bash
#SBATCH -J qwen3-merge                 # Job name for Qwen3 LoRA merge
#SBATCH -A teknogrp6
#SBATCH -p kolyoz-cuda                # GPU partition, as merging usually loads the base model onto GPU
#SBATCH -N 1
#SBATCH --gpus-per-node=1             # Request 1 GPU (to load the base model)
#SBATCH --cpus-per-task=16             # 8 CPU cores should be sufficient for merging
#SBATCH --mem=60G                     # Increased memory. Qwen3-8B + LoRA can consume significant RAM/VRAM during merge.
#SBATCH -C H200                       # Constraint for H200 GPU
#SBATCH -t 01:00:00                   # Increased time limit to 1 hour (adjust as needed if base model download/load is slow)
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail # Exit immediately if a command exits with a non-zero status or on unbound variables

# --- Modules (match your training sbatch environment) ---
module purge
module load comp/python/miniconda3
module load apps/truba-ai/gpu-2024.0 # Essential for PyTorch with CUDA on TRUBA
module load lib/cuda/12.4          # Essential for PyTorch with CUDA on TRUBA

# --- Conda from module ---
eval "$(conda shell.bash hook)"
conda activate tf-training || { echo "Conda env 'tf-training' not found"; exit 1; }

# --- Runtime env (H200/NCCL-friendly) ---
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" # Ensure quotes for safety
export NCCL_DEBUG=WARN
export NCCL_IB_HCA=mlx5
export NCCL_NVLS_ENABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=2
export TRANSFORMERS_NO_TF=1
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONUNBUFFERED=1

# --- Paths ---
# CRITICAL: Specify the Qwen3 base model
BASE_MODEL="Qwen/Qwen3-8B-Base"
# Path to your BEST LoRA checkpoint directory from training
LORA_CHECKPOINT_DIR="/arf/scratch/teknogrp6/qwen3/checkpoints/best"
# Output directory for the *fully merged* Qwen3 model
MERGE_OUT_DIR="/arf/scratch/teknogrp6/qwen3/merge_results"

mkdir -p "$MERGE_OUT_DIR" logs # Ensure output and log directories exist

# Sanity prints (helpful in logs)
echo "--- Environment Info ---"
which python
python -V
# CRITICAL FIX: Changed to f-string to prevent bash syntax error
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')"
echo "------------------------"

# --- Run the merge script ---
echo "Starting LoRA merge process..."
echo "Base Model: $BASE_MODEL"
echo "LoRA Checkpoint: $LORA_CHECKPOINT_DIR"
echo "Merged Model Output: $MERGE_OUT_DIR"

python -u code/merge.py \
  --base "$BASE_MODEL" \
  --lora_dir "$LORA_CHECKPOINT_DIR" \
  --out_dir "$MERGE_OUT_DIR" \
  --dtype bf16 \
  --device cuda:0 \
  --max_shard_size "40GB"

echo "Merge job finished."