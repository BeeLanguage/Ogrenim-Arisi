#!/bin/bash
#SBATCH -J llama31-8b-h100
#SBATCH -A teknogrp6
#SBATCH -p kolyoz-cuda
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH -C H200
#SBATCH -t 00:10:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

# --- Modules (match what you see interactively) ---
module purge
module load comp/python/miniconda3
module load apps/truba-ai/gpu-2024.0
module load lib/cuda/12.4

# --- Conda from module (DON'T hardcode $HOME/miniconda3) ---
eval "$(conda shell.bash hook)"
conda activate tf-training || { echo "Conda env 'tf-training' not found"; exit 1; }

# --- Runtime env (H100/NCCL-friendly) ---
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export NCCL_IB_HCA=mlx5
export NCCL_NVLS_ENABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=2
export TRANSFORMERS_NO_TF=1
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONUNBUFFERED=1


# --- Paths ---
MODEL="meta-llama/Llama-3.1-8B"
DATA="/arf/scratch/teknogrp6/llama31/data/tokenized"
OUT="/arf/scratch/teknogrp6/llama31/final"

mkdir -p "$(dirname "$OUT")" logs

# Sanity prints (helpful in logs)
which python
python -V
python -c "import torch; print('PyTorch', torch.__version__, 'CUDA', torch.version.cuda)"

# --- Run (use module form so we don't rely on the torchrun shim) ---

# Fast merge on a big NVIDIA/AMD GPU
python code/ckpt_maker.py \
  --base meta-llama/Llama-3.1-8B \
  --lora /arf/scratch/teknogrp6/llama31/checkpoints/best \
  --out ./merged_model.ckpt \
  --dtype bf16