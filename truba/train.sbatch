#!/bin/bash
#SBATCH -J qwen3-8b-h200                 # Job name for Qwen3, H200
#SBATCH -A teknogrp6
#SBATCH -p kolyoz-cuda
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G                       # Increased system RAM for safety (H200 has 141GB VRAM, but CPU processes need RAM)
#SBATCH -C H200                          # Constraint for H200 GPU
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

# --- Modules (match what you see interactively) ---
module purge
module load comp/python/miniconda3
module load apps/truba-ai/gpu-2024.0
module load lib/cuda/12.4

# --- Conda from module ---
eval "$(conda shell.bash hook)"
conda activate tf-training || { echo "Conda env 'tf-training' not found"; exit 1; }

# --- Runtime env (tuned for H100/H200/NCCL) ---
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" # Ensure quotes are here for safety
export NCCL_DEBUG=WARN
export NCCL_IB_HCA=mlx5
export NCCL_NVLS_ENABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=2
export TRANSFORMERS_NO_TF=1
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONUNBUFFERED=1


# --- Paths ---
# CRITICAL FIX: Removed the stray quote at the end of this line
MODEL="Qwen/Qwen3-8B-Base"
DATA="/arf/scratch/teknogrp6/qwen3/data/tokenized"
OUT="/arf/scratch/teknogrp6/qwen3/checkpoints"

mkdir -p "$(dirname "$OUT")" logs

# Sanity prints (helpful in logs)
echo "--- Environment Info ---" # Added header for clarity
which python
python -V
# This line is now correctly formatted with an f-string and should work.
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')"
echo "------------------------" # Added footer for clarity

# --- Run (use module form so we don't rely on the torchrun shim) ---

# Note on batch_size:
# --batch_size 64 (global) and --micro_batch_size 1 (per-device) on 1 GPU
# means gradient_accumulation_steps will be 64.
# If `auto_micro_batch` successfully increases micro_batch_size,
# gradient_accumulation_steps will be reduced accordingly.
# This setup is valid for gradient accumulation.
# Consider starting with a lower global batch_size for initial testing (e.g., 8 or 16)
# to confirm it runs before scaling up.

  python -u code/training.py \
  --model_name_or_path "$MODEL" \
  --tokenized_dir "$DATA" \
  --out_dir "$OUT" \
  --num_train_epochs 30 \
  --batch_size 64 \
  --micro_batch_size 1 \
  --max_seq_len 2048 \
  --auto_micro_batch \
  --max_grad_norm 1.0 \
  --learning_rate 4e-4 \
  --weight_decay 0.05 \
  --lr_schedule cosine_restart \
  --warmup_ratio 0.02 \
  --num_cycles 8 \
  --save_steps 250 \
  --log_steps 25 \
  --resume \
  --bf16 \
  --use_lora \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.10 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj \
  --eval_ratio 0.05 \
  --eval_every 500 \
  --early_stop_patience 10 \
  --min_delta 0.002 \
  --save_best \
  --load_best_at_end \
  --loader_safe