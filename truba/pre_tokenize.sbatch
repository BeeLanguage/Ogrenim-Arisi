#!/bin/bash
#SBATCH -J qwen3-tokenize             # Job name for Qwen3 tokenization
#SBATCH -A teknogrp6
#SBATCH -p orfoz                      # CPU partition (assuming 'orfoz' is the correct CPU partition for your group)
#SBATCH -N 1
#SBATCH --cpus-per-task=16            # Request 16 CPU cores (good for `num_proc` in tokenization)
#SBATCH --mem=48G                     # Increased memory. Tokenization can be memory intensive for large datasets.
#SBATCH -t 00:05:00                   # Increased time limit to 2 hours (adjust as needed for your dataset size)
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH -C weka

# --- IMPORTANT: Remove -C H200 (or -C weka if it was accidentally leftover) ---
# The -C H200 or -C weka constraint is for specific hardware or filesystems.
# For CPU-only tokenization, it's generally not needed. If your system requires
# a specific CPU type constraint, check TRUBA docs and add it (e.g., -C broadwell).
# For now, I'm removing it.

set -euo pipefail # Exit immediately if a command exits with a non-zero status or on unbound variables

# --- Modules (match your training sbatch for consistency in environment) ---
module purge
module load comp/python/miniconda3
# module load apps/truba-ai/gpu-2024.0 # Not needed for CPU-only tokenization, remove if it's GPU-specific
# module load lib/cuda/12.4          # Not needed for CPU-only tokenization, remove

# --- Conda from module ---
eval "$(conda shell.bash hook)"
# Assuming 'tf-training' *does* have transformers and datasets installed.
# If not, create a new env or install them: conda create -n llm_cpu_env python=3.10 && conda activate llm_cpu_env && pip install transformers datasets
conda activate tf-training || { echo "Conda env 'tf-training' not found"; exit 1; }

# --- Runtime env ---
export TOKENIZERS_PARALLELISM=false # Recommended for Hugging Face tokenizers

# --- Paths & params ---
# CRITICAL: Use the Qwen3 model name for its tokenizer
MODEL="Qwen/Qwen3-8B-Base"
RAW_DATA="/arf/scratch/teknogrp6/qwen3/data/alpaca.jsonl" # Changed to reflect raw data
TOK_OUT="/arf/scratch/teknogrp6/qwen3/data/tokenized" # Output directory for tokenized data
MAX_LEN=2048 # Max sequence length for tokenization
NUM_PROC="${SLURM_CPUS_PER_TASK:-8}" # Use allocated CPUs, fallback to 8 if not in Slurm

mkdir -p "$TOK_OUT" logs # Ensure output and log directories exist

# Sanity prints for debugging
echo "--- Environment Info ---"
which python
python -V
python -c "import transformers; print('Transformers version:', transformers.__version__)"
python -c "import datasets; print('Datasets version:', datasets.__version__)"
echo "------------------------"

# --- Run the tokenization script ---
echo "Starting tokenization with model: $MODEL"
echo "Raw data: $RAW_DATA"
echo "Output directory: $TOK_OUT"

python code/pre_tokenize.py \
  --model_name "$MODEL" \
  --dataset_path "$RAW_DATA" \
  --out_dir "$TOK_OUT" \
  --max_seq_length "$MAX_LEN" \
  --num_proc "$NUM_PROC" # Pass num_proc to the script

echo "Tokenization job finished."