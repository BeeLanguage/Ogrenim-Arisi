#!/bin/bash
#SBATCH -J llama31-8b-infer
#SBATCH -A teknogrp6
#SBATCH -p kolyoz-cuda
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH -t 00:10:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

module purge
module load comp/python/miniconda3
module load apps/truba-ai/gpu-2024.0
# Eğer gerçekten gerekiyorsa açın; yoksa kaldırın:
module load lib/cuda/12.4

eval "$(conda shell.bash hook)"
# PyTorch odaklı bir env daha sağlıklı olur ama elinizdekiyle devam:
conda activate tf-training || { echo "Conda env 'tf-training' not found"; exit 1; }

# Gürültülü TF loglarını kapat
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0

# Hız/istikrar env
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export NCCL_IB_HCA=mlx5
export NCCL_NVLS_ENABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
# HF cache’i scratch’a al (quota sorunlarını azaltır)
export HF_HOME=/arf/scratch/$USER/hf-cache
export HF_HUB_CACHE=$HF_HOME/hub

# export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets

export TRANSFORMERS_NO_TF=1
export TF_CPP_MIN_LOG_LEVEL=3
export TF_FORCE_GPU_ALLOW_GROWTH=1

mkdir -p "$HF_HOME" logs

# --- Parametreler ---
MODEL="meta-llama/Llama-3.1-8B"                           # veya /arf/scratch/.../merged
LORA="/arf/scratch/teknogrp6/llama31/checkpoints/best"          # merged kullanıyorsanız boş bırakın
INP="/arf/scratch/teknogrp6/llama31/inference/inference_in.jsonl"
OUT="/arf/scratch/teknogrp6/llama31/inference/preds_out.jsonl"

which python
python -V
python -c "import torch; print('PyTorch', torch.__version__, 'CUDA', torch.version.cuda, 'CUDA avail', torch.cuda.is_available())"


python code/inference.py \
  --model_path meta-llama/Llama-3.1-8B \
  --lora_path  "$LORA" \
  --input_jsonl "$INP" \
  --out "$OUT" \
  --out_key output \
  --num_beams 1 --top_p 1 --top_k 0 \
  --repetition_penalty 1.0 --min_new_tokens 2 --max_new_tokens 512
