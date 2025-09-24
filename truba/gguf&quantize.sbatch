#!/bin/bash
#SBATCH -J qwen3-gguf-quantize        # Job name for Qwen3 GGUF conversion & quantization
#SBATCH -A teknogrp6
#SBATCH -p orfoz                      # Use a CPU partition for this task (e.g., 'orfoz')
#SBATCH -N 1
#SBATCH --cpus-per-task=16            # Request many CPU cores for conversion and quantization
#SBATCH --mem=96G                     # CRITICAL: Request ample RAM. 8B model + overhead needs a lot.
#SBATCH -t 04:00:00                   # Generous time limit (4 hours, adjust based on observed runtime)
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

# --- User-defined paths and settings ---
LLAMA_CPP_DIR="/arf/scratch/teknogrp6/llama.cpp_repo" # Path where you cloned/built llama.cpp
MERGED_MODEL_DIR="/arf/scratch/teknogrp6/qwen3/merge_results" # Directory of your merged Qwen3 model
GGUF_OUTPUT_DIR="/arf/scratch/teknogrp6/qwen3/gguf_quantized" # Output directory for GGUF files

# --- Create necessary directories ---
mkdir -p "$GGUF_OUTPUT_DIR" logs

# --- Load Modules (Use your confirmed available versions here) ---
module purge
module load comp/python/miniconda3
module load comp/gcc/14.1.0 # <<< Use your *actual* available GCC version
module load comp/cmake/3.31.1   # <<< Use your *actual* available CMake version

# --- Activate Conda Environment ---
eval "$(conda shell.bash hook)"
conda activate tf-training || { echo "Conda env 'tf-training' not found"; exit 1; }

# --- Navigate to llama.cpp directory ---
echo "Navigating to llama.cpp directory: $LLAMA_CPP_DIR"
if [ ! -d "$LLAMA_CPP_DIR" ]; then
    echo "ERROR: llama.cpp directory not found at $LLAMA_CPP_DIR. Please check path or clone it."
    exit 1
fi
cd "$LLAMA_CPP_DIR" || { echo "ERROR: Failed to change directory to $LLAMA_CPP_DIR"; exit 1; }

# --- Build llama.cpp using CMake (FIXED PROCESS for CPU instructions) ---
echo "Checking for llama.cpp build..."

BUILD_DIR="build"
QUANTIZE_EXECUTABLE="$BUILD_DIR/bin/quantize" # Common location after CMake build

if [ ! -f "$QUANTIZE_EXECUTABLE" ]; then
    echo "llama.cpp executables not found. Attempting to build with CMake..."

    # Ensure a clean build by removing any existing build directory
    echo "Removing existing build directory: $BUILD_DIR"
    rm -rf "$BUILD_DIR" || { echo "WARN: Failed to remove old build directory. May cause issues."; } # Added warning if rm fails

    mkdir -p "$BUILD_DIR" || { echo "ERROR: Failed to create build directory $BUILD_DIR"; exit 1; }
    cd "$BUILD_DIR" || { echo "ERROR: Failed to change directory to $BUILD_DIR"; exit 1; }

    # --- CRITICAL FIX: Direct CPU architecture targeting ---
    # We will explicitly disable ALL AVX512 features and set a strong baseline.
    # The 'vpdpbusd' and 'vpdpwssd' errors are AVX512_VNNI.
    # The 'vmovw' errors could be related to AVX512_IFMA or other wider vector operations.
    # We aim for AVX2/FMA/F16C, which are widely supported on most modern server CPUs.
    echo "Configuring CMake for a compatible CPU architecture (forcing -march, disabling AVX512)..."
    cmake .. \
        -DGGML_AVX512=OFF \
        -DGGML_AVX512_VNNI=OFF \
        -DGGML_AVX512_VBMI=OFF \
        -DGGML_AVX2=ON \
        -DGGML_FMA=ON \
        -DGGML_F16C=ON \
        -DGGML_K_QUANT=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_FLAGS="-march=haswell" \
        -DCMAKE_CXX_FLAGS="-march=haswell" \
        || { echo "ERROR: CMake configuration failed."; exit 1; }
    # Note: 'haswell' is a good baseline as it supports AVX2/FMA/F16C but NOT AVX512.
    # If 'haswell' still causes issues, try '-march=sandybridge' (older AVX1) or even omit -march for generic.

    # Build (compile) with multiple jobs
    make -j$(nproc) || { echo "ERROR: CMake build failed."; exit 1; }

    echo "llama.cpp build complete."
    # After building, switch back to the main llama.cpp_repo directory for running python scripts
    cd .. || { echo "ERROR: Failed to change directory back to $LLAMA_CPP_DIR"; exit 1; }
else
    echo "llama.cpp executables already exist. Skipping build."
fi

# --- Step 1: Convert the merged model to GGUF (float16 type) ---
BASE_GGUF_MODEL="$GGUF_OUTPUT_DIR/qwen3-8b-merged-f16.gguf"
echo "Starting conversion to GGUF (f16 type)..."
python convert.py "$MERGED_MODEL_DIR" \
    --outfile "$GGUF_OUTPUT_DIR/qwen3-8b-merged-f16.gguf" \
    --outtype f16 \
    --vocab-only-from-tokenizer \
    --trust-remote-code
echo "Conversion to GGUF (f16) complete."


# --- Step 2: Quantize the GGUF model to different formats ---
echo "Starting quantization..."
QUANTIZE_PATH="$LLAMA_CPP_DIR/$QUANTIZE_EXECUTABLE"

# Q5_K_M quantization
echo "Quantizing to Q5_K_M..."
"$QUANTIZE_PATH" "$BASE_GGUF_MODEL" "$GGUF_OUTPUT_DIR/qwen3-8b-merged-Q5_K_M.gguf" Q5_K_M
echo "Q5_K_M quantization complete."

echo "All GGUF conversion and quantization jobs finished."
echo "Quantized models are located in: $GGUF_OUTPUT_DIR"