#!/bin/bash

# ES Fine-tuning Script with LoRA/vLLM support
# Usage: ./run_es.sh [lora|fullparam]

set -e  # Exit on error

# ============================================
# Environment Setup
# ============================================
eval "$(conda shell.bash hook)"
source ~/.bashrc
conda activate es_env
cd ~/ES_LLM

# ============================================
# Safe GPU Cleanup (won't kill this script)
# ============================================
echo "Cleaning up GPU processes..."

# Only target vLLM worker processes specifically
pgrep -f "vllm.worker" | xargs -r kill -9 2>/dev/null || true
pgrep -f "ray::VLLM" | xargs -r kill -9 2>/dev/null || true

# Stop Ray gracefully
ray stop --force 2>/dev/null || true
sleep 2

# Verify cleanup (don't do nuclear kill that might hit this script)
echo "GPU status after cleanup:"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
    awk -F',' '{printf "GPU %s: %s MB\n", $1, $2}'
echo "=========================================="

# Set visible GPU explicitly (vLLM handles multi-GPU internally)
export CUDA_VISIBLE_DEVICES=0

# Default mode
MODE="${1:-lora}"

# Common settings
MODEL_NAME="Qwen/Qwen2.5-7B"
HF_CACHE_DIR="./huggingface_cache"
PRECISION="bfloat16"

# ES Hyperparameters
NUM_ITERATIONS=1000
POPULATION_SIZE=30
SIGMA=0.001
ALPHA=0.0005
MAX_NEW_TOKENS=100
INITIAL_SEED=33

# LoRA settings
LORA_R=32
LORA_ALPHA=64
LORA_DROPOUT=0.1
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

# vLLM settings
VLLM_GPU_MEMORY=0.7
VLLM_MAX_MODEL_LEN=2048
VLLM_TENSOR_PARALLEL=1

# Full-param settings
GPU_THREADS=4

echo "=========================================="
echo "ES Fine-tuning Script"
echo "=========================================="
echo "Mode: $MODE"
echo "Model: $MODEL_NAME"
echo "Population: $POPULATION_SIZE"
echo "Iterations: $NUM_ITERATIONS"
echo "Sigma: $SIGMA, Alpha: $ALPHA"
echo "=========================================="
echo ""

if [ "$MODE" == "lora" ]; then
    echo "Running in LoRA+vLLM mode..."
    echo "LoRA rank: $LORA_R, alpha: $LORA_ALPHA"
    echo ""
    
    python conciseness/cvllm2.py \
        --use_lora \
        --model_name "$MODEL_NAME" \
        --hf_cache_dir "$HF_CACHE_DIR" \
        --precision "$PRECISION" \
        --num_iterations $NUM_ITERATIONS \
        --population_size $POPULATION_SIZE \
        --sigma $SIGMA \
        --alpha $ALPHA \
        --max_new_tokens $MAX_NEW_TOKENS \
        --initial_seed $INITIAL_SEED \
        --lora_r $LORA_R \
        --lora_alpha $LORA_ALPHA \
        --lora_dropout $LORA_DROPOUT \
        --lora_target_modules "$LORA_TARGET_MODULES" \
        --vllm_gpu_memory_utilization $VLLM_GPU_MEMORY \
        --vllm_max_model_len $VLLM_MAX_MODEL_LEN \
        --vllm_tensor_parallel_size $VLLM_TENSOR_PARALLEL \
        --verbose

elif [ "$MODE" == "fullparam" ]; then
    echo "Running in Full-Param+Transformers mode..."
    echo "GPU threads: $GPU_THREADS"
    echo ""
    
    python conciseness/cvllm2.py \
        --model_name "$MODEL_NAME" \
        --hf_cache_dir "$HF_CACHE_DIR" \
        --precision "$PRECISION" \
        --num_iterations $NUM_ITERATIONS \
        --population_size $POPULATION_SIZE \
        --sigma $SIGMA \
        --alpha $ALPHA \
        --max_new_tokens $MAX_NEW_TOKENS \
        --initial_seed $INITIAL_SEED \
        --gpu_threads $GPU_THREADS \
        --verbose

else
    echo "Error: Invalid mode '$MODE'"
    echo "Usage: $0 [lora|fullparam]"
    echo ""
    echo "Examples:"
    echo "  $0 lora       # Run with LoRA+vLLM (recommended)"
    echo "  $0 fullparam  # Run with full-param+Transformers (original)"
    exit 1
fi

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="