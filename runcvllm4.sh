#!/bin/bash

# ES Fine-tuning with LoRA/vLLM
# Usage: ./run_es.sh

set -e  # Exit on error

# ============================================
# Environment Setup
# ============================================
eval "$(conda shell.bash hook)"
source ~/.bashrc
conda activate es_env
cd ~/ES_LLM

# ============================================
# GPU Cleanup
# ============================================
echo "Cleaning up GPU processes..."

# Target vLLM worker processes
pgrep -f "vllm.worker" | xargs -r kill -9 2>/dev/null || true
pgrep -f "ray::VLLM" | xargs -r kill -9 2>/dev/null || true

# Stop Ray gracefully
ray stop --force 2>/dev/null || true
sleep 2

# Show GPU status
echo "GPU status:"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
    awk -F',' '{printf "GPU %s: %s MB\n", $1, $2}'
echo "=========================================="

# Set visible GPU (vLLM handles multi-GPU internally via tensor_parallel_size)
export CUDA_VISIBLE_DEVICES=0

# ============================================
# Configuration
# ============================================

# Model settings
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
HF_CACHE_DIR="./huggingface_cache"
PRECISION="bfloat16"

# ES Hyperparameters
NUM_ITERATIONS=1000
POPULATION_SIZE=30
SIGMA=0.0001
ALPHA=0.001
INITIAL_SEED=33
CHECKPOINT_INTERVAL=10

# Generation settings
MAX_NEW_TOKENS=100
DO_SAMPLE=""  # Add "--do_sample" to enable sampling

# LoRA settings
LORA_R=32
LORA_ALPHA=64
LORA_DROPOUT=0.1
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

# vLLM settings
VLLM_GPU_MEMORY=0.7
VLLM_MAX_MODEL_LEN=2048
VLLM_TENSOR_PARALLEL=1  # Set to 4 for 4-GPU setup

# ============================================
# Run Training
# ============================================

echo "=========================================="
echo "ES Fine-tuning with LoRA + vLLM"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Population: $POPULATION_SIZE | Iterations: $NUM_ITERATIONS"
echo "Sigma: $SIGMA | Alpha: $ALPHA"
echo "LoRA: r=$LORA_R, alpha=$LORA_ALPHA"
echo "Checkpoint interval: $CHECKPOINT_INTERVAL"
echo "Tensor parallel: $VLLM_TENSOR_PARALLEL"
echo "=========================================="
echo ""

python conciseness/cvllm3.2.py \
    --model_name "$MODEL_NAME" \
    --hf_cache_dir "$HF_CACHE_DIR" \
    --precision "$PRECISION" \
    --num_iterations $NUM_ITERATIONS \
    --population_size $POPULATION_SIZE \
    --sigma $SIGMA \
    --alpha $ALPHA \
    --max_new_tokens $MAX_NEW_TOKENS \
    --initial_seed $INITIAL_SEED \
    --checkpoint_interval $CHECKPOINT_INTERVAL \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --lora_target_modules "$LORA_TARGET_MODULES" \
    --vllm_gpu_memory_utilization $VLLM_GPU_MEMORY \
    --vllm_max_model_len $VLLM_MAX_MODEL_LEN \
    --vllm_tensor_parallel_size $VLLM_TENSOR_PARALLEL \
    $DO_SAMPLE

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="