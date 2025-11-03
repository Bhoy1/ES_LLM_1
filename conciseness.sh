#!/bin/bash

# ES Fine-tuning Runner Script
# Usage: ./run_es_finetuning.sh

# Set error handling
set -e

# Configuration
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
HF_CACHE_DIR="huggingface_cache"
PRECISION="bf16"  # Options: bf16, fp16, fp32
GPU_THREADS=1
NUM_GPUS=1
CHECKPOINT_INTERVAL=10  # Evaluate on test set every N iterations

# Optional: Set verbose mode
VERBOSE_FLAG=""  # Add "--verbose" to enable verbose logging

# Create cache directory if it doesn't exist
mkdir -p "$HF_CACHE_DIR"

# Export Python unbuffered mode for real-time output
export PYTHONUNBUFFERED=1

# Run with accelerate
echo "Starting ES fine-tuning with $NUM_GPUS GPUs and $GPU_THREADS threads per GPU..."
echo "Model: $MODEL_NAME"
echo "Precision: $PRECISION"
echo "Checkpoint interval: $CHECKPOINT_INTERVAL"
echo ""

accelerate launch \
    --num_processes=$NUM_GPUS \
    --mixed_precision=$PRECISION \
    conciseness/es_fullparam.py \
    --model_name "$MODEL_NAME" \
    --hf_cache_dir "$HF_CACHE_DIR" \
    --precision "$PRECISION" \
    --gpu_threads $GPU_THREADS \
    --checkpoint_interval $CHECKPOINT_INTERVAL \
    $VERBOSE_FLAG

echo ""
echo "Training complete!"