#!/bin/bash
# Launch script for run_forge.py
# This follows the same pattern as quick0.sh

set -e

echo "Activating conda environment..."
conda activate forge

echo "Changing to forge directory..."
cd /home/ubuntu/arihan/weight-transfer/forge

echo "Setting up environment variables..."
export HYPERACTOR_CODEC_MAX_FRAME_LENGTH=134217728
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}
export WANDB_MODE=offline

# Optional: Set custom model and paths
export MODEL_ID=${MODEL_ID:-"Qwen/Qwen2.5-0.5B-Instruct"}
export CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-"./checkpoints"}
export LATEST_POINTER=${LATEST_POINTER:-"./checkpoints/LATEST"}
export TOTAL_OUTER_STEPS=${TOTAL_OUTER_STEPS:-"100"}
export CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-"20"}
export LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE:-"2"}
export SEQ_LEN=${SEQ_LEN:-"1024"}
export LEARNING_RATE=${LEARNING_RATE:-"5e-6"}

echo "Environment variables:"
echo "  MODEL_ID=$MODEL_ID"
echo "  CHECKPOINT_ROOT=$CHECKPOINT_ROOT"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  TOTAL_OUTER_STEPS=$TOTAL_OUTER_STEPS"

echo "Running run_forge.py..."
python ../arihan/run_forge.py
