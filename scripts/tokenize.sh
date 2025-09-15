#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
# Set the path to your trained model checkpoint
CHECKPOINT_PATH="runs/shapenet_runs/ShapeNet_VQ_2025-09-14_21-05-38/best.pt"

# Set the directory where you want to save the token files
OUTPUT_DIR="./storage/tokenized_data"

# Model parameters (must match the trained model)
SIZE=128
N_EMBED=512
EMBED_DIM=64

# --- Script ---
echo "Starting dataset tokenization..."
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Output Directory: ${OUTPUT_DIR}"

uv run python tokenize.py \
    --checkpoint "${CHECKPOINT_PATH}" \
    --out-dir "${OUTPUT_DIR}" \
    --size ${SIZE} \
    --n-embed ${N_EMBED} \
    --embed-dim ${EMBED_DIM} \
    --batch-size 16 \
    --num-workers 8

echo "Tokenization script finished successfully."