#!/bin/bash
#
# ShapeNet VQModel3D Evaluation Script
#
# This script launches a single-GPU evaluation job.
#
# Usage:
#   1. Set the CHECKPOINT_PATH variable below to your trained model file.
#   2. Run the script:
#      chmod +x eval.sh
#      ./eval.sh
#
#   - To run on a specific GPU (e.g., GPU 2):
#     GPU_ID=2 ./eval.sh
#

# --- CONFIGURATION ---
# !!! IMPORTANT !!!
# Set this to the path of the checkpoint file (.pt) you want to evaluate.
CHECKPOINT_PATH="runs/shapenet_runs/ShapeNet_VQ_2025-09-14_21-05-38/best.pt"

# Set the GPU to use for evaluation.
GPU_ID=0

# --- Evaluation Parameters ---
BATCH_SIZE=8       # Batch size for inference. Adjust based on GPU memory.
NUM_WORKERS=16     # Number of CPU workers for data loading.
N_POINTS=2048      # Number of points to sample for metrics calculation.

# --- Model Parameters ---
# These parameters MUST match the parameters used to train the model specified in CHECKPOINT_PATH.
VOXEL_SIZE=128
N_EMBED=512
EMBED_DIM=64

# --- SCRIPT LOGIC ---
# Do not edit below this line.

# Check if the checkpoint file exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint file not found at '$CHECKPOINT_PATH'"
    echo "Please update the CHECKPOINT_PATH variable in this script."
    exit 1
fi

# Create an output directory for results inside the checkpoint's run folder
RUN_DIR=$(dirname $(dirname "$CHECKPOINT_PATH"))
OUT_DIR="$RUN_DIR/evaluation"
mkdir -p "$OUT_DIR"

echo "--- Starting Evaluation ---"
echo "GPU ID:          $GPU_ID"
echo "Checkpoint:      $CHECKPOINT_PATH"
echo "Batch Size:      $BATCH_SIZE"
echo "Output Dir:      $OUT_DIR"
echo "Model Params:    size=$VOXEL_SIZE, n_embed=$N_EMBED, embed_dim=$EMBED_DIM"
echo "---------------------------"

# Set the visible CUDA device for the python script
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Launch the evaluation script
uv run python eval.py \
    --checkpoint="$CHECKPOINT_PATH" \
    --batch-size=$BATCH_SIZE \
    --num-workers=$NUM_WORKERS \
    --n-points=$N_POINTS \
    --out-dir="$OUT_DIR" \
    --size=$VOXEL_SIZE \
    --n-embed=$N_EMBED \
    --embed-dim=$EMBED_DIM

echo "--- Evaluation Finished. Results saved in $OUT_DIR ---"