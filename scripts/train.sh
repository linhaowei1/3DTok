#!/bin/bash
#
# ShapeNet VQModel3D Training Script
#
# This script launches a distributed training job using torchrun.
# Key parameters can be configured in the "CONFIGURATION" section below.
#
# Usage:
#   - To run on a single machine with all available GPUs:
#     chmod +x train.sh
#     ./train.sh
#
#   - To run on a single machine with a specific number of GPUs (e.g., 2):
#     NPROC_PER_NODE=2 ./train.sh
#

# --- CONFIGURATION ---
# Set the number of GPUs to use per node. Defaults to all available GPUs if not set.
# You can override this from the command line, e.g., `NPROC_PER_NODE=2 ./train.sh`
if [ -z "$NPROC_PER_NODE" ]; then
  # Corrected the command to ensure only the first line is read, preventing errors
  # from nvidia-smi providing multi-line output.
  NPROC_PER_NODE=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n 1)
fi

# --- Training Hyperparameters ---
EPOCHS=500
BATCH_SIZE=2 # This is the batch size *per GPU*. Total batch size will be BATCH_SIZE * NPROC_PER_NODE.
LEARNING_RATE=1e-4
GRAD_ACCUM=1       # Gradient accumulation steps.
AMP_ENABLED=false   # Set to false to disable Automatic Mixed Precision (adds --no-amp flag).

# --- Model Parameters ---
N_EMBED=512      # Number of embeddings in the codebook.
EMBED_DIM=64     # Dimension of each embedding vector.
VOXEL_SIZE=128

# --- Logging and Checkpointing ---
# A unique run name is generated using the current date and time.
RUN_NAME="ShapeNet_VQ_$(date +'%Y-%m-%d_%H-%M-%S')"
OUT_DIR="runs/shapenet_runs" # Base directory for all runs.
WANDB_PROJECT="3DTok-ShapeNet-Experiments"

# Set to "offline" to run wandb without syncing to the cloud.
# WANDB_MODE="offline" 
WANDB_MODE="online"

# --- Distributed Training Setup (for multi-node, otherwise defaults are fine) ---
MASTER_ADDR="localhost"
MASTER_PORT="29502"

# --- SCRIPT LOGIC ---
# Do not edit below this line unless you know what you are doing.

# Create the full output directory for the current run
RUN_OUT_DIR="$OUT_DIR/$RUN_NAME"
mkdir -p "$RUN_OUT_DIR"
echo "--- Starting training run: $RUN_NAME ---"
echo "Using $NPROC_PER_NODE GPUs per node."
echo "Configuration:"
echo "  - Epochs: $EPOCHS"
echo "  - Per-GPU Batch Size: $BATCH_SIZE"
echo "  - Total Batch Size: $(($BATCH_SIZE * $NPROC_PER_NODE))"
echo "  - Learning Rate: $LEARNING_RATE"
echo "  - Output Directory: $RUN_OUT_DIR"
echo "  - Wandb Project: $WANDB_PROJECT"

# Construct the --no-amp flag if AMP is disabled
AMP_FLAG=""
if [ "$AMP_ENABLED" = false ]; then
    AMP_FLAG="--no-amp"
fi

# Set Wandb mode
export WANDB_MODE

# Launch the distributed training job using torchrun
# - --nproc_per_node: Number of processes (GPUs) to launch on this machine.
# - --rdzv_backend: Rendezvous backend for process coordination.
# - --rdzv_endpoint: The master address and port for coordination.
uv run python -m torch.distributed.run \
    --nproc_per_node=$NPROC_PER_NODE \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py \
    --epochs=$EPOCHS \
    --batch-size=$BATCH_SIZE \
    --lr=$LEARNING_RATE \
    --grad-accum=$GRAD_ACCUM \
    --size=$VOXEL_SIZE \
    --n-embed=$N_EMBED \
    --embed-dim=$EMBED_DIM \
    --out-dir="$RUN_OUT_DIR" \
    --wandb-project="$WANDB_PROJECT" \
    --wandb-name="$RUN_NAME" \
    $AMP_FLAG

echo "--- Training run $RUN_NAME finished. ---"
