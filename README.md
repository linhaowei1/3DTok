# 3DTok

A VQ-based 3D Tokenizer implementation for ShapeNet data.

## Overview

3DTok is a PyTorch implementation of a Vector Quantization (VQ) based 3D tokenizer trained on ShapeNet voxel data. The project includes a VQ-VAE model that learns to encode 3D voxel representations into discrete tokens and reconstruct them back to voxel space.

## Features

- VQ-VAE 3D model architecture
- Support for distributed training (DDP)
- Weights & Biases integration for experiment tracking
- ShapeNet voxel dataset handling
- Configurable training parameters

## Project Structure

```
3DTok/
├── train.py              # Main training script
├── configs/
│   └── config.py         # Training configuration
├── data/
│   └── shapenet_dataset.py  # ShapeNet dataset loader
├── model/
│   └── model.py          # VQModel3D implementation
├── utils/
│   ├── trainer.py        # Training loop
│   ├── train_utils.py    # Training utilities
│   ├── wandb_utils.py    # Weights & Biases integration
│   └── visualizer.py     # Visualization utilities
├── third_party/          # External dependencies
└── storage/              # Data storage (ShapeNet processed data)
```

## Installation

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

## Usage

### Training

```bash
python train.py \
    --size 128 \
    --batch-size 4 \
    --epochs 100 \
    --lr 1e-4 \
    --n-embed 512 \
    --embed-dim 256 \
    --wandb-project "3DTok-ShapeNet"
```

### Key Arguments

- `--size`: Voxel grid size (default: 128)
- `--batch-size`: Training batch size
- `--epochs`: Number of training epochs
- `--lr`: Learning rate
- `--n-embed`: Number of embedding vectors
- `--embed-dim`: Embedding dimension
- `--wandb-project`: Weights & Biases project name

## Requirements

- Python >= 3.13
- PyTorch >= 2.8.0
- CUDA-capable GPU
- See `pyproject.toml` for full dependency list

## License

MIT License - see LICENSE file for details.
