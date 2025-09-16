"""
Configuration dataclass for 3D voxel tokenizer training.
"""

from dataclasses import dataclass


@dataclass
class TrainConfig:
    """Configuration for 3D voxel tokenizer training."""
    size: int = 128
    batch_size: int = 1
    num_workers: int = 4
    epochs: int = 5
    lr: float = 1e-4
    grad_accum: int = 1
    beta_commit: float = 0.25
    out_dir: str = "runs/"
    amp: bool = True
    seed: int = 0
    n_embed: int = 1024
    embed_dim: int = 64
    category: str = "airplane"