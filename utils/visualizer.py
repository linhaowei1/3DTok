"""
Visualization utilities for 3D voxel tokenizer training (point cloud generation).
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

# Local project imports
from configs.config import TrainConfig

# --- HELPER FUNCTIONS (UNCHANGED) ---

def _save_point_cloud_ply(path: str, points: np.ndarray):
    """Saves a point cloud to a simple ASCII PLY file."""
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n"
    )
    with open(path, 'w') as f:
        f.write(header)
        np.savetxt(f, points, fmt='%.6f')


def sample_points_from_voxels(
    voxels_bool: np.ndarray,
    n_points: int = 2048,
    normalize: bool = True,
    seed: int | None = None
) -> np.ndarray:
    """
    Uniformly samples points from occupied voxels with random jitter.
    """
    v = np.asarray(voxels_bool, dtype=bool)
    assert v.ndim == 3, "voxels_bool must be a 3D boolean array"

    rng = np.random.default_rng(seed)
    flat_true_idx = np.flatnonzero(v.ravel())
    if flat_true_idx.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    replace = flat_true_idx.size < n_points
    chosen = rng.choice(flat_true_idx, size=n_points, replace=replace)

    zyx = np.column_stack(np.unravel_index(chosen, v.shape)).astype(np.float32)
    jitter = rng.random((n_points, 3), dtype=np.float32)
    points_zyx = zyx + jitter

    if normalize:
        scale = np.array(v.shape, dtype=np.float32)
        points_zyx /= scale

    points = points_zyx[:, ::-1].copy()  # zyx -> xyz
    return points


def _visualize_matplotlib_pointcloud(path: str, points_xyz: np.ndarray):
    """Renders and saves a point cloud visualization using Matplotlib."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    if points_xyz.shape[0] > 0:
        ax.scatter(points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2], s=2, alpha=0.9)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect((1, 1, 1))
    ax.set_title("Point Cloud Visualization")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

# --- REVISED VISUALIZATION FUNCTION ---

def visualize_reconstructions(
    recon_voxels: np.ndarray,
    gt_voxels: np.ndarray,
    out_dir: str,
    epoch: int,
    cfg: TrainConfig,
    threshold: float = 0.5,
):
    """
    Creates point cloud visualizations from pre-generated voxel grids without
    performing any model inference.
    
    Args:
        recon_voxels (np.ndarray): Batch of reconstructed voxel probabilities.
        gt_voxels (np.ndarray): Batch of ground truth boolean voxels.
        out_dir (str): Directory to save visualizations.
        epoch (int): Current epoch number for file naming.
        cfg (TrainConfig): Training configuration for seeding.
        threshold (float): Threshold to binarize reconstructed voxels.
    """
    # Only run on the main process in a distributed setup
    if hasattr(torch.distributed, 'is_initialized') and torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return

    os.makedirs(out_dir, exist_ok=True)
    num_to_vis = recon_voxels.shape[0]

    # Generate and save point clouds for each example
    for i in range(num_to_vis):
        volumes_to_process = {"recon": recon_voxels[i], "gt": gt_voxels[i]}
        for name, volume_data in volumes_to_process.items():
            try:
                # Binarize the volume (ground truth is already binary)
                volume_bin = (volume_data >= threshold)
                if not volume_bin.any():
                    print(f"[Visualization] No occupied voxels for e{epoch}_s{i}_{name}, skipping.")
                    continue

                points = sample_points_from_voxels(volume_bin, n_points=4096, normalize=True, seed=cfg.seed + i)
                
                base_path = os.path.join(out_dir, f"vis_e{epoch:03d}_s{i:02d}_{name}")
                _save_point_cloud_ply(f"{base_path}.ply", points)
                _visualize_matplotlib_pointcloud(f"{base_path}.png", points)

            except Exception as e:
                print(f"[Visualization] Failed for e{epoch}_s{i}_{name}: {e}")