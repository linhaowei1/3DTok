
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from configs.config import TrainConfig
from model.model import multichannel_to_boolean
from third_party.metrics import compute_all_metrics

def _batch_sample_from_voxels_torch(
    voxels_bool: torch.Tensor,
    n_points: int,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Uniformly samples points from occupied voxels for a whole batch on the GPU.

    Args:
        voxels_bool: A boolean tensor of shape (B, D, H, W).
        n_points: The number of points to sample for each item in the batch.
        normalize: If True, normalizes points to the [0, 1] range.

    Returns:
        A tensor of shape (B, n_points, 3) containing sampled points (xyz).
    """
    if not voxels_bool.is_cuda:
        print("Warning: _batch_sample_from_voxels_torch is much more efficient on a GPU.")

    B, D, H, W = voxels_bool.shape
    device = voxels_bool.device

    # Pre-allocate output tensor with zeros for cases with no occupied voxels
    all_points = torch.zeros((B, n_points, 3), dtype=torch.float32, device=device)

    for i in range(B):
        # Find coordinates of occupied voxels for the i-th sample
        coords = torch.nonzero(voxels_bool[i], as_tuple=False)  # Shape: (N, 3) -> z, y, x
        num_occupied = coords.shape[0]

        if num_occupied == 0:
            continue

        # Determine if we need to sample with replacement
        replace = num_occupied < n_points

        # Generate random indices to select from the occupied voxel coordinates
        if replace:
            rand_indices = torch.randint(0, num_occupied, (n_points,), device=device)
        else:
            # Efficiently sample without replacement
            perm = torch.randperm(num_occupied, device=device)
            rand_indices = perm[:n_points]

        # Select the coordinates and add random jitter
        selected_coords_zyx = coords[rand_indices].float()
        jitter = torch.rand((n_points, 3), device=device)
        points_zyx = selected_coords_zyx + jitter

        if normalize:
            scale = torch.tensor([D, H, W], dtype=torch.float32, device=device)
            points_zyx /= scale

        # Convert zyx -> xyz and store in the batch output
        all_points[i] = torch.flip(points_zyx, dims=[1])

    return all_points

@torch.no_grad()
def reconstruct_point_clouds(
    module: nn.Module,
    recon_loader: DataLoader,
    gt_loader: DataLoader,
    cfg: "TrainConfig",
    device: torch.device,
    n_points: int = 2048,
    threshold: float = 0.5,
    forward_microbatch: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstructs point clouds using a vectorized, GPU-accelerated approach.
    """
    module.eval()

    all_recon_pcs, all_gt_pcs = [], []
    device_type = device.type if isinstance(device, torch.device) else "cuda"

    pbar = tqdm(
        zip(recon_loader, gt_loader),
        desc="Reconstructing point clouds",
        total=min(len(recon_loader), len(gt_loader))
    )
    for (x, _), (_, y) in pbar:
        x = x.to(device)
        y = y.to(device) # Move ground truth to GPU for batch processing

        # --- Forward in micro-batches to avoid upsample overflow ---
        logits_list = []
        step = max(1, forward_microbatch)
        for i in range(0, x.shape[0], step):
            xi = x[i : i + step]
            with torch.amp.autocast(device_type=device_type, enabled=cfg.amp):
                li, *_ = module(xi)
            logits_list.append(li)
        logits = torch.cat(logits_list, dim=0)

        probs = torch.sigmoid(logits)
        # Assuming multichannel_to_boolean returns a tensor of shape (B, 1, D, H, W)
        voxel_preds = multichannel_to_boolean(probs).squeeze(1)

        # --- Vectorized Sampling on GPU ---
        # Process the entire batch at once, removing the inefficient inner Python loop
        recon_bin_batch = (voxel_preds >= threshold)
        recon_points = _batch_sample_from_voxels_torch(
            recon_bin_batch, n_points=n_points
        )
        all_recon_pcs.append(recon_points.cpu().numpy())

        gt_bin_batch = (y.squeeze(1) >= threshold)
        gt_points = _batch_sample_from_voxels_torch(
            gt_bin_batch, n_points=n_points
        )
        all_gt_pcs.append(gt_points.cpu().numpy())

    if not all_recon_pcs:
        print("Warning: No point clouds were reconstructed.")
        return np.zeros((0, n_points, 3)), np.zeros((0, n_points, 3))

    # Use np.concatenate as we are now appending arrays of shape (B, n_points, 3)
    return np.concatenate(all_recon_pcs, axis=0), np.concatenate(all_gt_pcs, axis=0)

def compute_metrics(
    recon_pcs: np.ndarray,
    gt_pcs: np.ndarray,
    device: torch.device,
    metrics_batch_size: int = 32,
    accelerated_cd: bool = False,
) -> dict:
    """Computes point cloud evaluation metrics given reconstructed and ground truth arrays."""
    recon_pcs_tensor = torch.from_numpy(recon_pcs).float().to(device)
    gt_pcs_tensor = torch.from_numpy(gt_pcs).float().to(device)

    print(f"\nComputing metrics for {recon_pcs_tensor.shape[0]} point cloud pairs...")
    results = compute_all_metrics(
        sample_pcs=recon_pcs_tensor,
        ref_pcs=gt_pcs_tensor,
        batch_size=metrics_batch_size,
        accelerated_cd=accelerated_cd,
        verbose=False
    )

    return results