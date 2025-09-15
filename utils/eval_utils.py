
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from .visualizer import sample_points_from_voxels
from configs.config import TrainConfig
from model.model import multichannel_to_boolean
from third_party.metrics import compute_all_metrics

@torch.no_grad()
def reconstruct_point_clouds(
    module: nn.Module,
    recon_loader: DataLoader,
    gt_loader: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
    n_points: int = 2048,
    threshold: float = 0.5,
    forward_microbatch: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstructs point clouds using inputs from `recon_loader` and ground truth
    from `gt_loader`, iterating through both in parallel.
    """

    module.eval()

    all_recon_pcs, all_gt_pcs = [], []
    device_type = device.type if isinstance(device, torch.device) else "cuda"

    # Zip the loaders to pair reconstruction inputs with ground truth targets
    pbar = tqdm(
        zip(recon_loader, gt_loader),
        desc="Reconstructing point clouds",
        total=min(len(recon_loader), len(gt_loader))
    )
    for (x, _), (_, y) in pbar:  # x from recon_loader, y from gt_loader
        x = x.to(device)

        # --- Forward in micro-batches to avoid upsample overflow ---
        logits_list = []
        step = max(1, forward_microbatch)
        for i in range(0, x.shape[0], step):
            xi = x[i:i + step]
            with torch.amp.autocast(device_type=device_type, enabled=cfg.amp):
                li, *_ = module(xi)
            logits_list.append(li)
        logits = torch.cat(logits_list, dim=0)

        probs = torch.sigmoid(logits)
        voxel_preds = multichannel_to_boolean(probs)

        recon_np = voxel_preds.squeeze(1).cpu().numpy()
        gt_np = y.squeeze(1).cpu().numpy()

        # Iterate only up to the size of the smaller batch to handle the final partial batch
        num_pairs_in_batch = min(recon_np.shape[0], gt_np.shape[0])
        for i in range(num_pairs_in_batch):
            recon_bin = (recon_np[i] >= threshold)
            recon_points = sample_points_from_voxels(
                recon_bin, n_points=n_points, seed=getattr(cfg, "seed", 0)
            ) if recon_bin.any() else np.zeros((n_points, 3), dtype=np.float32)
            all_recon_pcs.append(recon_points)

            gt_bin = (gt_np[i] >= threshold)
            gt_points = sample_points_from_voxels(
                gt_bin, n_points=n_points, seed=getattr(cfg, "seed", 0)
            ) if gt_bin.any() else np.zeros((n_points, 3), dtype=np.float32)
            all_gt_pcs.append(gt_points)

    if not all_recon_pcs:
        print("Warning: No point clouds were reconstructed.")
        return np.zeros((0, n_points, 3)), np.zeros((0, n_points, 3))

    return np.stack(all_recon_pcs), np.stack(all_gt_pcs)


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