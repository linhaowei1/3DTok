# training/train_utils.py

"""
Generic utility functions for distributed training and metrics.
"""

import os
import torch
import torch.distributed as dist
import random
import numpy as np

def setup_ddp():
    """Initializes the distributed process group."""
    use_ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if not use_ddp:
        return 0, 1, 0  # rank, world_size, local_rank

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank

def cleanup_ddp():
    """Cleans up the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process():
    """Checks if the current process is the main one (rank 0)."""
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0

def barrier():
    """Synchronizes all processes."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

def setup_seeds(seed, rank):
    """Sets random seeds for reproducibility across all processes."""
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Optional: for full determinism, but may impact performance
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def reduce_mean(t: torch.Tensor):
    """Reduces a tensor across all processes by averaging."""
    if not (dist.is_available() and dist.is_initialized()):
        return t
    rt = t.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

def compute_iou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    """
    Computes Intersection over Union (IoU) for binary voxel predictions.
    Args:
        pred: Predicted voxels (B, 1, D, H, W), boolean or float in {0,1}.
        target: Target voxels (B, 1, D, H, W), boolean or float in {0,1}.
    Returns:
        Batch mean IoU.
    """
    pred_bool = pred.bool()
    target_bool = target.bool()
    
    intersection = (pred_bool & target_bool).sum(dim=(1, 2, 3, 4)).float()
    union = (pred_bool | target_bool).sum(dim=(1, 2, 3, 4)).float()
    
    iou = (intersection + eps) / (union + eps)
    return iou.mean()