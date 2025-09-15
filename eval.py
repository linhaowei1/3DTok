import os
import argparse
import torch
from torch.utils.data import DataLoader
from collections import OrderedDict
from dataclasses import dataclass
import json

# Local imports
from data.shapenet_dataset import ShapeNetVoxelDataset
from model.model import VQModel3D
from utils.eval_utils import reconstruct_point_clouds, compute_metrics

@dataclass
class EvalConfig:
    """Configuration for the evaluation script."""
    checkpoint_path: str
    size: int = 128
    batch_size: int = 8
    num_workers: int = 16
    amp: bool = True
    seed: int = 0
    n_embed: int = 512
    embed_dim: int = 64
    n_points: int = 2048
    metrics_batch_size: int = 32

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained VQModel3D on ShapeNet.")
    
    # --- Evaluation Arguments ---
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint (.pt file).')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for evaluation.')
    parser.add_argument('--num-workers', type=int, default=16, help='Number of workers for the DataLoader.')
    parser.add_argument('--n-points', type=int, default=2048, help='Number of points to sample from voxels for metrics.')
    parser.add_argument('--out-dir', type=str, default=None, help='Optional: Directory to save evaluation results (JSON).')
    parser.add_argument('--no-amp', action='store_true', help='Disable automatic mixed precision.')
    
    # --- Model Arguments (must match the trained model) ---
    parser.add_argument('--size', type=int, default=128, help='Voxel grid resolution.')
    parser.add_argument('--n-embed', type=int, default=512, help='Number of codebook embeddings.')
    parser.add_argument('--embed-dim', type=int, default=64, help='Dimension of codebook embeddings.')

    args = parser.parse_args()

    # --- 1. SETUP DEVICE AND CONFIG ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cfg = EvalConfig(
        checkpoint_path=args.checkpoint,
        size=args.size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        amp=not args.no_amp,
        n_embed=args.n_embed,
        embed_dim=args.embed_dim,
        n_points=args.n_points
    )

    # --- 2. PREPARE DATASET ---
    train_set = ShapeNetVoxelDataset(split="train")
    val_set = ShapeNetVoxelDataset(split="val")

    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=False
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg.batch_size, 
        num_workers=cfg.num_workers, pin_memory=True, drop_last=False
    )
    print(f"Loaded validation dataset with {len(val_set)} samples.")

    # --- 3. BUILD AND LOAD MODEL ---
    # The model architecture must be identical to the one used for training.
    ddconfig = dict(
        ch=64, out_ch=8, ch_mult=(1, 1, 2, 4), num_res_blocks=2,
        attn_resolutions=(16,), dropout=0.0, resamp_with_conv=True,
        in_channels=8, resolution=cfg.size, z_channels=64,
    )
    model = VQModel3D(ddconfig, n_embed=cfg.n_embed, embed_dim=cfg.embed_dim)
    
    print(f"Loading checkpoint from: {cfg.checkpoint_path}")
    state_dict = torch.load(cfg.checkpoint_path, map_location="cpu")
    state_dict = state_dict['model']
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # --- 4. RUN EVALUATION ---
    # Reconstruct point clouds from the validation set
    recon_pcs, gt_pcs = reconstruct_point_clouds(
        module=model,
        recon_loader=train_loader,
        gt_loader=val_loader,
        cfg=cfg,
        device=device,
        n_points=cfg.n_points,
        forward_microbatch=args.batch_size
    )

    # Compute evaluation metrics
    if recon_pcs.shape[0] > 0:
        metrics = compute_metrics(
            recon_pcs=recon_pcs,
            gt_pcs=gt_pcs,
            device=device,
            metrics_batch_size=cfg.metrics_batch_size
        )

        # --- 5. PRINT AND SAVE RESULTS ---
        print("\n--- Evaluation Metrics ---")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.6f}")
            else:
                print(f"{key}: {value}")
        print("--------------------------\n")

        # Save results to a JSON file if an output directory is specified
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            results_path = os.path.join(args.out_dir, "evaluation_results.json")
            with open(results_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            print(f"Results saved to {results_path}")
    else:
        print("Evaluation failed: No point clouds were generated.")

if __name__ == '__main__':
    main()