# train.py

import os
import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Local imports
from data.shapenet_dataset import ShapeNetVoxelDataset
from model.model import VQModel3D
from utils.trainer import Trainer
from utils.train_utils import setup_ddp, setup_seeds, cleanup_ddp
from utils.wandb_utils import init_wandb, finish_wandb
from configs.config import TrainConfig

def main():
    parser = argparse.ArgumentParser()
    # --- Arguments remain the same, no changes needed here ---
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--grad-accum', type=int, default=1)
    parser.add_argument('--beta-commit', type=float, default=0.25)
    parser.add_argument('--out-dir', type=str, default='runs/shapenet_tokenizer')
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n-embed', type=int, default=512)
    parser.add_argument('--embed-dim', type=int, default=256)
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from')

    # Wandb arguments
    parser.add_argument('--wandb-project', type=str, default='3DTok-ShapeNet')
    parser.add_argument('--wandb-name', type=str, default=None)
    parser.add_argument('--wandb-offline', action='store_true')
    parser.add_argument('--wandb-tags', type=str, nargs='*', default=None)
    parser.add_argument('--wandb-notes', type=str, default=None)

    args = parser.parse_args()

    # --- 1. SETUP DDP, DEVICE, AND SEEDS ---
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    setup_seeds(args.seed, rank)
    os.makedirs(args.out_dir, exist_ok=True)

    # --- 2. PREPARE CONFIG AND DATASET ---
    cfg = TrainConfig(
        size=args.size, batch_size=args.batch_size, num_workers=args.num_workers,
        epochs=args.epochs, lr=args.lr, grad_accum=args.grad_accum,
        beta_commit=args.beta_commit, out_dir=args.out_dir, amp=not args.no_amp,
        seed=args.seed, n_embed=args.n_embed, embed_dim=args.embed_dim
    )
    
    train_set = ShapeNetVoxelDataset(split="train")
    val_set = ShapeNetVoxelDataset(split="val")

    train_sampler = DistributedSampler(train_set, shuffle=True, drop_last=False)
    val_sampler = DistributedSampler(val_set, shuffle=False, drop_last=False)

    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, sampler=train_sampler,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=False
    )
    val_loader = DataLoader(
        val_set, batch_size=1, sampler=val_sampler,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=False
    )

    if rank == 0:
        print("Train dataset stats:", train_set.get_stats())
        print("Val dataset stats:", val_set.get_stats())
        print(f"Starting ShapeNet training with world_size={world_size}, amp={cfg.amp}")

    # --- 3. BUILD MODEL AND OPTIMIZER ---
    # Model building is moved here from train_utils, as it's specific to this training script.
    ddconfig = dict(
        ch=64, out_ch=8, ch_mult=(1, 1, 2, 4), num_res_blocks=2,
        attn_resolutions=(16,), dropout=0.0, resamp_with_conv=True,
        in_channels=8, resolution=cfg.size, z_channels=64,
    )
    model = VQModel3D(ddconfig, n_embed=cfg.n_embed, embed_dim=cfg.embed_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999), weight_decay=0.05)
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.amp)
    
    # --- 4. INITIALIZE WANDB (Main process only) ---
    wandb_run = None
    if rank == 0:
        wandb_config = cfg.__dict__
        wandb_config.update({
            'train_samples': len(train_set), 'val_samples': len(val_set),
            'world_size': world_size
        })
        wandb_run = init_wandb(
            project_name=args.wandb_project, run_name=args.wandb_name, config=wandb_config,
            tags=args.wandb_tags, notes=args.wandb_notes, offline=args.wandb_offline
        )

    # --- 5. INITIALIZE AND RUN THE TRAINER ---
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        train_loader=train_loader,
        val_loader=val_loader,
        train_sampler=train_sampler,
        val_sampler=val_sampler,
        cfg=cfg,
        device=device,
        rank=rank,
        wandb_run=wandb_run,
        resume_path=args.resume
    )
    
    trainer.train()

    # --- 6. CLEANUP ---
    if rank == 0 and wandb_run:
        finish_wandb(wandb_run)
    
    cleanup_ddp()
    print("ShapeNet training complete.")

if __name__ == '__main__':
    main()