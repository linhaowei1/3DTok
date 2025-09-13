# training/trainer.py

import os
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from .train_utils import reduce_mean, compute_iou, barrier, is_main_process
from .wandb_utils import log_epoch_metrics
from configs.config import TrainConfig
from model.model import multichannel_to_boolean
from utils.visualizer import visualize_reconstructions

class Trainer:
    def __init__(self, model, optimizer, scaler, train_loader, val_loader, 
                 train_sampler, val_sampler, cfg: TrainConfig, device, 
                 rank, wandb_run=None, resume_path=None):
        self.model_unwrapped = model
        self.model = DDP(model, device_ids=[device.index], find_unused_parameters=False)
        self.optimizer = optimizer
        self.scaler = scaler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.cfg = cfg
        self.device = device
        self.rank = rank
        self.wandb_run = wandb_run

        self.start_epoch = 1
        self.best_iou = -1.0
        self.rec_loss_fn = nn.BCEWithLogitsLoss(reduction='mean')

        if resume_path:
            self._load_checkpoint(resume_path)

    def _load_checkpoint(self, path):
        if not os.path.isfile(path):
            if is_main_process():
                print(f"=> No checkpoint found at '{path}', starting from scratch.")
            return

        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
        checkpoint = torch.load(path, map_location=map_location)
        
        self.model.module.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scaler.load_state_dict(checkpoint['scaler'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_iou = checkpoint.get('best_iou', -1.0)
        
        if is_main_process():
            print(f"=> Loaded checkpoint '{path}' (epoch {checkpoint['epoch']})")

    def _save_checkpoint(self, epoch, is_best):
        if not is_main_process():
            return
            
        # Save a general checkpoint for resuming
        if epoch % 5 == 0 or epoch == self.cfg.epochs:
            checkpoint_path = os.path.join(self.cfg.out_dir, 'checkpoint.pt')
            print(f"[Checkpoint] Saving checkpoint to {checkpoint_path}")
            torch.save({
                'epoch': epoch, 'model': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(), 'scaler': self.scaler.state_dict(),
                'best_iou': self.best_iou, 'config': self.cfg.__dict__
            }, checkpoint_path)
        
        # Save the best model based on IoU
        if is_best:
            best_model_path = os.path.join(self.cfg.out_dir, 'best.pt')
            print(f"[Checkpoint] New best IoU={self.best_iou:.4f}, model saved to {best_model_path}")
            torch.save({'model': self.model.module.state_dict(), 'config': self.cfg.__dict__}, best_model_path)

    def _train_one_epoch(self, epoch):
        self.model.train()
        running_metrics = torch.zeros(3, device=self.device)  # [loss, rec_loss, commit_loss]

        # Setup progress bar for the main process
        data_iterator = self.train_loader
        if is_main_process():
            data_iterator = tqdm(data_iterator, desc=f"Epoch {epoch} Training")
        
        self.optimizer.zero_grad(set_to_none=True)
        
        for i, (x, _) in enumerate(data_iterator):
            x = x.to(self.device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=self.cfg.amp):
                recon, target, commit_loss, _ = self.model(x)
                rec_loss = self.rec_loss_fn(recon, target)
                loss = rec_loss + self.cfg.beta_commit * commit_loss
            
            self.scaler.scale(loss / self.cfg.grad_accum).backward()

            if (i + 1) % self.cfg.grad_accum == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            
            running_metrics += torch.tensor([loss.item(), rec_loss.item(), commit_loss.item()], device=self.device)
        
        # Aggregate metrics across all processes
        avg_metrics = reduce_mean(running_metrics) / len(self.train_loader)
        
        train_stats = {
            'loss': avg_metrics[0].item(),
            'reconstruction_loss': avg_metrics[1].item(),
            'commitment_loss': avg_metrics[2].item()
        }
        
        if is_main_process():
            print(f"[Train][Epoch {epoch}] Loss: {train_stats['loss']:.4f}, Rec: {train_stats['reconstruction_loss']:.4f}, Commit: {train_stats['commitment_loss']:.4f}")
        
        return train_stats

    @torch.no_grad()
    def _evaluate(self, epoch):
        self.model.eval()
        running_metrics = torch.zeros(2, device=self.device)  # [loss, iou]
        
        data_iterator = self.val_loader
        if is_main_process():
            data_iterator = tqdm(data_iterator, desc=f"Epoch {epoch} Evaluation")

        for x, _ in data_iterator:
            x = x.to(self.device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=self.cfg.amp):
                recon_logits, target, commit_loss, _ = self.model(x)
                loss = self.rec_loss_fn(recon_logits, target) + self.cfg.beta_commit * commit_loss
                
                # Convert to boolean voxels for IoU calculation
                voxel_preds = multichannel_to_boolean(torch.sigmoid(recon_logits))
                voxel_gt = multichannel_to_boolean(target)
                iou = compute_iou(voxel_preds, voxel_gt)
            
            running_metrics += torch.tensor([loss.item(), iou.item()], device=self.device)

        avg_metrics = reduce_mean(running_metrics) / len(self.val_loader)
        
        val_stats = {'loss': avg_metrics[0].item(), 'iou': avg_metrics[1].item()}
        
        if is_main_process():
            print(f"[Eval][Epoch {epoch}] Loss: {val_stats['loss']:.4f}, IoU: {val_stats['iou']:.4f}")
            
        return val_stats

    @torch.no_grad()
    def _visualize_epoch(self, epoch):
        if not is_main_process():
            return

        print(f"[Visualization] Generating visualizations for epoch {epoch}...")
        self.model.eval()
        try:
            x, y = next(iter(self.val_loader))
            x = x.to(self.device)

            with torch.amp.autocast('cuda', enabled=self.cfg.amp):
                recon_logits, _, _, _ = self.model(x)
            
            recon_probs = torch.sigmoid(recon_logits)
            voxel_preds = multichannel_to_boolean(recon_probs)

            visualize_reconstructions(
                recon_voxels=voxel_preds.squeeze(1).cpu().numpy(),
                gt_voxels=y.squeeze(1).cpu().numpy(),
                out_dir=os.path.join(self.cfg.out_dir, "visuals"),
                epoch=epoch,
                cfg=self.cfg
            )
        except StopIteration:
            print("[Visualization] Validation loader is empty, skipping.")
        except Exception as e:
            print(f"[Visualization] An error occurred: {e}")

    def train(self):
        for epoch in range(self.start_epoch, self.cfg.epochs + 1):
            self.train_sampler.set_epoch(epoch)
            self.val_sampler.set_epoch(epoch)

            train_metrics = self._train_one_epoch(epoch)
            val_metrics = self._evaluate(epoch)
            
            barrier() # Sync before checkpointing and logging

            if is_main_process():
                if self.wandb_run:
                    log_epoch_metrics(self.wandb_run, epoch, train_metrics, val_metrics)
                
                is_best = val_metrics['iou'] > self.best_iou
                if is_best:
                    self.best_iou = val_metrics['iou']
                
                self._save_checkpoint(epoch, is_best)

            if epoch % 5 == 0 or epoch == self.cfg.epochs:
                self._visualize_epoch(epoch)

            barrier()