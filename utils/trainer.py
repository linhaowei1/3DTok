# utils/trainer.py

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

### ADDED ###
# Helper function to compute gradient norm
def get_grad_norm(parameters, norm_type=2.0):
    """Computes the total norm of the gradients of parameters."""
    parameters = [p for p in parameters if p.grad is not None]
    if not parameters:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

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
        ### REVISED ###
        # Added grad_norm to tracked metrics
        running_metrics = torch.zeros(4, device=self.device) # [loss, rec_loss, commit_loss, grad_norm]
        grad_norm_steps = 0

        data_iterator = self.train_loader
        if is_main_process():
            data_iterator = tqdm(data_iterator, desc=f"Epoch {epoch} Training")
        
        self.optimizer.zero_grad(set_to_none=True)
        
        for i, (x, _) in enumerate(data_iterator):
            x = x.to(self.device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=self.cfg.amp):
                recon, target, commit_loss, _ = self.model(x)
                rec_loss = self.rec_loss_fn(recon.float(), target.float())
                
                if torch.isnan(recon).any():
                    print("!!! NaN detected in model reconstruction output !!!")
                    raise SystemExit("Stopping training due to NaN in model output.")

                loss = rec_loss + self.cfg.beta_commit * commit_loss
            
            if torch.isnan(loss):
                print(f"!!! NaN loss detected! rec_loss: {rec_loss.item()}, commit_loss: {commit_loss.item()} !!!")
                raise SystemExit("Stopping training due to NaN loss.")

            self.scaler.scale(loss / self.cfg.grad_accum).backward()

            if (i + 1) % self.cfg.grad_accum == 0:
                self.scaler.unscale_(self.optimizer)
                
                ### REVISED ###
                # Calculate grad norm before clipping
                grad_norm = get_grad_norm(self.model.parameters())
                running_metrics[3] += grad_norm.item()
                grad_norm_steps += 1
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            
            running_metrics[0] += loss.item()
            running_metrics[1] += rec_loss.item()
            running_metrics[2] += commit_loss.item()
        
        # --- Codebook Usage Aggregation ---
        barrier() # Ensure all processes have finished the epoch's training steps
        # Get usage counts from the local model replica
        local_codebook_usage = self.model.module.get_codebook_usage()
        # Sum the usage counts across all DDP processes
        torch.distributed.all_reduce(local_codebook_usage, op=torch.distributed.ReduceOp.SUM)
        # Now, `local_codebook_usage` on all processes holds the total usage for the epoch
        total_usage = local_codebook_usage.cpu()
        active_codes = (total_usage > 0).sum().item()
        percent_active = (active_codes / total_usage.numel()) * 100 if total_usage.numel() > 0 else 0.0

        # Aggregate metrics across all processes
        avg_metrics = reduce_mean(running_metrics)
        
        # Correctly average the accumulated values
        num_batches = len(self.train_loader)
        avg_loss = avg_metrics[0] / num_batches
        avg_rec_loss = avg_metrics[1] / num_batches
        avg_commit_loss = avg_metrics[2] / num_batches
        avg_grad_norm = avg_metrics[3] / grad_norm_steps if grad_norm_steps > 0 else 0.0

        train_stats = {
            'loss': avg_loss,
            'reconstruction_loss': avg_rec_loss,
            'commitment_loss': avg_commit_loss,
            'grad_norm': avg_grad_norm,
            'codebook_usage': total_usage,
            'active_codes_percent': percent_active
        }
        
        if is_main_process():
            print(f"[Train][Epoch {epoch}] Loss: {train_stats['loss']:.4f}, Rec: {train_stats['reconstruction_loss']:.4f}, Commit: {train_stats['commitment_loss']:.4f}, GradNorm: {train_stats['grad_norm']:.4f}, Codebook Usage: {train_stats['active_codes_percent']:.2f}%")
        
        return train_stats

    @torch.no_grad()
    def _evaluate(self, epoch):
        self.model.eval()
        ### REVISED ###
        # Track more detailed loss components
        running_metrics = torch.zeros(4, device=self.device)  # [loss, rec_loss, commit_loss, iou]
        
        data_iterator = self.val_loader
        if is_main_process():
            data_iterator = tqdm(data_iterator, desc=f"Epoch {epoch} Evaluation")

        for x, _ in data_iterator:
            x = x.to(self.device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=self.cfg.amp):
                recon_logits, target, commit_loss, _ = self.model(x)
                rec_loss = self.rec_loss_fn(recon_logits, target)
                loss = rec_loss + self.cfg.beta_commit * commit_loss
                
                voxel_preds = multichannel_to_boolean(torch.sigmoid(recon_logits))
                voxel_gt = multichannel_to_boolean(target)
                iou = compute_iou(voxel_preds, voxel_gt)
            
            ### REVISED ###
            # Accumulate all relevant metrics
            running_metrics += torch.tensor([loss.item(), rec_loss.item(), commit_loss.item(), iou.item()], device=self.device)

        avg_metrics = reduce_mean(running_metrics) / len(self.val_loader)
        
        ### REVISED ###
        # Return a more detailed dictionary for validation
        val_stats = {
            'loss': avg_metrics[0].item(),
            'reconstruction_loss': avg_metrics[1].item(),
            'commitment_loss': avg_metrics[2].item(),
            'iou': avg_metrics[3].item()
        }
        
        if is_main_process():
            print(f"[Eval][Epoch {epoch}] Loss: {val_stats['loss']:.4f}, IoU: {val_stats['iou']:.4f}, Rec: {val_stats['reconstruction_loss']:.4f}, Commit: {val_stats['commitment_loss']:.4f}")
            
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

            self.model.module.reset_codebook_usage()
            
            self.train_sampler.set_epoch(epoch)
            self.val_sampler.set_epoch(epoch)

            train_metrics = self._train_one_epoch(epoch)
            val_metrics = self._evaluate(epoch)
            
            barrier()

            if is_main_process():
                if self.wandb_run:
                    ### REVISED ###
                    # Pass the optimizer to log learning rate
                    log_epoch_metrics(self.wandb_run, epoch, train_metrics, val_metrics, self.optimizer)
                
                is_best = val_metrics['iou'] > self.best_iou
                if is_best:
                    self.best_iou = val_metrics['iou']
                
                self._save_checkpoint(epoch, is_best)

            if epoch % 10 == 0 or epoch == self.cfg.epochs:
                self._visualize_epoch(epoch)

            barrier()