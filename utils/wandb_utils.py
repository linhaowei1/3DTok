# utils/wandb_utils.py

import wandb
import torch

def init_wandb(project_name, run_name, config, tags, notes, offline):
    """Initializes a new WandB run."""
    return wandb.init(
        project=project_name,
        name=run_name,
        config=config,
        tags=tags,
        notes=notes,
        mode="offline" if offline else "online"
    )

def log_epoch_metrics(run, epoch, train_metrics, val_metrics, optimizer):
    """
    Logs metrics for a completed epoch to WandB, now including codebook usage.
    """
    if not run:
        return

    lr = optimizer.param_groups[0]['lr']
    
    # --- Prepare metrics dictionary for logging ---
    metrics_to_log = {
        'epoch': epoch,
        'learning_rate': lr,
        # Training metrics
        'train/loss': train_metrics['loss'],
        'train/reconstruction_loss': train_metrics['reconstruction_loss'],
        'train/commitment_loss': train_metrics['commitment_loss'],
        'train/grad_norm': train_metrics['grad_norm'],
        # Validation metrics
        'val/loss': val_metrics['loss'],
        'val/reconstruction_loss': val_metrics['reconstruction_loss'],
        'val/commitment_loss': val_metrics['commitment_loss'],
        'val/iou': val_metrics['iou'],
    }


    # --- Log codebook usage statistics ---
    # Log the percentage of the codebook that was used at least once
    if 'active_codes_percent' in train_metrics:
        metrics_to_log['train/active_codes_percent'] = train_metrics['active_codes_percent']
    
    # Log a histogram of the usage counts for the codes that were actually used
    if 'codebook_usage' in train_metrics:
        codebook_usage_counts = train_metrics['codebook_usage']
        # Filter for codes that were used (count > 0)
        used_codes_counts = codebook_usage_counts[codebook_usage_counts > 0]
        if used_codes_counts.numel() > 0:
            try:
                metrics_to_log['train/codebook_usage_hist'] = wandb.Histogram(used_codes_counts.numpy())
            except:
                print("ERROR IN LOGGING HISTROGRAM")

    run.log(metrics_to_log)


def finish_wandb(run):
    """Finishes the WandB run."""
    if run:
        run.finish()