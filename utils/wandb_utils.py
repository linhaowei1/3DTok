# wandb_utils.py

"""
Utility functions for Weights & Biases (wandb) logging.
"""

import wandb
import torch # Added for type hinting

def init_wandb(project_name, run_name=None, config=None, tags=None, notes=None, offline=False):
    """
    Initializes a new wandb run.
    """
    mode = 'offline' if offline else 'online'
    
    run = wandb.init(
        project=project_name,
        name=run_name,
        config=config,
        tags=tags,
        notes=notes,
        mode=mode
    )
    return run

### REVISED ###
# Updated function to log more detailed metrics
def log_epoch_metrics(run, epoch, train_metrics, val_metrics, optimizer: torch.optim.Optimizer):
    """
    Logs training and validation metrics for a given epoch.

    Args:
        run: The active wandb run object.
        epoch (int): The current epoch number.
        train_metrics (dict): Dictionary of training metrics.
        val_metrics (dict): Dictionary of validation metrics.
        optimizer (torch.optim.Optimizer): The optimizer, to log learning rate.
    """
    if not run:
        return

    # Prepare a dictionary with all metrics to log
    log_data = {
        'epoch': epoch,
        'learning_rate': optimizer.param_groups[0]['lr'],
        
        # Training metrics
        'train/loss': train_metrics.get('loss'),
        'train/reconstruction_loss': train_metrics.get('reconstruction_loss'),
        'train/commitment_loss': train_metrics.get('commitment_loss'),
        'train/grad_norm': train_metrics.get('grad_norm'),
        
        # Validation metrics
        'val/loss': val_metrics.get('loss'),
        'val/reconstruction_loss': val_metrics.get('reconstruction_loss'),
        'val/commitment_loss': val_metrics.get('commitment_loss'),
        'val/iou': val_metrics.get('iou'),
    }

    # Filter out any None values in case a metric is missing
    log_data = {k: v for k, v in log_data.items() if v is not None}
    
    # Log the data to wandb
    run.log(log_data)


def finish_wandb(run):
    """
    Finishes the active wandb run.
    """
    if run:
        run.finish()