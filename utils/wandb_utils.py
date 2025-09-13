"""
Utility functions for Weights & Biases (wandb) logging.
"""

import wandb

def init_wandb(project_name, run_name=None, config=None, tags=None, notes=None, offline=False):
    """
    Initializes a new wandb run.

    Args:
        project_name (str): The name of the wandb project.
        run_name (str, optional): The name for this specific run. Auto-generated if None.
        config (dict, optional): A dictionary of hyperparameters to log.
        tags (list, optional): A list of tags to associate with the run.
        notes (str, optional): A longer description for the run.
        offline (bool): If True, run wandb in 'offline' mode.

    Returns:
        A wandb run object.
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

def log_epoch_metrics(run, epoch, train_metrics, val_metrics):
    """
    Logs training and validation metrics for a given epoch.

    Args:
        run: The active wandb run object.
        epoch (int): The current epoch number.
        train_metrics (dict): A dictionary containing training metrics (e.g., 'recon_loss', 'iou').
        val_metrics (dict): A dictionary containing validation metrics (e.g., 'recon_loss', 'iou').
    """
    if run:
        # Prepare a dictionary with metrics to log
        log_data = {
            'epoch': epoch,
            'train/loss': train_metrics.get('recon_loss', float('nan')),
            'train/iou': train_metrics.get('iou', float('nan')),
            'val/loss': val_metrics.get('recon_loss', float('nan')),
            'val/iou': val_metrics.get('iou', float('nan')),
        }
        # Log the data to wandb
        run.log(log_data)


def finish_wandb(run):
    """
    Finishes the active wandb run.

    Args:
        run: The active wandb run object.
    """
    if run:
        run.finish()