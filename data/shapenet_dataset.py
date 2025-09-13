"""
ShapeNet voxel dataset for training the voxel tokenizer.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import json


class ShapeNetVoxelDataset(Dataset):
    """
    Dataset for loading pre-voxelized ShapeNet data from train/val splits.
    Loads .npy files containing binary occupancy volumes.
    Output tensor shape: (1, D, H, W) with {0, 1} floats.
    """
    
    def __init__(self, root_dir: str = "./storage/ShapeNet_Processed", category: str = "airplane", split: str = "train"):
        """
        Args:
            root_dir (str): The root directory of the processed ShapeNet data, 
            category (str): The object category to load, e.g., 'airplane'.
            split (str): The dataset split to load, "train" or "val".
        """
        super().__init__()
        self.root_dir = Path(root_dir)
        self.category = category
        self.split = split
        
        # Construct the path to the specific split directory
        self.data_dir = self.root_dir / self.category / "normalized_vox" / self.split
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Discover all .npy files in the directory
        self.file_paths = sorted(list(self.data_dir.glob("*.npy")))
        
        if not self.file_paths:
            raise FileNotFoundError(f"No .npy files found in {self.data_dir}")
            
        # Infer resolution from the first sample
        try:
            sample_voxel = np.load(self.file_paths[0])
            self.resolution = sample_voxel.shape
        except Exception as e:
            raise IOError(f"Could not load or read shape from {self.file_paths[0]}: {e}")

        print(f"✅ Loaded ShapeNet dataset ({self.category}/{self.split}): "
              f"{len(self.file_paths)} samples, resolution={self.resolution}")

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a voxel sample at the given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            A tuple (input_tensor, target_tensor), where both are the same
            voxel data with shape (1, D, H, W).
        """
        voxel_file_path = self.file_paths[idx]
        
        try:
            # Load the voxel data from the .npy file
            voxels = np.load(voxel_file_path)  # Expected shape: (D, H, W)
        except Exception as e:
            raise IOError(f"Error loading voxel file {voxel_file_path}: {e}")
        
        # Convert to a Float tensor and add a channel dimension
        # Final shape: (1, D, H, W)
        x = torch.from_numpy(voxels).float().unsqueeze(0)
        
        # For an autoencoder, the input is the same as the target
        return x, x.clone()

    def get_stats(self) -> dict:
        """
        Calculates and returns statistics about the dataset.
        For efficiency, this is calculated on a small subset of the data.
        """
        # Limit stats calculation to a small number of samples for speed
        num_samples_for_stats = min(20, len(self))
        if num_samples_for_stats == 0:
            return {
                "num_samples": 0,
                "resolution": self.resolution,
                "avg_occupancy_ratio": 0
            }

        total_voxels = 0
        total_occupied = 0
        
        for i in range(num_samples_for_stats):
            x, _ = self[i]
            total_voxels += x.numel()
            total_occupied += x.sum().item()
        
        occupancy_ratio = total_occupied / total_voxels if total_voxels > 0 else 0
        
        return {
            "num_samples": len(self),
            "resolution": self.resolution,
            "avg_occupancy_ratio": occupancy_ratio
        }

# --- Example Usage ---
if __name__ == '__main__':
    # Define the path to your processed ShapeNet data
    ROOT_DATA_DIR = "./storage/ShapeNet_Processed"
    CATEGORY = "airplane"

    try:
        # 1. Initialize the training dataset
        print("Initializing training dataset...")
        train_dataset = ShapeNetVoxelDataset(root_dir=ROOT_DATA_DIR, category=CATEGORY, split="train")
        
        # 2. Get a sample item
        if len(train_dataset) > 0:
            input_sample, target_sample = train_dataset[0]
            print(f"\nSample shape: {input_sample.shape}")
            print(f"Sample dtype: {input_sample.dtype}")
        
        # 3. Get dataset statistics
        stats = train_dataset.get_stats()
        print("\nTraining Set Statistics:")
        print(f"  - Total Samples: {stats['num_samples']}")
        print(f"  - Voxel Resolution: {stats['resolution']}")
        print(f"  - Average Occupancy Ratio (from subset): {stats['avg_occupancy_ratio']:.4f}")

        # 4. Initialize the validation dataset
        print("\nInitializing validation dataset...")
        val_dataset = ShapeNetVoxelDataset(root_dir=ROOT_DATA_DIR, category=CATEGORY, split="val")
        if len(val_dataset) > 0:
            val_stats = val_dataset.get_stats()
            print("\nValidation Set Statistics:")
            print(f"  - Total Samples: {val_stats['num_samples']}")

    except (FileNotFoundError, IOError) as e:
        print(f"\nError: {e}")
        print("Please ensure the `ROOT_DATA_DIR` and `CATEGORY` are correct and the data is structured as expected.")