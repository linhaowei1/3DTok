# tokenize.py

import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataclasses import dataclass

# Local imports
from data.shapenet_dataset import ShapeNetVoxelDataset
from model.model import VQModel3D, boolean_to_multichannel, multichannel_to_boolean

@dataclass
class TokenizeConfig:
    """Configuration for the tokenization script."""
    checkpoint_path: str
    out_dir: str
    size: int = 128
    batch_size: int = 8
    num_workers: int = 16
    n_embed: int = 512
    embed_dim: int = 64

def tokenize_split(model, loader, device, output_file):
    """Iterates through a dataset split, tokenizes it, and saves the result."""
    all_indices = []
    latent_shape = None
    
    print(f"Tokenizing {len(loader.dataset)} samples...")
    for batch in tqdm(loader):
        voxels = batch['voxel'].to(device)
        
        # Get the token indices for the current batch
        # The new get_tokens method handles the conversion and encoding
        indices, shape = model.get_tokens(voxels, reshape=True)
        if latent_shape is None:
            latent_shape = shape
            print(f"Detected latent grid shape: {latent_shape}")

        all_indices.append(indices.cpu().numpy())

    # Concatenate all batch results into a single NumPy array
    full_dataset_indices = np.concatenate(all_indices, axis=0)
    
    # Save the tokens to a file
    print(f"Saving {full_dataset_indices.shape[0]} tokenized samples to {output_file}")
    np.save(output_file, full_dataset_indices)
    
    return output_file, latent_shape

def verify_detokenization(model, original_dataset, tokens_path, latent_shape, device, num_samples=4):
    """Loads saved tokens, decodes them, and compares with original data."""
    print("\n--- Running Verification ---")
    
    # 1. Load the saved tokens
    print(f"Loading tokens from {tokens_path}...")
    saved_tokens = np.load(tokens_path)
    if saved_tokens.shape[0] < num_samples:
        print(f"Warning: Only {saved_tokens.shape[0]} samples available for verification.")
        num_samples = saved_tokens.shape[0]

    token_subset = torch.from_numpy(saved_tokens[:num_samples]).long().to(device)
    
    # 2. Decode the tokens back into voxel representation
    print(f"Decoding {num_samples} samples using model.decode_code...")
    with torch.no_grad():
        # Use the model's decode_code method
        recon_multichannel = model.decode_code(token_subset, shape=latent_shape)
        # Convert back to boolean voxels
        recon_voxels = multichannel_to_boolean(recon_multichannel)

    # 3. Load the corresponding original samples
    print("Loading original voxels for comparison...")
    original_voxels = torch.stack([original_dataset[i]['voxel'] for i in range(num_samples)]).to(device)

    # 4. Compare
    are_equal = torch.equal(original_voxels, recon_voxels)
    
    if are_equal:
        print("✅ Success: De-tokenized voxels perfectly match the original samples.")
    else:
        # Calculate similarity for more detailed feedback
        similarity = (original_voxels == recon_voxels).float().mean().item()
        print(f"❌ Failure: De-tokenized voxels do not match the originals.")
        print(f"   Pixel-wise similarity: {similarity * 100:.2f}%")
        print("   This can happen if the VQ-VAE is not perfectly lossless.")

    print("--------------------------\n")


def main():
    parser = argparse.ArgumentParser(description="Tokenize the ShapeNet dataset using a pre-trained VQModel3D.")
    
    # --- Arguments ---
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint (.pt file).')
    parser.add_argument('--out-dir', type=str, required=True, help='Directory to save the tokenized .npy files.')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for tokenization.')
    parser.add_argument('--num-workers', type=int, default=8, help='Number of workers for the DataLoader.')
    
    # --- Model Arguments (must match the trained model) ---
    parser.add_argument('--size', type=int, default=128, help='Voxel grid resolution.')
    parser.add_argument('--n-embed', type=int, default=512, help='Number of codebook embeddings.')
    parser.add_argument('--embed-dim', type=int, default=64, help='Dimension of codebook embeddings.')

    args = parser.parse_args()

    # --- 1. SETUP ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.out_dir, exist_ok=True)

    # --- 2. BUILD AND LOAD MODEL ---
    ddconfig = dict(
        ch=64, out_ch=8, ch_mult=(1, 1, 2, 4), num_res_blocks=2,
        attn_resolutions=(16,), dropout=0.0, resamp_with_conv=True,
        in_channels=8, resolution=args.size, z_channels=64,
    )
    model = VQModel3D(ddconfig, n_embed=args.n_embed, embed_dim=args.embed_dim)
    
    print(f"Loading checkpoint from: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state_dict['model'])
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # --- 3. PREPARE DATASETS ---
    train_set = ShapeNetVoxelDataset(split="train")
    val_set = ShapeNetVoxelDataset(split="val")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=False, # No need to shuffle for tokenization
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    print(f"Loaded train set ({len(train_set)} samples) and val set ({len(val_set)} samples).")

    # --- 4. TOKENIZE DATA SPLITS ---
    print("\n--- Starting Train Set Tokenization ---")
    train_tokens_path, latent_shape = tokenize_split(model, train_loader, device, os.path.join(args.out_dir, "train_tokens.npy"))
    
    print("\n--- Starting Validation Set Tokenization ---")
    val_tokens_path, _ = tokenize_split(model, val_loader, device, os.path.join(args.out_dir, "val_tokens.npy"))

    # --- 5. VERIFY DE-TOKENIZATION ---
    # Use the validation set for verification as it's smaller and faster
    verify_detokenization(model, val_set, val_tokens_path, latent_shape, device)
    
    print("Tokenization complete.")

if __name__ == '__main__':
    main()