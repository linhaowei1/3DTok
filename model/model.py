import torch
import torch.nn as nn
import torch.nn.functional as F

from vector_quantize_pytorch import VectorQuantize
from model.layers import Encoder3D, Decoder3D
from einops import repeat

@torch.no_grad()
def zero_patch_mask_3d(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Computes a mask for patches that are entirely zero.

    Args:
        x (torch.Tensor): Input tensor of shape (B, C, D, H, W).
        patch_size (int): The size of the patches to check.

    Returns:
        torch.Tensor: A boolean mask of shape (B, 1, D/patch, H/patch, W/patch),
                      where True indicates an all-zero patch.
    """
    assert x.dim() == 5, "Input tensor must be 5D."
    B, C, D, H, W = x.shape
    assert D % patch_size == 0 and H % patch_size == 0 and W % patch_size == 0, \
        "Input dimensions must be divisible by patch size."

    # Check if any channel in a voxel is non-zero
    nonzero_any = x.ne(0).any(dim=1, keepdim=True).float()
    
    # Pool over patches: if max is 0, all voxels in the patch were 0
    pooled = F.max_pool3d(nonzero_any, kernel_size=patch_size, stride=patch_size)
    
    # A patch is all-zero if the max-pooled value is 0
    mask = pooled < 0.5
    return mask

## Helper Functions for Voxel-to-Channel Conversion
def boolean_to_multichannel(voxel_data: torch.Tensor) -> torch.Tensor:
    """
    Converts a boolean voxel tensor (B, 1, D, H, W) to a multichannel tensor
    (B, 8, D/4, H/4, W/4). Each 4x4x4 boolean patch is encoded into 8 channels.
    
    The encoding maps the 64 boolean values (2^64 patterns) to 8 bytes (2^(8*8)).
    The mapping is arbitrary but consistent, using bit packing.
    """
    B, _, D, H, W = voxel_data.shape
    assert D % 4 == 0 and H % 4 == 0 and W % 4 == 0
    
    # Unfold the tensor into 4x4x4 patches
    patches = voxel_data.unfold(2, 4, 4).unfold(3, 4, 4).unfold(4, 4, 4)
    patches = patches.permute(0, 2, 3, 4, 1, 5, 6, 7) # (B, D', H', W', C, d, h, w)
    patches = patches.reshape(B, D // 4, H // 4, W // 4, 64) # (B, D', H', W', 64)
    
    # Convert boolean to integer (0 or 1)
    patches = patches.long()
    
    # Bit pack 64 bits into 8 bytes (channels)
    multichannel_data = torch.zeros(B, D // 4, H // 4, W // 4, 8, dtype=torch.uint8, device=voxel_data.device)
    
    powers_of_2 = 2**torch.arange(8, device=voxel_data.device).view(1, 1, 1, 1, 8)
    for i in range(8):
        bits = patches[:, :, :, :, i*8:(i+1)*8]
        multichannel_data[:, :, :, :, i] = (bits * powers_of_2).sum(dim=-1)

    # Permute to (B, C, D, H, W) and normalize to [0,1] for loss calculation
    return multichannel_data.permute(0, 4, 1, 2, 3).contiguous().float() / 255.0

def multichannel_to_boolean(multichannel_data: torch.Tensor) -> torch.Tensor:
    """
    Converts a multichannel tensor (B, 8, D/4, H/4, W/4) back to a boolean
    voxel tensor (B, 1, D, H, W). This involves non-differentiable operations.
    """
    B, C, d_prime, h_prime, w_prime = multichannel_data.shape
    assert C == 8
    
    # Denormalize and convert to uint8, rounding probabilities to 0 or 1 effectively
    multichannel_data = (multichannel_data * 255.0).round().byte()
    
    multichannel_data = multichannel_data.permute(0, 2, 3, 4, 1).reshape(B, d_prime, h_prime, w_prime, 8)
    
    # Unpack bytes into 64 bits
    bits = torch.zeros(B, d_prime, h_prime, w_prime, 64, dtype=torch.bool, device=multichannel_data.device)
    powers_of_2 = 2**torch.arange(8, device=multichannel_data.device)
    
    for i in range(8):
        byte_values = multichannel_data[:, :, :, :, i].unsqueeze(-1)
        bits[:, :, :, :, i*8:(i+1)*8] = (byte_values & powers_of_2) != 0
        
    # Reshape back to original voxel dimensions
    voxel_data = bits.reshape(B, d_prime, h_prime, w_prime, 4, 4, 4)
    voxel_data = voxel_data.permute(0, 1, 4, 2, 5, 3, 6).reshape(B, 1, d_prime*4, h_prime*4, w_prime*4)
    
    return voxel_data

class VQModel3D(nn.Module):
    def __init__(
        self,
        ddconfig: dict,
        n_embed: int,
        embed_dim: int,
        ckpt_path: str | None = None,
        ignore_keys: list[str] = [],
        replace: bool = True,
        **kwargs,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.use_replacement = replace

        enc_cfg = dict(ddconfig)
        enc_cfg["double_z"] = False
        self.encoder = Encoder3D(**enc_cfg)
        self.decoder = Decoder3D(**ddconfig)

        zc = ddconfig["z_channels"]
        self.quant_conv = nn.Conv3d(zc, embed_dim, kernel_size=1)
        self.post_quant_conv = nn.Conv3d(embed_dim, zc, kernel_size=1)

        self.quantize = VectorQuantize(
            dim=embed_dim,
            codebook_size=n_embed,
            decay=0.8,
            commitment_weight=.25,
            channel_last=True,
        )
        
        self.total_codebook_size = n_embed + 1 if self.use_replacement else n_embed
        self.replacement_idx = 0 if self.use_replacement else -1
        if self.use_replacement:
            self.replacement_token = nn.Parameter(torch.randn(1, embed_dim) * 0.02)

        downsample_factor = 2**(len(ddconfig['ch_mult']) - 1)
        latent_res = ddconfig['resolution'] // downsample_factor
        d, h, w = latent_res, latent_res, latent_res
        
        self.pos_embedding = nn.Parameter(torch.randn(1, embed_dim, d, h, w))
        self.ddconfig = ddconfig

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list(), logging=True):
        state = torch.load(path, map_location="cpu")
        model_weight = state["state_dict"] if "state_dict" in state else state
        keys = list(model_weight.keys())
        for k in keys:
            if any(k.startswith(ik) for ik in ignore_keys):
                if logging:
                    print(f"Deleting key {k} from state_dict.")
                del model_weight[k]
        missing, unexpected = self.load_state_dict(model_weight, strict=False)
        if logging:
            print(f"Restored from {path}")
            print(f"Missing Keys in State Dict: {missing}")
            print(f"Unexpected Keys in State Dict: {unexpected}")

    def _flatten_spatial(self, x: torch.Tensor):
        b, c, d, h, w = x.shape
        x = x.view(b, c, d * h * w).permute(0, 2, 1).contiguous()
        return x, (d, h, w)

    def _unflatten_spatial(self, x: torch.Tensor, dhw: tuple[int, int, int]):
        d, h, w = dhw
        b, n, c = x.shape
        assert n == d * h * w, "Sequence length does not match target spatial dims"
        x = x.permute(0, 2, 1).contiguous().view(b, c, d, h, w)
        return x

    def _get_full_codebook(self) -> torch.Tensor:
        main_codebook = self.quantize.codebook
        if not self.use_replacement:
            return main_codebook
        
        if main_codebook.dim() == 3:
            replacement = repeat(self.replacement_token, '1 d -> h 1 d', h=main_codebook.shape[0])
            return torch.cat([replacement, main_codebook], dim=1)
        else:
            return torch.cat([self.replacement_token, main_codebook], dim=0)

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        h = self.quant_conv(h)
        seq, dhw = self._flatten_spatial(h)           # (B, N, C) with channel_last=True expectation

        if not self.use_replacement:
            # ---  FP32 VQ ---
            with torch.amp.autocast('cuda', enabled=False):
                quant_seq, indices, commit_loss_out = self.quantize(seq.float())
                commit_loss = commit_loss_out.mean()
            quant = self._unflatten_spatial(quant_seq.to(h.dtype), dhw)
            return quant, commit_loss, indices

        d_latent, _, _ = dhw
        patch_size = x.shape[-3] // d_latent
        zero_mask_flat = zero_patch_mask_3d(x, patch_size=patch_size).view(x.shape[0], -1)
        nonzero_mask_flat = ~zero_mask_flat
        non_mask_frac = nonzero_mask_flat.float().mean()

        if non_mask_frac == 0:
            full_indices = torch.full((seq.shape[0], seq.shape[1]),
                                    self.replacement_idx, dtype=torch.long, device=seq.device)
            full_codebook = self._get_full_codebook()
            with torch.amp.autocast('cuda', enabled=False):
                quant_seq = F.embedding(full_indices, full_codebook).float()
            quant = self._unflatten_spatial(quant_seq.to(h.dtype), dhw)
            commit_loss = seq.new_zeros(())
            return quant, commit_loss, full_indices

        with torch.amp.autocast('cuda', enabled=False):
            seq_nonzero = seq[nonzero_mask_flat].float().view(-1, self.embed_dim)
            quant_seq_nonzero, indices_nonzero, commit_loss_out = self.quantize(seq_nonzero)
            commit_loss = commit_loss_out.mean() * non_mask_frac

        full_indices = torch.full((seq.shape[0], seq.shape[1]),
                                self.replacement_idx, dtype=torch.long, device=seq.device)
        full_indices[nonzero_mask_flat] = indices_nonzero + 1

        full_codebook = self._get_full_codebook()
        with torch.amp.autocast('cuda', enabled=False):
            quant_seq = F.embedding(full_indices, full_codebook).float()

        quant = self._unflatten_spatial(quant_seq.to(h.dtype), dhw)
        return quant, commit_loss, full_indices


    def decode(self, quant: torch.Tensor):
        quant = quant + self.pos_embedding
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    @torch.no_grad()
    def decode_code(self, code_b: torch.Tensor, shape: tuple[int, int, int]):
        full_codebook = self._get_full_codebook()
        quant_seq = F.embedding(code_b, full_codebook)
        quant = self._unflatten_spatial(quant_seq, shape)
        return self.decode(quant)

    def forward(self, x: torch.Tensor):
        """
        Full autoencoding pass. The reconstruction loss is computed in the
        multichannel space for stable gradients.

        Args:
            x (torch.Tensor): Input boolean voxel volume (B, 1, 512, 512, 512).
            
        Returns:
            multichannel_recon (torch.Tensor): Reconstructed multichannel volume (logits).
            multichannel_data (torch.Tensor): Target multichannel volume (normalized floats).
            commit_loss (torch.Tensor): Scalar commitment loss.
            indices (torch.Tensor): Flattened code indices (B, D'*H'*W').
        """
        # --- Stage 1: Voxel to Multichannel Conversion ---
        # This becomes the target for the reconstruction loss
        multichannel_data = boolean_to_multichannel(x)
        
        # --- Stage 2: Autoencoding on Multichannel Data ---
        quant, commit_loss, indices = self.encode(multichannel_data)
        multichannel_recon = self.decode(quant).float() # This is expected to be logits
        
        return multichannel_recon, multichannel_data, commit_loss, indices

    @property
    def latent_spatial_dim(self) -> int:
        """Returns the spatial dimension (D, H, or W) of the latent grid."""
        downsample_factor = 2**(len(self.ddconfig['ch_mult']) - 1)
        latent_res = self.ddconfig['resolution'] // downsample_factor
        return latent_res
        
    @torch.no_grad()
    def get_tokens(self, x: torch.Tensor, reshape: bool = True):
        """
        Encodes an input tensor into discrete codebook indices (tokens).

        Args:
            x (torch.Tensor): Input boolean voxel tensor (B, 1, D, H, W).
            reshape (bool): If True, reshapes the flat indices to their spatial
                            grid form (B, D', H', W').

        Returns:
            torch.Tensor: A tensor of integer indices.
            tuple[int, int, int]: The spatial shape of the latent grid (D', H', W').
        """
        # The VQ-VAE works on a multichannel representation
        multichannel_data = boolean_to_multichannel(x)
        
        # Encode to get quantized vectors and indices
        _, _, indices = self.encode(multichannel_data) # We only need the indices
        
        d = self.latent_spatial_dim
        latent_shape = (d, d, d)

        if reshape:
            return indices.view(x.shape[0], *latent_shape), latent_shape
        
        return indices, latent_shape
        
def test_lossless_conversion():
    """
    Tests if the conversion from boolean -> multichannel -> boolean is lossless.
    """
    print("Running test for lossless voxel conversion...")
    
    # Create a random boolean tensor.
    # Dimensions must be divisible by 4 for the patch-based conversion.
    batch_size = 2
    depth, height, width = 32, 64, 32
    
    # Original boolean voxel data
    original_voxels = torch.rand(batch_size, 1, depth, height, width) > 0.5
    print(f"Original boolean tensor shape: {original_voxels.shape}")
    
    # 1. Convert from boolean to multichannel representation
    multichannel_representation = boolean_to_multichannel(original_voxels)
    print(f"Intermediate multichannel tensor shape: {multichannel_representation.shape}")
    
    # 2. Convert back from multichannel to boolean representation
    reconstructed_voxels = multichannel_to_boolean(multichannel_representation)
    print(f"Reconstructed boolean tensor shape: {reconstructed_voxels.shape}")
    
    # 3. Verify that the original and reconstructed tensors are identical
    are_equal = torch.equal(original_voxels, reconstructed_voxels)
    
    assert are_equal, "Test Failed: The reconstructed voxel data does not match the original."
    
    print("\n✅ Test Passed: Voxel conversion is lossless.")

if __name__ == '__main__':

    test_lossless_conversion()