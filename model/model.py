# model/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from vector_quantize_pytorch import VectorQuantize
from model.layers import Encoder3D, Decoder3D
from einops import repeat


@torch.no_grad()
def zero_patch_mask_3d(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Compute a mask for patches that are entirely zero.

    Args:
        x (torch.Tensor): Input tensor of shape (B, C, D, H, W).
        patch_size (int): The patch size used to pool (D/H/W must be divisible by it).

    Returns:
        torch.Tensor: Boolean mask of shape (B, 1, D/patch, H/patch, W/patch),
                      where True indicates an all-zero patch.
    """
    assert x.dim() == 5, "Input tensor must be 5D."
    B, C, D, H, W = x.shape
    assert D % patch_size == 0 and H % patch_size == 0 and W % patch_size == 0, \
        "Input dimensions must be divisible by patch size."

    # A voxel is nonzero if any channel is nonzero
    nonzero_any = x.ne(0).any(dim=1, keepdim=True).float()

    # Max-pool over patches: if the pooled max is 0, the entire patch is zero
    pooled = F.max_pool3d(nonzero_any, kernel_size=patch_size, stride=patch_size)

    # A patch is all-zero if the max is 0
    mask = pooled < 0.5
    return mask


# -------- Helper functions for voxel-to-channel conversion --------

def boolean_to_multichannel(voxel_data: torch.Tensor) -> torch.Tensor:
    """
    Convert a boolean voxel tensor (B, 1, D, H, W) to a multichannel tensor
    (B, 8, D/4, H/4, W/4). Each 4x4x4 boolean patch is encoded into 8 bytes.

    The 64 bits (2^64 patterns) are bit-packed into 8 uint8 channels (8 * 8 bits).
    The mapping is arbitrary but consistent (bit packing).
    """
    B, _, D, H, W = voxel_data.shape
    assert D % 4 == 0 and H % 4 == 0 and W % 4 == 0

    # Unfold into 4x4x4 patches
    patches = voxel_data.unfold(2, 4, 4).unfold(3, 4, 4).unfold(4, 4, 4)
    patches = patches.permute(0, 2, 3, 4, 1, 5, 6, 7)  # (B, D', H', W', C, d, h, w)
    patches = patches.reshape(B, D // 4, H // 4, W // 4, 64)  # (B, D', H', W', 64)

    # Convert boolean to integer (0 or 1)
    patches = patches.long()

    # Bit-pack 64 bits into 8 bytes (channels)
    multichannel_data = torch.zeros(
        B, D // 4, H // 4, W // 4, 8, dtype=torch.uint8, device=voxel_data.device
    )

    powers_of_2 = 2**torch.arange(8, device=voxel_data.device).view(1, 1, 1, 1, 8)
    for i in range(8):
        bits = patches[:, :, :, :, i*8:(i+1)*8]
        multichannel_data[:, :, :, :, i] = (bits * powers_of_2).sum(dim=-1)

    # To (B, C, D, H, W) and normalize to [0, 1] for loss computation
    return multichannel_data.permute(0, 4, 1, 2, 3).contiguous().float() / 255.0


def multichannel_to_boolean(multichannel_data: torch.Tensor) -> torch.Tensor:
    """
    Convert a multichannel tensor (B, 8, D/4, H/4, W/4) back to a boolean
    voxel tensor (B, 1, D, H, W). This is non-differentiable.
    """
    B, C, d_prime, h_prime, w_prime = multichannel_data.shape
    assert C == 8

    # De-normalize to uint8, rounding probabilities to {0, 1}
    multichannel_data = (multichannel_data * 255.0).round().byte()
    multichannel_data = multichannel_data.permute(0, 2, 3, 4, 1).reshape(
        B, d_prime, h_prime, w_prime, 8
    )

    # Unpack 8 bytes -> 64 bits
    bits = torch.zeros(
        B, d_prime, h_prime, w_prime, 64, dtype=torch.bool, device=multichannel_data.device
    )
    powers_of_2 = 2**torch.arange(8, device=multichannel_data.device)

    for i in range(8):
        byte_values = multichannel_data[:, :, :, :, i].unsqueeze(-1)
        bits[:, :, :, :, i*8:(i+1)*8] = (byte_values & powers_of_2) != 0

    # Fold back to original voxel layout
    voxel_data = bits.reshape(B, d_prime, h_prime, w_prime, 4, 4, 4)
    voxel_data = voxel_data.permute(0, 1, 4, 2, 5, 3, 6).reshape(
        B, 1, d_prime*4, h_prime*4, w_prime*4
    )
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
        # --- New: vector-quantizer hyperparameters focused on utilization ---
        codebook_dim: int = 8,
        use_cosine: bool = True,
        vq_decay: float = 0.95,
        vq_commitment_weight: float = 0.5,
        vq_kmeans_init: bool = True,
        vq_threshold_ema_dead_code: int = 2,
        vq_diversity_weight: float = 0.1,
        vq_diversity_temp: float = 75.0,
        vq_orthogonal_weight: float = 1e-2,
        vq_sample_temp: float = 1.0,
        vq_stochastic: bool = True,
        # You can expose heads/separate codebooks later if needed
        **kwargs,
    ):
        """
        3D VQ-VAE model with utilization-oriented defaults:
        - K-means init + dead-code recycling
        - Diversity (entropy) loss
        - Orthogonal regularization
        - Gumbel sampling with adjustable temperature
        """
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

        # -------- VectorQuantize with utilization-friendly defaults --------
        # - cosine distance (good for scale-invariant latent vectors)
        # - k-means initialization to avoid early collapse
        # - threshold_ema_dead_code to recycle unused codes
        # - diversity loss and orthogonal regularization to spread usage
        self.quantize = VectorQuantize(
            dim=embed_dim,
            codebook_size=n_embed,
            decay=vq_decay,
            commitment_weight=vq_commitment_weight,
            channel_last=True,
            use_cosine_sim=use_cosine,
            codebook_dim=codebook_dim,
            kmeans_init=vq_kmeans_init,
            # threshold_ema_dead_code=vq_threshold_ema_dead_code,
            codebook_diversity_loss_weight=vq_diversity_weight,
            codebook_diversity_temperature=vq_diversity_temp,
            orthogonal_reg_weight=vq_orthogonal_weight,
            # stochastic_sample_codes=vq_stochastic,
            # sample_codebook_temp=vq_sample_temp
        )

        # Track the codebook dimensionality (used by the replacement token)
        self.codebook_dim = codebook_dim

        # Total number of indices including the special replacement index
        self.total_codebook_size = n_embed + 1 if self.use_replacement else n_embed

        # Usage counter (sum across processes in DDP if needed)
        self.register_buffer('codebook_usage', torch.zeros(n_embed))

        # Replacement token index and its learnable code (in codebook space)
        self.replacement_idx = 0 if self.use_replacement else -1
        if self.use_replacement:
            # The replacement token lives in the codebook space (codebook_dim)
            self.replacement_token_code = nn.Parameter(
                torch.randn(1, self.codebook_dim) * 0.02
            )

        # Positional embedding in latent space
        downsample_factor = 2**(len(ddconfig['ch_mult']) - 1)
        latent_res = ddconfig['resolution'] // downsample_factor
        d, h, w = latent_res, latent_res, latent_res
        self.pos_embedding = nn.Parameter(torch.randn(1, embed_dim, d, h, w))

        self.ddconfig = ddconfig

        # A small buffer to hold the current sampling temperature (anneal in training loop)
        self.register_buffer("code_temp", torch.tensor(float(vq_sample_temp)), persistent=True)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    # ------------------------- Utilities -------------------------

    def set_code_temperature(self, t: float):
        """Set the Gumbel/argmax sampling temperature for the codebook."""
        self.code_temp.fill_(float(t))

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

    def _get_full_codebook_outputs(self) -> torch.Tensor:
        """
        Returns a (total_codebook_size, embed_dim) matrix of decoder-side code vectors.

        The first row is the replacement token (if enabled), followed by the
        projected outputs of the main codebook indices.
        """
        device = self.pos_embedding.device

        # Main codebook outputs (already projected to embed_dim)
        main_indices = torch.arange(self.quantize.codebook_size, device=device)
        main_outputs = self.quantize.get_output_from_indices(main_indices)  # (N, embed_dim)

        if not self.use_replacement:
            return main_outputs

        # Replacement output: parameter lives in codebook space -> project_out to embed_dim
        repl_out = self.quantize.project_out(self.replacement_token_code)  # (1, embed_dim)

        full_outputs = torch.cat([repl_out, main_outputs], dim=0)  # (N+1, embed_dim)
        return full_outputs

    # -------------------------- Encode / Decode --------------------------

    def encode(self, x: torch.Tensor):
        """
        Encode an input volume to quantized latents and indices.
        Returns:
            quant (B, C, D', H', W'), commit_loss (scalar), indices (B, N)
        """
        h = self.encoder(x)
        h = self.quant_conv(h)
        seq, dhw = self._flatten_spatial(h)  # (B, N, C) with channel_last=True

        # Path without replacement token (all positions are quantized normally)
        if not self.use_replacement:
            # Use FP32 for the VQ operation to avoid numerical issues
            with torch.amp.autocast('cuda', enabled=False):
                quant_seq, indices, _, breakdown = self.quantize(
                    seq.float(),
                    sample_codebook_temp=float(self.code_temp.item()),
                    return_loss_breakdown=True
                )
                commit_loss = breakdown.commitment.mean()

            if self.training:
                with torch.no_grad():
                    usage_counts = torch.bincount(indices.flatten(), minlength=self.quantize.codebook_size)
                    self.codebook_usage.add_(usage_counts)

            quant = self._unflatten_spatial(quant_seq.to(h.dtype), dhw)
            return quant, commit_loss, indices

        # Replacement-aware path (e.g., ignore truly empty patches)
        d_latent, _, _ = dhw
        patch_size = x.shape[-3] // d_latent

        zero_mask_flat = zero_patch_mask_3d(x, patch_size=patch_size).view(x.shape[0], -1)  # (B, N)
        nonzero_mask_flat = ~zero_mask_flat
        non_mask_frac = nonzero_mask_flat.float().mean()

        full_outputs = self._get_full_codebook_outputs()  # (total_codebook_size, embed_dim)

        # If all patches are zero, directly return the replacement token everywhere
        if non_mask_frac == 0:
            full_indices = torch.full(
                (seq.shape[0], seq.shape[1]),
                self.replacement_idx, dtype=torch.long, device=seq.device
            )
            with torch.amp.autocast('cuda', enabled=False):
                quant_seq = F.embedding(full_indices, full_outputs).float()   # (B, N, embed_dim)
            quant = self._unflatten_spatial(quant_seq.to(h.dtype), dhw)
            commit_loss = seq.new_zeros(())
            return quant, commit_loss, full_indices

        # Quantize the non-zero positions only (fp32 VQ for stability)
        with torch.amp.autocast('cuda', enabled=False):
            seq_nonzero = seq[nonzero_mask_flat].float().view(-1, self.embed_dim)
            quant_seq_nonzero, indices_nonzero, _, breakdown = self.quantize(
                seq_nonzero,
                sample_codebook_temp=float(self.code_temp.item()),
                return_loss_breakdown=True
            )
            commit_loss = breakdown.commitment.mean() * non_mask_frac

        if self.training:
            with torch.no_grad():
                usage_counts = torch.bincount(indices_nonzero.flatten(), minlength=self.quantize.codebook_size)
                self.codebook_usage.add_(usage_counts)

        # Stitch back: fill zeros with replacement_idx, others with (indices + 1)
        full_indices = torch.full(
            (seq.shape[0], seq.shape[1]),
            self.replacement_idx, dtype=torch.long, device=seq.device
        )
        full_indices[nonzero_mask_flat] = indices_nonzero + 1  # shift by 1 due to replacement at index 0

        with torch.amp.autocast('cuda', enabled=False):
            quant_seq = F.embedding(full_indices, full_outputs).float()  # (B, N, embed_dim)

        quant = self._unflatten_spatial(quant_seq.to(h.dtype), dhw)
        return quant, commit_loss, full_indices

    def decode(self, quant: torch.Tensor):
        """
        Decode quantized latents back to the multichannel volume space.
        """
        quant = quant + self.pos_embedding
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    @torch.no_grad()
    def decode_code(self, code_b: torch.Tensor, shape: tuple[int, int, int]):
        """
        Decode a sequence of code indices back to the voxel space, using the
        pre-projected (to embed_dim) code output matrix (compatible with replacement idx).
        """
        full_outputs = self._get_full_codebook_outputs()  # (total_codebook_size, embed_dim)
        quant_seq = F.embedding(code_b, full_outputs)     # (B, N, embed_dim)
        quant = self._unflatten_spatial(quant_seq, shape)
        return self.decode(quant)

    # --------------------------- End-to-end ---------------------------

    def forward(self, x: torch.Tensor):
        """
        Full autoencoding pass. The reconstruction loss should be computed
        on the multichannel space outside this module (decoder outputs logits).

        Args:
            x (torch.Tensor): Input boolean voxel volume (B, 1, 512, 512, 512).

        Returns:
            multichannel_recon (torch.Tensor): Reconstructed multichannel volume (logits).
            multichannel_data (torch.Tensor): Target multichannel volume (normalized floats).
            commit_loss (torch.Tensor): Scalar commitment loss ONLY.
            indices (torch.Tensor): Flattened code indices (B, D'*H'*W').
        """
        # 1) Boolean -> multichannel
        multichannel_data = boolean_to_multichannel(x)

        # 2) VQ encode + decode on multichannel
        quant, commit_loss, indices = self.encode(multichannel_data)
        multichannel_recon = self.decode(quant).float()  # logits

        return multichannel_recon, multichannel_data, commit_loss, indices

    # --------------------------- Introspection ---------------------------

    @property
    def latent_spatial_dim(self) -> int:
        """Return the spatial dimension (D, H, or W) of the latent grid."""
        downsample_factor = 2**(len(self.ddconfig['ch_mult']) - 1)
        latent_res = self.ddconfig['resolution'] // downsample_factor
        return latent_res

    @torch.no_grad()
    def get_tokens(self, x: torch.Tensor, reshape: bool = True):
        """
        Encode an input tensor into discrete codebook indices (tokens).

        Args:
            x (torch.Tensor): Input boolean voxel tensor (B, 1, D, H, W).
            reshape (bool): If True, reshape the flat indices to (B, D', H', W').

        Returns:
            torch.Tensor: Integer indices.
            tuple[int, int, int]: Spatial shape of the latent grid (D', H', W').
        """
        multichannel_data = boolean_to_multichannel(x)
        _, _, indices = self.encode(multichannel_data)

        d = self.latent_spatial_dim
        latent_shape = (d, d, d)

        if reshape:
            return indices.view(x.shape[0], *latent_shape), latent_shape

        return indices, latent_shape

    @torch.no_grad()
    def get_codebook_usage(self):
        """
        Return the accumulated codebook usage counts.
        In DDP, sum across processes outside this function.
        """
        return self.codebook_usage.clone()

    @torch.no_grad()
    def reset_codebook_usage(self):
        """Reset the usage counter to zero."""
        self.codebook_usage.zero_()


# ------------------------------ Quick test ------------------------------

def test_lossless_conversion():
    """
    Quick check that boolean -> multichannel -> boolean is lossless.
    """
    print("Running test for lossless voxel conversion...")

    # Create a random boolean tensor. D/H/W must be divisible by 4.
    batch_size = 2
    depth, height, width = 32, 64, 32

    original_voxels = (torch.rand(batch_size, 1, depth, height, width) > 0.5)
    print(f"Original boolean tensor shape: {original_voxels.shape}")

    multichannel_representation = boolean_to_multichannel(original_voxels)
    print(f"Intermediate multichannel tensor shape: {multichannel_representation.shape}")

    reconstructed_voxels = multichannel_to_boolean(multichannel_representation)
    print(f"Reconstructed boolean tensor shape: {reconstructed_voxels.shape}")

    are_equal = torch.equal(original_voxels, reconstructed_voxels)
    assert are_equal, "Test Failed: reconstructed voxel data does not match the original."

    print("\n✅ Test Passed: Voxel conversion is lossless.")


if __name__ == '__main__':
    test_lossless_conversion()
