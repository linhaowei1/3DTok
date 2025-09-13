import math
import torch
import torch.nn as nn
import numpy as np


# -------------------- helpers --------------------

def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels: int):
    # Guard against small channel counts that don't divide 32 nicely
    num_groups = min(32, in_channels)
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


# -------------------- blocks (3D) --------------------

class Upsample3D(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv3d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )

    def forward(self, x):
        # scale all three spatial dims by 2
        x = torch.nn.functional.interpolate(x, scale_factor=(2.0, 2.0, 2.0), mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample3D(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # No asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv3d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=2,
                padding=0,
            )

    def forward(self, x):
        if self.with_conv:
            # pad order for F.pad on 5D is (W_left, W_right, H_left, H_right, D_left, D_right)
            pad = (0, 1, 0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool3d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock3D(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int | None = None,
        conv_shortcut: bool = False,
        dropout: float,
        temb_channels: int = 512,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        else:
            self.temb_proj = None

        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv3d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = torch.nn.Conv3d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if (temb is not None) and (self.temb_proj is not None):
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock3D(nn.Module):
    """Self-attention over 3D spatial volume (D*H*W). Be mindful of memory: O((DHW)^2)."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    # In layers.py

class AttnBlock3D(nn.Module):
    """Self-attention over 3D spatial volume (D*H*W). Be mindful of memory: O((DHW)^2)."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, d, h, w = q.shape
        # Reshape and permute, then make contiguous
        q = q.reshape(b, c, d * h * w).permute(0, 2, 1).contiguous()  # b, dhw, c
        k = k.reshape(b, c, d * h * w)  # b, c, dhw
        attn = torch.bmm(q, k)  # b, dhw, dhw
        attn = attn * (c**-0.5)
        attn = torch.nn.functional.softmax(attn, dim=2)

        # attend to values
        v = v.reshape(b, c, d * h * w)
        # Permute, then make contiguous
        attn = attn.permute(0, 2, 1).contiguous()  # b, dhw, dhw (first dhw of k, second of q)
        h_ = torch.bmm(v, attn)  # b, c, dhw
        h_ = h_.reshape(b, c, d, h, w)
        h_ = self.proj_out(h_)
        return x + h_


# -------------------- Encoder / Decoder (3D) --------------------

class Encoder3D(nn.Module):
    def __init__(
        self,
        *,
        ch: int,
        out_ch: int,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks: int = 2,
        attn_resolutions=(),
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        in_channels: int = 1,
        resolution: int = 32,
        z_channels: int = 4,
        double_z: bool = True,
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv3d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock3D(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock3D(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample3D(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock3D(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        )
        self.mid.attn_1 = AttnBlock3D(block_in)
        self.mid.block_2 = ResnetBlock3D(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        )

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv3d(
            block_in, 2 * z_channels if double_z else z_channels, kernel_size=3, stride=1, padding=1
        )
        self.double_z = double_z
        self.z_channels = z_channels

    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder3D(nn.Module):
    def __init__(
        self,
        *,
        ch: int,
        out_ch: int,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks: int = 2,
        attn_resolutions=(),
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        in_channels: int = 1,
        resolution: int = 32,
        z_channels: int = 4,
        give_pre_end: bool = False,
        **ignorekwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res, curr_res)
        print(f"Working with z of shape {self.z_shape} = {int(np.prod(self.z_shape))} dimensions.")

        # z to block_in
        self.conv_in = torch.nn.Conv3d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock3D(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        )
        self.mid.attn_1 = AttnBlock3D(block_in)
        self.mid.block_2 = ResnetBlock3D(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock3D(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock3D(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample3D(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv3d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    cfg = dict(
        ch=32,
        out_ch=1,
        ch_mult=(1, 1, 2, 2, 4),     
        num_res_blocks=2,
        attn_resolutions=(),        
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=1,
        resolution=256,
        z_channels=8,                
    )

    print("\nBuilding Encoder3D and Decoder3D...")
    enc = Encoder3D(double_z=True, **cfg).bfloat16().to(device)
    dec = Decoder3D(**{k: v for k, v in cfg.items() if k != "z_channels"},
                    z_channels=cfg["z_channels"]).bfloat16().to(device)

    from replacer import TokenReplacer3D, zero_patch_mask_3d

    replacer = TokenReplacer3D(z_channels=cfg["z_channels"]).to(device).bfloat16()

    x = torch.randn(1, cfg["in_channels"], cfg["resolution"], cfg["resolution"], cfg["resolution"],
                    device=device).bfloat16()

    x[:, :, :128, :128, :128] = 0
    print(f"Input x: {tuple(x.shape)}")


    mask16 = zero_patch_mask_3d(x, patch_size=16)  # (1,1,32,32,32)
    
    with torch.no_grad():
        enc_out = enc(x)  # (B, 2*zc, 32, 32, 32) 
    print(f"Encoder output: {tuple(enc_out.shape)}")


    z_mean = enc_out[:, :cfg["z_channels"], ...]
    z_logv = enc_out[:, cfg["z_channels"]:, ...]


    z_replaced = replacer(z_mean, mask16)  # (B, 2*zc, 32, 32, 32)

    z_for_dec = z_replaced  
    with torch.no_grad():
        y = dec(z_for_dec)
    print(f"Decoder output: {tuple(y.shape)}")
