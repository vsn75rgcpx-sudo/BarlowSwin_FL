"""
model_swin3d.swin3d_block
-------------------------
Implementation of Swin Transformer 3D block with support for
NAS (MixedOp3D) and shifted window attention.

This block uses:
- LayerNorm3D
- WindowAttention3D
- Shift-window mechanism
- Patch partition + reverse
- MixedOp3D for NAS-enhanced MLP/FFN

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import (
    window_partition,
    window_reverse,
    WindowAttention3D,
    compute_attn_mask,
)
from .nas_ops import MixedOp3D


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
class LayerNormChannel(nn.Module):
    """
    LayerNorm over channel dimension for 3D features.
    Input: (B, C, D, H, W)
    """
    def __init__(self, num_channels):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels)

    def forward(self, x):
        # x: B,C,D,H,W -> B,D,H,W,C -> LN -> B,C,D,H,W
        x_perm = x.permute(0,2,3,4,1).contiguous()
        x_norm = self.ln(x_perm)
        return x_norm.permute(0,4,1,2,3).contiguous()


class DropPath(nn.Module):
    """Stochastic depth (drop path)."""
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,)*(x.dim()-1)
        random = torch.rand(shape, dtype=x.dtype, device=x.device)
        random = random < keep_prob
        return x * random / keep_prob


# ------------------------------------------------------------
# Swin Transformer 3D Block
# ------------------------------------------------------------
class Swin3DBlock(nn.Module):
    """
    A single Swin block with:
    - LN
    - (Shifted) Window Attention
    - Residual
    - NAS-augmented MLP (MixedOp3D)

    Args:
        dim: channels
        input_resolution: (D,H,W)
        num_heads: attention heads
        window_size: (wd,wh,ww)
        shift_size: (sd,sh,sw)
        mlp_ratio: expansion ratio for FFN
        drop_path: drop path prob.
        nas: bool, use MixedOp3D for FFN if True
    """
    def __init__(
        self,
        dim,
        input_resolution,
        num_heads=4,
        window_size=(2,7,7),
        shift_size=(0,0,0),
        mlp_ratio=4.0,
        drop_path=0.0,
        nas=True,
    ):
        super().__init__()
        self.dim = dim
        self.D, self.H, self.W = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.nas = nas

        # ----------------------------
        # Layer Norms
        # ----------------------------
        self.norm1 = LayerNormChannel(dim)
        self.norm2 = LayerNormChannel(dim)

        # ----------------------------
        # Window Attention
        # ----------------------------
        self.attn = WindowAttention3D(
            dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=0.0,
            proj_drop=0.0,
        )

        # ----------------------------
        # DropPath for residual
        # ----------------------------
        self.drop_path = DropPath(drop_path)

        # ----------------------------
        # MLP / FFN
        # If NAS is enabled → use MixedOp3D
        # otherwise standard MLP (in channel-last)
        # ----------------------------
        hidden_dim = int(dim * mlp_ratio)

        if nas:
            # MixedOp3D operates on (B,C,D,H,W) so we need Conv3D-like ops.
            self.ffn = MixedOp3D(dim, dim)  # output keeps same channels
        else:
            # Standard FFN layer
            self.ffn_fc1 = nn.Conv3d(dim, hidden_dim, kernel_size=1)
            self.ffn_act = nn.GELU()
            self.ffn_fc2 = nn.Conv3d(hidden_dim, dim, kernel_size=1)

        # ----------------------------
        # Mask cache for SW-MSA
        # ----------------------------
        self.register_buffer("attn_mask", None, persistent=False)

    # --------------------------------------------------------
    # Helper: Calculate attention mask for SW-MSA (once only)
    # --------------------------------------------------------
    def calculate_mask(self, x):
        """
        Compute mask only once and cache it.
        Shifted window attention needs a mask.
        """
        if self.attn_mask is not None:
            return self.attn_mask

        D, H, W = self.D, self.H, self.W
        wd, wh, ww = self.window_size
        sd, sh, sw = self.shift_size

        if sd == 0 and sh == 0 and sw == 0:
            self.attn_mask = None
            return None

        # Use helper from layers.py
        mask = compute_attn_mask(D, H, W, self.window_size, self.shift_size, x.device)
        self.attn_mask = mask
        return mask

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------
    def forward(self, x):
        """
        x: (B, C, D, H, W)
        """
        B, C, D, H, W = x.shape
        assert D == self.D and H == self.H and W == self.W, \
            "input resolution mismatch"

        # ----------------------------
        # Norm1 &  Window Attention
        # ----------------------------
        shortcut = x
        x = self.norm1(x)

        # Apply shift
        sd, sh, sw = self.shift_size
        if sd or sh or sw:
            shifted_x = torch.roll(x, shifts=(-sd, -sh, -sw), dims=(2,3,4))
        else:
            shifted_x = x

        # Partition windows
        windows = window_partition(shifted_x, self.window_size)
        # convert to sequence (B*nW, N, C)
        wd, wh, ww = self.window_size
        N = wd * wh * ww
        windows_flat = windows.view(-1, N, C)

        # Attention: use mask only for SW-MSA
        mask = self.calculate_mask(x)
        attn_windows = self.attn(windows_flat, mask=mask)

        # Restore window shape
        attn_windows = attn_windows.view(-1, wd, wh, ww, C)

        # Reverse windows
        shifted_x = window_reverse(attn_windows, self.window_size, B, D, H, W)

        # Reverse shift
        if sd or sh or sw:
            x_attn = torch.roll(shifted_x, shifts=(sd, sh, sw), dims=(2,3,4))
        else:
            x_attn = shifted_x

        # Residual connection
        x = shortcut + self.drop_path(x_attn)

        # ----------------------------
        # FFN / NAS-FFN
        # ----------------------------
        shortcut2 = x
        x = self.norm2(x)

        if self.nas:
            # MixedOp3D expects (B,C,D,H,W)
            x = self.ffn(x)
        else:
            x = self.ffn_fc1(x)
            x = self.ffn_act(x)
            x = self.ffn_fc2(x)

        # Residual
        x = shortcut2 + self.drop_path(x)

        return x
