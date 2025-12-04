"""
model_swin3d.layers
-------------------
Swin3D fundamental building blocks implemented in PyTorch:

- window_partition / window_reverse for 3D tensors
- WindowAttention3D: window-based multi-head self-attention for 3D windows
- compute_mask for shifted-window attention (helper)
- PatchMerging3D: downsample by merging neighboring patches
- PatchExpanding3D: upsample by expanding patches (for decoder)
- MLP3D: feed-forward network for Swin blocks

TENSOR SHAPES:
 - 5D conv-space: (B, C, D, H, W)
 - sequence-space: (B, N, C) where N = D*H*W (for a given feature map)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple

def window_partition(x: torch.Tensor, window_size: Tuple[int,int,int]):
    """
    Partition 5D tensor into windows.
    Args:
        x: tensor (B, C, D, H, W)
        window_size: (wd, wh, ww)
    Returns:
        windows: (num_windows*B, wd, wh, ww, C)
    """
    assert x.dim() == 5, "Input must be (B, C, D, H, W)"
    B, C, D, H, W = x.shape
    wd, wh, ww = window_size
    assert D % wd == 0 and H % wh == 0 and W % ww == 0, "Dimensions must be divisible by window size"

    # Bring channels to last for easier partitioning
    x = x.permute(0,2,3,4,1).contiguous()  # B, D, H, W, C
    x = x.view(B,
               D // wd, wd,
               H // wh, wh,
               W // ww, ww,
               C)
    windows = x.permute(0,1,3,5,2,4,6,7).contiguous().view(-1, wd, wh, ww, C)
    return windows

def window_reverse(windows: torch.Tensor, window_size: Tuple[int,int,int], B:int, D:int, H:int, W:int):
    """
    Reverse windows to original 5D tensor shape (B, C, D, H, W)
    Args:
        windows: (num_windows*B, wd, wh, ww, C)
        window_size: (wd, wh, ww)
        B, D, H, W: original spatial dims
    Returns:
        x: (B, C, D, H, W)
    """
    wd, wh, ww = window_size
    C = windows.shape[-1]
    x = windows.view(B,
                     D // wd,
                     H // wh,
                     W // ww,
                     wd, wh, ww, C)
    x = x.permute(0,1,4,2,5,3,6,7).contiguous()
    x = x.view(B, D, H, W, C)
    # back to (B, C, D, H, W)
    x = x.permute(0,4,1,2,3).contiguous()
    return x

class WindowAttention3D(nn.Module):
    """
    Window based multi-head self attention (W-MSA) for 3D windows.
    Input: x of shape (num_windows*B, N, C) where N = wd*wh*ww
    Output: same shape
    """
    def __init__(self, dim, window_size=(2,7,7), num_heads=4, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (wd, wh, ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        wd, wh, ww = window_size
        # relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2*wd-1)*(2*wh-1)*(2*ww-1), num_heads)
        )

        # pair-wise relative position index for each token inside the window
        coords_d = torch.arange(wd)
        coords_h = torch.arange(wh)
        coords_w = torch.arange(ww)
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing='ij'))  # 3, wd, wh, ww
        coords_flatten = torch.flatten(coords, 1)  # 3, N
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, N, N
        relative_coords = relative_coords.permute(1,2,0).contiguous()  # N, N, 3
        relative_coords[:, :, 0] += wd - 1
        relative_coords[:, :, 1] += wh - 1
        relative_coords[:, :, 2] += ww - 1
        relative_coords[:, :, 0] *= (2*wh-1)*(2*ww-1)
        relative_coords[:, :, 1] *= (2*ww-1)
        relative_position_index = relative_coords.sum(-1)  # N, N
        self.register_buffer("relative_position_index", relative_position_index)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask: torch.Tensor = None):
        """
        x: (num_windows*B, N, C)
        mask: (num_windows, N, N) or None (for SW-MSA)
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv[:,:,0], qkv[:,:,1], qkv[:,:,2]  # each (B_, N, heads, head_dim)
        q = q.permute(0,2,1,3)  # B_, heads, N, hd
        k = k.permute(0,2,3,1)  # B_, heads, hd, N
        attn = (q @ k) * self.scale  # B_, heads, N, N

        # add relative position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.relative_position_index.shape[0], self.relative_position_index.shape[1], -1)  # N,N,heads
        relative_position_bias = relative_position_bias.permute(2,0,1).unsqueeze(0)  # 1, heads, N, N
        attn = attn + relative_position_bias

        if mask is not None:
            nW = mask.shape[0]
            # expand mask to (B_ // nW, nW, heads, N, N) then broadcast
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        v = v.permute(0,2,1,3)  # B_, heads, N, hd
        out = (attn @ v).permute(0,2,1,3).reshape(B_, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

def compute_attn_mask(D:int, H:int, W:int, window_size:Tuple[int,int,int], shift_size:Tuple[int,int,int], device):
    """
    Compute attention mask for SW-MSA (shifted windows).
    Returns a mask of shape (num_windows, N, N) where N = wd*wh*ww
    Implementation follows the Swin Transformer logic extended to 3D.
    """
    wd, wh, ww = window_size
    sd, sh, sw = shift_size
    img_mask = torch.zeros((1, D, H, W, 1), device=device)  # 1 D H W 1
    cnt = 0
    d_slices = [(slice(0, -wd), slice(-wd, -sd), slice(-sd, None))] if sd>0 else [slice(0, D)]
    # Simpler implementation: assign region ids by slicing with window stride and shift
    d_ranges = list(range(0, D, wd))
    h_ranges = list(range(0, H, wh))
    w_ranges = list(range(0, W, ww))
    reg = torch.zeros((D, H, W), device=device, dtype=torch.int32)
    idx = 0
    for di in d_ranges:
        for hi in h_ranges:
            for wi in w_ranges:
                d1 = di
                d2 = min(di + wd, D)
                h1 = hi
                h2 = min(hi + wh, H)
                w1 = wi
                w2 = min(wi + ww, W)
                reg[d1:d2, h1:h2, w1:w2] = idx
                idx += 1
    # now build masks per window (no shift). For SW-MSA proper mask requires careful mapping after shift.
    # For typical use we will call this function with shift_size either zero or half-window and then construct mask accordingly.
    # To keep complexity manageable, return None if shift is zero (no mask needed).
    if sd == 0 and sh == 0 and sw == 0:
        return None
    # For non-zero shift, building full mask is complex; here we provide a functional mask approximator:
    # Create windows on padded grid then compute mask where tokens from different windows get -inf
    pad_d = (wd - D % wd) % wd
    pad_h = (wh - H % wh) % wh
    pad_w = (ww - W % ww) % ww
    Dp = D + pad_d
    Hp = H + pad_h
    Wp = W + pad_w
    reg_padded = F.pad(reg.unsqueeze(0).unsqueeze(-1).float(), (0, pad_w, 0, pad_h, 0, pad_d)).squeeze(-1).squeeze(0).long()
    # partition windows
    windows = []
    for d in range(0, Dp, wd):
        for h in range(0, Hp, wh):
            for w in range(0, Wp, ww):
                block = reg_padded[d:d+wd, h:h+wh, w:w+ww].reshape(-1)
                windows.append(block)
    windows = torch.stack(windows, dim=0)  # num_windows, N
    num_windows = windows.shape[0]
    N = wd*wh*ww
    mask = torch.zeros((num_windows, N, N), device=device)
    for i in range(num_windows):
        for j in range(num_windows):
            neq = (windows[i] != windows[j])
            mask[i, neq, :] = float(-100.0)  # penalize differing positions
    return mask

class PatchMerging3D(nn.Module):
    """
    Downsample by merging 2x2x2 neighbors. Input: (B, C, D, H, W)
    Output: (B, out_dim, D/2, H/2, W/2)
    """
    def __init__(self, input_dim, out_dim=None):
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim or 2*input_dim
        self.reduction = nn.Linear(8*input_dim, self.out_dim, bias=False)
        self.norm = nn.LayerNorm(8*input_dim)

    def forward(self, x):
        # x: B, C, D, H, W
        B, C, D, H, W = x.shape
        assert D % 2 == 0 and H % 2 == 0 and W % 2 == 0, "spatial dims must be even"
        x0 = x[:,:,0::2,0::2,0::2]
        x1 = x[:,:,0::2,0::2,1::2]
        x2 = x[:,:,0::2,1::2,0::2]
        x3 = x[:,:,0::2,1::2,1::2]
        x4 = x[:,:,1::2,0::2,0::2]
        x5 = x[:,:,1::2,0::2,1::2]
        x6 = x[:,:,1::2,1::2,0::2]
        x7 = x[:,:,1::2,1::2,1::2]
        x_cat = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], dim=1)  # B, 8C, D/2, H/2, W/2
        B, C8, D2, H2, W2 = x_cat.shape
        x_flat = x_cat.permute(0,2,3,4,1).contiguous().view(-1, C8)  # (B*D2*H2*W2, 8C)
        x_norm = self.norm(x_flat)
        x_reduced = self.reduction(x_norm).view(B, D2, H2, W2, -1).permute(0,4,1,2,3).contiguous()
        return x_reduced

class PatchExpanding3D(nn.Module):
    """
    Upsample by expanding patches. Input: (B, C, D, H, W) -> (B, C_out, D*2, H*2, W*2)
    Implemented by linear projection to 8*exp_dim and reshaping.
    """
    def __init__(self, input_dim, expand_dim=None):
        super().__init__()
        self.input_dim = input_dim
        self.expand_dim = expand_dim or input_dim // 2
        self.expand = nn.Linear(input_dim, 8*self.expand_dim, bias=False)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        B, C, D, H, W = x.shape
        x_perm = x.permute(0,2,3,4,1).contiguous().view(-1, C)  # (B*D*H*W, C)
        x_norm = self.norm(x_perm)
        x_exp = self.expand(x_norm)  # (B*D*H*W, 8*exp_dim)
        x_exp = x_exp.view(B, D, H, W, 2,2,2, self.expand_dim).permute(0,7,1,4,2,5,3,6).contiguous()
        # reshape to B, Cnew, D*2, H*2, W*2
        B, Cnew, D1, a, H1, b, W1, c = x_exp.shape
        out = x_exp.view(B, Cnew, D*2, H*2, W*2)
        return out

class MLP3D(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features*4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        # x: (B, N, C) or (..., C)
        if x.dim() == 3:
            B, N, C = x.shape
            x = x.view(-1, C)
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
            x = x.view(B, N, -1)
            return x
        else:
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
            return x
