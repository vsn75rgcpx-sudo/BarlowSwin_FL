"""
model_swin3d.nas_ops (EXTENDED)
--------------------------------
Provides:
 - candidate primitive ops for FFN (Conv1/Conv3/Conv5/Depthwise/Identity)
 - MixedOp3D: softmax mix of FFN ops (as before)
 - MixedAttention3D: softmax mix of multiple WindowAttention3D variants
 - AlphaManager: manage alpha parameters for all searchable choices:
     for each Swin block we will keep:
        - alpha_ffn (len = num_ffn_choices)
        - alpha_attn (len = num_attn_choices)
        - alpha_gate (scalar)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from .layers import WindowAttention3D
import numpy as np

# -----------------------
# FFN candidate ops (3D)
# -----------------------
class Conv3x3x3(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_c),
            nn.GELU()
        )
    def forward(self, x): return self.op(x)

class Conv1x1x1(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_c),
            nn.GELU()
        )
    def forward(self, x): return self.op(x)

class Conv5x5x5(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm3d(out_c),
            nn.GELU()
        )
    def forward(self, x): return self.op(x)

class DepthwiseSeparable3D(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.dw = nn.Conv3d(in_c, in_c, kernel_size=3, padding=1, groups=in_c, bias=False)
        self.pw = nn.Conv3d(in_c, out_c, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm3d(out_c)
        self.act = nn.GELU()
    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)

class IdentityOp(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        if in_c == out_c:
            self.op = nn.Identity()
        else:
            self.op = nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_c)
            )
    def forward(self, x): return self.op(x)

# helper to build ffn op by name
def build_ffn_op(name: str, in_c: int, out_c: int):
    name = name.lower()
    if name == "conv3":
        return Conv3x3x3(in_c, out_c)
    if name == "conv1":
        return Conv1x1x1(in_c, out_c)
    if name == "conv5":
        return Conv5x5x5(in_c, out_c)
    if name == "dw" or name == "dwconv":
        return DepthwiseSeparable3D(in_c, out_c)
    if name == "identity":
        return IdentityOp(in_c, out_c)
    raise ValueError(f"Unknown ffn op: {name}")

# Alias for backward compatibility
build_op_3d = build_ffn_op

# -----------------------
# Mixed FFN Op (softmax over choices)
# -----------------------
class MixedOp3D(nn.Module):
    """
    MixedOp3D using Gumbel-Softmax sampling.
    - self.alpha: nn.Parameter logits for choices
    - forward supports:
        * soft: softmax(alpha) weighted sum (for validation arch evaluation)
        * gumbel sampling: gumbel_softmax(alpha/temperature, hard=hard) -> weight vector
          (use this for weight updates; hard=True gives ST-estimator near one-hot)
    - expected_flops() returns sum_i softprob_i * flops_i
    """
    def __init__(self, in_c, out_c, op_names=None):
        super().__init__()
        if op_names is None:
            op_names = ["conv3","conv1","dw","identity","conv5"]
        self.op_names = op_names
        self.ops = nn.ModuleList([build_ffn_op(n, in_c, out_c) for n in op_names])
        self.num_ops = len(self.ops)
        
        # architecture logits (alpha) - will be set externally by AlphaManager
        # For backward compatibility, we keep alpha as None initially
        self.alpha = None

        # approximate flops per op (simple heuristic)
        self._flops = [self._approx_flops(n, in_c, out_c) for n in op_names]

    def _approx_flops(self, name, in_c, out_c):
        # crude flops per output voxel
        if "conv3" in name:
            k = 3
            return in_c * out_c * (k**3)
        if "conv1" in name:
            k = 1
            return in_c * out_c
        if "conv5" in name:
            k = 5
            return in_c * out_c * (k**3)
        if "dw" in name:
            k = 3
            return in_c * (k**3) + in_c * out_c
        if "identity" in name:
            return 0.0
        return in_c * out_c * 27

    def forward_soft(self, x):
        """soft forward: weighted sum using softmax(alpha)"""
        if self.alpha is None:
            # uniform average fallback
            out = sum(op(x) for op in self.ops) / len(self.ops)
            return out
        
        probs = torch.softmax(self.alpha, dim=0)
        out = 0
        for p, op in zip(probs, self.ops):
            out = out + p * op(x)
        return out

    def forward_gumbel(self, x, temp=1.0, hard=False):
        """
        Gumbel-Softmax forward:
        - temp: temperature
        - hard: if True, returns a one-hot in forward (ST), else soft weights
        Returns: (output, weights) where weights can be used for logging
        """
        if self.alpha is None:
            # uniform average fallback
            out = sum(op(x) for op in self.ops) / len(self.ops)
            uniform_w = torch.ones(self.num_ops, device=x.device) / self.num_ops
            return out, uniform_w
        
        logits = self.alpha.unsqueeze(0)  # shape (1, K)
        g = F.gumbel_softmax(logits, tau=temp, hard=hard, dim=1).squeeze(0)  # shape (K,)
        
        # g is a weight vector summing to 1
        out = 0
        for wi, op in zip(g, self.ops):
            out = out + wi * op(x)
        return out, g  # return tensor and weights (weights for logging)

    def forward(self, x, sampled_idx=None):
        """
        Backward compatibility: if sampled_idx provided, use it; else use soft forward
        """
        if sampled_idx is not None:
            return self.ops[sampled_idx](x)
        return self.forward_soft(x)

    def expected_flops(self, temp=1.0):
        """expected flops under softmax(alpha/temp)"""
        if self.alpha is None:
            probs = np.array([1.0 / len(self._flops)] * len(self._flops))
        else:
            probs = torch.softmax(self.alpha / temp, dim=0).detach().cpu().numpy()
        return float(np.sum(np.array(self._flops) * probs))

# -----------------------
# Mixed Attention (softmax over attention variants)
# -----------------------
class MixedAttention3D(nn.Module):
    """
    Holds several WindowAttention3D variants (different window_size / num_heads).
    Each candidate returns same shape (B, N, C) → mix by alpha softmax.
    """
    def __init__(self, dim, candidates: List[dict]):
        """
        candidates: list of dicts each with keys: {"window_size": (wd,wh,ww), "num_heads": h}
        """
        super().__init__()
        self.atts = nn.ModuleList([
            WindowAttention3D(dim, window_size=cand["window_size"], num_heads=cand["num_heads"])
            for cand in candidates
        ])
        self.num_ops = len(self.atts)
        self.alpha = None
        self.dim = dim
        self.candidates = candidates

    def forward_soft(self, x, mask=None):
        """soft forward: weighted sum using softmax(alpha)"""
        if self.alpha is None:
            out = sum(att(x, mask=mask) for att in self.atts) / self.num_ops
            return out
        
        probs = torch.softmax(self.alpha, dim=0)
        out = 0
        for p, att in zip(probs, self.atts):
            out = out + p * att(x, mask=mask)
        return out

    def forward_gumbel(self, x, mask=None, temp=1.0, hard=False):
        """
        Gumbel-Softmax forward:
        - temp: temperature
        - hard: if True, returns a one-hot in forward (ST), else soft weights
        Returns: (output, weights) where weights can be used for logging
        """
        if self.alpha is None:
            out = sum(att(x, mask=mask) for att in self.atts) / self.num_ops
            uniform_w = torch.ones(self.num_ops, device=x.device) / self.num_ops
            return out, uniform_w
        
        logits = self.alpha.unsqueeze(0)  # shape (1, K)
        g = F.gumbel_softmax(logits, tau=temp, hard=hard, dim=1).squeeze(0)  # shape (K,)
        
        # g is a weight vector summing to 1
        out = 0
        for wi, att in zip(g, self.atts):
            out = out + wi * att(x, mask=mask)
        return out, g  # return tensor and weights

    def forward(self, x, mask=None, sampled_idx=None):
        """
        Backward compatibility: if sampled_idx provided, use it; else use soft forward
        """
        if sampled_idx is not None:
            return self.atts[sampled_idx](x, mask=mask)
        return self.forward_soft(x, mask=mask)
    
    def expected_flops(self, temp=1.0):
        """
        Return expected FLOPs (approximate) under softmax(alpha/temp) distribution.
        Attention FLOPs: O(N^2 * C) per head, where N = window_size product
        """
        if self.alpha is None:
            probs = np.array([1.0 / self.num_ops] * self.num_ops)
        else:
            probs = torch.softmax(self.alpha / temp, dim=0).detach().cpu().numpy()
        
        total_flops = 0.0
        for i, cand in enumerate(self.candidates):
            wd, wh, ww = cand["window_size"]
            num_heads = cand["num_heads"]
            N = wd * wh * ww
            # Approximate: QK^T and AV operations
            # QK^T: N * N * (C/num_heads) per head
            # AV: N * N * (C/num_heads) per head
            attn_flops = 2 * N * N * (self.dim // num_heads) * num_heads
            total_flops += probs[i] * attn_flops
        
        return float(total_flops)

# -----------------------
# Alpha Manager (extended)
# -----------------------

class AlphaManager(nn.Module):
    """
    Manage alpha groups. We will create alphas in order:
      for each Swin block:
         - alpha_ffn (len = num_ffn_choices)
         - alpha_attn (len = num_attn_choices)
         - alpha_gate (len = 1)
    We store them in a flat list for assignment.
    """
    def __init__(self, alpha_groups: List[int]):
        """
        alpha_groups: list of integers, each integer is the length of one alpha vector.
                      e.g. [4, 3, 1, 4, 3, 1, ...] repeating per block
        """
        super().__init__()
        self.alpha_list = nn.ParameterList([nn.Parameter(1e-3*torch.randn(n)) for n in alpha_groups])

    def forward(self):
        return list(self.alpha_list)

    def num_alpha(self):
        return len(self.alpha_list)

    def get_alpha(self, idx):
        return self.alpha_list[idx]

    def as_numpy(self):
        return [p.detach().cpu().numpy() for p in self.alpha_list]

