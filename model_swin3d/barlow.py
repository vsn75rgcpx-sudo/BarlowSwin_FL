"""
model_swin3d/barlow.py
----------------------
Components for Barlow Twins self-supervised pretraining:
 - BarlowProjector: MLP head (optimized for 3D small batch)
 - BarlowLoss: cross-correlation loss
 - 3D data augmentations (flip / intensity / crop-or-resize)
"""

import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.ndimage as ndi


# -------------------------
# Projector MLP for Barlow Twins
# -------------------------
class BarlowProjector(nn.Module):
    """
    Projector for Barlow Twins.
    For 3D medical data with small batches, use out_dim=512 or 1024 (not 2048).
    Uses LayerNorm instead of BatchNorm to support batch_size=1.
    """
    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),  # LayerNorm works with batch_size=1
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C) flattened features
        Returns:
            (N, out_dim) projected features
        """
        return self.net(x)


# -------------------------
# Barlow Twins Loss (batch-level cross-correlation)
# -------------------------
def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    """Return flattened off-diagonal elements of a square matrix."""
    n = x.shape[0]
    assert x.shape[0] == x.shape[1]
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowLoss(nn.Module):
    """
    Barlow Twins loss: encourages on-diagonal correlation = 1, off-diagonal = 0.
    """
    def __init__(self, lambda_offdiag: float = 0.005, eps: float = 1e-12):
        super().__init__()
        self.lambda_offdiag = lambda_offdiag
        self.eps = eps

    def forward(self, z1: torch.Tensor, z2: torch.Tensor):
        """
        Args:
            z1, z2: (N, D) projected features (not normalized yet)
        Returns:
            loss: scalar tensor
            diag_vals: numpy array of diagonal of cross-corr
            off_diag_val: numpy scalar of off-diagonal penalty (for logging)
        """
        N, D = z1.shape

        # For batch_size=1, Barlow Twins loss is not well-defined
        # We use a simplified version: cosine similarity loss
        if N == 1:
            # Single sample: use cosine similarity as proxy
            z1_norm = z1 / (z1.norm(dim=1, keepdim=True) + self.eps)
            z2_norm = z2 / (z2.norm(dim=1, keepdim=True) + self.eps)
            cosine_sim = (z1_norm * z2_norm).sum(dim=1)
            loss = (1 - cosine_sim).mean()
            # Return dummy values for diag and off_diag
            diag_vals = np.array([cosine_sim.item()])
            off_diag_val = np.array(0.0)
            return loss, diag_vals, off_diag_val

        # Normalize each feature dimension
        # Use unbiased=False for better stability with small batches
        z1_mean = z1.mean(0, keepdim=True)
        z2_mean = z2.mean(0, keepdim=True)
        z1_std = z1.std(0, unbiased=False, keepdim=True) + self.eps
        z2_std = z2.std(0, unbiased=False, keepdim=True) + self.eps
        
        z1 = (z1 - z1_mean) / z1_std
        z2 = (z2 - z2_mean) / z2_std

        # Cross-correlation matrix
        c = (z1.T @ z2) / N  # (D, D)

        # On-diagonal: should be close to 1
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        # Off-diagonal: should be close to 0
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambda_offdiag * off_diag

        # Check for NaN
        if torch.isnan(loss):
            # Fallback to cosine similarity if NaN
            z1_norm = z1 / (z1.norm(dim=1, keepdim=True) + self.eps)
            z2_norm = z2 / (z2.norm(dim=1, keepdim=True) + self.eps)
            cosine_sim = (z1_norm * z2_norm).sum(dim=1)
            loss = (1 - cosine_sim).mean()
            diag_vals = np.array([cosine_sim.mean().item()])
            off_diag_val = np.array(0.0)
            return loss, diag_vals, off_diag_val

        return loss, torch.diag(c).detach().cpu().numpy(), off_diag.detach().cpu().numpy()


# -------------------------
# 3D augmentations for medical volumes (two views)
# -------------------------
def rand_flip(vol: np.ndarray) -> np.ndarray:
    """Flip along random axes."""
    if random.random() < 0.5:
        vol = np.flip(vol, axis=0).copy()
    if random.random() < 0.5:
        vol = np.flip(vol, axis=1).copy()
    if random.random() < 0.5:
        vol = np.flip(vol, axis=2).copy()
    return vol


def rand_intensity_scale_shift(
    vol: np.ndarray,
    scale_range: Tuple[float, float] = (0.9, 1.1),
    shift_range: Tuple[float, float] = (-0.1, 0.1),
) -> np.ndarray:
    """Random intensity scaling and shifting."""
    s = random.uniform(*scale_range)
    sh = random.uniform(*shift_range)
    vol = vol * s + sh
    return vol


def random_crop_or_resize(vol: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """If vol shape >= target shape, random crop; else resize via zoom."""
    d, h, w = vol.shape
    td, th, tw = target_shape
    if d >= td and h >= th and w >= tw:
        sd = random.randint(0, d - td)
        sh = random.randint(0, h - th)
        sw = random.randint(0, w - tw)
        return vol[sd : sd + td, sh : sh + th, sw : sw + tw].copy()
    else:
        factors = (td / d, th / h, tw / w)
        return ndi.zoom(vol, factors, order=1)


def apply_barlow_augment(vol: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """Apply a sequence of augmentations for Barlow Twins."""
    v = vol.copy()
    v = rand_flip(v)
    v = rand_intensity_scale_shift(v)
    v = random_crop_or_resize(v, target_shape)
    return v
