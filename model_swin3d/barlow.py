"""
model_swin3d/barlow.py
----------------------
Components for Barlow Twins self-supervised pretraining:
 - BarlowProjector: MLP head (Updated to use BatchNorm1d for larger batches)
 - BarlowLoss: cross-correlation loss
 - 3D data augmentations
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
    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            # 第一层
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            # 第二层 (新增)
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            # 第三层 (输出层，通常不加 ReLU，原论文也不加 BN，但加上有助于训练稳定性)
            nn.Linear(hidden_dim, out_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        """
        # === [修改] 强制转换为 float32 以避免 FP16 下的数值不稳定 ===
        z1 = z1.float()
        z2 = z2.float()
        # =======================================================

        N, D = z1.shape

        # 保护机制：如果 Batch=1，BN 和相关性计算都会失败
        if N == 1:
            # 回退到简单的余弦相似度
            z1_norm = z1 / (z1.norm(dim=1, keepdim=True) + self.eps)
            z2_norm = z2 / (z2.norm(dim=1, keepdim=True) + self.eps)
            cosine_sim = (z1_norm * z2_norm).sum(dim=1)
            loss = (1 - cosine_sim).mean()
            return loss, np.array([1.0]), np.array(0.0)

        # ---------------------------------------------------------
        # [关键归一化步骤]：沿着 Batch 维度 (dim=0) 进行归一化
        # 这使得每个特征维度在当前 Batch 内均值为 0，标准差为 1
        # ---------------------------------------------------------
        z1_mean = z1.mean(0, keepdim=True)
        z2_mean = z2.mean(0, keepdim=True)
        z1_std = z1.std(0, unbiased=False, keepdim=True) + self.eps
        z2_std = z2.std(0, unbiased=False, keepdim=True) + self.eps

        z1_norm = (z1 - z1_mean) / z1_std
        z2_norm = (z2 - z2_mean) / z2_std

        # 计算互相关矩阵 C = (Z1_norm.T @ Z2_norm) / N
        c = (z1_norm.T @ z2_norm) / N  # 形状: (D, D)

        # 1. 对角线项：希望 C_ii 接近 1 (Invariance)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()

        # 2. 非对角线项：希望 C_ij 接近 0 (Redundancy Reduction)
        off_diag = off_diagonal(c).pow_(2).sum()

        loss = on_diag + self.lambda_offdiag * off_diag

        return loss, torch.diag(c).detach().cpu().numpy(), off_diag.detach().cpu().numpy()


# -------------------------
# 3D augmentations
# -------------------------
def rand_flip(vol: np.ndarray) -> np.ndarray:
    if random.random() < 0.5:
        vol = np.flip(vol, axis=0).copy()
    if random.random() < 0.5:
        vol = np.flip(vol, axis=1).copy()
    if random.random() < 0.5:
        vol = np.flip(vol, axis=2).copy()
    return vol


def rand_intensity_scale_shift(vol: np.ndarray) -> np.ndarray:
    s = random.uniform(0.9, 1.1)
    sh = random.uniform(-0.1, 0.1)
    vol = vol * s + sh
    return vol


def random_crop_or_resize(vol: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    d, h, w = vol.shape
    td, th, tw = target_shape
    if d >= td and h >= th and w >= tw:
        sd = random.randint(0, d - td)
        sh = random.randint(0, h - th)
        sw = random.randint(0, w - tw)
        return vol[sd: sd + td, sh: sh + th, sw: sw + tw].copy()
    else:
        factors = (td / d, th / h, tw / w)
        return ndi.zoom(vol, factors, order=1)


def apply_barlow_augment(vol: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    v = vol.copy()
    v = rand_flip(v)
    v = rand_intensity_scale_shift(v)
    v = random_crop_or_resize(v, target_shape)
    return v