"""
metrics.py
----------
Evaluation metrics for 3D segmentation:
- dice_score: Compute Dice coefficient per class and mean
- psnr: Peak Signal-to-Noise Ratio
- ssim3d: Structural Similarity Index for 3D volumes
"""

import torch
import torch.nn.functional as F
import numpy as np


def dice_score(preds, targets):
    """
    Compute Dice score for multi-class 3D segmentation.
    
    Args:
        preds: (N, C, D, H, W) raw logits
        targets: (N, D, H, W) long type (0..C-1)
    
    Returns:
        dice_list: list of dice scores per class
        mean_dice: average dice score across all classes
    """
    num_classes = preds.shape[1]
    pred_soft = F.softmax(preds, dim=1)
    pred_label = torch.argmax(pred_soft, dim=1)

    dice_per_class = []
    eps = 1e-5

    for c in range(num_classes):
        p = (pred_label == c).float()
        t = (targets == c).float()

        inter = (p * t).sum()
        union = p.sum() + t.sum()
        dice = (2 * inter + eps) / (union + eps)
        dice_per_class.append(dice.item())

    return dice_per_class, float(np.mean(dice_per_class))


def psnr(pred, target, max_val=1.0):
    """
    Compute Peak Signal-to-Noise Ratio.
    
    Args:
        pred: predicted tensor (any shape)
        target: target tensor (same shape as pred)
        max_val: maximum value of the signal (default 1.0)
    
    Returns:
        psnr_value: PSNR in dB
    """
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return 100.0
    return 20 * np.log10(max_val / np.sqrt(mse))


def ssim3d(pred, target, eps=1e-5):
    """
    Compute simplified SSIM for 3D volumetric data.
    
    Args:
        pred: predicted tensor (N, D, H, W) or (D, H, W)
        target: target tensor (same shape as pred)
        eps: small epsilon for numerical stability
    
    Returns:
        ssim_value: SSIM score (0-1)
    """
    # Flatten to compute statistics
    pred_flat = pred.flatten().float()
    target_flat = target.flatten().float()
    
    mu_x = pred_flat.mean()
    mu_y = target_flat.mean()
    sig_x = pred_flat.var()
    sig_y = target_flat.var()
    sig_xy = ((pred_flat - mu_x) * (target_flat - mu_y)).mean()

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_val = ((2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (sig_x + sig_y + C2) + eps)

    return ssim_val.item()

