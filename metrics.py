"""
metrics.py
----------
Evaluation metrics for 3D segmentation:
- dice_score (F1): Multi-class Dice loss / F1 Score
- jaccard_score: Intersection over Union (IoU)
- pixel_accuracy: Pixel Accuracy (PA)
- hd95: Hausdorff Distance 95%
- ssim3d: Structural Similarity Index
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from scipy import ndimage


def get_one_hot(preds, num_classes):
    return F.one_hot(preds.argmax(dim=1), num_classes).permute(0, 4, 1, 2, 3)


def dice_score(preds, targets):
    """
    Compute Dice score (F1 Score) for multi-class 3D segmentation.
    Dice = 2 * TP / (2 * TP + FP + FN)
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

    return float(np.mean(dice_per_class))


def f1_score(preds, targets):
    """Alias for Dice Score in segmentation tasks"""
    return dice_score(preds, targets)


def jaccard_score(preds, targets):
    """
    Compute Jaccard Coefficient (IoU).
    IoU = TP / (TP + FP + FN)
    """
    num_classes = preds.shape[1]
    pred_soft = F.softmax(preds, dim=1)
    pred_label = torch.argmax(pred_soft, dim=1)

    iou_per_class = []
    eps = 1e-5

    for c in range(num_classes):
        p = (pred_label == c).float()
        t = (targets == c).float()

        inter = (p * t).sum()
        union = p.sum() + t.sum() - inter  # IoU formula
        iou = (inter + eps) / (union + eps)
        iou_per_class.append(iou.item())

    return float(np.mean(iou_per_class))


def pixel_accuracy(preds, targets):
    """
    Compute Pixel Accuracy (PA).
    PA = (TP + TN) / Total_Pixels
    """
    pred_label = torch.argmax(preds, dim=1)
    correct = (pred_label == targets).float().sum()
    total = torch.numel(targets)
    return float(correct / total)


def compute_hd95(preds, targets, num_classes=None):
    """
    Compute Hausdorff Distance 95% (HD95).
    Note: This can be slow for large 3D volumes.
    """
    if num_classes is None:
        num_classes = preds.shape[1]

    pred_label = torch.argmax(preds, dim=1).cpu().numpy()
    target_label = targets.cpu().numpy()

    hd95_per_class = []

    for c in range(num_classes):
        # Skip background if class 0 is background and rarely segmented,
        # but here we compute all.
        p = (pred_label == c)
        t = (target_label == c)

        # If either is empty, distance is undefined (or max)
        if np.sum(p) == 0 or np.sum(t) == 0:
            # Assign a large value or skip
            # hd95_per_class.append(100.0)
            continue

            # Compute surface distances using directed hausdorff
        # Since exact HD95 on raw voxels is hard with scipy alone efficiently,
        # we stick to a simplified version or use robust approximation.
        # For speed in Python loop, we use a border extraction approach.

        # Extract boundaries
        p_border = p ^ ndimage.binary_erosion(p)
        t_border = t ^ ndimage.binary_erosion(t)

        p_points = np.argwhere(p_border)
        t_points = np.argwhere(t_border)

        if len(p_points) == 0 or len(t_points) == 0:
            continue

        # Calculate forward and backward directed Hausdorff distances
        d_p_to_t = directed_hausdorff(p_points, t_points)[0]
        d_t_to_p = directed_hausdorff(t_points, p_points)[0]

        # True HD is max(d_forward, d_backward), but for HD95 we typically need
        # the percentile of distances. SciPy doesn't give percentile directly.
        # We approximate HD95 here as the max distance (Standard HD) for simplicity
        # without external C++ libraries like MedPy.
        # If MedPy is strictly required, install `medpy` and use `metric.binary.hd95`.
        # Here we return the max HD as a proxy or use avg.
        hd95_per_class.append(max(d_p_to_t, d_t_to_p))

    if len(hd95_per_class) == 0:
        return 100.0  # High penalty if empty

    return float(np.mean(hd95_per_class))


def ssim3d(preds, targets, eps=1e-5):
    """
    Compute SSIM for 3D volumetric data (class-averaged).
    Assumes inputs are standardized.
    """
    num_classes = preds.shape[1]
    pred_soft = F.softmax(preds, dim=1)

    # We can compute SSIM on the probability maps for each class
    ssim_per_class = []

    for c in range(num_classes):
        p = pred_soft[:, c, ...].float()
        t = (targets == c).float()

        # Flatten to compute statistics globally for the volume
        pred_flat = p.flatten()
        target_flat = t.flatten()

        mu_x = pred_flat.mean()
        mu_y = target_flat.mean()
        sig_x = pred_flat.var()
        sig_y = target_flat.var()
        sig_xy = ((pred_flat - mu_x) * (target_flat - mu_y)).mean()

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_val = ((2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)) / \
                   ((mu_x ** 2 + mu_y ** 2 + C1) * (sig_x + sig_y + C2) + eps)

        ssim_per_class.append(ssim_val.item())

    return float(np.mean(ssim_per_class))


def psnr(pred, target, max_val=1.0):
    # Keep existing if needed, though usually SSIM is preferred for segmentation
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return 100.0
    return 20 * np.log10(max_val / np.sqrt(mse))