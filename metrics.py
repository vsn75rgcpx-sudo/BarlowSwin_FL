"""
metrics.py
----------
Evaluation metrics for 3D segmentation:
- dice_score (F1)
- jaccard_score (mIoU)
- pixel_accuracy (PA)
- mean_pixel_accuracy (MPA)
- hd95
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from scipy import ndimage


def get_one_hot(preds, num_classes):
    return F.one_hot(preds.argmax(dim=1), num_classes).permute(0, 4, 1, 2, 3)


def dice_score(preds, targets):
    """Dice score (F1 Score)"""
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


def jaccard_score(preds, targets):
    """Jaccard Coefficient (mIoU)"""
    num_classes = preds.shape[1]
    pred_soft = F.softmax(preds, dim=1)
    pred_label = torch.argmax(pred_soft, dim=1)
    iou_per_class = []
    eps = 1e-5
    for c in range(num_classes):
        p = (pred_label == c).float()
        t = (targets == c).float()
        inter = (p * t).sum()
        union = p.sum() + t.sum() - inter
        iou = (inter + eps) / (union + eps)
        iou_per_class.append(iou.item())
    return float(np.mean(iou_per_class))


def pixel_accuracy(preds, targets):
    """Global Pixel Accuracy (PA)"""
    pred_label = torch.argmax(preds, dim=1)
    correct = (pred_label == targets).float().sum()
    total = torch.numel(targets)
    return float(correct / (total + 1e-8))


def mean_pixel_accuracy(preds, targets):
    """Mean Pixel Accuracy (MPA) per class"""
    num_classes = preds.shape[1]
    pred_label = torch.argmax(preds, dim=1)
    acc_per_class = []
    for c in range(num_classes):
        t = (targets == c).float()
        p = (pred_label == c).float()

        total_c = t.sum()
        if total_c == 0:
            continue

        correct_c = (p * t).sum()
        acc = correct_c / total_c
        acc_per_class.append(acc.item())

    if len(acc_per_class) == 0:
        return 0.0
    return float(np.mean(acc_per_class))


def compute_hd95(preds, targets, num_classes=None):
    if num_classes is None:
        num_classes = preds.shape[1]
    pred_label = torch.argmax(preds, dim=1).cpu().numpy()
    target_label = targets.cpu().numpy()
    hd95_per_class = []
    for c in range(num_classes):
        p = (pred_label == c)
        t = (target_label == c)
        if np.sum(p) == 0 or np.sum(t) == 0:
            continue
        p_border = p ^ ndimage.binary_erosion(p)
        t_border = t ^ ndimage.binary_erosion(t)
        p_points = np.argwhere(p_border)
        t_points = np.argwhere(t_border)
        if len(p_points) == 0 or len(t_points) == 0:
            continue
        d_p_to_t = directed_hausdorff(p_points, t_points)[0]
        d_t_to_p = directed_hausdorff(t_points, p_points)[0]
        hd95_per_class.append(max(d_p_to_t, d_t_to_p))
    if len(hd95_per_class) == 0:
        return 100.0
    return float(np.mean(hd95_per_class))