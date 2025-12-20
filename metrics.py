import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from scipy import ndimage


def get_brats_regions(tensor_mask):
    """
    将标签 (0, 1, 2, 3) 转换为 BraTS 区域 (WT, TC, ET)。
    假设标签: 0=背景, 1=坏死/非增强, 2=水肿, 3=增强肿瘤
    WT (Whole Tumor) = 1 + 2 + 3
    TC (Tumor Core)  = 1 + 3
    ET (Enhancing)   = 3
    """
    wt = (tensor_mask > 0).float()
    tc = ((tensor_mask == 1) | (tensor_mask == 3)).float()
    et = (tensor_mask == 3).float()
    return wt, tc, et


def compute_dice_single(pred, target):
    """计算单个二值 mask 的 Dice"""
    eps = 1e-5
    inter = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2 * inter + eps) / (union + eps)


def compute_hd95_single(pred, target):
    """使用 Scipy 计算单个二值 mask 的 HD95"""
    pred_np = pred.cpu().numpy().astype(bool)
    target_np = target.cpu().numpy().astype(bool)

    if not np.any(pred_np) or not np.any(target_np):
        return 374.0  # 如果一方为空，给予最大惩罚

    # 如果完全相同，距离为0
    if np.array_equal(pred_np, target_np):
        return 0.0

    d_p_to_t = directed_hausdorff(pred_np, target_np)[0]
    d_t_to_p = directed_hausdorff(target_np, pred_np)[0]
    return max(d_p_to_t, d_t_to_p)


def calculate_brats_metrics(preds, targets):
    """
    计算 WT, TC, ET 的 Dice 和 HD95。
    Args:
        preds: (N, C, D, H, W) 模型输出的 logits
        targets: (N, D, H, W) 真实标签 (0,1,2,3)
    Returns:
        dict: 包含各项指标的字典
    """
    pred_labels = torch.argmax(preds, dim=1)  # (N, D, H, W)

    dice_scores = {"wt": [], "tc": [], "et": []}
    hd95_scores = {"wt": [], "tc": [], "et": []}

    batch_size = preds.shape[0]

    for i in range(batch_size):
        # 提取三个区域的 mask
        p_wt, p_tc, p_et = get_brats_regions(pred_labels[i])
        t_wt, t_tc, t_et = get_brats_regions(targets[i])

        # 计算 Dice
        dice_scores["wt"].append(compute_dice_single(p_wt, t_wt).item())
        dice_scores["tc"].append(compute_dice_single(p_tc, t_tc).item())
        dice_scores["et"].append(compute_dice_single(p_et, t_et).item())

        # 计算 HD95 (注意：计算量较大，如果训练太慢可考虑只在验证时计算)
        hd95_scores["wt"].append(compute_hd95_single(p_wt, t_wt))
        hd95_scores["tc"].append(compute_hd95_single(p_tc, t_tc))
        hd95_scores["et"].append(compute_hd95_single(p_et, t_et))

    results = {
        "dice_wt": np.mean(dice_scores["wt"]),
        "dice_tc": np.mean(dice_scores["tc"]),
        "dice_et": np.mean(dice_scores["et"]),
        "dice_mean": np.mean(dice_scores["wt"] + dice_scores["tc"] + dice_scores["et"]),
        "hd95_wt": np.mean(hd95_scores["wt"]),
        "hd95_tc": np.mean(hd95_scores["tc"]),
        "hd95_et": np.mean(hd95_scores["et"]),
        "hd95_mean": np.mean(hd95_scores["wt"] + hd95_scores["tc"] + hd95_scores["et"])
    }
    return results


# 兼容旧代码的简单接口
def dice_score(preds, targets):
    res = calculate_brats_metrics(preds, targets)
    return res["dice_mean"]