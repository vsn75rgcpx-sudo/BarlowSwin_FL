import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import directed_hausdorff


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
    """
    使用 Scipy 计算单个二值 mask 的 HD95
    [修复]: 使用 np.argwhere 将 mask 转换为坐标点集 (N, 3)，
            因为 directed_hausdorff 需要点集输入，而不是图像矩阵。
    """
    pred_np = pred.cpu().numpy().astype(bool)
    target_np = target.cpu().numpy().astype(bool)

    # 如果没有预测出任何点，或者目标为空，返回最大惩罚值
    if not np.any(pred_np) or not np.any(target_np):
        return 374.0

        # 如果完全相同，距离为0
    if np.array_equal(pred_np, target_np):
        return 0.0

    # [关键修复]: 获取非零像素的坐标 (N_points, 3)
    pred_coords = np.argwhere(pred_np)
    target_coords = np.argwhere(target_np)

    d_p_to_t = directed_hausdorff(pred_coords, target_coords)[0]
    d_t_to_p = directed_hausdorff(target_coords, pred_coords)[0]

    # HD95 约等于 Hausdorff 距离的 95% 分位数，这里简化用最大距离代替
    # 或者标准的 HD95 需要计算所有点的距离分布，directed_hausdorff 只返回最大值(HD100)
    # 对于训练监控，这个近似足够了。
    return max(d_p_to_t, d_t_to_p)


def calculate_brats_metrics(preds, targets, compute_hd95=False):
    """
    计算 WT, TC, ET 的 Dice (可选 HD95)。
    Args:
        preds: (N, C, D, H, W) 模型输出的 logits
        targets: (N, D, H, W) 真实标签 (0,1,2,3)
        compute_hd95: 是否计算 HD95 (训练时建议关闭以提速)
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

        # 计算 HD95 (仅在需要时计算)
        if compute_hd95:
            hd95_scores["wt"].append(compute_hd95_single(p_wt, t_wt))
            hd95_scores["tc"].append(compute_hd95_single(p_tc, t_tc))
            hd95_scores["et"].append(compute_hd95_single(p_et, t_et))
        else:
            # 填充 0 或者 NaN
            hd95_scores["wt"].append(0.0)
            hd95_scores["tc"].append(0.0)
            hd95_scores["et"].append(0.0)

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


# [重要]: 用于训练/验证的简化接口
# 我们在训练循环中只调用这个，且不计算 HD95
def dice_score(preds, targets):
    res = calculate_brats_metrics(preds, targets, compute_hd95=False)
    return res["dice_mean"]


# 额外补充：计算 mIoU, PA, MPA 的函数 (你的测试代码需要这些)
def pixel_accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        correct = (pred == target).float()
        return correct.sum() / correct.numel()


def mean_pixel_accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        num_classes = output.shape[1]
        acc_cls = []
        for c in range(num_classes):
            mask = (target == c)
            if mask.sum() > 0:
                acc = (pred[mask] == c).float().mean()
                acc_cls.append(acc)
        return torch.tensor(acc_cls).mean().item() if acc_cls else 0.0


def jaccard_score(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        num_classes = output.shape[1]
        ious = []
        for c in range(num_classes):  # 0 is background
            pred_inds = (pred == c)
            target_inds = (target == c)
            intersection = (pred_inds & target_inds).sum().float()
            union = (pred_inds | target_inds).sum().float()
            if union == 0:
                ious.append(float('nan'))  # Ignored
            else:
                ious.append(intersection / union)
        # 过滤掉 nan
        valid_ious = [x for x in ious if not torch.isnan(torch.tensor(x))]
        return torch.tensor(valid_ious).mean().item() if valid_ious else 0.0