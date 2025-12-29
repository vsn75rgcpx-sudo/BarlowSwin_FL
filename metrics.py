import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from scipy import ndimage


def get_brats_regions(tensor_mask):
    """
    将标签 (0, 1, 2, 3) 转换为 BraTS 区域 (WT, TC, ET)。
    输入可以是 Tensor 或 Numpy array。
    """
    # 兼容 Numpy
    if isinstance(tensor_mask, np.ndarray):
        tensor_mask = torch.from_numpy(tensor_mask)

    wt = (tensor_mask > 0).float()
    tc = ((tensor_mask == 1) | (tensor_mask == 3)).float()
    et = (tensor_mask == 3).float()
    return wt, tc, et


def compute_dice_single(pred, target):
    """计算单个二值 mask 的 Dice"""
    eps = 1e-5
    # 确保是 Tensor
    if isinstance(pred, np.ndarray): pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray): target = torch.from_numpy(target)

    inter = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2 * inter + eps) / (union + eps)


def compute_hd95_single(pred, target):
    """计算单个二值 mask 的 HD95"""
    if isinstance(pred, torch.Tensor):
        pred_np = pred.detach().cpu().numpy().astype(bool)
    else:
        pred_np = pred.astype(bool)

    if isinstance(target, torch.Tensor):
        target_np = target.detach().cpu().numpy().astype(bool)
    else:
        target_np = target.astype(bool)

    # 边界情况处理
    if not np.any(pred_np) or not np.any(target_np):
        return 374.0  # 最大惩罚距离

    if np.array_equal(pred_np, target_np):
        return 0.0

    pred_coords = np.argwhere(pred_np)
    target_coords = np.argwhere(target_np)

    d_p_to_t = directed_hausdorff(pred_coords, target_coords)[0]
    d_t_to_p = directed_hausdorff(target_coords, pred_coords)[0]

    return max(d_p_to_t, d_t_to_p)


# === 核心修复：重写后处理函数 ===
def postprocess_remove_small_objects(pred_mask, min_size=200):
    """
    后处理：移除小于 min_size 个体素的连通域。
    pred_mask: (D, H, W) numpy array, 值为类别标签 (0,1,2,3)
    """
    # 策略：对每个前景类别分别处理
    result = np.copy(pred_mask)
    for c in [1, 2, 3]:
        binary_class = (pred_mask == c)
        if not np.any(binary_class):
            continue

        # 标记连通域
        labeled_array, num_features = ndimage.label(binary_class)

        # 如果没有连通域，跳过
        if num_features == 0:
            continue

        # 计算每个连通域的大小 (index 1 to num_features)
        component_sizes = ndimage.sum(binary_class, labeled_array, range(1, num_features + 1))

        # 找到太小的连通域 label
        # np.where 返回 tuple, 取 [0] 得到索引数组
        small_indices = np.where(component_sizes < min_size)[0] + 1

        if len(small_indices) > 0:
            # 使用 np.isin 生成布尔掩码，标记所有属于小连通域的像素
            remove_mask = np.isin(labeled_array, small_indices)
            # 将这些位置置为背景 (0)
            result[remove_mask] = 0

    return result


def calculate_metrics_from_mask(pred_mask, target_mask):
    """
    [新函数] 从最终的 Mask 计算所有 BraTS 指标
    pred_mask: (D, H, W) 预测标签 (0,1,2,3)
    target_mask: (D, H, W) 真实标签
    """
    p_wt, p_tc, p_et = get_brats_regions(pred_mask)
    t_wt, t_tc, t_et = get_brats_regions(target_mask)

    results = {}

    # Dice
    results["Dice_WT"] = compute_dice_single(p_wt, t_wt).item()
    results["Dice_TC"] = compute_dice_single(p_tc, t_tc).item()
    results["Dice_ET"] = compute_dice_single(p_et, t_et).item()
    results["Dice_Mean"] = (results["Dice_WT"] + results["Dice_TC"] + results["Dice_ET"]) / 3.0

    # HD95 (测试时必算)
    results["HD95_WT"] = compute_hd95_single(p_wt, t_wt)
    results["HD95_TC"] = compute_hd95_single(p_tc, t_tc)
    results["HD95_ET"] = compute_hd95_single(p_et, t_et)
    results["HD95_Mean"] = (results["HD95_WT"] + results["HD95_TC"] + results["HD95_ET"]) / 3.0

    return results


# 保留旧接口兼容训练
def calculate_brats_metrics(preds, targets, compute_hd95=False):
    pred_labels = torch.argmax(preds, dim=1)
    # 这里只取 Batch 中的第一个做演示，通常 FL 训练 batch_size 较小
    # 实际上训练日志只需要一个粗略的 Mean Dice
    res = calculate_metrics_from_mask(pred_labels[0], targets[0])
    return {
        "dice_mean": res["Dice_Mean"],
        "dice_wt": res["Dice_WT"],
        "dice_tc": res["Dice_TC"],
        "dice_et": res["Dice_ET"]
    }


def dice_score(preds, targets):
    res = calculate_brats_metrics(preds, targets)
    return res["dice_mean"]


# 辅助指标
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
        for c in range(num_classes):
            pred_inds = (pred == c)
            target_inds = (target == c)
            intersection = (pred_inds & target_inds).sum().float()
            union = (pred_inds | target_inds).sum().float()
            if union == 0:
                ious.append(float('nan'))
            else:
                ious.append(intersection / union)
        valid_ious = [x for x in ious if not torch.isnan(torch.tensor(x))]
        return torch.tensor(valid_ious).mean().item() if valid_ious else 0.0


def sliding_window_inference(inputs, model, window_size=(96, 96, 96), num_classes=4, overlap=0.5):
    """
    3D 滑动窗口推理
    inputs: (B, C, D, H, W)  (通常 B=1)
    model: 训练好的网络
    window_size: 模型训练时的切片大小
    num_classes: 输出类别数
    overlap: 窗口重叠比例 (0.5 表示每次移动半个窗口)
    """
    B, C, D, H, W = inputs.shape

    # 1. 初始化输出容器 (累计 Logits 和 计数器)
    output_sum = torch.zeros((B, num_classes, D, H, W), device=inputs.device)
    count_map = torch.zeros((B, num_classes, D, H, W), device=inputs.device)

    # 2. 计算步长
    strides = [int(w * (1 - overlap)) for w in window_size]

    # 3. 确定 Patch 坐标
    # 逻辑：从 0 开始，每次加 stride，直到覆盖全图
    # 如果最后一步超出了，强制对齐到最右/下/后边缘
    d_steps = list(range(0, D - window_size[0] + 1, strides[0]))
    if d_steps[-1] != D - window_size[0]: d_steps.append(D - window_size[0])

    h_steps = list(range(0, H - window_size[1] + 1, strides[1]))
    if h_steps[-1] != H - window_size[1]: h_steps.append(H - window_size[1])

    w_steps = list(range(0, W - window_size[2] + 1, strides[2]))
    if w_steps[-1] != W - window_size[2]: w_steps.append(W - window_size[2])

    # 处理图像比窗口小的情况 (Padding) - 简单起见假设输入已 Pad 或足够大
    # BraTS 原始大小 240x240x155，窗口 96x96x96，通常不需要额外 Pad 除非被切过

    model.eval()
    with torch.no_grad():
        for d in d_steps:
            for h in h_steps:
                for w in w_steps:
                    # 截取 Patch
                    patch = inputs[..., d:d + window_size[0], h:h + window_size[1], w:w + window_size[2]]

                    # 推理
                    pred_patch = model(patch)  # logits

                    # 累加结果
                    output_sum[..., d:d + window_size[0], h:h + window_size[1], w:w + window_size[2]] += pred_patch
                    count_map[..., d:d + window_size[0], h:h + window_size[1], w:w + window_size[2]] += 1.0

    # 4. 平均
    output = output_sum / count_map
    return output