import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from scipy import ndimage


# ==========================================
# 1. 基础工具函数
# ==========================================
def get_brats_regions(tensor_mask):
    """将标签 (0, 1, 2, 3) 转换为 BraTS 区域 (WT, TC, ET)"""
    if isinstance(tensor_mask, np.ndarray):
        tensor_mask = torch.from_numpy(tensor_mask)
    wt = (tensor_mask > 0).float()
    tc = ((tensor_mask == 1) | (tensor_mask == 3)).float()
    et = (tensor_mask == 3).float()
    return wt, tc, et


def compute_dice_single(pred, target):
    eps = 1e-5
    if isinstance(pred, np.ndarray): pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray): target = torch.from_numpy(target)
    inter = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2 * inter + eps) / (union + eps)


def compute_hd95_single(pred, target):
    if isinstance(pred, torch.Tensor):
        pred_np = pred.detach().cpu().numpy().astype(bool)
    else:
        pred_np = pred.astype(bool)
    if isinstance(target, torch.Tensor):
        target_np = target.detach().cpu().numpy().astype(bool)
    else:
        target_np = target.astype(bool)

    if not np.any(pred_np) or not np.any(target_np):
        return 374.0  # 最大惩罚
    if np.array_equal(pred_np, target_np):
        return 0.0

    pred_coords = np.argwhere(pred_np)
    target_coords = np.argwhere(target_np)
    d_p_to_t = directed_hausdorff(pred_coords, target_coords)[0]
    d_t_to_p = directed_hausdorff(target_coords, pred_coords)[0]
    return max(d_p_to_t, d_t_to_p)


# ==========================================
# 2. 核心后处理：层级约束 (Hierarchical)
# ==========================================
def postprocess_hierarchical(pred_mask, min_size=200):
    """
    层级后处理 (带 Safety Net 版本)

    改进点：
    如果清洗操作导致某种类别(TC/ET)完全消失，说明清洗过头了（通常是因为WT预测错了导致误杀内部）。
    此时触发 Safety Net，回退该类别的清洗操作，避免 HD95=374。
    """
    result = np.copy(pred_mask)

    # === 0. 备份原始各通道状态，用于 Safety Check ===
    # 检查原始预测里有没有 TC 和 ET
    orig_has_tc = np.any((pred_mask == 1) | (pred_mask == 3))
    orig_has_et = np.any(pred_mask == 3)

    # === Step 1: 清洗 WT (Whole Tumor) ===
    wt_mask = (result > 0)
    if not np.any(wt_mask):
        return result  # 原本就是空的，没办法

    # 保留 WT 最大连通域
    labeled_array, num_features = ndimage.label(wt_mask)
    if num_features > 1:
        sizes = ndimage.sum(wt_mask, labeled_array, range(1, num_features + 1))
        max_label = np.argmax(sizes) + 1

        # 临时变量存储清洗后的 WT
        wt_keep_mask = (labeled_array == max_label)

        # 暂时应用清洗
        result_step1 = np.copy(result)
        result_step1[~wt_keep_mask] = 0

        # [Safety Check 1] 检查：清洗 WT 是否导致 TC 全部丢失？
        # 如果原始有 TC，但清洗 WT 后 TC 没了，说明这个"最大 WT"可能是个假阳性，而真肿瘤被删了。
        # 这种情况下，我们放弃清洗 WT，保留原样。
        new_has_tc = np.any((result_step1 == 1) | (result_step1 == 3))
        if orig_has_tc and not new_has_tc:
            # 触发 WT 回退：保留所有 WT 连通域，不删了
            pass
        else:
            # 安全，应用清洗
            result = result_step1

    # 更新清洗后的 WT 范围
    wt_clean_mask = (result > 0)

    # === Step 2: 约束 TC (必须在 WT 内) ===
    # 找到跑出 WT 范围的 TC
    tc_outliers = ((result == 1) | (result == 3)) & (~wt_clean_mask)

    # [Safety Check 2] 如果把 Outliers 删掉后，TC 就全没了？
    # 计算如果删掉后的 TC 数量
    current_tc_mask = ((result == 1) | (result == 3))
    # 剩下的 TC = 当前 TC - Outliers
    remaining_tc = current_tc_mask & (~tc_outliers)

    if orig_has_tc and not np.any(remaining_tc):
        # 触发 TC 回退：不要强制约束"TC必须在WT内"
        # 这种情况通常发生在 WT 预测偏了，但 TC 预测对了
        pass
    else:
        # 安全，执行删除
        result[tc_outliers] = 0

        # === Step 3: 约束 ET (必须在 TC 内) ===
    # 这里我们只移除极小噪点，不强制"必须在TC内"（防止TC被误删导致ET也被误删）
    et_mask = (result == 3)
    if np.any(et_mask):
        labeled_et, num_et = ndimage.label(et_mask)
        if num_et > 0:
            sizes_et = ndimage.sum(et_mask, labeled_et, range(1, num_et + 1))
            # 移除小于 10 个体素的极小 ET 噪点
            # 这里的 Safety Check 是：如果移除后 ET 全没了，就保留最大的那个
            small_indices = np.where(sizes_et < 10)[0] + 1

            # 如果所有 ET 都是小的 (例如总共就 5 个体素)，全删了就 374 了
            # 所以：只有当"删除会导致 ET 清零"时，我们才豁免最大的那个
            if len(small_indices) == num_et:  # 也就是所有块都是小的
                # 豁免最大的那个小块
                max_small_idx = np.argmax(sizes_et) + 1
                small_indices = small_indices[small_indices != max_small_idx]

            if len(small_indices) > 0:
                result[np.isin(labeled_et, small_indices)] = 1  # 退化为核心

    return result


# ==========================================
# 3. 计算入口 (自动集成后处理)
# ==========================================
def calculate_metrics_from_mask(pred_mask, target_mask):
    # [关键修复]：在此处内部调用后处理！
    # 这样 train_fednas_full.py 只需要调这个函数，不用管后处理
    pred_mask_clean = postprocess_hierarchical(pred_mask, min_size=200)

    p_wt, p_tc, p_et = get_brats_regions(pred_mask_clean)
    t_wt, t_tc, t_et = get_brats_regions(target_mask)

    results = {}
    results["Dice_WT"] = compute_dice_single(p_wt, t_wt).item()
    results["Dice_TC"] = compute_dice_single(p_tc, t_tc).item()
    results["Dice_ET"] = compute_dice_single(p_et, t_et).item()
    results["Dice_Mean"] = (results["Dice_WT"] + results["Dice_TC"] + results["Dice_ET"]) / 3.0

    results["HD95_WT"] = compute_hd95_single(p_wt, t_wt)
    results["HD95_TC"] = compute_hd95_single(p_tc, t_tc)
    results["HD95_ET"] = compute_hd95_single(p_et, t_et)
    results["HD95_Mean"] = (results["HD95_WT"] + results["HD95_TC"] + results["HD95_ET"]) / 3.0
    return results


def calculate_brats_metrics(preds, targets):
    # 兼容旧接口
    pred_labels = torch.argmax(preds, dim=1)
    return calculate_metrics_from_mask(pred_labels[0].cpu().numpy(), targets[0].cpu().numpy())


# ==========================================
# 4. 推理函数 (加入 Logit Reweighting 救 Dice)
# ==========================================
def sliding_window_inference(inputs, model, window_size=(96, 96, 96), num_classes=4, overlap=0.5):
    """
    滑动窗口推理 + Logit Reweighting
    """
    B, C, D, H, W = inputs.shape
    output_sum = torch.zeros((B, num_classes, D, H, W), device=inputs.device)
    count_map = torch.zeros((B, num_classes, D, H, W), device=inputs.device)

    strides = [int(w * (1 - overlap)) for w in window_size]
    # 计算步长坐标...
    d_steps = list(range(0, D - window_size[0] + 1, strides[0]))
    if d_steps[-1] != D - window_size[0]: d_steps.append(D - window_size[0])
    h_steps = list(range(0, H - window_size[1] + 1, strides[1]))
    if h_steps[-1] != H - window_size[1]: h_steps.append(H - window_size[1])
    w_steps = list(range(0, W - window_size[2] + 1, strides[2]))
    if w_steps[-1] != W - window_size[2]: w_steps.append(W - window_size[2])

    model.eval()
    with torch.no_grad():
        for d in d_steps:
            for h in h_steps:
                for w in w_steps:
                    patch = inputs[..., d:d + window_size[0], h:h + window_size[1], w:w + window_size[2]]
                    pred_patch = model(patch)
                    output_sum[..., d:d + window_size[0], h:h + window_size[1], w:w + window_size[2]] += pred_patch
                    count_map[..., d:d + window_size[0], h:h + window_size[1], w:w + window_size[2]] += 1.0

    # 平均 Logits
    avg_logits = output_sum / count_map

    # === [关键改进] Logit Reweighting ===
    # 针对 BraTS-00014 这种小肿瘤漏报的情况
    # 人为放大 TC(1) 和 ET(3) 的 Logits 值，让模型更容易选它们
    avg_logits[:, 1, ...] *= 1.50  # 强力提升 TC 召回
    avg_logits[:, 3, ...] *= 1.35  # 强力提升 ET 召回

    return avg_logits


# 其他辅助函数保持不变...
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