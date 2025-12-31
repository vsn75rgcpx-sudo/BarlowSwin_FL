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
def postprocess_hierarchical(pred_mask, min_size_wt=50, min_size_tc=30, min_size_et=10):
    """
    层级后处理 (修复 HD95 版)
    策略:
    1. WT (Whole Tumor): 严格只保留最大连通域 (解决 HD95 飙升问题)。
    2. TC (Tumor Core): 严格只保留最大连通域，且必须在 WT 内。
    3. ET (Enhancing): 允许存在多个小块(如多发强化)，但极小的噪点退化为 TC。
    """
    result = np.copy(pred_mask)

    # 0. 备份原始状态用于 Safety Net
    orig_has_tc = np.any((pred_mask == 1) | (pred_mask == 3))

    # === Step 1: WT (Whole Tumor) 处理 ===
    wt_mask = (result > 0)
    if np.any(wt_mask):
        labeled_array, num_features = ndimage.label(wt_mask)
        sizes = ndimage.sum(wt_mask, labeled_array, range(1, num_features + 1))

        # [核心修正] 只保留【最大】的 WT 连通域
        # 除非最大的那个都小于 min_size_wt (极罕见)，否则只取最大
        if num_features > 0:
            max_idx = np.argmax(sizes) + 1
            max_size = sizes[max_idx - 1]

            if max_size < min_size_wt:
                # 如果最大的都认为是噪点，那就全删了 (但在 Safety Net 可能会救回来)
                wt_keep_mask = np.zeros_like(wt_mask, dtype=bool)
            else:
                wt_keep_mask = (labeled_array == max_idx)

            # 执行清洗
            result_step1 = np.copy(result)
            result_step1[~wt_keep_mask] = 0

            # [Safety Net] 如果清洗 WT 导致 TC 全部丢失，说明预测的主 WT 可能是水肿，而真 TC 在另一个小 WT 里
            # 这种情况下，放弃本次 WT 清洗，保留原样
            new_has_tc = np.any((result_step1 == 1) | (result_step1 == 3))
            if orig_has_tc and not new_has_tc:
                pass  # 触发回退，不做操作
            else:
                result = result_step1

    # 更新 WT 掩膜
    wt_clean_mask = (result > 0)

    # === Step 2: TC (Tumor Core) 处理 ===
    # 先清除跑出 WT 范围的 TC (设为 2: Edema，保持 WT 形状完整)
    tc_outliers = ((result == 1) | (result == 3)) & (~wt_clean_mask)
    result[tc_outliers] = 2

    # 处理 TC 连通域
    tc_mask = (result == 1) | (result == 3)
    if np.any(tc_mask):
        labeled_tc, num_tc = ndimage.label(tc_mask)
        sizes_tc = ndimage.sum(tc_mask, labeled_tc, range(1, num_tc + 1))

        # [核心修正] TC 也只保留最大连通域 (防止离散噪点)
        if num_tc > 0:
            max_tc_idx = np.argmax(sizes_tc) + 1
            # 将非最大的 TC 区域降级为 Edema (Label 2)
            # 这样既去除了 TC 噪点，又不会让 WT 出现空洞
            mask_not_max = (labeled_tc != max_tc_idx) & tc_mask
            result[mask_not_max] = 2

    # === Step 3: ET (Enhancing Tumor) 处理 ===
    # ET 必须在 TC 内 (Step 2 已经确立了 TC 范围)
    current_tc_mask = (result == 1) | (result == 3)
    et_mask = (result == 3)

    # 清除跑出 TC 范围的 ET (设为 1: NCR)
    et_outliers = et_mask & (~current_tc_mask)
    result[et_outliers] = 1

    # ET 允许保留多个块 (应对多发强化)，但去除极小噪点
    et_mask = (result == 3)  # 更新 mask
    if np.any(et_mask):
        labeled_et, num_et = ndimage.label(et_mask)
        sizes_et = ndimage.sum(et_mask, labeled_et, range(1, num_et + 1))

        # 找到小于阈值的噪点
        small_et_indices = np.where(sizes_et < min_size_et)[0] + 1

        # 如果所有 ET 都是小的，保留最大的那个 (防止 ET Dice 归零)
        if len(small_et_indices) == num_et and num_et > 0:
            max_idx = np.argmax(sizes_et) + 1
            small_et_indices = small_et_indices[small_et_indices != max_idx]

        # 将噪点 ET 退化为 NCR (Label 1)
        mask_remove = np.isin(labeled_et, small_et_indices)
        result[mask_remove] = 1

    return result

# ==========================================
# 3. 计算入口 (自动集成后处理)
# ==========================================
def calculate_metrics_from_mask(pred_mask, target_mask):
    # [关键修复]：在此处内部调用后处理！
    # 这样 train_fednas_full.py 只需要调这个函数，不用管后处理
    pred_mask_clean = postprocess_hierarchical(pred_mask, min_size_wt=50, min_size_tc=30, min_size_et=10)

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
def sliding_window_inference(inputs, model, window_size=(96, 96, 96), num_classes=4, overlap=0.5,sw_weights=None):
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

    # === Logit Reweighting ===
    if sw_weights is not None:
        # 使用传入的权重
        w_tensor = torch.tensor(sw_weights, device=inputs.device).view(1, num_classes, 1, 1, 1)
        avg_logits = avg_logits * w_tensor
    else:
        # [建议修改] 默认使用更温和的权重
        # 1.2 倍足以提升召回，又不至于产生太多假阳性
        # BG=1.0, NCR=1.2, ED=1.0, ET=1.2
        avg_logits[:, 1, ...] *= 1.2
        avg_logits[:, 3, ...] *= 1.2

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