import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt as distance


def compute_edt_map(mask):
    """计算二值 mask 的符号距离图 (Signed Distance Map)"""
    if mask.sum() == 0:
        return np.ones_like(mask).astype(np.float32)
    if mask.sum() == mask.size:
        return -np.ones_like(mask).astype(np.float32)
    dist_out = distance(1 - mask)
    dist_in = distance(mask)
    return (dist_out - dist_in)


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        # alpha 可以是列表，对应每一类的权重
        self.alpha = alpha

    def forward(self, inputs, targets):
        # inputs: (N, C, ...)
        # targets: (N, ...)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class BoundaryLoss(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, preds, targets):
        probs = F.softmax(preds, dim=1)
        batch_size = preds.shape[0]
        loss = 0.0
        # 只计算前景类 (1, 2, 3)
        valid_classes = range(1, self.num_classes)

        for b in range(batch_size):
            t_np = targets[b].detach().cpu().numpy()
            for c in valid_classes:
                mask_c = (t_np == c).astype(np.uint8)
                if mask_c.sum() == 0: continue  # 忽略不存在的类别

                sdf = compute_edt_map(mask_c)
                sdf_tensor = torch.from_numpy(sdf).float().to(preds.device)
                loss += torch.mean(sdf_tensor * probs[b, c])

        return loss / (batch_size * len(valid_classes) + 1e-8)


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        num_classes = preds.shape[1]
        pred_soft = F.softmax(preds, dim=1)
        target_onehot = F.one_hot(targets, num_classes).permute(0, 4, 1, 2, 3).float()

        # 聚合除了 batch 和 channel 以外的维度
        dims = tuple(range(2, preds.dim()))
        intersection = (pred_soft * target_onehot).sum(dim=dims)
        union = pred_soft.sum(dim=dims) + target_onehot.sum(dim=dims)

        # 计算每个类的 Dice，然后取平均 (忽略背景类0，或者全算)
        # 这里计算所有类的平均
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class NewCombinedLoss(nn.Module):
    """
    新损失函数:
    L = L_Dice + (Focal_Loss if use_focal else CE) + w_bound * L_Boundary
    """

    def __init__(self, weight_dice=1.0, weight_ce=1.0, weight_boundary=0.01, use_focal=False):
        super().__init__()
        self.dice = SoftDiceLoss()

        if use_focal:
            # Focal Loss 替代 CE，gamma=2.0 专注于难样本
            self.ce_or_focal = FocalLoss(gamma=2.0)
        else:
            self.ce_or_focal = nn.CrossEntropyLoss()

        self.boundary = BoundaryLoss(num_classes=4)

        self.w_dice = weight_dice
        self.w_ce = weight_ce
        self.w_bound = weight_boundary
        self.use_focal = use_focal

    def forward(self, preds, targets):
        l_dice = self.dice(preds, targets)
        l_main = self.ce_or_focal(preds, targets)

        # 只有当 boundary 权重 > 0 时才计算，节省时间
        if self.w_bound > 0:
            l_bound = self.boundary(preds, targets)
        else:
            l_bound = 0.0

        return self.w_dice * l_dice + self.w_ce * l_main + self.w_bound * l_bound