import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt as distance


def compute_edt_map(mask):
    if mask.sum() == 0:
        return np.ones_like(mask).astype(np.float32)
    if mask.sum() == mask.size:
        return -np.ones_like(mask).astype(np.float32)
    dist_out = distance(1 - mask)
    dist_in = distance(mask)
    return (dist_out - dist_in)


# === [新增] Tversky Loss 替代软 Dice ===
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-5):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha  # 惩罚 FP (假阳性)
        self.beta = beta  # 惩罚 FN (漏报) -> 调高这个让模型更拼命找小目标
        self.smooth = smooth

    def forward(self, inputs, targets):
        # inputs: logits (N, C, ...)
        # targets: labels (N, ...)
        num_classes = inputs.shape[1]
        probs = F.softmax(inputs, dim=1)
        targets_onehot = F.one_hot(targets, num_classes).permute(0, 4, 1, 2, 3).float()

        # 聚合空间维度
        dims = tuple(range(2, inputs.dim()))

        TP = (probs * targets_onehot).sum(dim=dims)
        FP = (probs * (1 - targets_onehot)).sum(dim=dims)
        FN = ((1 - probs) * targets_onehot).sum(dim=dims)

        tversky_score = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)

        # 返回 1 - Mean Tversky
        return 1.0 - tversky_score.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets):
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
        valid_classes = range(1, self.num_classes)
        for b in range(batch_size):
            t_np = targets[b].detach().cpu().numpy()
            for c in valid_classes:
                mask_c = (t_np == c).astype(np.uint8)
                if mask_c.sum() == 0: continue
                sdf = compute_edt_map(mask_c)
                sdf_tensor = torch.from_numpy(sdf).float().to(preds.device)
                loss += torch.mean(sdf_tensor * probs[b, c])
        return loss / (batch_size * len(valid_classes) + 1e-8)


class NewCombinedLoss(nn.Module):
    """
    升级版 Loss: Tversky + Focal + Boundary
    """

    def __init__(self, weight_dice=1.0, weight_ce=1.0, weight_boundary=0.01, use_focal=True):
        super().__init__()
        # 使用 Tversky 替代普通 Dice，beta=0.7 强调召回率
        self.tversky = TverskyLoss(alpha=0.3, beta=0.7)

        if use_focal:
            self.ce_or_focal = FocalLoss(gamma=2.0)
        else:
            self.ce_or_focal = nn.CrossEntropyLoss()

        self.boundary = BoundaryLoss(num_classes=4)
        self.w_dice = weight_dice
        self.w_ce = weight_ce
        self.w_bound = weight_boundary

    def forward(self, preds, targets):
        l_dice = self.tversky(preds, targets)  # 使用 Tversky
        l_main = self.ce_or_focal(preds, targets)

        if self.w_bound > 0:
            l_bound = self.boundary(preds, targets)
        else:
            l_bound = 0.0

        return self.w_dice * l_dice + self.w_ce * l_main + self.w_bound * l_bound