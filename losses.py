import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt as distance


def compute_edt_map(mask):
    """
    计算二值 mask 的符号距离图 (Signed Distance Map)。
    内部为负距离，外部为正距离。
    """
    if mask.sum() == 0:
        return np.ones_like(mask).astype(np.float32)
    if mask.sum() == mask.size:
        return -np.ones_like(mask).astype(np.float32)

    dist_out = distance(1 - mask)
    dist_in = distance(mask)
    return (dist_out - dist_in)


class BoundaryLoss(nn.Module):
    """
    Boundary Loss: 最小化预测概率与 GT 距离图的乘积积分。
    """

    def __init__(self, num_classes=4):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, preds, targets):
        probs = F.softmax(preds, dim=1)
        batch_size = preds.shape[0]
        loss = 0.0

        # 假设 BraTS 标签为 0,1,2,3，通常只计算前景类的边界损失 (1,2,3)
        valid_classes = range(1, self.num_classes)

        for b in range(batch_size):
            t_np = targets[b].detach().cpu().numpy()

            for c in valid_classes:
                # 生成类别 c 的二值 mask
                mask_c = (t_np == c).astype(np.uint8)

                # 计算距离图 (在 CPU 上进行，因为 scipy 没有 GPU 版)
                sdf = compute_edt_map(mask_c)

                # 转回 Tensor
                sdf_tensor = torch.from_numpy(sdf).float().to(preds.device)

                # Loss = mean( SDF * Prob_c )
                loss += torch.mean(sdf_tensor * probs[b, c])

        return loss / (batch_size * len(valid_classes))


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        num_classes = preds.shape[1]
        pred_soft = F.softmax(preds, dim=1)

        target_onehot = F.one_hot(targets, num_classes).permute(0, 4, 1, 2, 3).float()

        intersection = (pred_soft * target_onehot).sum(dim=(2, 3, 4))
        union = pred_soft.sum(dim=(2, 3, 4)) + target_onehot.sum(dim=(2, 3, 4))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    # 旧的 CombinedLoss 保留以防万一
    def __init__(self, weight_ce=1.0, weight_dice=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = SoftDiceLoss()
        self.wce = weight_ce
        self.wdice = weight_dice

    def forward(self, preds, targets):
        return self.wce * self.ce(preds, targets) + self.wdice * self.dice(preds, targets)


class NewCombinedLoss(nn.Module):
    """
    新损失函数: L = L_Dice + L_CE + lambda * L_Boundary
    """

    def __init__(self, weight_dice=1.0, weight_ce=1.0, weight_boundary=0.01):
        super().__init__()
        self.dice = SoftDiceLoss()
        self.ce = nn.CrossEntropyLoss()
        self.boundary = BoundaryLoss(num_classes=4)

        self.w_dice = weight_dice
        self.w_ce = weight_ce
        self.w_bound = weight_boundary

    def forward(self, preds, targets):
        l_dice = self.dice(preds, targets)
        l_ce = self.ce(preds, targets)
        l_bound = self.boundary(preds, targets)

        return self.w_dice * l_dice + self.w_ce * l_ce + self.w_bound * l_bound