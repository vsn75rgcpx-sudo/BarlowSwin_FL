"""
losses.py
---------
Loss functions for 3D segmentation:
- SoftDiceLoss: Multi-class Dice loss
- CombinedLoss: CrossEntropy + Dice combination
"""

import torch
import torch.nn.functional as F


class SoftDiceLoss(torch.nn.Module):
    """
    Soft Dice Loss for multi-class 3D segmentation.
    """
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        """
        Args:
            preds: (N, C, D, H, W) raw logits
            targets: (N, D, H, W) long type (0..C-1)
        Returns:
            dice_loss: scalar tensor
        """
        num_classes = preds.shape[1]
        pred_soft = F.softmax(preds, dim=1)

        # convert target to one-hot
        target_onehot = F.one_hot(targets, num_classes)
        target_onehot = target_onehot.permute(0, 4, 1, 2, 3).float()

        # flatten
        pred_flat = pred_soft.contiguous().view(preds.shape[0], num_classes, -1)
        target_flat = target_onehot.contiguous().view(preds.shape[0], num_classes, -1)

        intersection = (pred_flat * target_flat).sum(2)
        union = pred_flat.sum(2) + target_flat.sum(2)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        loss = 1.0 - dice.mean()

        return loss


class CombinedLoss(torch.nn.Module):
    """
    Combined loss: CrossEntropy + Dice
    """
    def __init__(self, weight_ce=1.0, weight_dice=1.0):
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss()
        self.dice = SoftDiceLoss()

        self.wce = weight_ce
        self.wdice = weight_dice

    def forward(self, preds, targets):
        """
        Args:
            preds: (N, C, D, H, W) raw logits
            targets: (N, D, H, W) long type
        Returns:
            combined_loss: scalar tensor
        """
        loss_ce = self.ce(preds, targets)
        loss_dice = self.dice(preds, targets)
        return self.wce * loss_ce + self.wdice * loss_dice

