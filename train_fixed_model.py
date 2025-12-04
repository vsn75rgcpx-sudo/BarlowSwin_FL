"""
train_fixed_model.py
--------------------
Non-Federated final training script for SwinUNet3D_Fixed.

Pipeline:
 1. Load fixed architecture JSON (best_arch.json)
 2. Construct SwinUNet3D_Fixed
 3. Train on MRI dataset (BraTS / Toy / Custom)
 4. Save best model (based on validation loss or Dice)

This script is stand-alone and does NOT involve NAS or FL.
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from datasets.custom_multimodal_dataset import MultiModalSingleFolderDataset
from glob import glob

from model_swin3d.swin3d_unet_fixed import SwinUNet3D_Fixed


# ------------------------------------------------------------
# Example Dataset (replace with BraTS / custom loader)
# ------------------------------------------------------------

class ToyMRI3DDataset(torch.utils.data.Dataset):
    """
    Replace this with your actual dataset loader.
    """
    def __init__(self, n=20, shape=(1,64,128,128), num_classes=4):
        self.n = n
        self.shape = shape
        self.num_classes = num_classes

    def __len__(self): return self.n

    def __getitem__(self, idx):
        vol = torch.randn(*self.shape)
        seg = torch.randint(0, self.num_classes, self.shape[1:])
        return vol.float(), seg.long()


# ------------------------------------------------------------
# Dice Loss for multi-class segmentation
# ------------------------------------------------------------
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        """
        logits: (B, C, D, H, W)
        targets: (B, D, H, W)
        """
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        targets_onehot = torch.nn.functional.one_hot(targets, num_classes)
        targets_onehot = targets_onehot.permute(0,4,1,2,3).float()

        dims = (0,2,3,4)
        intersection = torch.sum(probs * targets_onehot, dims)
        cardinality = torch.sum(probs + targets_onehot, dims)
        dice = (2 * intersection + self.eps) / (cardinality + self.eps)
        return 1 - dice.mean()


# ------------------------------------------------------------
# Training loop
# ------------------------------------------------------------
def train_one_epoch(model, loader, optim, criterion, device):
    model.train()
    total_loss = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optim.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()

    return total_loss / len(loader)


# ------------------------------------------------------------
# Main training
# ------------------------------------------------------------
def main(args):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------------
    # Load fixed architecture
    # -----------------------------
    model = SwinUNet3D_Fixed(
        in_channels=4,
        num_classes=args.num_classes,
        dims=(48,96,192,384),
        depths=(2,2,2,2),
        arch_json=args.arch_path
    ).to(device)

    print(f"[OK] Loaded fixed architecture: {args.arch_path}")
    print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # -----------------------------
    # Dataset
    # Replace with BraTS dataset loader
    # -----------------------------
    train_set = ToyMRI3DDataset(n=40)
    val_set = ToyMRI3DDataset(n=8)

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1)

    # -----------------------------
    # Optimizer and Loss
    # -----------------------------
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs
    )

    criterion = DiceLoss()

    # -----------------------------
    # Training loop
    # -----------------------------
    best_loss = 1e9
    os.makedirs("checkpoints_fixed", exist_ok=True)

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optim, criterion, device)
        val_loss = eval_one_epoch(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch[{epoch+1}/{args.epochs}]  "
              f"Train: {train_loss:.4f}  Val: {val_loss:.4f}")

        # save best
        if val_loss < best_loss:
            best_loss = val_loss
            save_path = f"checkpoints_fixed/best_model.pth"
            torch.save(model.state_dict(), save_path)
            print(f"  [OK] Saved best model to {save_path}")

    print("\nTraining finished!")
    print("Best Val Loss =", best_loss)


# ------------------------------------------------------------
# Args
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--arch_path", type=str, default="best_arch.json")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_classes", type=int, default=4)

    args = parser.parse_args()
    main(args)
