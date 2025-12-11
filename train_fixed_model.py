"""
train_fixed_model.py (Final Version with Real Data)
--------------------
Non-Federated final training script for SwinUNet3D_Fixed.

Pipeline:
 1. Load fixed architecture JSON (best_arch.json)
 2. Construct SwinUNet3D_Fixed
 3. Train on MRI dataset (Real BraTS Data)
 4. Save best model (based on validation loss or Dice)
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import nibabel as nib
import glob

from model_swin3d.swin3d_unet_fixed import SwinUNet3D_Fixed


# ============================================================
# BraTS Dataset (Copy from train_fednas_full.py)
# ============================================================
class BraTSDataset(Dataset):
    def __init__(self, root_dir, case_ids, target_shape=(96, 96, 96), augment=False):
        self.root_dir = root_dir
        self.case_ids = case_ids
        self.target_shape = target_shape
        self.augment = augment
        self.suffixes = ['t1n', 't1c', 't2w', 't2f']
        self.seg_suffix = 'seg'

    def __len__(self):
        return len(self.case_ids)

    def normalize(self, vol):
        mask = vol > 0
        if mask.sum() > 0:
            mean = vol[mask].mean()
            std = vol[mask].std()
            vol = (vol - mean) / (std + 1e-8)
            vol[~mask] = 0
        return vol

    def crop_or_pad(self, vol, seg):
        D, H, W = vol.shape[1:]
        tD, tH, tW = self.target_shape

        if self.augment:
            d_start = random.randint(0, max(0, D - tD))
            h_start = random.randint(0, max(0, H - tH))
            w_start = random.randint(0, max(0, W - tW))
        else:
            d_start = (D - tD) // 2
            h_start = (H - tH) // 2
            w_start = (W - tW) // 2

        d_start = max(0, d_start)
        h_start = max(0, h_start)
        w_start = max(0, w_start)

        d_end = min(D, d_start + tD)
        h_end = min(H, h_start + tH)
        w_end = min(W, w_start + tW)

        vol_crop = vol[:, d_start:d_end, h_start:h_end, w_start:w_end]
        seg_crop = seg[d_start:d_end, h_start:h_end, w_start:w_end]

        pad_d = tD - vol_crop.shape[1]
        pad_h = tH - vol_crop.shape[2]
        pad_w = tW - vol_crop.shape[3]

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            vol_crop = np.pad(vol_crop, ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
            seg_crop = np.pad(seg_crop, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')

        return torch.from_numpy(vol_crop).float(), torch.from_numpy(seg_crop).long()

    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        case_path = os.path.join(self.root_dir, case_id)

        imgs = []
        for suf in self.suffixes:
            fpath = os.path.join(case_path, f"{case_id}-{suf}.nii.gz")
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"Missing file: {fpath}")

            img_obj = nib.load(fpath)
            img_data = img_obj.get_fdata().astype(np.float32)
            img_data = self.normalize(img_data)
            imgs.append(img_data)

        vol = np.stack(imgs, axis=0)

        seg_path = os.path.join(case_path, f"{case_id}-{self.seg_suffix}.nii.gz")
        if os.path.exists(seg_path):
            seg_obj = nib.load(seg_path)
            seg_data = seg_obj.get_fdata().astype(np.longlong)
            seg_data[seg_data == 4] = 3
        else:
            seg_data = np.zeros(vol.shape[1:], dtype=np.longlong)

        return self.crop_or_pad(vol, seg_data)


# ------------------------------------------------------------
# Dice Loss for multi-class segmentation
# ------------------------------------------------------------
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        targets_onehot = torch.nn.functional.one_hot(targets, num_classes)
        targets_onehot = targets_onehot.permute(0, 4, 1, 2, 3).float()

        dims = (0, 2, 3, 4)
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
    # 优先使用 MPS (Mac) 或 CUDA
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
        print("[Init] Using Apple MPS acceleration.")
    else:
        device = "cpu"

    # -----------------------------
    # 1. Prepare Data
    # -----------------------------
    data_root = "dataset"
    if not os.path.exists(data_root):
        raise ValueError("Dataset folder not found!")

    possible_dirs = glob.glob(os.path.join(data_root, "*"))
    all_case_ids = sorted([
        os.path.basename(d) for d in possible_dirs
        if os.path.isdir(d) and len(glob.glob(os.path.join(d, "*.nii.gz"))) > 0
    ])

    if len(all_case_ids) == 0:
        raise ValueError("No valid BraTS cases found in dataset/.")

    print(f"[Init] Found {len(all_case_ids)} cases.")

    # 简单划分 Train/Val (80/20)
    split_idx = int(len(all_case_ids) * 0.8)
    train_ids = all_case_ids[:split_idx]
    val_ids = all_case_ids[split_idx:]

    train_set = BraTSDataset(data_root, train_ids, target_shape=(96, 96, 96), augment=True)
    val_set = BraTSDataset(data_root, val_ids, target_shape=(96, 96, 96), augment=False)

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1)  # 验证集 batch=1 更稳妥

    # -----------------------------
    # 2. Load fixed architecture
    # -----------------------------
    model = SwinUNet3D_Fixed(
        in_channels=4,
        num_classes=args.num_classes,
        dims=(48, 96, 192, 384),
        depths=(2, 2, 2, 2),
        arch_json=args.arch_path,
        window_size=(2, 6, 6)
    ).to(device)

    print(f"[OK] Loaded fixed architecture: {args.arch_path}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # -----------------------------
    # 3. Optimizer and Loss
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
    # 4. Training loop
    # -----------------------------
    best_loss = 1e9
    os.makedirs("checkpoints_fixed", exist_ok=True)

    print("Start training on Real Data...")
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optim, criterion, device)
        val_loss = eval_one_epoch(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch[{epoch + 1}/{args.epochs}]  "
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
    parser.add_argument("--epochs", type=int, default=50)  # 真实训练可以多跑一些 epoch
    parser.add_argument("--batch", type=int, default=2)  # 显存够的话可以开大一点
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_classes", type=int, default=4)

    args = parser.parse_args()
    main(args)