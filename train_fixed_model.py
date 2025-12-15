"""
train_fixed_model.py (Enhanced Metrics Version)
--------------------
Pipeline:
 1. Load fixed architecture JSON (best_arch.json)
 2. Train on MRI dataset (Real BraTS Data)
 3. Evaluate F1, Jaccard, HD95, PA, SSIM
 4. Save metrics plots and checkpoints to dedicated folders
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
import matplotlib.pyplot as plt
import json

from model_swin3d.swin3d_unet_fixed import SwinUNet3D_Fixed
# 导入我们刚刚更新的 metrics 模块
from metrics import f1_score, jaccard_score, pixel_accuracy, compute_hd95, ssim3d


# ============================================================
# Directory Setup Helper
# ============================================================
def setup_output_dirs(base_dir="output_results"):
    dirs = {
        "f1": os.path.join(base_dir, "F1_Score"),
        "jaccard": os.path.join(base_dir, "Jaccard_IoU"),
        "hd95": os.path.join(base_dir, "HD95"),
        "pa": os.path.join(base_dir, "Pixel_Accuracy"),
        "ssim": os.path.join(base_dir, "SSIM"),
        "ckpt": os.path.join(base_dir, "Checkpoints"),
        "plots": os.path.join(base_dir, "Plots_Overview")
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


# ============================================================
# BraTS Dataset
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
# Dice Loss
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
# Training & Eval Functions
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


def eval_full_metrics(model, loader, criterion, device):
    """
    Evaluate model and return dictionary of all metrics.
    """
    model.eval()
    metrics = {
        "val_loss": [], "f1": [], "jaccard": [],
        "hd95": [], "pa": [], "ssim": []
    }

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)

            # 1. Loss
            loss = criterion(logits, y)
            metrics["val_loss"].append(loss.item())

            # 2. Metrics
            metrics["f1"].append(f1_score(logits, y))
            metrics["jaccard"].append(jaccard_score(logits, y))
            metrics["pa"].append(pixel_accuracy(logits, y))
            metrics["ssim"].append(ssim3d(logits, y))

            # HD95 比较慢，如果数据量大可以考虑每N个Epoch算一次
            # 这里默认每次都算
            metrics["hd95"].append(compute_hd95(logits, y))

    # Calculate means
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    return avg_metrics


def save_metric_plot(history, metric_name, save_dir, filename):
    plt.figure(figsize=(10, 6))
    rounds = range(1, len(history) + 1)
    plt.plot(rounds, history, 'o-', label=f'{metric_name}')
    plt.title(f'{metric_name} Curve')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main(args):
    # Setup Output Dirs
    out_dirs = setup_output_dirs(args.output_dir)

    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
        print("[Init] Using Apple MPS acceleration.")
    else:
        device = "cpu"

    # 1. Prepare Data
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

    split_idx = int(len(all_case_ids) * 0.8)
    train_ids = all_case_ids[:split_idx]
    val_ids = all_case_ids[split_idx:]

    train_set = BraTSDataset(data_root, train_ids, target_shape=(96, 96, 96), augment=True)
    val_set = BraTSDataset(data_root, val_ids, target_shape=(96, 96, 96), augment=False)

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1)

    # 2. Load Model
    model = SwinUNet3D_Fixed(
        in_channels=4,
        num_classes=args.num_classes,
        dims=(48, 96, 192, 384),
        depths=(2, 2, 2, 2),
        arch_json=args.arch_path,
        window_size=(2, 6, 6)
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    criterion = DiceLoss()

    # History Recording
    history = {
        "train_loss": [], "val_loss": [],
        "f1": [], "jaccard": [], "hd95": [], "pa": [], "ssim": []
    }

    best_score = -1  # Based on F1

    print("Start training...")
    for epoch in range(args.epochs):
        # Train
        t_loss = train_one_epoch(model, train_loader, optim, criterion, device)

        # Eval
        metrics = eval_full_metrics(model, val_loader, criterion, device)

        scheduler.step()

        # Update History
        history["train_loss"].append(t_loss)
        history["val_loss"].append(metrics["val_loss"])
        history["f1"].append(metrics["f1"])
        history["jaccard"].append(metrics["jaccard"])
        history["hd95"].append(metrics["hd95"])
        history["pa"].append(metrics["pa"])
        history["ssim"].append(metrics["ssim"])

        print(f"Epoch[{epoch + 1}/{args.epochs}] "
              f"T-Loss:{t_loss:.4f} V-Loss:{metrics['val_loss']:.4f} | "
              f"F1:{metrics['f1']:.4f} IoU:{metrics['jaccard']:.4f} "
              f"HD95:{metrics['hd95']:.2f} PA:{metrics['pa']:.4f} SSIM:{metrics['ssim']:.4f}")

        # Save Best Model (Based on F1 Score)
        if metrics["f1"] > best_score:
            best_score = metrics["f1"]
            save_path = os.path.join(out_dirs["ckpt"], "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  [*] Best model saved (F1={best_score:.4f})")

        # Save Regular Checkpoint
        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(out_dirs["ckpt"], f"epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), ckpt_path)

        # ----------------------------------------
        # Plotting & Saving every epoch
        # ----------------------------------------
        # Save plots to their specific folders
        save_metric_plot(history["f1"], "F1 Score", out_dirs["f1"], "f1_curve.png")
        save_metric_plot(history["jaccard"], "Jaccard Index", out_dirs["jaccard"], "jaccard_curve.png")
        save_metric_plot(history["hd95"], "HD95", out_dirs["hd95"], "hd95_curve.png")
        save_metric_plot(history["pa"], "Pixel Accuracy", out_dirs["pa"], "pa_curve.png")
        save_metric_plot(history["ssim"], "SSIM", out_dirs["ssim"], "ssim_curve.png")

        # Save overview plot
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 3, 1);
        plt.plot(history["f1"]);
        plt.title("F1")
        plt.subplot(2, 3, 2);
        plt.plot(history["jaccard"]);
        plt.title("IoU")
        plt.subplot(2, 3, 3);
        plt.plot(history["hd95"]);
        plt.title("HD95")
        plt.subplot(2, 3, 4);
        plt.plot(history["pa"]);
        plt.title("PA")
        plt.subplot(2, 3, 5);
        plt.plot(history["ssim"]);
        plt.title("SSIM")
        plt.subplot(2, 3, 6);
        plt.plot(history["train_loss"], label='Train');
        plt.plot(history["val_loss"], label='Val');
        plt.title("Loss");
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dirs["plots"], "metrics_overview.png"))
        plt.close()

    # Save Logs to JSON
    with open(os.path.join(args.output_dir, "training_metrics.json"), "w") as f:
        json.dump(history, f, indent=4)

    print("\nTraining finished! All results saved to", args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch_path", type=str, default="best_arch.json")
    parser.add_argument("--output_dir", type=str, default="output_results")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_classes", type=int, default=4)

    args = parser.parse_args()
    main(args)