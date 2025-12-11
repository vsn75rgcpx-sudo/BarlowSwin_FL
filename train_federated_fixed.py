"""
train_federated_fixed.py
------------------------
Federated Retraining with Fixed Architecture.
Skips the search phase and trains a fixed model defined in best_arch.json.
"""

import os
import random
import torch
import numpy as np
import glob

# 复用已有的模块
from federated.fl_server import FederatedServer
from federated.fl_client import FederatedClient
from model_swin3d.swin3d_unet_fixed import SwinUNet3D_Fixed  # 使用固定模型
from datasets.custom_multimodal_dataset import MultiModalSingleFolderDataset  # 假设你有这个，或者用 BraTSDataset
from analysis.log_helpers import append_round_log

# ============================================================
# BraTS Dataset (复用之前的定义)
# ============================================================
from torch.utils.data import Dataset
import nibabel as nib


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


# ============================================================
# Main Logic
# ============================================================
def main():
    seed = 42
    random.seed(seed);
    np.random.seed(seed);
    torch.manual_seed(seed)

    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"[Init] Using device: {device}")

    # Config
    NUM_CLIENTS = 4
    ROUNDS = 50  # 训练轮数
    ARCH_JSON = "best_arch.json"  # 必须存在

    if not os.path.exists(ARCH_JSON):
        raise FileNotFoundError(f"Cannot find {ARCH_JSON}. Please run search phase first or provide a json.")

    # 1. Prepare Data
    data_root = "dataset"
    possible_dirs = glob.glob(os.path.join(data_root, "*"))
    case_dirs = [d for d in possible_dirs if os.path.isdir(d) and len(glob.glob(os.path.join(d, "*.nii.gz"))) > 0]
    all_case_ids = sorted([os.path.basename(d) for d in case_dirs])

    if len(all_case_ids) == 0:
        raise ValueError("No data found!")

    client_splits = np.array_split(all_case_ids, NUM_CLIENTS)
    datasets = []
    for split in client_splits:
        ds = BraTSDataset(data_root, split, target_shape=(96, 96, 96), augment=True)
        datasets.append(ds)

    # 2. Model Factory (Fixed Architecture)
    def create_fixed_model():
        return SwinUNet3D_Fixed(
            in_channels=4,
            num_classes=4,
            dims=(48, 96, 192, 384),
            depths=(2, 2, 2, 2),
            arch_json=ARCH_JSON,
            window_size=(2, 6, 6)  # 必须与 search 时一致
        )

    # 3. Server
    server = FederatedServer(
        model_fn=create_fixed_model,
        num_clients=NUM_CLIENTS,
        device=device,
        compress=False
    )

    # 4. Clients
    clients = []
    for i in range(NUM_CLIENTS):
        client = FederatedClient(
            cid=i,
            model_fn=create_fixed_model,
            dataset=datasets[i],
            batch_size=2,
            epochs=1,
            device=device,
            lr=1e-4,
            lr_alpha=0.0,  # 关键：Alpha 学习率为 0
            val_split_ratio=0.0  # 不需要验证集来更新架构
        )
        clients.append(client)

    # 5. Training Loop
    print("===== START FEDERATED TRAINING (FIXED ARCH) =====")
    for rnd in range(ROUNDS):
        print(f"\n--- Round {rnd + 1}/{ROUNDS} ---")

        global_weights = server.get_global_weights()
        results = {}
        losses = []

        for cid in range(NUM_CLIENTS):
            # 这里的 global_alpha 传空列表即可，因为模型是 Fixed 的
            res = clients[cid].train(global_weights, [])
            results[cid] = res
            if res.get("loss"): losses.append(res["loss"])

        # Aggregate
        weights_list = [results[c]["weights"] for c in results]
        sizes = [results[c]["size"] for c in results]
        new_state = server.aggregate_params(weights_list, sizes)
        server.set_global_weights(new_state)

        # Log
        avg_loss = np.mean(losses) if losses else 0
        print(f" Round {rnd + 1} Avg Loss: {avg_loss:.4f}")

        # Save Checkpoint
        ckpt_path = f"fed_fixed_round_{rnd + 1}.pth"
        server.save(ckpt_path)


if __name__ == "__main__":
    main()