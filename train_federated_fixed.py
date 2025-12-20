import os
import random
import torch
import torch.nn as nn
import numpy as np
import glob
from torch.utils.data import Dataset, DataLoader
import nibabel as nib

# 复用已有的模块
from federated.fl_server import FederatedServer
from federated.fl_client import FederatedClient
from model_swin3d.swin3d_unet_fixed import SwinUNet3D_Fixed
from metrics import calculate_brats_metrics  # [新增] 导入新指标函数


class DataParallelPassthrough(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


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

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"[Init] Using device: {device}")

    # Config
    NUM_CLIENTS = 4
    ROUNDS = 50
    ARCH_JSON = "best_arch.json"

    if not os.path.exists(ARCH_JSON):
        raise FileNotFoundError(f"Cannot find {ARCH_JSON}. Please run search phase first.")

    # 1. Prepare Data
    data_root = "dataset"
    possible_dirs = glob.glob(os.path.join(data_root, "*"))
    case_dirs = [d for d in possible_dirs if os.path.isdir(d) and len(glob.glob(os.path.join(d, "*.nii.gz"))) > 0]
    all_case_ids = sorted([os.path.basename(d) for d in case_dirs])

    if len(all_case_ids) == 0:
        raise ValueError("No data found!")

    # 简单划分全局验证集 (取最后10个或20%作为验证)
    val_count = min(len(all_case_ids) // 5, 10)
    train_ids = all_case_ids[:-val_count]
    val_ids = all_case_ids[-val_count:]

    val_dataset = BraTSDataset(data_root, val_ids, target_shape=(96, 96, 96), augment=False)
    # 验证 batch_size 可以小一点以防 OOM
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

    client_splits = np.array_split(train_ids, NUM_CLIENTS)
    datasets = []
    for split in client_splits:
        ds = BraTSDataset(data_root, split, target_shape=(96, 96, 96), augment=True)
        datasets.append(ds)

    # 2. Model Factory
    def create_fixed_model():
        model = SwinUNet3D_Fixed(
            in_channels=4,
            num_classes=4,
            dims=(48, 96, 192, 384),
            depths=(2, 2, 2, 2),
            arch_json=ARCH_JSON,
            window_size=(2, 6, 6)
        )
        if torch.cuda.device_count() > 1:
            model = DataParallelPassthrough(model)
        return model

    # 3. Server
    server = FederatedServer(create_fixed_model, NUM_CLIENTS, device, compress=False)

    # 4. Clients
    clients = []
    for i in range(NUM_CLIENTS):
        client = FederatedClient(
            cid=i,
            model_fn=create_fixed_model,
            dataset=datasets[i],
            batch_size=16,
            epochs=5,
            device=device,
            lr=1e-4,  # 初始学习率
            lr_alpha=0.0,
            val_split_ratio=0.0  # 使用全局验证集
        )
        clients.append(client)

    # 5. Helper: Global Evaluation
    def evaluate_global(model, loader):
        model.eval()
        metrics_accum = {
            "dice_wt": [], "dice_tc": [], "dice_et": [],
            "hd95_wt": [], "hd95_tc": [], "hd95_et": []
        }

        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                preds = model(x)

                # 计算该 batch 的指标
                batch_res = calculate_brats_metrics(preds, y)

                metrics_accum["dice_wt"].append(batch_res["dice_wt"])
                metrics_accum["dice_tc"].append(batch_res["dice_tc"])
                metrics_accum["dice_et"].append(batch_res["dice_et"])
                metrics_accum["hd95_wt"].append(batch_res["hd95_wt"])
                metrics_accum["hd95_tc"].append(batch_res["hd95_tc"])
                metrics_accum["hd95_et"].append(batch_res["hd95_et"])

        # 计算平均值
        final = {k: np.mean(v) for k, v in metrics_accum.items()}
        final["dice_avg"] = (final["dice_wt"] + final["dice_tc"] + final["dice_et"]) / 3.0
        final["hd95_avg"] = (final["hd95_wt"] + final["hd95_tc"] + final["hd95_et"]) / 3.0
        return final

    # 6. Training Loop
    print("===== START FEDERATED TRAINING (FIXED ARCH - FAST MODE) =====")
    for rnd in range(ROUNDS):
        print(f"\n--- Round {rnd + 1}/{ROUNDS} ---")

        global_weights = server.get_global_weights()
        results = {}
        losses = []

        # Client Training
        for cid in range(NUM_CLIENTS):
            res = clients[cid].train(global_weights, [])
            results[cid] = res
            if res.get("loss"): losses.append(res["loss"])
            print(f" Client {cid} loss: {res['loss']:.4f}")

        # Server Aggregation
        weights_list = [results[c]["weights"] for c in results]
        sizes = [results[c]["size"] for c in results]
        new_state = server.aggregate_params(weights_list, sizes)
        server.set_global_weights(new_state)

        # Log
        avg_loss = np.mean(losses) if losses else 0
        print(f" [Train] Round Avg Loss: {avg_loss:.4f}")

        # Evaluation (每轮都验证)
        if (rnd + 1) % 1 == 0:
            print(" [Eval] Validating...")
            mets = evaluate_global(server.global_model, val_loader)
            print(f" >>> Round {rnd + 1} Metrics:")
            print(f"     Dice(WT): {mets['dice_wt']:.4f}  | HD95(WT): {mets['hd95_wt']:.2f}")
            print(f"     Dice(TC): {mets['dice_tc']:.4f}  | HD95(TC): {mets['hd95_tc']:.2f}")
            print(f"     Dice(ET): {mets['dice_et']:.4f}  | HD95(ET): {mets['hd95_et']:.2f}")
            print(f"     Mean Dice: {mets['dice_avg']:.4f} | Mean HD95: {mets['hd95_avg']:.2f}")

        # Save Checkpoint
        ckpt_path = f"fed_fixed_round_{rnd + 1}.pth"
        server.save(ckpt_path)


if __name__ == "__main__":
    main()