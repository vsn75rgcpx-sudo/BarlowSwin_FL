"""
train_fednas_full.py (50 Rounds + Multi-GPU Fix Version)
--------------------
Integrated pipeline:
  1. FedNAS search stage (Loads 'encoder_pretrained.pth' if available)
  2. Export best architecture
  3. Federated retraining
  4. Auto-plotting

Modified:
 - [FIX] Added DataParallelPassthrough to fix 'AttributeError: arch_parameters'.
"""

import os
import torch
import numpy as np
import random
import glob
import nibabel as nib
from torch.utils.data import Dataset
from scipy import ndimage

from federated.fl_server import FederatedServer
from federated.fl_client import FederatedClient

from model_swin3d.swin3d_unet_nas import SwinUNet3D_NAS
from model_swin3d.export_architecture import export_best_architecture
from model_swin3d.swin3d_unet_fixed import SwinUNet3D_Fixed

from analysis.log_helpers import append_round_log
from analysis import plot_metrics


# ============================================================
# [FIX] Custom DataParallel to expose underlying methods
# ============================================================
class DataParallelPassthrough(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            # If DataParallel doesn't have it, ask the inner module
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


def infer_input_shape(dataset):
    sample_img, _ = dataset[0]
    C, D, H, W = sample_img.shape
    return C, (D, H, W)


# ============================================================
# Model Factory Helper (Updated with Fix)
# ============================================================
def get_model_factory(in_channels, resolution, num_classes=4, init_barlow_path=None, arch_json=None, mode="nas"):
    def create_nas():
        window_candidates = [
            {"window_size": (2, 6, 6), "num_heads": 2},
            {"window_size": (2, 6, 6), "num_heads": 4},
            {"window_size": (2, 6, 6), "num_heads": 6},
        ]
        model = SwinUNet3D_NAS(
            in_channels=in_channels,
            num_classes=num_classes,
            dims=(48, 96, 192, 384),
            depths=(2, 2, 2, 2),
            resolution=resolution,
            window_candidates=window_candidates,
            nas=True,
            drop_path_rate=0.1
        )
        if init_barlow_path and os.path.exists(init_barlow_path):
            try:
                ckpt = torch.load(init_barlow_path, map_location="cpu")
                model_dict = model.state_dict()
                pretrained_dict = {k: v for k, v in ckpt.items()
                                   if k in model_dict and v.size() == model_dict[k].size()}
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
                print(
                    f"[Factory] Loaded PRETRAINED weights from {init_barlow_path} ({len(pretrained_dict)} layers matched)")
            except Exception as e:
                print(f"[Factory] Warning: Failed to load Barlow weights: {e}")

        # [FIX] Use Custom DataParallelPassthrough
        if torch.cuda.device_count() > 1:
            print(f"[Factory-NAS] Detected {torch.cuda.device_count()} GPUs. Wrapping in DataParallelPassthrough.")
            model = DataParallelPassthrough(model)

        return model

    def create_fixed():
        if arch_json is None:
            raise ValueError("arch_json required for fixed mode")
        model = SwinUNet3D_Fixed(
            in_channels=in_channels,
            num_classes=num_classes,
            dims=(48, 96, 192, 384),
            depths=(2, 2, 2, 2),
            arch_json=arch_json,
            window_size=(2, 6, 6)
        )

        # [FIX] Use Custom DataParallelPassthrough
        if torch.cuda.device_count() > 1:
            print(f"[Factory-Fixed] Detected {torch.cuda.device_count()} GPUs. Wrapping in DataParallelPassthrough.")
            model = DataParallelPassthrough(model)

        return model

    if mode == "nas":
        return create_nas
    else:
        return create_fixed


# ============================================================
# Step 1 — FedNAS Search Stage
# ============================================================
def federated_search_stage(num_clients, datasets, rounds, device, in_channels, resolution):
    # 尝试加载 encoder_pretrained.pth
    barlow_ckpt = "encoder_pretrained.pth"
    if not os.path.exists(barlow_ckpt):
        print("[Search] No pretrained weights found. Starting from scratch.")
        barlow_ckpt = None
    else:
        print(f"[Search] Using pretrained weights: {barlow_ckpt}")

    model_fn = get_model_factory(in_channels, resolution, init_barlow_path=barlow_ckpt, mode="nas")

    server = FederatedServer(
        model_fn=model_fn,
        num_clients=num_clients,
        device=device,
        compress=False,
        alpha_lr=0.5
    )

    clients = [
        FederatedClient(
            cid=i,
            model_fn=model_fn,
            dataset=datasets[i],
            device=device,
            epochs=1,
            lr=1e-3,
            lr_alpha=1e-4,
            sample_temperature=5.0,
            lambda_flops=1e-4,
            val_split_ratio=0.1,
            batch_size=8
        ) for i in range(num_clients)
    ]

    print("===== START FEDNAS SEARCH =====")
    if torch.cuda.device_count() > 1:
        print(f"[System] Multi-GPU Mode Enabled: Using {torch.cuda.device_count()} GPUs.")

    for rnd in range(rounds):
        print(f"\n===== SEARCH ROUND {rnd + 1}/{rounds} =====")

        global_weights = server.get_global_weights()
        global_alpha = server.global_alphas
        results = {}
        losses = []

        for cid in range(num_clients):
            res = clients[cid].train(global_weights, global_alpha)
            results[cid] = res
            if res.get("loss") is not None: losses.append(res["loss"])

        server.federated_round(list(range(num_clients)), results)

        ckpt_path = f"fednas_round_{rnd + 1}.pth"
        server.save(ckpt_path)

        comm_bytes = os.path.getsize(ckpt_path)
        train_loss = float(np.mean(losses)) if losses else None

        val_dice = np.mean([r.get("val_dice") for r in results.values() if r.get("val_dice")])
        expected_flops = np.mean([r.get("expected_flops") for r in results.values() if r.get("expected_flops")])
        avg_temp = np.mean([r.get("temperature") for r in results.values() if r.get("temperature")])

        if np.isnan(val_dice): val_dice = 0.0
        if np.isnan(expected_flops): expected_flops = 0.0

        server.val_dice_history.append(val_dice)
        server.train_dice_history.append(None)

        append_round_log(
            round_idx=rnd + 1, stage="search", ckpt_path=ckpt_path,
            train_loss=train_loss, val_loss=None, val_dice=val_dice,
            alpha_list=server.global_alphas, comm_bytes=comm_bytes,
            expected_flops=expected_flops, temperature=avg_temp
        )
        print(f"  [LOG] loss={train_loss:.3f}, val_dice={val_dice:.3f}, FLOPS={expected_flops:.2e}")

    return server


# ============================================================
# Step 2 — Export Best Architecture
# ============================================================
def export_arch_step(server, path_json="best_arch.json"):
    print("\n===== EXPORT BEST ARCHITECTURE =====")

    # Unwrap DataParallel if exists
    raw_model = server.global_model.module if isinstance(server.global_model,
                                                         torch.nn.DataParallel) else server.global_model

    if hasattr(raw_model, 'set_alpha'):
        raw_model.set_alpha()

    export_best_architecture(raw_model, output_path=path_json)

    full_state = raw_model.state_dict()
    clean_state = {}
    for k, v in full_state.items():
        if "alpha" not in k and "gate" not in k:
            if "head.conv." in k:
                k = k.replace("head.conv.", "head.")
            clean_state[k] = v
    torch.save(clean_state, "fixed_model_initial.pth")
    return path_json


# ============================================================
# Step 3 — Federated Retraining
# ============================================================
def federated_retrain(num_clients, datasets, rounds, arch_json, device, in_channels, resolution):
    print("\n===== START RETRAINING (FIXED ARCH) =====")

    model_fn = get_model_factory(in_channels, resolution, arch_json=arch_json, mode="fixed")

    server = FederatedServer(
        model_fn=model_fn,
        num_clients=num_clients,
        device=device,
        compress=False
    )

    if os.path.exists("fixed_model_initial.pth"):
        try:
            ckpt = torch.load("fixed_model_initial.pth", map_location=device)
            raw_model = server.global_model.module if isinstance(server.global_model,
                                                                 torch.nn.DataParallel) else server.global_model
            raw_model.load_state_dict(ckpt, strict=False)
            print("[Retrain] Loaded warm-start weights.")
        except Exception as e:
            print(f"[Retrain] Warm start failed: {e}")

    clients = [
        FederatedClient(
            cid=i,
            model_fn=model_fn,
            dataset=datasets[i],
            device=device,
            epochs=2,
            lr=3e-4,
            lr_alpha=0.0,
            val_split_ratio=0.0,
            batch_size=8
        ) for i in range(num_clients)
    ]

    for rnd in range(rounds):
        print(f"\n===== RETRAIN ROUND {rnd + 1}/{rounds} =====")
        global_weights = server.get_global_weights()
        results = {}
        losses = []

        for cid in range(num_clients):
            res = clients[cid].train(global_weights, [])
            results[cid] = res
            if res.get("loss") is not None: losses.append(res["loss"])

        weights_list = [results[c]["weights"] for c in results]
        sizes = [results[c]["size"] for c in results]
        new_state = server.aggregate_params(weights_list, sizes)
        server.set_global_weights(new_state)

        ckpt_path = f"retrain_round_{rnd + 1}.pth"
        server.save(ckpt_path)

        train_loss = float(np.mean(losses)) if losses else None

        append_round_log(
            round_idx=rnd + 1, stage="retrain", ckpt_path=ckpt_path,
            train_loss=train_loss, alpha_list=None, comm_bytes=os.path.getsize(ckpt_path)
        )
        print(f"  [LOG] retrain_loss={train_loss:.3f}")

    return server


# ============================================================
# Main
# ============================================================
def main():
    def _patch_plot_metrics():
        def main_with_args(log_path, ckpt_pattern, out_dir, show=False):
            import sys
            original_argv = sys.argv.copy()
            try:
                sys.argv = ["plot_metrics", "--log", log_path, "--ckpt_pattern", ckpt_pattern, "--out_dir", out_dir]
                plot_metrics.main()
            finally:
                sys.argv = original_argv

        setattr(plot_metrics, "main_with_args", main_with_args)

    _patch_plot_metrics()

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

    num_clients = 4

    # -------------------------------------------------
    # Dataset Discovery (BraTS Folder Structure)
    # -------------------------------------------------
    datasets = []
    data_root = "dataset"
    if os.path.exists(data_root):
        possible_dirs = glob.glob(os.path.join(data_root, "*"))
        case_dirs = [d for d in possible_dirs if os.path.isdir(d) and len(glob.glob(os.path.join(d, "*.nii.gz"))) > 0]

        if len(case_dirs) > 0:
            print(f"[Init] Found {len(case_dirs)} case folders in {data_root}. Using BraTS Dataset.")
            all_case_ids = sorted([os.path.basename(d) for d in case_dirs])

            client_case_splits = np.array_split(all_case_ids, num_clients)

            for cid, case_list in enumerate(client_case_splits):
                if len(case_list) == 0: continue
                ds = BraTSDataset(root_dir=data_root, case_ids=case_list, target_shape=(96, 96, 96), augment=True)
                datasets.append(ds)
        else:
            print("[Init] No data found in dataset/.")
    else:
        print("[Init] dataset/ directory not found.")

    if not datasets: raise ValueError("No valid data found.")

    in_channels, resolution = infer_input_shape(datasets[0])
    print(f"[Auto-Detect] in_channels={in_channels}, resolution={resolution}")

    # 1. Search (50 Rounds)
    server_search = federated_search_stage(num_clients, datasets, 50, device, in_channels, resolution)

    # 2. Export
    json_path = export_arch_step(server_search)

    # 3. Retrain (50 Rounds)
    federated_retrain(num_clients, datasets, 50, json_path, device, in_channels, resolution)

    # 4. Plot
    try:
        plot_metrics.main_with_args("training_log.json", "fednas_round_*.pth", "plots")
        print("[OK] Plots saved.")
    except Exception as e:
        print(f"[Plotting] Failed: {e}")


if __name__ == "__main__":
    main()