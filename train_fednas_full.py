"""
train_full_pipeline.py (Optimized Version)
-------------------------------------------
Pipeline:
 1. [Stage 0] Barlow Twins Pretraining (Self-Supervised) -> encoder_pretrained.pth
 2. [Stage 1] FedNAS Search (With Warmup) -> fednas_round_X.pth
 3. [Stage 2] Export Best Architecture -> best_arch.json
 4. [Stage 3] Federated Retraining (Top-5 Avg) -> retrain_averaged_model.pth
 5. [Stage 4] Final Test & Visualization -> results/

Modifications:
 - Search Stage: Warmup for 10 rounds (freeze alpha, train weights).
 - Retrain Stage: Ignore Dice first 50%, then keep Top-5 best models and Average them.
 - Test Stage: Explicitly print mIoU and MPA.
"""

import os
import torch
import numpy as np
import random
import glob
import json
import copy  # Added for weight averaging
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import nibabel as nib
from torch.utils.data import Dataset, DataLoader

# --- Imports from your project ---
from federated.fl_server import FederatedServer
from federated.fl_client import FederatedClient
from federated.barlow_client import BarlowClient
from federated.barlow_server import BarlowServer

from model_swin3d.swin3d_unet_nas import SwinUNet3D_NAS
from model_swin3d.swin3d_unet_fixed import SwinUNet3D_Fixed
from model_swin3d.export_architecture import export_best_architecture
from datasets.barlow_pair_dataset import BarlowPairDataset
import metrics

# ============================================================
# 0. Infrastructure & Configuration
# ============================================================
CHECKPOINT_DIR = "checkpoints"
RESULT_DIR = "results"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# A40 Optimization Config
CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_clients": 4,
    "num_workers": 0,

    # Stage 0: Barlow Twins
    "barlow_rounds": 20,
    "barlow_local_epochs": 5,
    "barlow_batch_size": 16,

    # Stage 1: Search (NAS)
    "search_rounds": 50,
    "search_local_epochs": 2,
    "search_batch_size": 8,
    "search_warmup_rounds": 10,  # [NEW] Lock alpha for first 10 rounds

    # Stage 3: Retrain
    "retrain_rounds": 100,
    "retrain_local_epochs": 2,
    "retrain_batch_size": 8,
    "retrain_lr": 1e-4,
    "retrain_top_k": 5  # [NEW] Average top 5 models
}


class EarlyStopping:
    def __init__(self, patience=15, delta=0.001, mode='max', verbose=True):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.mode = mode
        self.verbose = verbose

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        else:
            if self.mode == 'max':
                improve = score > self.best_score + self.delta
            else:
                improve = score < self.best_score - self.delta

            if improve:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
                if self.verbose: print(
                    f"    [EarlyStop] Counter: {self.counter}/{self.patience}. Best: {self.best_score:.4f}")
                if self.counter >= self.patience: self.early_stop = True


class DataParallelPassthrough(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


# ============================================================
# 1. Dataset Class
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
            mean = vol[mask].mean();
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
            d_start = (D - tD) // 2;
            h_start = (H - tH) // 2;
            w_start = (W - tW) // 2

        d_start = max(0, d_start);
        h_start = max(0, h_start);
        w_start = max(0, w_start)
        d_end = min(D, d_start + tD);
        h_end = min(H, h_start + tH);
        w_end = min(W, w_start + tW)

        vol_crop = vol[:, d_start:d_end, h_start:h_end, w_start:w_end]
        seg_crop = seg[d_start:d_end, h_start:h_end, w_start:w_end]

        pad_d = tD - vol_crop.shape[1];
        pad_h = tH - vol_crop.shape[2];
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


def infer_input_shape(dataset):
    sample_img, _ = dataset[0]
    C, D, H, W = sample_img.shape
    return C, (D, H, W)


def get_model_factory(in_channels, resolution, num_classes=4, init_barlow_path=None, arch_json=None, mode="nas"):
    def create_nas():
        window_candidates = [{"window_size": (2, 6, 6), "num_heads": h} for h in [2, 4, 6]]
        model = SwinUNet3D_NAS(
            in_channels=in_channels, num_classes=num_classes,
            dims=(48, 96, 192, 384), depths=(2, 2, 2, 2),
            resolution=resolution, window_candidates=window_candidates, nas=True
        )
        if init_barlow_path and os.path.exists(init_barlow_path):
            try:
                ckpt = torch.load(init_barlow_path, map_location="cpu")
                model_dict = model.state_dict()
                pretrained_dict = {k: v for k, v in ckpt.items() if k in model_dict and v.shape == model_dict[k].shape}
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict, strict=False)
                print(f"[Factory] Loaded {len(pretrained_dict)} layers from Barlow Pretraining")
            except Exception as e:
                print(f"[Factory] Warning: {e}")
        if torch.cuda.device_count() > 1: model = DataParallelPassthrough(model)
        return model

    def create_fixed():
        model = SwinUNet3D_Fixed(
            in_channels=in_channels, num_classes=num_classes,
            dims=(48, 96, 192, 384), depths=(2, 2, 2, 2),
            arch_json=arch_json, window_size=(2, 6, 6)
        )
        if torch.cuda.device_count() > 1: model = DataParallelPassthrough(model)
        return model

    return create_nas if mode == "nas" else create_fixed


# ============================================================
# STAGE 0: Barlow Twins Pretraining
# ============================================================
def run_barlow_pretraining(data_root, train_ids, device):
    save_path = "encoder_pretrained.pth"
    if os.path.exists(save_path):
        print(f"\n[Stage 0] {save_path} already exists. Skipping pretraining.")
        return save_path

    print("\n===== STAGE 0: BARLOW TWINS PRETRAINING =====")
    args_mock = type('', (), {})()
    args_mock.barlow_target = (96, 96, 96)
    args_mock.barlow_hidden = 512
    args_mock.barlow_out = 1024
    args_mock.barlow_bs = CONFIG["barlow_batch_size"]
    args_mock.barlow_lr = 1e-4
    args_mock.barlow_lambda = 0.005
    args_mock.weight_decay = 1e-4
    args_mock.grad_clip = 5.0
    args_mock.device = device

    client_splits = np.array_split(train_ids, CONFIG["num_clients"])
    clients = []
    sizes = []

    for i, ids in enumerate(client_splits):
        ds = BarlowPairDataset(folder=data_root, target_size=args_mock.barlow_target, augment=True)
        ds.ids = list(ids)
        window_candidates = [{"window_size": (2, 6, 6), "num_heads": h} for h in [2, 4, 6]]
        encoder = SwinUNet3D_NAS(
            in_channels=1, num_classes=2, dims=(48, 96, 192, 384), depths=(2, 2, 2, 2),
            resolution=args_mock.barlow_target, window_candidates=window_candidates, nas=False
        )
        c = BarlowClient(client_id=i, model_encoder=encoder, dataset=ds, device=device, args=args_mock)
        clients.append(c)
        sizes.append(len(ids))

    window_candidates = [{"window_size": (2, 6, 6), "num_heads": h} for h in [2, 4, 6]]
    server_encoder = SwinUNet3D_NAS(
        in_channels=1, num_classes=2, dims=(48, 96, 192, 384), depths=(2, 2, 2, 2),
        resolution=args_mock.barlow_target, window_candidates=window_candidates, nas=False
    )
    server = BarlowServer(server_encoder, device, args_mock)

    history_loss = []
    for r in range(CONFIG["barlow_rounds"]):
        print(f"--- Barlow Round {r + 1}/{CONFIG['barlow_rounds']} ---")
        client_states = []
        losses = []

        for i, client in enumerate(clients):
            client.set_state_dicts({"encoder": server.global_encoder.state_dict()})
            loss = 0.0
            for _ in range(CONFIG["barlow_local_epochs"]):
                loss += client.local_epoch()
            loss /= CONFIG["barlow_local_epochs"]
            losses.append(loss)
            client_states.append(client.get_state_dicts())

        new_state = server.aggregate(client_states, sizes)
        server.global_encoder.load_state_dict(new_state)
        avg_loss = np.mean(losses)
        history_loss.append(avg_loss)
        print(f"  [Log] Barlow Loss: {avg_loss:.4f}")

    server.save(save_path)
    plt.figure()
    plt.plot(history_loss, label='Barlow Loss')
    plt.savefig(os.path.join(RESULT_DIR, 'barlow_loss.png'))
    plt.close()
    return save_path


# ============================================================
# STAGE 1: FedNAS Search (With Warmup)
# ============================================================
def federated_search_stage(num_clients, datasets, device, in_channels, resolution):
    print("\n===== STAGE 1: FEDNAS SEARCH =====")
    barlow_ckpt = "encoder_pretrained.pth"
    model_fn = get_model_factory(in_channels, resolution, init_barlow_path=barlow_ckpt, mode="nas")

    # Initialize Server
    server = FederatedServer(model_fn, num_clients, device=device, alpha_lr=0.5)

    clients = [FederatedClient(cid=i, model_fn=model_fn, dataset=datasets[i],
                               device=device, batch_size=CONFIG["search_batch_size"],
                               num_workers=CONFIG["num_workers"], epochs=CONFIG["search_local_epochs"])
               for i in range(num_clients)]

    dice_history = []
    warmup_rounds = CONFIG["search_warmup_rounds"]

    for rnd in range(CONFIG["search_rounds"]):
        print(f"\n--- Search Round {rnd + 1}/{CONFIG['search_rounds']} ---")

        # [MODIFICATION] Warmup Logic: Freeze alpha for first 10 rounds
        if rnd < warmup_rounds:
            print(f"  [Warmup] Round {rnd + 1}: Freezing Alpha, Training Weights Only (Focus on Supernet Convergence).")
            # Try to zero out alpha learning rate if optimizer exists
            if hasattr(server, 'alpha_optimizer'):
                for param_group in server.alpha_optimizer.param_groups:
                    param_group['lr'] = 0.0
        elif rnd == warmup_rounds:
            print(f"  [Warmup End] Round {rnd + 1}: Unfreezing Alpha. Starting Architecture Search.")
            # Restore learning rate
            if hasattr(server, 'alpha_optimizer'):
                for param_group in server.alpha_optimizer.param_groups:
                    param_group['lr'] = 0.5

        global_weights = server.get_global_weights()
        global_alpha = server.global_alphas
        results, losses = {}, []

        for cid in range(num_clients):
            res = clients[cid].train(global_weights, global_alpha)
            results[cid] = res
            if res.get("loss"): losses.append(res["loss"])

        server.federated_round(list(range(num_clients)), results)

        val_dice = np.mean([r.get("val_dice", 0) for r in results.values()])

        if rnd < warmup_rounds:
            print(f"  [Log] Val Dice: {val_dice:.4f} (Ignored during Warmup)")
        else:
            print(f"  [Log] Val Dice: {val_dice:.4f}")

        dice_history.append(val_dice)

        # Save Checkpoint
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"fednas_round_{rnd + 1}.pth")
        server.save(ckpt_path)

    plt.figure()
    plt.plot(dice_history, label='Search Dice')
    plt.savefig(os.path.join(RESULT_DIR, 'search_dice.png'))
    plt.close()

    return server


# ============================================================
# STAGE 3: Retrain (Top-K Averaging)
# ============================================================
def average_model_weights(top_k_checkpoints, device):
    """
    Averages the weights of the models in the top_k_checkpoints list.
    """
    print(f"\n[Averaging] Averaging weights of {len(top_k_checkpoints)} models...")

    # Load first model to initialize sum
    avg_state_dict = torch.load(top_k_checkpoints[0]['path'], map_location=device)
    if "model" in avg_state_dict: avg_state_dict = avg_state_dict["model"]

    # Sum remaining models
    for i in range(1, len(top_k_checkpoints)):
        ckpt = torch.load(top_k_checkpoints[i]['path'], map_location=device)
        state = ckpt["model"] if "model" in ckpt else ckpt
        for key in avg_state_dict:
            avg_state_dict[key] += state[key]

    # Divide by N
    for key in avg_state_dict:
        avg_state_dict[key] = avg_state_dict[key] / len(top_k_checkpoints)

    return avg_state_dict


def federated_retrain(num_clients, datasets, arch_json, device, in_channels, resolution):
    print("\n===== STAGE 3: RETRAINING =====")

    model_fn = get_model_factory(in_channels, resolution, arch_json=arch_json, mode="fixed")
    server = FederatedServer(model_fn, num_clients, device=device)

    # Warm-start from previous round (optional)
    if os.path.exists("fixed_model_initial.pth"):
        try:
            ckpt = torch.load("fixed_model_initial.pth", map_location=device)
            server.global_model.load_state_dict(ckpt, strict=False)
            print("[Retrain] Loaded warm-start weights.")
        except:
            pass

    clients = []
    for i in range(num_clients):
        c = FederatedClient(cid=i, model_fn=model_fn, dataset=datasets[i],
                            device=device, batch_size=CONFIG["retrain_batch_size"],
                            num_workers=CONFIG["num_workers"],
                            epochs=CONFIG["retrain_local_epochs"],
                            lr=CONFIG["retrain_lr"])
        clients.append(c)

    # [MODIFICATION] Top-K List
    # Stores tuples: {'dice': float, 'path': str}
    top_k_models = []
    start_monitoring_round = CONFIG["retrain_rounds"] // 2  # Start monitoring after 50%
    dice_history = []

    for rnd in range(CONFIG["retrain_rounds"]):
        print(f"\n--- Retrain Round {rnd + 1}/{CONFIG['retrain_rounds']} ---")
        global_weights = server.get_global_weights()
        results, losses = {}, []

        for cid in range(num_clients):
            res = clients[cid].train(global_weights, [])
            results[cid] = res
            if res.get("loss"): losses.append(res["loss"])

        # Aggregate
        weights_list = [results[c]["weights"] for c in results]
        sizes = [results[c]["size"] for c in results]
        new_state = server.aggregate_params(weights_list, sizes)
        server.set_global_weights(new_state)

        val_dice = np.mean([r.get("val_dice", 0) for r in results.values()])
        dice_history.append(val_dice)
        print(f"  [Log] Val Dice: {val_dice:.4f}")

        # [MODIFICATION] Custom Saving Logic
        # Always save current round for resumption
        current_ckpt_path = os.path.join(CHECKPOINT_DIR, f"retrain_round_{rnd + 1}.pth")
        server.save(current_ckpt_path)

        # Only care about best models after 50% rounds
        if rnd >= start_monitoring_round:
            print(f"  [Monitor] Analyzing for Top-{CONFIG['retrain_top_k']} candidates...")

            # Add current to list
            top_k_models.append({'dice': val_dice, 'path': current_ckpt_path})

            # Sort by dice descending
            top_k_models.sort(key=lambda x: x['dice'], reverse=True)

            # Keep only Top K
            if len(top_k_models) > CONFIG["retrain_top_k"]:
                # Remove the one that fell off the list (optional: delete file to save space)
                removed = top_k_models.pop()
                # os.remove(removed['path']) # Uncomment if disk space is tight

            print(f"  [Monitor] Current Top Dice values: {[round(m['dice'], 4) for m in top_k_models]}")
        else:
            print("  [Monitor] Warmup phase (first 50%), not tracking top models yet.")

    plt.figure()
    plt.plot(dice_history, label='Retrain Dice')
    plt.savefig(os.path.join(RESULT_DIR, 'retrain_dice.png'))
    plt.close()

    # [MODIFICATION] Final Averaging
    if len(top_k_models) > 0:
        print(f"\n[Final] Averaging Top-{len(top_k_models)} models...")
        averaged_weights = average_model_weights(top_k_models, device)

        # Save Averaged Model
        final_path = os.path.join(CHECKPOINT_DIR, "retrain_averaged_model.pth")
        server.global_model.load_state_dict(averaged_weights)
        server.save(final_path)
        print(f"[Final] Averaged model saved to: {final_path}")
        return final_path
    else:
        # Fallback if rounds < start_monitoring
        return os.path.join(CHECKPOINT_DIR, f"retrain_round_{CONFIG['retrain_rounds']}.pth")


# ============================================================
# STAGE 4: Final Test
# ============================================================
def final_test_phase(test_dataset, arch_json, device, in_channels, resolution, best_model_path=None):
    print("\n===== STAGE 4: FINAL TEST =====")

    # If not provided, try to find the averaged model first
    if best_model_path is None:
        best_model_path = os.path.join(CHECKPOINT_DIR, "retrain_averaged_model.pth")

    if not os.path.exists(best_model_path):
        print(f"[!] {best_model_path} not found. Trying best single model...")
        best_model_path = os.path.join(CHECKPOINT_DIR, "retrain_best_model.pth")  # Fallback

    print(f"[Test] Loading model from: {best_model_path}")
    model_fn = get_model_factory(in_channels, resolution, arch_json=arch_json, mode="fixed")
    model = model_fn().to(device)

    ckpt = torch.load(best_model_path, map_location=device)
    if "model" in ckpt: ckpt = ckpt["model"]
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    metrics_log = {"CaseID": [], "Dice": [], "mIoU": [], "PA": [], "MPA": []}

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            d = metrics.dice_score(outputs, targets)
            miou = metrics.jaccard_score(outputs, targets)
            pa = metrics.pixel_accuracy(outputs, targets)
            mpa = metrics.mean_pixel_accuracy(outputs, targets)

            case_id = test_dataset.case_ids[i]
            metrics_log["CaseID"].append(case_id)
            metrics_log["Dice"].append(d)
            metrics_log["mIoU"].append(miou)
            metrics_log["PA"].append(pa)
            metrics_log["MPA"].append(mpa)

    df = pd.DataFrame(metrics_log)
    csv_path = os.path.join(RESULT_DIR, "final_test_metrics.csv")
    df.to_csv(csv_path, index=False)

    # [MODIFICATION] Explicitly Print Metrics to Console
    print("\n" + "=" * 30)
    print("       FINAL TEST RESULTS       ")
    print("=" * 30)
    print(f"Mean Dice : {df['Dice'].mean():.4f}")
    print(f"Mean mIoU : {df['mIoU'].mean():.4f}")
    print(f"Mean PA   : {df['PA'].mean():.4f}")
    print(f"Mean MPA  : {df['MPA'].mean():.4f}")
    print("=" * 30 + "\n")

    # Generate Plots
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df[['Dice', 'mIoU']])
    plt.title("Dice & mIoU")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    sns.boxplot(data=df[['PA', 'MPA']])
    plt.title("Pixel Accuracy")
    plt.grid(True, alpha=0.3)

    plt.savefig(os.path.join(RESULT_DIR, "final_metrics_boxplot.png"))
    print(f"[Test] Detailed CSV saved to {csv_path}")


# ============================================================
# Main Execution
# ============================================================
def main():
    torch.backends.cudnn.benchmark = True
    seed = 42
    random.seed(seed);
    np.random.seed(seed);
    torch.manual_seed(seed)

    # 1. Dataset Setup
    data_root = "dataset"
    if not os.path.exists(data_root): raise ValueError("dataset/ folder not found")
    all_cases = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
    if not all_cases: raise ValueError("No cases found")

    random.shuffle(all_cases)
    split_idx = int(len(all_cases) * 0.8)
    train_ids = all_cases[:split_idx]
    test_ids = all_cases[split_idx:]

    print(f"[Init] Train: {len(train_ids)} | Test: {len(test_ids)} | Device: {CONFIG['device']}")

    client_splits = np.array_split(train_ids, CONFIG["num_clients"])
    datasets = [BraTSDataset(data_root, split, augment=True) for split in client_splits]
    in_channels, resolution = infer_input_shape(datasets[0])

    # --- PIPELINE ---

    # Stage 0: Barlow
    run_barlow_pretraining(data_root, train_ids, CONFIG["device"])

    # Stage 1: Search
    server_search = federated_search_stage(CONFIG["num_clients"], datasets, CONFIG["device"], in_channels, resolution)

    # Stage 2: Export
    json_path = export_best_architecture(server_search.global_model,
                                         output_path=os.path.join(RESULT_DIR, "best_arch.json"))

    # Stage 3: Retrain (Returns path to averaged model)
    best_model_path = federated_retrain(CONFIG["num_clients"], datasets, json_path, CONFIG["device"], in_channels,
                                        resolution)

    # Stage 4: Test
    test_ds = BraTSDataset(data_root, test_ids, augment=False)
    final_test_phase(test_ds, json_path, CONFIG["device"], in_channels, resolution, best_model_path)


if __name__ == "__main__":
    main()