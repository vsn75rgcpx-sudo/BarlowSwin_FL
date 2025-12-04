"""
train_fednas_full.py
--------------------
Integrated pipeline:
  1. FedNAS search stage (θ + α)
  2. Export best architecture from α
  3. Federated retraining (fixed arch)
  4. Auto-generate all plots at the end (α / loss / comm)

Generates:
 - training_log.json
 - plots/*.png
"""

import os
import torch
import numpy as np
import random
import time

from federated.fl_server import FederatedServer
from federated.fl_client import FederatedClient

from model_swin3d.swin3d_unet_nas import SwinUNet3D_NAS
from model_swin3d.export_architecture import export_best_model, export_best_architecture
from model_swin3d.swin3d_unet_fixed import SwinUNet3D_Fixed

from analysis.log_helpers import append_round_log
from analysis import plot_metrics  # <-- NEW: auto plotting
from datasets.custom_multimodal_dataset import MultiModalSingleFolderDataset
from glob import glob

def infer_input_shape(dataset):
    """
    Automatically infer (in_channels, resolution) from dataset[0].
    """
    sample_img, _ = dataset[0]   # shape: (C, D, H, W)
    C, D, H, W = sample_img.shape
    return C, (D, H, W)


# ============================================================
# Model Factories
# ============================================================

def nas_model(init_barlow_path: str | None = None):
    """
    Create NAS model.

    If init_barlow_path is provided and exists, encoder weights will be
    initialized from a Barlow Twins pretrained encoder state_dict.
    """
    # Use window candidates compatible with resolution after patch embed
    # After patch embed (patch_size=2), resolution becomes (D/2, H/2, W/2)
    # For (96,96,96) input -> (48,48,48) after patch embed
    # Window size (2,6,6) is compatible with 48
    # IMPORTANT: All candidates must use the same window_size for MixedAttention3D to work
    # We only search over num_heads to avoid relative_position_bias size mismatch
    window_candidates = [
        {"window_size": (2, 6, 6), "num_heads": 2},
        {"window_size": (2, 6, 6), "num_heads": 4},
        {"window_size": (2, 6, 6), "num_heads": 6},
    ]

    model = SwinUNet3D_NAS(
        in_channels=auto_in_channels,       # ★ 自动
        num_classes=4,
        dims=(48,96,192,384),
        depths=(2,2,2,2),
        resolution=auto_resolution,  # ★ 自动
        window_candidates=window_candidates,
        nas=True,
        drop_path_rate=0.1
    )

    # Optional: load Barlow Twins pretrained encoder weights
    if init_barlow_path is not None and os.path.exists(init_barlow_path):
        try:
            ckpt = torch.load(init_barlow_path, map_location="cpu")
            model_dict = model.state_dict()
            # filter matching keys
            pretrained_dict = {
                k: v
                for k, v in ckpt.items()
                if k in model_dict and v.size() == model_dict[k].size()
            }
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print(
                f"[nas_model] Loaded encoder weights from {init_barlow_path} "
                f"({len(pretrained_dict)} matching keys)"
            )
        except Exception as e:
            print(f"[WARNING] Failed to load Barlow encoder from {init_barlow_path}: {e}")

    return model


def fixed_model(arch_json):
    return SwinUNet3D_Fixed(
        in_channels=auto_in_channels,      # ★ 自动
        num_classes=4,
        dims=(48,96,192,384),
        depths=(2,2,2,2),
        arch_json=arch_json,
        window_size=(2, 6, 6)  # Compatible with resolution after patch embed
    )


# ============================================================
# Step 1 — FedNAS Search Stage
# ============================================================

def federated_search_stage(num_clients, datasets, rounds, device, init_barlow_path: str | None = None):
    """
    FedNAS search stage.
    If init_barlow_path is provided, NAS models will be initialized from
    Barlow Twins pretrained encoder weights.
    """

    # Wrap nas_model to capture init_barlow_path
    def model_fn():
        return nas_model(init_barlow_path=init_barlow_path)

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
            model_fn=nas_model,
            dataset=datasets[i],
            device=device,
            epochs=1,
            lr=1e-3,  # Weight learning rate (higher)
            lr_alpha=1e-4,  # Architecture learning rate (lower, 10% of weight LR)
            sample_temperature=5.0,  # Initial temperature for Gumbel-Softmax
            lambda_flops=1e-4,  # FLOPs regularization weight
            min_temp=0.5,  # Minimum temperature
            temp_decay=0.95  # Temperature decay per epoch
        ) for i in range(num_clients)
    ]

    print("===== START FEDNAS SEARCH =====")

    for rnd in range(rounds):
        print(f"\n===== SEARCH ROUND {rnd+1}/{rounds} =====")

        global_weights = server.get_global_weights()
        global_alpha = server.global_alphas

        results = {}
        losses = []

        # --- Local training on clients ---
        for cid in range(num_clients):
            res = clients[cid].train(global_weights, global_alpha)
            results[cid] = res
            if res.get("loss") is not None:
                losses.append(res["loss"])

        # --- Federated aggregation ---
        server.federated_round(list(range(num_clients)), results)

        # Save checkpoint
        ckpt_path = f"fednas_round_{rnd+1}.pth"
        server.save(ckpt_path)

        comm_bytes = os.path.getsize(ckpt_path)
        train_loss = float(np.mean(losses)) if losses else None
        
        # Extract metrics from results
        val_losses = [res.get("val_loss") for res in results.values() if res.get("val_loss") is not None]
        val_loss = float(np.mean(val_losses)) if val_losses else None
        
        val_dices = [res.get("val_dice") for res in results.values() if res.get("val_dice") is not None]
        val_dice = float(np.mean(val_dices)) if val_dices else None
        
        expected_flops_list = [res.get("expected_flops") for res in results.values() if res.get("expected_flops") is not None]
        expected_flops = float(np.mean(expected_flops_list)) if expected_flops_list else None
        
        temperatures = [res.get("temperature") for res in results.values() if res.get("temperature") is not None]
        avg_temperature = float(np.mean(temperatures)) if temperatures else None

        # Update server dice history
        if val_dice is not None:
            server.val_dice_history.append(val_dice)
        # For training dice, we can use train_loss as proxy or compute separately
        # For now, we'll use None for train_dice in search stage
        server.train_dice_history.append(None)

        # Log α evolution with FLOPs and temperature
        append_round_log(
            round_idx=rnd+1,
            stage="search",
            ckpt_path=ckpt_path,
            train_loss=train_loss,
            val_loss=val_loss,
            val_dice=val_dice,
            alpha_list=server.global_alphas,
            comm_bytes=comm_bytes,
            expected_flops=expected_flops,
            temperature=avg_temperature
        )

        print(f"  [LOG] train_loss={train_loss}, val_loss={val_loss}, val_dice={val_dice:.4f}, E[FLOPs]={expected_flops:.2e}, temp={avg_temperature:.3f}, comm={comm_bytes} bytes")

    return server


# ============================================================
# Step 2 — Export Best Architecture
# ============================================================

def export_best_architecture(server, path_json="best_arch.json"):
    print("\n===== EXPORT BEST ARCHITECTURE =====")
    
    # Ensure alpha is set before export
    if hasattr(server.global_model, 'set_alpha'):
        server.global_model.set_alpha()

    fixed_initial = export_best_model(server.global_model, json_path=path_json)
    
    # Convert state_dict to match SwinUNet3D_Fixed structure
    # 1. Remove alpha_mgr keys (not needed in fixed model)
    # 2. Convert head.conv.* to head.* (NAS uses SegmentationHead3D, Fixed uses nn.Conv3d directly)
    state_dict = fixed_initial.state_dict()
    filtered_state = {}
    for k, v in state_dict.items():
        if k.startswith("alpha_mgr."):
            continue  # Skip alpha_mgr keys
        elif k.startswith("head.conv."):
            # Convert head.conv.weight -> head.weight, head.conv.bias -> head.bias
            new_key = k.replace("head.conv.", "head.")
            filtered_state[new_key] = v
        else:
            filtered_state[k] = v
    
    torch.save(filtered_state, "fixed_model_initial.pth")

    print(f"[OK] saved best_arch.json & fixed_model_initial.pth")
    return path_json


# ============================================================
# Step 3 — Federated Retraining (Fixed Architecture)
# ============================================================

def federated_retrain(num_clients, datasets, rounds, arch_json, device):

    print("\n===== START RETRAINING (FIXED ARCH) =====")

    def fixed_fn():
        return fixed_model(arch_json)

    server = FederatedServer(
        model_fn=fixed_fn,
        num_clients=num_clients,
        device=device,
        compress=False
    )

    # warm start from NAS initialization
    try:
        checkpoint = torch.load("fixed_model_initial.pth", map_location=device)
        server.global_model.load_state_dict(checkpoint, strict=False)
    except FileNotFoundError:
        print("[WARNING] fixed_model_initial.pth not found, starting from scratch")
    except Exception as e:
        print(f"[WARNING] Failed to load checkpoint: {e}, starting from scratch")

    clients = [
        FederatedClient(
            cid=i,
            model_fn=fixed_fn,
            dataset=datasets[i],
            device=device,
            epochs=1,
            lr=1e-4,
            lr_alpha=0.0
        ) for i in range(num_clients)
    ]

    for rnd in range(rounds):
        print(f"\n===== RETRAIN ROUND {rnd+1}/{rounds} =====")

        global_weights = server.get_global_weights()

        results = {}
        losses = []

        for cid in range(num_clients):
            res = clients[cid].train(global_weights, [])
            results[cid] = res
            if res.get("loss") is not None:
                losses.append(res["loss"])

        # aggregate θ only
        weights_list = [results[c]["weights"] for c in results]
        sizes = [results[c]["size"] for c in results]
        new_state = server.aggregate_params(weights_list, sizes)
        server.set_global_weights(new_state)

        ckpt_path = f"retrain_round_{rnd+1}.pth"
        server.save(ckpt_path)

        comm_bytes = os.path.getsize(ckpt_path)
        train_loss = float(np.mean(losses)) if losses else None
        
        # Extract metrics from results
        val_losses = [res.get("val_loss") for res in results.values() if res.get("val_loss") is not None]
        val_loss = float(np.mean(val_losses)) if val_losses else None
        
        val_dices = [res.get("val_dice") for res in results.values() if res.get("val_dice") is not None]
        val_dice = float(np.mean(val_dices)) if val_dices else None

        # Update server dice history
        if val_dice is not None:
            server.val_dice_history.append(val_dice)
        server.train_dice_history.append(None)  # Can compute train_dice separately if needed

        append_round_log(
            round_idx=rnd+1,
            stage="retrain",
            ckpt_path=ckpt_path,
            train_loss=train_loss,
            val_loss=val_loss,
            val_dice=val_dice,
            alpha_list=None,
            comm_bytes=comm_bytes
        )

        val_dice_str = f"{val_dice:.4f}" if val_dice is not None else "None"
        print(f"  [LOG] retrain_loss={train_loss}, val_loss={val_loss}, val_dice={val_dice_str}, comm={comm_bytes} bytes")

    return server


# ============================================================
# Step 4 — AUTO-PLOTTING
# ============================================================

def auto_generate_plots():
    print("\n===== GENERATING PLOTS =====")
    try:
        plot_metrics.main_with_args(
            log_path="training_log.json",
            ckpt_pattern="fednas_round_*.pth",
            out_dir="plots",
            show=False
        )
        print("[OK] All plots saved in plots/")
    except Exception as e:
        print("[WARNING] Plotting failed:", e)


# Provide a wrapper used by main()
def _patch_plot_metrics():
    """
    Add a helper method to plot_metrics so we can call it like:
       plot_metrics.main_with_args(...)
    """

    def main_with_args(log_path, ckpt_pattern, out_dir, show=False):
        import sys
        # Save original argv
        original_argv = sys.argv.copy()
        try:
            # Set up argv to simulate command line arguments
            sys.argv = [
                "plot_metrics",
                "--log", log_path,
                "--ckpt_pattern", ckpt_pattern,
                "--out_dir", out_dir,
                "--max_ops_show", "6"
            ]
            if show:
                sys.argv.append("--show")
            # Call main which will parse these arguments
            plot_metrics.main()
        finally:
            # Restore original argv
            sys.argv = original_argv

    setattr(plot_metrics, "main_with_args", main_with_args)


# ============================================================
# Main
# ============================================================

def main():

    _patch_plot_metrics()

    seed = 0
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_clients = 4

    # 获取所有 case ID
    all_files = glob("dataset/*.nii.gz")
    if not all_files:
        raise ValueError("No .nii.gz files found in dataset/ directory")
    
    all_ids = sorted(list(set([os.path.basename(f).split("_")[0] for f in all_files])))
    print("All case IDs:", all_ids)
    
    if len(all_ids) == 0:
        raise ValueError("No valid case IDs found in dataset/")

    # 根据 client 数量切分
    client_ids = np.array_split(all_ids, num_clients)

    datasets = []
    for cid_list in client_ids:
        ds = MultiModalSingleFolderDataset(
            folder="dataset",
            target_size=(128, 128, 128),
            patch_size=(96, 96, 96),
            augment=True
        )
        datasets.append(ds)
        
        # ---- auto infer from first dataset ----
        if len(datasets) == 1:
            global auto_in_channels, auto_resolution
            auto_in_channels, auto_resolution = infer_input_shape(datasets[0])
            print(f"[Auto-Detect] in_channels={auto_in_channels}, resolution={auto_resolution}")



    # 1) SEARCH STAGE
    server_after_search = federated_search_stage(
        num_clients=num_clients,
        datasets=datasets,
        rounds=5,
        device=device
    )

    # 2) EXPORT BEST ARCH
    arch_json = export_best_architecture(server_after_search)

    # 3) RETRAIN STAGE
    federated_retrain(
        num_clients=num_clients,
        datasets=datasets,
        rounds=5,
        arch_json=arch_json,
        device=device
    )

    # 4) AUTO PLOT
    auto_generate_plots()


if __name__ == "__main__":
    main()
