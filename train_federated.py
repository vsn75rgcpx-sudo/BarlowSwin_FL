"""
train_federated.py
------------------
Main training script for NAS + SwinUNet3D federated learning.

To run:
    python train_federated.py
"""

import os
import random
import torch
import numpy as np

from federated.fl_server import FederatedServer
from federated.fl_client import FederatedClient

from model_swin3d.swin3d_unet_nas import SwinUNet3D_NAS
from metrics import dice_score, psnr, ssim3d
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Example Toy Dataset (placeholder)
# Replace with your MRI dataset (BraTS, IXI, etc.)
# ------------------------------------------------------------
class Toy3DDataset(torch.utils.data.Dataset):
    def __init__(self, n=10, shape=(1, 32, 64, 64), num_classes=4):
        self.n = n
        self.shape = shape
        self.num_classes = num_classes

        if n <= 0:
            raise ValueError(f"Dataset size must be > 0, got {n}")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.n:
            raise IndexError(f"Index {idx} out of range [0, {self.n})")
        vol = torch.randn(*self.shape)  # fake MRI
        seg = torch.randint(0, self.num_classes, (self.shape[1], self.shape[2], self.shape[3]))
        return vol, seg


# ------------------------------------------------------------
# Model factory
# ------------------------------------------------------------
def create_model():
    """
    Create NAS model. Note: resolution should match your actual data shape.
    For Toy3DDataset with shape (1, 32, 64, 64), we use (32, 64, 64).
    After patch embed (kernel_size=2, stride=2), resolution becomes (16, 32, 32).
    Window size must be divisible by (16, 32, 32), so we use (2, 4, 4) or (2, 8, 8).
    """
    # Use window candidates compatible with resolution after patch embed
    # After patch embed: (32, 64, 64) -> (16, 32, 32)
    # Window size (2, 4, 4) is compatible: 16%2=0, 32%4=0, 32%4=0
    window_candidates = [
        {"window_size": (2, 4, 4), "num_heads": 2},
        {"window_size": (2, 4, 4), "num_heads": 4},
        {"window_size": (2, 4, 4), "num_heads": 6},
    ]

    return SwinUNet3D_NAS(
        in_channels=1,
        num_classes=4,
        dims=(48, 96, 192, 384),
        depths=(2, 2, 2, 2),
        window_candidates=window_candidates,
        ffn_op_names=None,  # Will use default: ["conv3","conv1","dw","identity","conv5"]
        drop_path_rate=0.1,
        resolution=(32, 64, 64),  # Match Toy3DDataset shape (D, H, W)
        nas=True,
    )


# ------------------------------------------------------------
# Federated Training Loop
# ------------------------------------------------------------
def main():
    # -----------------------------
    # Reproducibility
    # -----------------------------
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 优先使用 MPS (Mac) 或 CUDA
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
        print("[Init] Using Apple MPS acceleration.")
    else:
        device = "cpu"

    # -----------------------------
    # Configuration
    # -----------------------------
    NUM_CLIENTS = 4
    ROUNDS = 10
    EPOCHS_PER_CLIENT = 1

    COMPRESS = False
    COMPRESS_MODE = "8bit"
    TOPK_RATIO = 0.1

    # Validation
    if NUM_CLIENTS <= 0:
        raise ValueError(f"NUM_CLIENTS must be > 0, got {NUM_CLIENTS}")
    if ROUNDS <= 0:
        raise ValueError(f"ROUNDS must be > 0, got {ROUNDS}")
    if EPOCHS_PER_CLIENT <= 0:
        raise ValueError(f"EPOCHS_PER_CLIENT must be > 0, got {EPOCHS_PER_CLIENT}")

    # -----------------------------
    # Create datasets for each client
    # Replace with your real MRI datasets
    # -----------------------------
    try:
        datasets = [Toy3DDataset(n=5) for _ in range(NUM_CLIENTS)]
        # Verify datasets are not empty
        for i, ds in enumerate(datasets):
            if len(ds) == 0:
                raise ValueError(f"Dataset for client {i} is empty")
    except Exception as e:
        raise RuntimeError(f"Failed to create datasets: {e}")

    # -----------------------------
    # Initialize server
    # -----------------------------
    try:
        server = FederatedServer(
            model_fn=create_model,
            num_clients=NUM_CLIENTS,
            device=device,
            compress=COMPRESS,
            compress_mode=COMPRESS_MODE,
            topk_ratio=TOPK_RATIO,
            alpha_lr=0.5
        )
        # Verify server initialized correctly
        if server.global_model is None:
            raise RuntimeError("Failed to initialize global model")
        if len(server.global_alphas) == 0:
            print("[WARNING] No architecture parameters found in model")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize server: {e}")

    # -----------------------------
    # Create clients
    # -----------------------------
    clients = []
    for cid in range(NUM_CLIENTS):
        try:
            client = FederatedClient(
                cid=cid,
                model_fn=create_model,
                dataset=datasets[cid],
                batch_size=1,
                epochs=EPOCHS_PER_CLIENT,
                device=device,
                lr=1e-4,
                lr_alpha=3e-3,
                weight_decay=1e-5,
                grad_clip=1.0,
                compress=COMPRESS,
                compress_mode=COMPRESS_MODE,
                topk_ratio=TOPK_RATIO,
                val_split_ratio=0.2  # Use 20% for validation
            )
            clients.append(client)
        except Exception as e:
            raise RuntimeError(f"Failed to create client {cid}: {e}")

    if len(clients) != NUM_CLIENTS:
        raise RuntimeError(f"Expected {NUM_CLIENTS} clients, but created {len(clients)}")

    # -----------------------------
    # Evaluation function
    # -----------------------------
    def evaluate(model, dataloader, device):
        """
        Evaluate model on validation data.
        Returns: (mean_dice, mean_psnr, mean_ssim)
        """
        model.eval()
        dices, psnrs, ssims = [], [], []

        if len(dataloader) == 0:
            return 0.0, 0.0, 0.0

        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                preds = model(x)

                # Dice
                _, mean_dice = dice_score(preds, y)
                dices.append(mean_dice)

                # PSNR / SSIM on argmax label
                pred_label = torch.argmax(preds, dim=1).float()
                y_float = y.float()

                # Normalize to [0, 1] for PSNR/SSIM
                num_classes = preds.shape[1]
                pred_normalized = pred_label / (num_classes - 1) if num_classes > 1 else pred_label
                y_normalized = y_float / (num_classes - 1) if num_classes > 1 else y_float

                psnrs.append(psnr(pred_normalized, y_normalized, max_val=1.0))
                ssims.append(ssim3d(pred_normalized, y_normalized))

        return (
            float(np.mean(dices)),
            float(np.mean(psnrs)),
            float(np.mean(ssims))
        )

    # -----------------------------
    # Metrics history
    # -----------------------------
    dice_history = []
    psnr_history = []
    ssim_history = []

    # -----------------------------
    # Federated training rounds
    # -----------------------------
    for rnd in range(ROUNDS):
        print(f"\n========== Federated Round {rnd + 1}/{ROUNDS} ==========")

        try:
            # Sample all clients or subset
            selected = list(range(NUM_CLIENTS))

            if len(selected) == 0:
                print("[WARNING] No clients selected, skipping round")
                continue

            # Broadcast global model
            global_state = server.get_global_weights()
            if not global_state:
                print("[WARNING] Empty global state, skipping round")
                continue

            # Broadcast α
            global_alpha = server.global_alphas
            if len(global_alpha) == 0:
                print("[WARNING] No global alpha parameters")

            # Results dict
            client_results = {}
            successful_clients = []

            # ---- Local updates ----
            for cid in selected:
                try:
                    print(f" Client {cid} training...")
                    result = clients[cid].train(global_state, global_alpha)

                    # Validate result
                    if result is None:
                        print(f"[WARNING] Client {cid} returned None, skipping")
                        continue
                    if "weights" not in result:
                        print(f"[WARNING] Client {cid} result missing 'weights', skipping")
                        continue
                    if "alpha" not in result:
                        print(f"[WARNING] Client {cid} result missing 'alpha', skipping")
                        continue
                    if "size" not in result:
                        print(f"[WARNING] Client {cid} result missing 'size', skipping")
                        continue

                    client_results[cid] = result
                    successful_clients.append(cid)

                except Exception as e:
                    print(f"[ERROR] Client {cid} training failed: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            # Check if we have any successful clients
            if len(successful_clients) == 0:
                print("[ERROR] No clients completed training, skipping aggregation")
                continue

            # ---- Server aggregation ----
            if len(successful_clients) < len(selected):
                print(f"[WARNING] Only {len(successful_clients)}/{len(selected)} clients succeeded")
                # Only aggregate successful clients
                filtered_results = {cid: client_results[cid] for cid in successful_clients}
                server.federated_round(successful_clients, filtered_results)
            else:
                server.federated_round(selected, client_results)

            # Save every round
            save_path = f"checkpoint_round_{rnd + 1}.pth"
            try:
                server.save(save_path)
                print(f" Saved checkpoint to {save_path}")
            except Exception as e:
                print(f"[ERROR] Failed to save checkpoint: {e}")
                import traceback
                traceback.print_exc()

            # Evaluate on validation set (use first client's validation set)
            try:
                # [Fix]: Use val_loader instead of dataloader
                eval_loader = clients[0].val_loader

                # If val_loader is empty, fallback to train_loader just for testing pipeline
                if len(eval_loader) == 0:
                    print("[WARNING] Val loader empty, falling back to train loader for eval")
                    eval_loader = clients[0].train_loader

                val_dice, val_psnr, val_ssim = evaluate(
                    server.global_model,
                    eval_loader,
                    device
                )
                dice_history.append(val_dice)
                psnr_history.append(val_psnr)
                ssim_history.append(val_ssim)
                print(f"[Round {rnd + 1}] Dice={val_dice:.4f}, PSNR={val_psnr:.3f}, SSIM={val_ssim:.4f}")
            except Exception as e:
                print(f"[WARNING] Evaluation failed: {e}")
                import traceback
                traceback.print_exc()
                # Append None to maintain history length
                dice_history.append(None)
                psnr_history.append(None)
                ssim_history.append(None)

        except KeyboardInterrupt:
            print("\n[INFO] Training interrupted by user")
            break
        except Exception as e:
            print(f"[ERROR] Round {rnd + 1} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("Training finished!")

    # -----------------------------
    # Plot metrics curves
    # -----------------------------
    def plot_curve(values, name, out_dir="plots_retrain"):
        """Plot a single metric curve"""
        os.makedirs(out_dir, exist_ok=True)

        # Filter out None values
        valid_values = [(i, v) for i, v in enumerate(values) if v is not None]
        if len(valid_values) == 0:
            print(f"[WARNING] No valid {name} values to plot")
            return

        rounds, vals = zip(*valid_values)

        plt.figure(figsize=(8, 5))
        plt.plot(rounds, vals, marker='o', linestyle='-', linewidth=2, markersize=6)
        plt.title(f"{name} Curve")
        plt.xlabel("Round")
        plt.ylabel(name)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{name.lower()}.png"), dpi=150)
        plt.close()
        print(f"  Saved {name} curve to {out_dir}/{name.lower()}.png")

    print("\n===== GENERATING METRICS PLOTS =====")
    plot_curve(dice_history, "Dice")
    plot_curve(psnr_history, "PSNR")
    plot_curve(ssim_history, "SSIM")
    print("Saved Dice/PSNR/SSIM curves to plots_retrain/")


if __name__ == "__main__":
    main()