"""
train_barlow_fed.py
-------------------
Federated Barlow Twins pretraining for SwinUNet3D encoder.

Workflow:
 1. Each client runs local Barlow Twins self-supervised training on its data.
 2. Server aggregates encoder weights via FedAvg.
 3. Saves global encoder weights to 'encoder_pretrained.pth'.
"""

import argparse
import json
import os
import time
from glob import glob

import numpy as np
import torch

from federated.barlow_client import BarlowClient
from federated.barlow_server import BarlowServer
from model_swin3d.swin3d_unet_nas import SwinUNet3D_NAS
from datasets.barlow_pair_dataset import BarlowPairDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, default="dataset")
    parser.add_argument("--num_clients", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--barlow_target",
        type=int,
        nargs=3,
        default=(96, 96, 96),
        help="Target crop size (D H W) for Barlow Twins augmentations",
    )
    parser.add_argument("--barlow_hidden", type=int, default=512)
    parser.add_argument("--barlow_out", type=int, default=1024, 
                        help="Projector output dim (512-1024 recommended for 3D small batch)")
    parser.add_argument("--barlow_bs", type=int, default=1, 
                        help="Batch size per client for BT training")
    parser.add_argument("--barlow_lr", type=float, default=1e-4)
    parser.add_argument("--barlow_lambda", type=float, default=0.005)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--use_mask", action="store_true", 
                        help="Use segmentation mask for mask-aware BT (optional)")
    parser.add_argument("--mask_alpha", type=float, default=5.0,
                        help="Weight for lesion-focused loss in mask-aware BT")
    parser.add_argument("--test_single_case", action="store_true",
                        help="Test mode: use only the first case found (for debugging)")
    parser.add_argument("--test_num_cases", type=int, default=None,
                        help="Test mode: use first N cases (overrides test_single_case if set)")
    parser.add_argument("--save_path", type=str, default="encoder_pretrained.pth")
    args = parser.parse_args()

    # convert barlow_target list to tuple
    args.barlow_target = tuple(args.barlow_target)
    return args


def build_clients(args, device):
    # Find all NIfTI files (recursively search subdirectories)
    all_files = glob(os.path.join(args.data_folder, "**/*.nii*"), recursive=True)
    
    # Extract case IDs from filenames
    # Handle both formats: "000_t1c.nii.gz" and "BraTS-GLI-00000-000-t1c.nii.gz"
    all_ids = set()
    for f in all_files:
        basename = os.path.basename(f)
        # Try to extract case ID (before first underscore or before last dash before modality)
        if "_" in basename:
            case_id = basename.split("_")[0]
        elif "-t1" in basename or "-t2" in basename or "-seg" in basename:
            # Format: BraTS-GLI-00000-000-t1c.nii.gz -> BraTS-GLI-00000-000
            parts = basename.split("-")
            if len(parts) >= 4:
                case_id = "-".join(parts[:-1])  # Remove last part (modality)
            else:
                case_id = basename.split("-")[0]
        else:
            continue
        all_ids.add(case_id)
    
    all_ids = sorted(list(all_ids))
    
    if len(all_ids) == 0:
        raise RuntimeError(f"No NIfTI files found in {args.data_folder} (searched recursively)")
    
    # Test mode: use first N cases
    test_num = getattr(args, 'test_num_cases', None)
    if test_num is not None and test_num > 0:
        test_num = min(test_num, len(all_ids))
        all_ids = all_ids[:test_num]
        print(f"[TEST MODE] Using first {test_num} cases: {all_ids}")
        # Auto-adjust num_clients to match test cases
        if args.num_clients > test_num:
            print(f"[TEST MODE] Reducing num_clients from {args.num_clients} to {test_num}")
            args.num_clients = test_num
    elif getattr(args, 'test_single_case', False):
        print(f"[TEST MODE] Using only first case: {all_ids[0]}")
        all_ids = [all_ids[0]]
        args.num_clients = 1  # Force single client in single case test mode
    
    # If we have fewer cases than clients, use all cases and reduce client count
    if len(all_ids) < args.num_clients:
        print(f"[WARNING] Only {len(all_ids)} cases found, reducing num_clients from {args.num_clients} to {len(all_ids)}")
        args.num_clients = len(all_ids)
    
    splits = np.array_split(all_ids, args.num_clients)

    clients = []
    sizes = []

    # window size must be compatible with resolution after patch embed (/2)
    # For barlow_target (D,H,W), after patch_embed -> (D/2, H/2, W/2)
    # We choose window_size=(2,6,6) which works for (96,96,96)->(48,48,48)
    window_candidates = [
        {"window_size": (2, 6, 6), "num_heads": 2},
        {"window_size": (2, 6, 6), "num_heads": 4},
        {"window_size": (2, 6, 6), "num_heads": 6},
    ]

    for i, ids in enumerate(splits):
        # Use BarlowPairDataset for modality-as-view strategy
        ds = BarlowPairDataset(
            folder=args.data_folder,
            target_size=args.barlow_target,
            use_mask=args.use_mask,
            augment=True,  # Enable augmentations for hybrid strategy
            seed=42 + i,  # Different seed per client
        )
        # Filter dataset to only include selected case_ids
        ds.ids = list(ids)

        # encoder model: SwinUNet3D_NAS used as encoder (we will flatten its output)
        encoder_model = SwinUNet3D_NAS(
            in_channels=1,
            num_classes=2,  # dummy
            dims=(48, 96, 192, 384),
            depths=(2, 2, 2, 2),
            resolution=args.barlow_target,
            window_candidates=window_candidates,
            nas=False,
        )

        client = BarlowClient(
            client_id=i, model_encoder=encoder_model, dataset=ds, device=device, args=args
        )
        clients.append(client)
        sizes.append(len(ds.ids))

    return clients, sizes


def main():
    args = parse_args()
    device = args.device

    print("[BT Fed] Using device:", device)
    print("[BT Fed] barlow_target:", args.barlow_target)
    print("[BT Fed] Strategy: Modality-as-View (different modalities as positive pairs)")
    print(f"[BT Fed] Projector: hidden={args.barlow_hidden}, out={args.barlow_out}")
    print(f"[BT Fed] Batch size per client: {args.barlow_bs}")

    clients, sizes = build_clients(args, device)

    # server initialized with same encoder architecture
    window_candidates = [
        {"window_size": (2, 6, 6), "num_heads": 2},
        {"window_size": (2, 6, 6), "num_heads": 4},
        {"window_size": (2, 6, 6), "num_heads": 6},
    ]
    server_encoder = SwinUNet3D_NAS(
        in_channels=1,
        num_classes=2,
        dims=(48, 96, 192, 384),
        depths=(2, 2, 2, 2),
        resolution=args.barlow_target,
        window_candidates=window_candidates,
        nas=False,
    )
    server = BarlowServer(server_encoder, device, args)

    history = {"rounds": []}

    for r in range(args.rounds):
        print(f"=== BT Fed Round {r+1}/{args.rounds} ===")
        client_states = []
        round_losses = []

        for i, client in enumerate(clients):
            # load global encoder into client
            client.set_state_dicts({"encoder": server.global_encoder.state_dict()})

            loss = 0.0
            for _ in range(args.local_epochs):
                loss += client.local_epoch()
            loss = loss / max(1, args.local_epochs)
            round_losses.append(loss)
            print(f"  client {i} local BT loss {loss:.4f}")

            client_states.append(client.get_state_dicts())

        # aggregate
        new_state = server.aggregate(client_states, sizes)
        server.global_encoder.load_state_dict(new_state)
        server.save(args.save_path)

        avg_loss = float(np.mean(round_losses)) if round_losses else None
        history["rounds"].append({"round": r + 1, "avg_loss": avg_loss})

    # save history
    with open("barlow_training_log.json", "w") as f:
        json.dump(history, f, indent=2)
    print("[BT Fed] Done. encoder saved to", args.save_path)


if __name__ == "__main__":
    main()


