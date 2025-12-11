"""
federated/barlow_client.py
--------------------------
Federated client for Barlow Twins self-supervised pretraining.

Modified:
 - Added Automatic Mixed Precision (AMP) support to reduce VRAM usage.
"""

import os
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Check for torch.amp (newer pytorch) or fallback
try:
    from torch.amp import autocast, GradScaler

    has_new_amp = True
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

    has_new_amp = False

from model_swin3d.barlow import BarlowProjector, BarlowLoss


class BarlowClient:
    def __init__(self, client_id: int, model_encoder: torch.nn.Module, dataset, device: str, args, logger=None):
        self.client_id = client_id
        self.device = device
        self.encoder = model_encoder.to(device)
        self.dataset = dataset
        self.args = args
        self.logger = logger

        # Initialize AMP Scaler if using CUDA
        if "cuda" in str(device) and torch.cuda.is_available():
            self.use_amp = True
            self.device_type = "cuda"
            self.scaler = GradScaler('cuda') if has_new_amp else GradScaler()
        else:
            self.use_amp = False
            self.device_type = "cpu"
            self.scaler = None

        # Projector initialization (Run one dummy forward pass to get dim)
        sample = self._sample_input().to(device)
        with torch.no_grad():
            # Use autocast for shape inference to avoid OOM during init
            with self._autocast_context():
                if hasattr(self.encoder, 'forward_encoder'):
                    fea = self.encoder.forward_encoder(sample)
                else:
                    fea = self.encoder(sample)
                    if isinstance(fea, (tuple, list)): fea = fea[0]

            feat_dim = int(np.prod(fea.shape[1:]))
            print(f"[Client {client_id}] Encoder output shape: {fea.shape}, Flattened: {feat_dim}")

        self.feat_dim = feat_dim

        self.projector = BarlowProjector(
            in_dim=self.feat_dim,
            hidden_dim=args.barlow_hidden,
            out_dim=args.barlow_out,
        ).to(device)

        self.criterion = BarlowLoss(lambda_offdiag=args.barlow_lambda)

        self.opt_encoder = torch.optim.AdamW(
            self.encoder.parameters(), lr=args.barlow_lr, weight_decay=args.weight_decay
        )
        self.opt_proj = torch.optim.AdamW(
            self.projector.parameters(), lr=args.barlow_lr, weight_decay=args.weight_decay
        )

    def _autocast_context(self):
        if self.use_amp:
            if has_new_amp:
                return autocast(device_type=self.device_type)
            else:
                return autocast()
        else:
            from contextlib import nullcontext
            return nullcontext()

    def _sample_input(self) -> torch.Tensor:
        item = self.dataset[0]
        if isinstance(item, (tuple, list)):
            view1 = item[0]
        else:
            view1 = item

        if not isinstance(view1, torch.Tensor):
            view1 = torch.tensor(view1).float()

        # Add batch dims if needed to match (1, 1, D, H, W) or (1, C, D, H, W)
        if view1.dim() == 3:
            view1 = view1.unsqueeze(0).unsqueeze(0)
        elif view1.dim() == 4:
            view1 = view1.unsqueeze(0)

        return view1

    def extract_feature_flat(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.encoder, 'forward_encoder'):
            f = self.encoder.forward_encoder(x)
        else:
            f = self.encoder(x)
            if isinstance(f, (tuple, list)): f = f[0]
            if f.dim() == 5 and f.shape[1] == self.encoder.num_classes:
                f = F.adaptive_avg_pool3d(f, (1, 1, 1)).flatten(1)
                return f
        return f.view(f.size(0), -1)

    def local_epoch(self) -> float:
        self.encoder.train()
        self.projector.train()
        total_loss = 0.0
        num_samples = 0

        dataloader = DataLoader(
            self.dataset,
            batch_size=getattr(self.args, 'barlow_bs', 2),
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )

        for batch in dataloader:
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                v1 = batch[0].to(self.device, non_blocking=True)
                v2 = batch[1].to(self.device, non_blocking=True)
            else:
                raise ValueError("Dataset must return tuple (view1, view2)")

            self.opt_encoder.zero_grad()
            self.opt_proj.zero_grad()

            # Mixed Precision Forward
            with self._autocast_context():
                f1 = self.extract_feature_flat(v1)
                f2 = self.extract_feature_flat(v2)
                z1 = self.projector(f1)
                z2 = self.projector(f2)
                loss, _, _ = self.criterion(z1, z2)

            # Mixed Precision Backward
            if self.use_amp and self.scaler:
                self.scaler.scale(loss).backward()

                # Unscale before clipping
                self.scaler.unscale_(self.opt_encoder)
                self.scaler.unscale_(self.opt_proj)

                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.projector.parameters()),
                    self.args.grad_clip
                )

                self.scaler.step(self.opt_encoder)
                self.scaler.step(self.opt_proj)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(...)
                self.opt_encoder.step()
                self.opt_proj.step()

            total_loss += loss.item()
            num_samples += v1.size(0)

        return total_loss / max(1, len(dataloader))  # Return avg loss per batch

    def get_state_dicts(self) -> Dict[str, Any]:
        return {"encoder": self.encoder.state_dict()}

    def set_state_dicts(self, state_dicts: Dict[str, Any]) -> None:
        if "encoder" in state_dicts:
            self.encoder.load_state_dict(state_dicts["encoder"])