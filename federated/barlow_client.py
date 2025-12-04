"""
federated/barlow_client.py
--------------------------
Federated client for Barlow Twins self-supervised pretraining.

Each client holds:
 - a local encoder (e.g. SwinUNet3D_NAS with nas=False)
 - a projector MLP
and runs local Barlow Twins training on its dataset.

Strategy: Modality-as-View - uses different modalities as positive pairs.
"""

import os
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model_swin3d.barlow import BarlowProjector, BarlowLoss


class BarlowClient:
    """
    Federated client for Barlow Twins pretraining using modality-as-view strategy.
    
    Dataset should return (view1, view2, seg) where:
    - view1, view2: (1, D, H, W) tensors - two different modalities (or augmented views)
    - seg: (D, H, W) tensor - segmentation mask (optional, for mask-aware BT)
    
    model_encoder: nn.Module encoder mapping input volume -> feature map.
    dataset: dataset providing (view1, view2, seg) tuples.
    """

    def __init__(self, client_id: int, model_encoder: torch.nn.Module, dataset, device: str, args, logger=None):
        self.client_id = client_id
        self.device = device
        self.encoder = model_encoder.to(device)
        self.dataset = dataset
        self.args = args
        self.logger = logger

        # Projector: determine feature dim by a single forward pass through encoder
        sample = self._sample_input()
        # Ensure sample has correct shape (B, C, D, H, W)
        if sample.dim() != 5:
            raise ValueError(f"Sample must be 5D (B,C,D,H,W), got shape {sample.shape}")
        with torch.no_grad():
            # Use forward_encoder to get bottleneck features (not full UNet output)
            if hasattr(self.encoder, 'forward_encoder'):
                fea = self.encoder.forward_encoder(sample.to(device))
            else:
                raise AttributeError("Encoder must have forward_encoder method for Barlow Twins")
            
            # Extract bottleneck feature and flatten
            if isinstance(fea, (tuple, list)):
                fea = fea[0]
            # fea shape: (B, C, D, H, W) -> flatten to get feature dimension
            feat_dim = int(np.prod(fea.shape[1:]))  # (B, C, D, H, W) -> C*D*H*W
            print(f"[Client {client_id}] Encoder bottleneck feature shape: {fea.shape}, flattened dim: {feat_dim}")
        self.feat_dim = feat_dim
        print(f"[Client {client_id}] Initializing projector with input_dim={self.feat_dim}, hidden={args.barlow_hidden}, out={args.barlow_out}")

        self.projector = BarlowProjector(
            in_dim=self.feat_dim,
            hidden_dim=args.barlow_hidden,
            out_dim=args.barlow_out,
        ).to(device)
        self.criterion = BarlowLoss(lambda_offdiag=args.barlow_lambda)

        # Optimizers
        self.opt_encoder = torch.optim.AdamW(
            self.encoder.parameters(), lr=args.barlow_lr, weight_decay=args.weight_decay
        )
        self.opt_proj = torch.optim.AdamW(
            self.projector.parameters(), lr=args.barlow_lr, weight_decay=args.weight_decay
        )

    # -------------------------
    # Helpers
    # -------------------------
    def _sample_input(self) -> torch.Tensor:
        """
        Sample first element from dataset and return view1 as (1, 1, D, H, W) tensor.
        Model expects (B, C, D, H, W) where B=batch, C=channels.
        Dataset returns (view1, view2, seg) where view1 is (1, D, H, W).
        """
        item = self.dataset[0]
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            view1 = item[0]  # (1, D, H, W) from dataset
        else:
            raise ValueError(f"Dataset should return (view1, view2, seg), got {type(item)}")
        
        if not isinstance(view1, torch.Tensor):
            view1 = torch.tensor(view1).float()
        
        # Ensure shape is (1, 1, D, H, W) for model input
        if view1.dim() == 3:
            # (D, H, W) -> (1, 1, D, H, W)
            view1 = view1.unsqueeze(0).unsqueeze(0)
        elif view1.dim() == 4:
            # (1, D, H, W) -> (1, 1, D, H, W)
            if view1.shape[0] == 1:
                view1 = view1.unsqueeze(0)  # Add batch dimension
            else:
                view1 = view1[0:1].unsqueeze(0)  # Take first and add batch
        elif view1.dim() == 5:
            # Already (B, C, D, H, W), just ensure batch=1
            if view1.shape[0] != 1:
                view1 = view1[0:1]
        
        return view1

    def extract_feature_flat(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract and flatten features from encoder bottleneck (for Barlow Twins).
        Args:
            x: (B, 1, D, H, W) input volume
        Returns:
            (B, feat_dim) flattened features
        """
        # Use forward_encoder to get bottleneck features instead of full UNet output
        if hasattr(self.encoder, 'forward_encoder'):
            f = self.encoder.forward_encoder(x)
        else:
            # Fallback: use full forward but extract encoder features manually
            # This is a workaround if forward_encoder is not available
            f = self.encoder(x)
            if isinstance(f, (tuple, list)):
                f = f[0]
            # If it's logits (B, num_classes, D, H, W), we need encoder features
            # For now, we'll use global average pooling on the feature map
            if f.dim() == 5 and f.shape[1] == self.encoder.num_classes:
                # This is logits, not encoder features - use adaptive pooling
                f = F.adaptive_avg_pool3d(f, (1, 1, 1)).squeeze(-1).squeeze(-1).squeeze(-1)
                return f
        
        # Flatten bottleneck feature map: (B, C, D, H, W) -> (B, C*D*H*W)
        f = f.view(f.size(0), -1)  # (B, feat_dim)
        return f

    # -------------------------
    # Local training
    # -------------------------
    def local_epoch(self) -> float:
        """
        Run one local epoch of Barlow training using modality-as-view strategy.
        Dataset returns (view1, view2, seg) where view1 and view2 are different modalities.
        """
        self.encoder.train()
        self.projector.train()
        total_loss = 0.0
        num_samples = 0

        # Use DataLoader for batching (helps with small batch sizes)
        batch_size = getattr(self.args, 'barlow_bs', 1)
        dataloader = DataLoader(
            self.dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            pin_memory=False
        )

        for batch_idx, batch in enumerate(dataloader):
            # batch is (view1, view2, seg) or list of tuples
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                v1_batch = batch[0].to(self.device)  # (B, 1, D, H, W)
                v2_batch = batch[1].to(self.device)   # (B, 1, D, H, W)
                seg_batch = batch[2].to(self.device) if len(batch) >= 3 else None  # (B, D, H, W)
            else:
                raise ValueError(f"Unexpected batch format: {type(batch)}")

            # Forward through encoder -> flatten features
            f1 = self.extract_feature_flat(v1_batch)  # (B, feat_dim)
            f2 = self.extract_feature_flat(v2_batch)  # (B, feat_dim)
            
            # Debug: check feature dimensions match projector input
            if batch_idx == 0:
                print(f"[Client {self.client_id}] Batch 0 - f1 shape: {f1.shape}, expected feat_dim: {self.feat_dim}")
                if f1.shape[1] != self.feat_dim:
                    raise ValueError(f"Feature dimension mismatch! Got {f1.shape[1]}, expected {self.feat_dim}")

            # Project to embedding space
            z1 = self.projector(f1)  # (B, out_dim)
            z2 = self.projector(f2)  # (B, out_dim)

            # Compute Barlow Twins loss on whole volume
            loss_whole, diag_vals, off = self.criterion(z1, z2)
            loss = loss_whole

            # Optional: Mask-aware Barlow (ROI-weighted loss)
            # If use_mask=True and seg is available, add lesion-focused loss
            if getattr(self.args, 'use_mask', False) and seg_batch is not None:
                # Simple approach: weight samples by lesion presence
                # More sophisticated: ROI pooling on encoder features (requires spatial info)
                # For now, we use a simple weighting: if lesion exists, add extra loss term
                lesion_weight = getattr(self.args, 'mask_alpha', 5.0)  # Weight for lesion loss
                
                # Check if any lesion exists in batch (seg > 0)
                has_lesion = (seg_batch > 0).any(dim=(1, 2, 3))  # (B,)
                
                if has_lesion.any():
                    # Compute additional loss only on samples with lesions
                    lesion_indices = torch.where(has_lesion)[0]
                    if len(lesion_indices) > 0:
                        z1_lesion = z1[lesion_indices]
                        z2_lesion = z2[lesion_indices]
                        loss_lesion, _, _ = self.criterion(z1_lesion, z2_lesion)
                        loss = loss_whole + lesion_weight * loss_lesion

            # Backward and update
            self.opt_encoder.zero_grad()
            self.opt_proj.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.projector.parameters()), 
                self.args.grad_clip
            )
            self.opt_encoder.step()
            self.opt_proj.step()

            total_loss += float(loss.item())
            num_samples += v1_batch.size(0)

        return total_loss / max(1, num_samples)

    # -------------------------
    # State dicts
    # -------------------------
    def get_state_dicts(self) -> Dict[str, Any]:
        """Return encoder weights; projector is normally not aggregated globally."""
        return {"encoder": self.encoder.state_dict()}

    def set_state_dicts(self, state_dicts: Dict[str, Any]) -> None:
        if "encoder" in state_dicts:
            self.encoder.load_state_dict(state_dicts["encoder"])


