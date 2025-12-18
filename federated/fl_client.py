"""
federated/fl_client.py
----------------------
Standard Federated Client.
Handles local training (both weights and architecture alphas).

Updated:
 - Added 'num_workers' to __init__ and DataLoader for A40 optimization.
 - Added default values for NAS parameters.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np


class FederatedClient:
    def __init__(
            self,
            cid,
            model_fn,
            dataset,
            device="cuda",
            batch_size=8,
            epochs=1,
            lr=1e-3,
            weight_decay=1e-4,
            # NAS parameters (with defaults)
            lr_alpha=0.0,  # 默认不更新 alpha (Retrain阶段用)
            sample_temperature=5.0,
            lambda_flops=0.0,
            val_split_ratio=0.1,
            # Optimization parameter
            num_workers=0  # [新增] 默认为0，防止CPU卡死
    ):
        self.cid = cid
        self.device = device
        self.dataset = dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_alpha = lr_alpha
        self.sample_temperature = sample_temperature
        self.lambda_flops = lambda_flops
        self.num_workers = num_workers  # [新增] 保存参数

        # Initialize model
        self.model = model_fn().to(device)

        # Split Train/Val (Simple split)
        total_len = len(dataset)
        val_len = int(total_len * val_split_ratio)
        train_len = total_len - val_len

        # 如果数据太少，就不分验证集了
        if val_len < 1:
            self.train_ds = dataset
            self.val_ds = None
        else:
            # 这里的 split 需要固定 seed，保证每轮一样，但在 FL 里通常 dataset 已经是切分好的
            # 这里简单起见，假设 dataset 已经是该 client 的全部数据
            self.train_ds, self.val_ds = torch.utils.data.random_split(
                dataset, [train_len, val_len],
                generator=torch.Generator().manual_seed(42 + cid)
            )

        # Optimizers (Placeholder, initialized in train)
        self.optimizer = None
        self.alpha_optimizer = None

    def train(self, global_weights, global_alphas=[]):
        """
        Local training loop.
        Args:
            global_weights: state_dict from server
            global_alphas: list of alpha tensors (for NAS)
        Returns:
            dict with weights, alpha, loss, etc.
        """
        # 1. Load Global Weights
        # Filter out alpha/gate keys just in case
        clean_state = {k: v for k, v in global_weights.items()
                       if not k.endswith('.alpha') and not k.endswith('.gate')}
        self.model.load_state_dict(clean_state, strict=False)

        # 2. Load Global Alphas (if NAS mode)
        # Check if model supports set_alpha (New NAS model)
        if hasattr(self.model, 'set_alpha') and len(global_alphas) > 0:
            # Manually set alphas if the model doesn't link them automatically
            # Usually FederatedServer.apply_alpha_to_model handles the global model,
            # but here we need to sync the local model.
            # However, for simplicity in this codebase, we assume alphas are passed
            # as arguments or parameters.
            # In standard DARTS/FedNAS, alphas are part of the optimizer.

            # Re-assign alphas from server to local model ops
            # Assuming model has ordered MixedOps
            ptr = 0
            for m in self.model.modules():
                if hasattr(m, 'alpha') and isinstance(m.alpha, torch.Tensor):
                    if ptr < len(global_alphas):
                        # Update local alpha data
                        with torch.no_grad():
                            m.alpha.copy_(global_alphas[ptr])
                        m.alpha.requires_grad_(True)
                        ptr += 1

        # 3. DataLoaders
        # [关键修改] 这里使用了 self.num_workers
        train_loader = DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True if self.device == "cuda" else False
        )

        # 4. Optimizers
        # Weights optimizer
        # Filter out alpha parameters from weight optimizer
        weight_params = [
            p for n, p in self.model.named_parameters()
            if 'alpha' not in n and p.requires_grad
        ]
        self.optimizer = optim.AdamW(weight_params, lr=self.lr, weight_decay=self.weight_decay)

        # Alpha optimizer (only if lr_alpha > 0)
        self.alpha_optimizer = None
        if self.lr_alpha > 0:
            alpha_params = [
                p for n, p in self.model.named_parameters()
                if 'alpha' in n and p.requires_grad
            ]
            if len(alpha_params) > 0:
                self.alpha_optimizer = optim.Adam(alpha_params, lr=self.lr_alpha, betas=(0.5, 0.999), weight_decay=1e-3)

        # 5. Training Loop
        self.model.train()
        epoch_losses = []

        # FLOPs penalty setup
        # For simplicity, we just train with CrossEntropy + Dice
        from losses import DiceLoss
        criterion_ce = nn.CrossEntropyLoss()
        criterion_dice = DiceLoss(n_classes=4)

        for ep in range(self.epochs):
            batch_losses = []
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)

                # --- Step A: Architecture Search (Alpha) ---
                if self.alpha_optimizer is not None:
                    self.alpha_optimizer.zero_grad()
                    # Forward with Gumbel Softmax (if implemented in model) or Softmax
                    # For NAS, we often use a separate batch or same batch
                    # Here we use same batch for simplicity
                    output = self.model(data)

                    loss_alpha = criterion_ce(output, target) + criterion_dice(output, target)

                    # Add FLOPs penalty if needed (skipped for brevity/stability)

                    loss_alpha.backward()
                    self.alpha_optimizer.step()

                # --- Step B: Weight Training ---
                self.optimizer.zero_grad()
                output = self.model(data)

                loss_ce = criterion_ce(output, target)
                loss_dice = criterion_dice(output, target)
                loss = loss_ce + loss_dice

                loss.backward()
                self.optimizer.step()

                batch_losses.append(loss.item())

            epoch_losses.append(np.mean(batch_losses))

        # 6. Validation (Optional, for Dice log)
        val_dice = 0.0
        if self.val_ds is not None:
            self.model.eval()
            val_loader = DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                                    num_workers=self.num_workers)
            import metrics
            dice_scores = []
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    d = metrics.dice_score(output, target)
                    dice_scores.append(d)
            val_dice = np.mean(dice_scores) if len(dice_scores) > 0 else 0.0

        # 7. Package Results
        # Extract updated alpha
        updated_alphas = []
        for m in self.model.modules():
            if hasattr(m, 'alpha') and isinstance(m.alpha, torch.Tensor):
                updated_alphas.append(m.alpha.detach().cpu())

        return {
            "weights": self.model.state_dict(),
            "alpha": updated_alphas,
            "loss": np.mean(epoch_losses) if len(epoch_losses) > 0 else 0.0,
            "val_dice": val_dice,
            "size": len(self.train_ds)
        }