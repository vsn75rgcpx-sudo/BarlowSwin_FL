"""
federated/fl_client.py
----------------------
Standard Federated Client.
Handles local training (both weights and architecture alphas).

Fixed:
 - Alpha synchronization logic now uses model.alpha_mgr to ensure correct order
   (FFN vs Attention vs Gate) avoiding shape mismatch errors.
 - Correctly imports SoftDiceLoss.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from losses import SoftDiceLoss  # 确保导入正确


class FederatedClient:
    def __init__(self, cid, model_fn, dataset, device="cuda",
                 batch_size=8, epochs=1, lr=1e-3, weight_decay=1e-4,
                 lr_alpha=0.0, sample_temperature=5.0, lambda_flops=0.0,
                 val_split_ratio=0.1, num_workers=0):
        self.cid = cid
        self.device = device
        self.dataset = dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_alpha = lr_alpha
        self.num_workers = num_workers

        self.model = model_fn().to(device)

        # Dataset Split
        total_len = len(dataset)
        val_len = int(total_len * val_split_ratio)
        train_len = total_len - val_len
        if val_len < 1:
            self.train_ds, self.val_ds = dataset, None
        else:
            self.train_ds, self.val_ds = torch.utils.data.random_split(
                dataset, [train_len, val_len],
                generator=torch.Generator().manual_seed(42 + cid)
            )

    def train(self, global_weights, global_alphas=[]):
        # 1. Load Global Weights (排除 alpha 相关参数)
        clean_state = {k: v for k, v in global_weights.items()
                       if not k.endswith('.alpha') and not k.endswith('.gate')}
        self.model.load_state_dict(clean_state, strict=False)

        # 2. Load Global Alphas (【修正逻辑】使用 AlphaManager 确保顺序一致)
        if hasattr(self.model, 'alpha_mgr') and self.model.alpha_mgr is not None and len(global_alphas) > 0:
            # 直接把 Server 发来的参数填入 alpha_mgr，顺序绝对对齐
            with torch.no_grad():
                for param, g_alpha in zip(self.model.alpha_mgr.parameters(), global_alphas):
                    if param.shape != g_alpha.shape:
                        print(
                            f"[Client {self.cid}] 警告: Alpha 形状不匹配! 本地 {param.shape} vs 全局 {g_alpha.shape}。已跳过。")
                        continue
                    param.copy_(g_alpha)
                    param.requires_grad_(True)
            # 同步到各个 Block
            if hasattr(self.model, 'set_alpha'):
                self.model.set_alpha()

        # 3. DataLoaders
        train_loader = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                                  num_workers=self.num_workers, pin_memory=(self.device == "cuda"))

        # 4. Optimizers
        # 权重优化器: 排除 alpha 和 gate
        weight_params = [p for n, p in self.model.named_parameters()
                         if 'alpha' not in n and 'gate' not in n and p.requires_grad]
        self.optimizer = optim.AdamW(weight_params, lr=self.lr, weight_decay=self.weight_decay)

        # 架构优化器: 直接优化 alpha_mgr 的参数
        self.alpha_optimizer = None
        if self.lr_alpha > 0 and hasattr(self.model, 'alpha_mgr') and self.model.alpha_mgr is not None:
            self.alpha_optimizer = optim.Adam(self.model.alpha_mgr.parameters(),
                                              lr=self.lr_alpha, betas=(0.5, 0.999), weight_decay=1e-3)

        # 5. Training Loop
        self.model.train()
        epoch_losses = []
        criterion_ce = nn.CrossEntropyLoss()
        criterion_dice = SoftDiceLoss()

        for ep in range(self.epochs):
            batch_losses = []
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)

                # --- Step A: Architecture Update ---
                if self.alpha_optimizer:
                    self.alpha_optimizer.zero_grad()
                    output = self.model(data)
                    loss = criterion_ce(output, target) + criterion_dice(output, target)
                    loss.backward()
                    self.alpha_optimizer.step()
                    # 再次同步 (虽不是必须，但保险)
                    if hasattr(self.model, 'set_alpha'):
                        self.model.set_alpha()

                # --- Step B: Weight Update ---
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = criterion_ce(output, target) + criterion_dice(output, target)
                loss.backward()
                self.optimizer.step()
                batch_losses.append(loss.item())
            epoch_losses.append(np.mean(batch_losses))

        # 6. Validation
        val_dice = 0.0
        if self.val_ds:
            self.model.eval()
            val_loader = DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                                    num_workers=self.num_workers)
            import metrics
            scores = []
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    out = self.model(data)
                    scores.append(metrics.dice_score(out, target))
            val_dice = np.mean(scores) if scores else 0.0

        # 7. Package Results (【修正逻辑】使用 AlphaManager 打包)
        updated_alphas = []
        if hasattr(self.model, 'alpha_mgr') and self.model.alpha_mgr is not None:
            for p in self.model.alpha_mgr.parameters():
                updated_alphas.append(p.detach().cpu())

        return {
            "weights": self.model.state_dict(),
            "alpha": updated_alphas,
            "loss": np.mean(epoch_losses) if epoch_losses else 0.0,
            "val_dice": val_dice,
            "size": len(self.train_ds)
        }