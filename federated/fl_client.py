import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from losses import NewCombinedLoss  # 导入新 Loss


class FederatedClient:
    def __init__(self, cid, model_fn, dataset, device="cuda",
                 batch_size=8, epochs=1, lr=1e-3, weight_decay=1e-4,
                 lr_alpha=0.0, sample_temperature=5.0, lambda_flops=0.0,
                 val_split_ratio=0.1, num_workers=0,
                 compress=False, compress_mode="8bit", topk_ratio=0.1,
                 grad_clip=1.0):
        self.cid = cid
        self.device = device
        self.dataset = dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_alpha = lr_alpha
        self.num_workers = num_workers
        self.grad_clip = grad_clip

        self.model = model_fn().to(device)

        # Dataset Split
        total_len = len(dataset)
        val_len = int(total_len * val_split_ratio)
        train_len = total_len - val_len
        if val_len < 1:
            self.train_ds, self.val_ds = dataset, None
            self.val_loader = None
        else:
            self.train_ds, self.val_ds = torch.utils.data.random_split(
                dataset, [train_len, val_len],
                generator=torch.Generator().manual_seed(42 + cid)
            )
            # 预先创建验证集 loader
            self.val_loader = DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                                         num_workers=self.num_workers)

        # 预先创建训练集 loader
        self.train_loader = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                                       num_workers=self.num_workers, pin_memory=(self.device == "cuda"))

    def train(self, global_weights, global_alphas=[]):
        # 1. Load Global Weights
        clean_state = {k: v for k, v in global_weights.items()
                       if not k.endswith('.alpha') and not k.endswith('.gate')}
        self.model.load_state_dict(clean_state, strict=False)

        # 2. Load Global Alphas
        if hasattr(self.model, 'alpha_mgr') and self.model.alpha_mgr is not None and len(global_alphas) > 0:
            with torch.no_grad():
                for param, g_alpha in zip(self.model.alpha_mgr.parameters(), global_alphas):
                    if param.shape != g_alpha.shape:
                        continue
                    param.copy_(g_alpha)
            if hasattr(self.model, 'set_alpha'):
                self.model.set_alpha()

        # 3. Optimizers
        weight_params = [p for n, p in self.model.named_parameters()
                         if 'alpha' not in n and 'gate' not in n and p.requires_grad]

        self.optimizer = optim.AdamW(weight_params, lr=self.lr, weight_decay=self.weight_decay)

        # [新增] 学习率调度器: 余弦退火
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-6)

        self.alpha_optimizer = None
        if self.lr_alpha > 0 and hasattr(self.model, 'alpha_mgr') and self.model.alpha_mgr is not None:
            self.alpha_optimizer = optim.Adam(self.model.alpha_mgr.parameters(),
                                              lr=self.lr_alpha, betas=(0.5, 0.999), weight_decay=1e-3)

        # 4. Loss Function (使用 Dice + CE + Boundary)
        # weight_boundary 可根据需要调整
        criterion = NewCombinedLoss(weight_dice=1.0, weight_ce=1.0, weight_boundary=0.01).to(self.device)

        # 5. Training Loop
        self.model.train()
        epoch_losses = []

        for ep in range(self.epochs):
            batch_losses = []
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)

                # Step A: Architecture Update (如果需要)
                if self.alpha_optimizer:
                    self.alpha_optimizer.zero_grad()
                    output = self.model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    self.alpha_optimizer.step()
                    if hasattr(self.model, 'set_alpha'):
                        self.model.set_alpha()

                # Step B: Weight Update
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()

                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(weight_params, self.grad_clip)

                self.optimizer.step()
                batch_losses.append(loss.item())

            # Step C: 更新学习率
            self.scheduler.step()

            epoch_losses.append(np.mean(batch_losses))

        # 6. Return Results
        updated_alphas = []
        if hasattr(self.model, 'alpha_mgr') and self.model.alpha_mgr is not None:
            for p in self.model.alpha_mgr.parameters():
                updated_alphas.append(p.detach().cpu())

        return {
            "weights": self.model.state_dict(),
            "alpha": updated_alphas,
            "loss": np.mean(epoch_losses) if epoch_losses else 0.0,
            "size": len(self.train_ds)
        }