import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from losses import NewCombinedLoss
import metrics


class FederatedClient:
    def __init__(self, cid, model_fn, dataset, device="cuda",
                 batch_size=8, epochs=1, lr=1e-3, weight_decay=1e-4,
                 lr_alpha=0.0, val_split_ratio=0.1, num_workers=0,
                 grad_clip=1.0, **kwargs):
        self.cid = cid
        self.device = device
        self.dataset = dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr  # 初始 LR
        self.weight_decay = weight_decay
        self.lr_alpha = lr_alpha
        self.num_workers = num_workers
        self.grad_clip = grad_clip

        self.model = model_fn().to(device)

        # Dataset Split (保持原有逻辑)
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
            self.val_loader = DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                                         num_workers=self.num_workers)

        self.train_loader = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                                       num_workers=self.num_workers, pin_memory=(self.device == "cuda"))

    def train(self, global_weights, global_alphas=[], current_lr=None, loss_config=None):
        """
        Args:
            current_lr: 当前轮次的学习率 (如果是 None 则使用 self.lr)
            loss_config: 包含 'weight_boundary' 和 'use_focal' 的字典
        """
        # 1. Load Global Weights
        clean_state = {k: v for k, v in global_weights.items()
                       if not k.endswith('.alpha') and not k.endswith('.gate')}
        self.model.load_state_dict(clean_state, strict=False)

        # 2. Load Global Alphas
        if hasattr(self.model, 'alpha_mgr') and self.model.alpha_mgr is not None and len(global_alphas) > 0:
            with torch.no_grad():
                for param, g_alpha in zip(self.model.alpha_mgr.parameters(), global_alphas):
                    if param.shape != g_alpha.shape: continue
                    param.copy_(g_alpha)
            if hasattr(self.model, 'set_alpha'):
                self.model.set_alpha()

        # 3. Optimizers & LR Strategy
        # 如果传入了新的 LR (全局衰减策略)，则更新
        actual_lr = current_lr if current_lr is not None else self.lr

        weight_params = [p for n, p in self.model.named_parameters()
                         if 'alpha' not in n and 'gate' not in n and p.requires_grad]
        self.optimizer = optim.AdamW(weight_params, lr=actual_lr, weight_decay=self.weight_decay)

        # 4. Dynamic Loss Configuration
        w_bound = 0.01
        use_focal = False
        if loss_config:
            w_bound = loss_config.get('weight_boundary', 0.01)
            use_focal = loss_config.get('use_focal', False)

        criterion = NewCombinedLoss(
            weight_dice=1.0,
            weight_ce=1.0,
            weight_boundary=w_bound,
            use_focal=use_focal
        ).to(self.device)

        # 5. Training Loop
        self.model.train()
        epoch_losses = []

        for ep in range(self.epochs):
            batch_losses = []
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)

                # Architecture Update (Optional)
                if self.lr_alpha > 0 and hasattr(self.model, 'alpha_mgr'):
                    # (省略 Alpha 更新代码以节省空间，与原代码一致)
                    pass

                # Weight Update
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()

                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(weight_params, self.grad_clip)

                self.optimizer.step()
                batch_losses.append(loss.item())

            epoch_losses.append(np.mean(batch_losses))

        # 6. Validation (Dice)
        val_dice = 0.0
        if self.val_loader:
            self.model.eval()
            val_dices = []
            with torch.no_grad():
                for data, target in self.val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    d = metrics.dice_score(output, target)
                    val_dices.append(d)
            val_dice = np.mean(val_dices) if len(val_dices) > 0 else 0.0
            self.model.train()

        # 7. Return Results
        updated_alphas = []
        if hasattr(self.model, 'alpha_mgr') and self.model.alpha_mgr is not None:
            for p in self.model.alpha_mgr.parameters():
                updated_alphas.append(p.detach().cpu())

        return {
            "weights": self.model.state_dict(),
            "alpha": updated_alphas,
            "loss": np.mean(epoch_losses) if epoch_losses else 0.0,
            "size": len(self.train_ds),
            "val_dice": val_dice
        }