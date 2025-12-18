"""
FL Client (Multi-Threaded Loader Version)
---------
Modified:
 - [FIX] DataLoader num_workers=8 to speed up data loading
 - Fixed AMP deprecation warnings
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import copy
import numpy as np

try:
    from torch.amp import autocast, GradScaler

    has_new_amp = True
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

    has_new_amp = False

from federated.fl_server import quantize_8bit, topk_sparsify
from losses import CombinedLoss
from metrics import dice_score


class DiceCELoss(nn.Module):
    def __init__(self, weight_ce=1.0, weight_dice=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice

    def forward(self, logits, target):
        ce_loss = self.ce(logits, target)
        num_classes = logits.shape[1]
        pred = torch.softmax(logits, dim=1)
        dice = 0.0
        for c in range(num_classes):
            p = pred[:, c]
            t = (target == c).float()
            inter = (p * t).sum()
            denom = (p + t).sum() + 1e-6
            dice += 1 - 2 * inter / denom
        dice /= num_classes
        return self.weight_ce * ce_loss + self.weight_dice * dice


class FederatedClient:
    def __init__(
            self,
            cid,
            model_fn,
            dataset,
            batch_size=1,
            epochs=1,
            device="cuda",
            lr=1e-4,
            lr_alpha=3e-3,
            weight_decay=1e-5,
            grad_clip=1.0,
            compress=False,
            compress_mode="8bit",
            topk_ratio=0.1,
            sample_temperature=5.0,
            lambda_flops=1e-4,
            min_temp=0.5,
            temp_decay=0.95,
            val_split_ratio=0.1,
    ):
        self.cid = cid
        self.device = device
        self.model_fn = model_fn
        self.dataset = dataset
        self.epochs = epochs
        self.sample_temperature = sample_temperature
        self.lambda_flops = lambda_flops
        self.min_temp = min_temp
        self.temp_decay = temp_decay

        self.lr = lr
        self.lr_alpha = lr_alpha
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip

        self.compress = compress
        self.compress_mode = compress_mode
        self.topk_ratio = topk_ratio

        self.criterion = CombinedLoss(weight_ce=1.0, weight_dice=1.0)

        if "cuda" in str(device) and torch.cuda.is_available():
            self.use_amp = True
            self.device_type = "cuda"
            self.scaler = GradScaler('cuda') if has_new_amp else GradScaler()
        else:
            self.use_amp = False
            self.device_type = "cpu"
            self.scaler = None

        total_len = len(dataset)
        val_len = int(val_split_ratio * total_len)
        if val_len < 1 and total_len > 1:
            val_len = 1
        train_len = total_len - val_len

        if val_len > 0:
            self.train_ds, self.val_ds = random_split(
                dataset, [train_len, val_len],
                generator=torch.Generator().manual_seed(42 + cid)
            )
        else:
            self.train_ds = dataset
            self.val_ds = dataset

        # [关键修改] num_workers=8 和 pin_memory=True
        # 利用服务器多核 CPU 预取数据，彻底解决 GPU 等待问题
        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True  # 保持线程活跃
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        self.val_iter = iter(self.val_loader)

    def load_global_model(self, server_state, compress=False):
        model = self.model_fn().to(self.device)
        if not compress:
            filtered_state = {k: v for k, v in server_state.items()
                              if not k.endswith('.alpha') and not k.endswith('.gate')}
            model.load_state_dict(filtered_state, strict=False)
            return model

        new_state = {}
        for k, item in server_state.items():
            if k.endswith('.alpha') or k.endswith('.gate'):
                continue
            if self.compress_mode == "8bit":
                q, scale, minv = item
                new_state[k] = (q.float() * scale + minv).to(self.device)
            elif self.compress_mode == "topk":
                sparse, mask = item
                new_state[k] = sparse.to(self.device)
        model.load_state_dict(new_state, strict=False)
        return model

    def _autocast_context(self):
        if self.use_amp:
            return autocast(device_type=self.device_type)
        else:
            from contextlib import nullcontext
            return nullcontext()

    def train(self, global_state, global_alpha):
        model = self.load_global_model(global_state, compress=False)

        # Skip alpha loading for fixed model retraining
        # ... (rest of the logic remains same, standard training loop) ...

        # Optimizers
        arch_params = list(model.arch_parameters())
        arch_param_ids = {id(p) for p in arch_params}
        model_params = [p for p in model.parameters() if id(p) not in arch_param_ids]

        opt = optim.AdamW(model_params, lr=self.lr, weight_decay=self.weight_decay)
        opt_alpha = None
        if len(arch_params) > 0:
            opt_alpha = optim.AdamW(arch_params, lr=self.lr_alpha, weight_decay=1e-3)

        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)

        model.train()
        total_loss = 0.0
        num_batches = 0

        for epoch in range(self.epochs):
            for vol, seg in self.train_loader:
                vol = vol.to(self.device)
                seg = seg.to(self.device).long()

                opt.zero_grad()

                with self._autocast_context():
                    if hasattr(model, 'forward_gumbel'):  # NAS mode
                        logits = model.forward_gumbel(vol, temp=self.sample_temperature)
                    else:  # Fixed mode
                        logits = model(vol)
                    loss = self.criterion(logits, seg)

                if self.use_amp and self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model_params, self.grad_clip)
                    self.scaler.step(opt)
                    self.scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model_params, self.grad_clip)
                    opt.step()

                total_loss += loss.item()
                num_batches += 1
            sch.step()

        # Simple return for fixed training
        final_state = model.state_dict()
        avg_loss = total_loss / max(1, num_batches)

        return {
            "weights": final_state,
            "alpha": [],
            "size": len(self.dataset),
            "loss": avg_loss
        }