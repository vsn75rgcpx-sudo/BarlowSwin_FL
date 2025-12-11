"""
FL Client
---------
Each client performs local training and returns:
 - updated model weights
 - updated α parameters
 - number of local samples
 - (optional) compressed weights

Modified:
 - Fixed AMP deprecation warnings (switched to torch.amp).
 - Auto-disable AMP on CPU/MPS to prevent errors/warnings.
 - Explicit Train/Val split for stable NAS architecture updates.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import copy
import numpy as np

# Use new torch.amp API if available, else fallback
try:
    from torch.amp import autocast, GradScaler

    has_new_amp = True
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

    has_new_amp = False

from federated.fl_server import quantize_8bit, topk_sparsify
from losses import CombinedLoss
from metrics import dice_score


# ------------------------------------------------------------
# Example 3D segmentation loss (Dice + CE)
# ------------------------------------------------------------
class DiceCELoss(nn.Module):
    def __init__(self, weight_ce=1.0, weight_dice=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice

    def forward(self, logits, target):
        ce_loss = self.ce(logits, target)

        # dice
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


# ------------------------------------------------------------
# FL Client
# ------------------------------------------------------------
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

        # --- optimization ---
        self.lr = lr
        self.lr_alpha = lr_alpha
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip

        self.compress = compress
        self.compress_mode = compress_mode
        self.topk_ratio = topk_ratio

        # loss
        self.criterion = CombinedLoss(weight_ce=1.0, weight_dice=1.0)

        # --- AMP Scaler & Device Config ---
        # Only enable AMP on CUDA devices to avoid CPU/MPS warnings and slowdowns
        if "cuda" in str(device) and torch.cuda.is_available():
            self.use_amp = True
            self.device_type = "cuda"
            # Use appropriate Scaler based on API version
            if has_new_amp:
                self.scaler = GradScaler('cuda')
            else:
                self.scaler = GradScaler()
        else:
            self.use_amp = False
            self.device_type = "cpu"
            self.scaler = None

        # --- Data Split ---
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

        self.train_loader = DataLoader(self.train_ds, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_ds, batch_size=batch_size, shuffle=False)
        self.val_iter = iter(self.val_loader)

    # --------------------------------------------------------
    # Set local model to server weights
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Context Manager for AMP (Helper)
    # --------------------------------------------------------
    def _autocast_context(self):
        if self.use_amp:
            if has_new_amp:
                return autocast(device_type=self.device_type)
            else:
                return autocast()  # Legacy
        else:
            # Null context manager
            from contextlib import nullcontext
            return nullcontext()

    # --------------------------------------------------------
    # Local training
    # --------------------------------------------------------
    def train(self, global_state, global_alpha):
        # 1. Load model
        model = self.load_global_model(global_state, compress=False)

        # Load α
        if hasattr(model, 'set_alpha') and hasattr(model, 'alpha_mgr'):
            if len(global_alpha) > 0:
                for i, alpha in enumerate(global_alpha):
                    if i < len(model.alpha_mgr.alpha_list):
                        model.alpha_mgr.alpha_list[i].data = alpha.clone().detach().to(self.device).requires_grad_(True)
                model.set_alpha()
        else:
            from model_swin3d.nas_ops import MixedOp3D
            ptr = 0
            for m in model.modules():
                if isinstance(m, MixedOp3D):
                    if ptr < len(global_alpha):
                        m.alpha = global_alpha[ptr].clone().detach().to(self.device).requires_grad_(True)
                        ptr += 1

        # 2. Optimizers
        arch_params = list(model.arch_parameters())
        arch_param_ids = {id(p) for p in arch_params}
        model_params = [p for p in model.parameters() if id(p) not in arch_param_ids]

        opt = optim.AdamW(model_params, lr=self.lr, weight_decay=self.weight_decay)

        opt_alpha = None
        if len(arch_params) > 0:
            opt_alpha = optim.AdamW(arch_params, lr=self.lr_alpha, weight_decay=1e-3)

        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        sch_alpha = optim.lr_scheduler.CosineAnnealingLR(opt_alpha,
                                                         T_max=self.epochs) if opt_alpha is not None else None

        # 3. Training loop
        model.train()
        total_loss = 0.0
        num_batches = 0

        min_temp = getattr(self, 'min_temp', 0.5)
        temp_decay = getattr(self, 'temp_decay', 0.95)
        current_temp = self.sample_temperature
        hard_flag = current_temp < 0.5

        for epoch in range(self.epochs):
            for vol, seg in self.train_loader:
                vol = vol.to(self.device)
                seg = seg.to(self.device).long()

                opt.zero_grad()

                # Forward
                with self._autocast_context():
                    if hasattr(model, 'forward_gumbel'):
                        logits = model.forward_gumbel(vol, temp=current_temp, hard=hard_flag)
                    else:
                        logits = model(vol)
                    loss = self.criterion(logits, seg)

                # Backward
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
            current_temp = max(current_temp * temp_decay, min_temp)
            hard_flag = current_temp < 0.5

        # 4. Architecture update
        val_data = None
        val_target = None

        if opt_alpha is not None and len(arch_params) > 0:
            try:
                val_data, val_target = next(self.val_iter)
            except StopIteration:
                self.val_iter = iter(self.val_loader)
                val_data, val_target = next(self.val_iter)

            val_data = val_data.to(self.device)
            val_target = val_target.to(self.device).long()

            model.eval()
            for p in model_params: p.requires_grad = False
            for p in arch_params: p.requires_grad = True

            opt_alpha.zero_grad()

            with self._autocast_context():
                if hasattr(model, 'forward_soft'):
                    logits_val = model.forward_soft(val_data)
                else:
                    logits_val = model(val_data)
                val_loss = self.criterion(logits_val, val_target)

                if hasattr(model, 'expected_flops'):
                    expected_flops = model.expected_flops(temp=current_temp)
                else:
                    expected_flops = torch.tensor(0.0, device=self.device)

                arch_loss = val_loss + self.lambda_flops * expected_flops

            if self.use_amp and self.scaler:
                self.scaler.scale(arch_loss).backward()
                self.scaler.unscale_(opt_alpha)
                nn.utils.clip_grad_norm_(arch_params, self.grad_clip)
                self.scaler.step(opt_alpha)
                self.scaler.update()
            else:
                arch_loss.backward()
                nn.utils.clip_grad_norm_(arch_params, self.grad_clip)
                opt_alpha.step()

            for p in model_params: p.requires_grad = True
            if sch_alpha is not None:
                sch_alpha.step()

            with torch.no_grad():
                _, val_dice = dice_score(logits_val, val_target)
            self._last_val_loss = val_loss.item()
            self._last_val_dice = val_dice
            self._last_expected_flops = expected_flops.item()
            self._last_temperature = current_temp

        elif len(self.val_loader) > 0:
            # Metric calc only
            try:
                val_data, val_target = next(self.val_iter)
            except StopIteration:
                self.val_iter = iter(self.val_loader)
                val_data, val_target = next(self.val_iter)
            val_data = val_data.to(self.device)
            val_target = val_target.to(self.device).long()

            model.eval()
            with torch.no_grad(), self._autocast_context():
                if hasattr(model, 'forward_soft'):
                    logits_val = model.forward_soft(val_data)
                else:
                    logits_val = model(val_data)
                val_loss = self.criterion(logits_val, val_target)
                _, val_dice = dice_score(logits_val, val_target)

            self._last_val_loss = val_loss.item()
            self._last_val_dice = val_dice
            self._last_expected_flops = None
            self._last_temperature = None
        else:
            self._last_val_loss = None
            self._last_val_dice = None
            self._last_expected_flops = None
            self._last_temperature = None

        final_state = model.state_dict()
        if hasattr(model, 'alpha_mgr') and model.alpha_mgr is not None:
            alphas = [p.detach().cpu() for p in model.alpha_mgr.parameters()]
        else:
            from model_swin3d.nas_ops import MixedOp3D
            alphas = []
            for m in model.modules():
                if isinstance(m, MixedOp3D) and m.alpha is not None:
                    alphas.append(m.alpha.detach().cpu())

        if self.compress:
            compressed = {}
            for k, v in final_state.items():
                v_cpu = v.cpu()
                if self.compress_mode == "8bit":
                    q, scale, minv = quantize_8bit(v_cpu)
                    compressed[k] = (q, scale, minv)
                elif self.compress_mode == "topk":
                    sparse, mask = topk_sparsify(v_cpu, self.topk_ratio)
                    compressed[k] = (sparse, mask)
            return {
                "weights": compressed,
                "alpha": alphas,
                "size": len(self.dataset)
            }
        else:
            avg_loss = total_loss / max(1, num_batches)
            return {
                "weights": final_state,
                "alpha": alphas,
                "size": len(self.dataset),
                "loss": avg_loss,
                "val_loss": getattr(self, '_last_val_loss', None),
                "val_dice": getattr(self, '_last_val_dice', None),
                "expected_flops": getattr(self, '_last_expected_flops', None),
                "temperature": getattr(self, '_last_temperature', self.sample_temperature)
            }