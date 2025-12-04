"""
FL Client
---------
Each client performs local training and returns:
 - updated model weights
 - updated α parameters
 - number of local samples
 - (optional) compressed weights

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
import numpy as np

from federated.fl_server import quantize_8bit, topk_sparsify
from losses import CombinedLoss
from metrics import dice_score


# ------------------------------------------------------------
# Example 3D segmentation loss (Dice + CE)
# You may replace with your own
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
        sample_temperature=5.0,  # Initial temperature for Gumbel-Softmax
        lambda_flops=1e-4,  # FLOPs regularization weight
        min_temp=0.5,  # Minimum temperature (temperature won't go below this)
        temp_decay=0.95,  # Temperature decay factor per epoch
    ):
        """
        Args:
            cid: client id
            model_fn: function that returns new model instance
            dataset: local dataset
        """
        self.cid = cid
        self.device = device
        self.model_fn = model_fn
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
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

        # loss: use CombinedLoss (CrossEntropy + Dice)
        self.criterion = CombinedLoss(weight_ce=1.0, weight_dice=1.0)

    # --------------------------------------------------------
    # Set local model to server weights
    # --------------------------------------------------------
    def load_global_model(self, server_state, compress=False):
        """
        server_state can be:
         (1) raw state_dict
         (2) compressed dict: each item = (q,scale,min) or (sparse,mask)
        """
        model = self.model_fn().to(self.device)
        if not compress:
            # Filter out alpha and gate keys (they're not registered parameters)
            filtered_state = {k: v for k, v in server_state.items() 
                            if not k.endswith('.alpha') and not k.endswith('.gate')}
            model.load_state_dict(filtered_state, strict=False)
            return model

        # --- decompress ---
        new_state = {}
        for k, item in server_state.items():
            # Skip alpha and gate keys
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
    # Retrieve α parameters
    # --------------------------------------------------------
    def get_alpha_params(self, model):
        return [p for p in model.arch_parameters()]

    # --------------------------------------------------------
    # Local training
    # --------------------------------------------------------
    def train(self, global_state, global_alpha):
        """
        Args:
            global_state: server model parameters
            global_alpha: list of α parameter tensors
        Return:
            {
              "weights": state_dict,
              "alpha": list_of_alpha,
              "size": num_samples
            }
        """
        # 1. Load model
        model = self.load_global_model(global_state, compress=False)

        # Load α
        # Check if model has set_alpha method (new NAS model)
        if hasattr(model, 'set_alpha') and hasattr(model, 'alpha_mgr'):
            # New NAS model: assign alphas to alpha_mgr
            if len(global_alpha) > 0:
                for i, alpha in enumerate(global_alpha):
                    if i < len(model.alpha_mgr.alpha_list):
                        model.alpha_mgr.alpha_list[i].data = alpha.clone().detach().to(self.device).requires_grad_(True)
                model.set_alpha()
        else:
            # Old NAS model: manually assign to MixedOp3D
            from model_swin3d.nas_ops import MixedOp3D
            ptr = 0
            for m in model.modules():
                if isinstance(m, MixedOp3D):
                    if ptr < len(global_alpha):
                        m.alpha = global_alpha[ptr].clone().detach().to(self.device).requires_grad_(True)
                        ptr += 1

        # 2. Optimizers
        # model params (except alpha)
        arch_params = list(model.arch_parameters())
        arch_param_ids = {id(p) for p in arch_params}
        model_params = [p for p in model.parameters() if id(p) not in arch_param_ids]

        # Weight optimizer: higher LR
        opt = optim.AdamW(model_params, lr=self.lr, weight_decay=self.weight_decay)
        
        # Architecture optimizer: lower LR, with weight decay
        opt_alpha = None
        if len(arch_params) > 0:
            # Use smaller LR for architecture parameters
            opt_alpha = optim.AdamW(arch_params, lr=self.lr_alpha, weight_decay=1e-3)

        # schedulers
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        sch_alpha = optim.lr_scheduler.CosineAnnealingLR(opt_alpha, T_max=self.epochs) if opt_alpha is not None else None

        # 3. Prepare validation data (use first batch from dataloader as validation)
        val_data = None
        val_target = None
        for vol, seg in self.dataloader:
            val_data = vol.to(self.device)
            val_target = seg.to(self.device).long()
            break

        # 4. Training loop: Weight update phase (Gumbel-Softmax)
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Temperature schedule: anneal from initial value to min_temp
        min_temp = getattr(self, 'min_temp', 0.5)
        temp_decay = getattr(self, 'temp_decay', 0.95)
        current_temp = self.sample_temperature
        
        # hard flag: set to True when temp is low (e.g., temp < 0.5)
        hard_flag = current_temp < 0.5
        
        for epoch in range(self.epochs):
            for vol, seg in self.dataloader:
                vol = vol.to(self.device)
                seg = seg.to(self.device).long()

                # Forward with Gumbel-Softmax (for weight updates)
                if hasattr(model, 'forward_gumbel'):
                    logits = model.forward_gumbel(vol, temp=current_temp, hard=hard_flag)
                else:
                    logits = model(vol)
                loss = self.criterion(logits, seg)

                # Backward for weight parameters only
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model_params, self.grad_clip)
                opt.step()

                total_loss += loss.item()
                num_batches += 1

            sch.step()
            
            # Anneal temperature per epoch
            current_temp = max(current_temp * temp_decay, min_temp)
            hard_flag = current_temp < 0.5

        # 5. Architecture update phase (validation + FLOPs penalty)
        if opt_alpha is not None and len(arch_params) > 0 and val_data is not None:
            model.eval()
            
            # Compute validation loss under soft architecture
            if hasattr(model, 'forward_soft'):
                logits_val = model.forward_soft(val_data)
            else:
                logits_val = model(val_data)
            val_loss = self.criterion(logits_val, val_target)

            # Compute Dice Score for validation
            _, val_dice = dice_score(logits_val, val_target)

            # Compute expected FLOPs (use current temperature)
            if hasattr(model, 'expected_flops'):
                expected_flops = model.expected_flops(temp=current_temp)
            else:
                expected_flops = torch.tensor(0.0, device=self.device)

            # Architecture loss = validation loss + FLOPs penalty
            arch_loss = val_loss + self.lambda_flops * expected_flops

            # Backward only to architecture parameters
            # Temporarily disable gradients for weight parameters
            for p in model.parameters():
                p.requires_grad = False
            for ap in arch_params:
                ap.requires_grad = True

            opt_alpha.zero_grad()
            arch_loss.backward()
            nn.utils.clip_grad_norm_(arch_params, self.grad_clip)
            opt_alpha.step()

            # Restore requires_grad
            for p in model.parameters():
                p.requires_grad = True

            if sch_alpha is not None:
                sch_alpha.step()
            
            # Store metrics for logging
            self._last_val_loss = val_loss.item()
            self._last_val_dice = val_dice
            self._last_expected_flops = expected_flops.item()
            self._last_temperature = current_temp
        elif val_data is not None:
            # Even without architecture update, compute validation metrics for fixed model
            model.eval()
            with torch.no_grad():
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

        # ----------------------------------------------------
        # Return updated weights + α
        # ----------------------------------------------------
        final_state = model.state_dict()

        # Extract α as list
        # Check if model has alpha_mgr (new NAS model)
        if hasattr(model, 'alpha_mgr') and model.alpha_mgr is not None:
            # New NAS model: extract from alpha_mgr
            alphas = [p.detach().cpu() for p in model.alpha_mgr.parameters()]
        else:
            # Old NAS model: extract from MixedOp3D modules
            from model_swin3d.nas_ops import MixedOp3D
            alphas = []
            for m in model.modules():
                if isinstance(m, MixedOp3D) and m.alpha is not None:
                    alphas.append(m.alpha.detach().cpu())

        # compression (to reduce communication)
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
            avg_loss = total_loss / num_batches if num_batches > 0 else None
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
