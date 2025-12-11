"""
FL Server Module (FedAvg + FedNAS + Compression)
------------------------------------------------

This server handles:
  - Client registration
  - Model broadcasting
  - Aggregation of model parameters
  - Aggregation of architecture parameters (α)
  - Weighted averaging (FedAvg)
  - Compression (8-bit quant / top-k)
  - Global training loop control

Modified:
 - Fixed device mismatch error in aggregate_alpha (CPU vs MPS/CUDA).
"""

import copy
import torch
import torch.nn as nn
import numpy as np


# ------------------------------------------------------------
# Utilities for compression
# ------------------------------------------------------------
def quantize_8bit(tensor: torch.Tensor):
    """
    Uniform 8-bit quantization for communication compression.
    """
    min_val = tensor.min()
    max_val = tensor.max()
    # Handle case where all values are the same (avoid division by zero)
    if max_val == min_val:
        scale = 1.0
        q = torch.zeros_like(tensor, dtype=torch.uint8)
    else:
        scale = (max_val - min_val) / 255.0
        q = torch.clamp(((tensor - min_val) / scale).round(), 0, 255).to(torch.uint8)
    return q, scale, min_val


def dequantize_8bit(q: torch.Tensor, scale: torch.Tensor, min_val: torch.Tensor):
    """
    Restore quantized tensor.
    """
    return q.float() * scale + min_val


def topk_sparsify(tensor: torch.Tensor, k=0.1):
    """
    Keep top-k% amplitudes, zero the rest.
    """
    num = tensor.numel()
    kk = int(num * k)

    if kk <= 0:
        return torch.zeros_like(tensor), None

    flat = tensor.view(-1)
    vals, idx = torch.topk(flat.abs(), kk, sorted=False)
    mask = torch.zeros_like(flat)
    mask[idx] = 1.0
    mask = mask.view_as(tensor)
    sparse = tensor * mask
    return sparse, mask


# ------------------------------------------------------------
# FL Server
# ------------------------------------------------------------
class FederatedServer:

    def __init__(
            self,
            model_fn,
            num_clients,
            device="cuda",
            compress=False,
            compress_mode="8bit",  # "8bit" or "topk"
            topk_ratio=0.1,
            alpha_lr=0.5,  # alpha update factor β
    ):
        """
        Args:
            model_fn: a function that returns NEW model instance: model_fn()
            num_clients: number of clients
            compress: enable model compression
            compress_mode: "8bit" or "topk"
            topk_ratio: ratio for top-k sparsification
            alpha_lr: α update rate (β)
        """
        self.num_clients = num_clients
        self.device = device
        self.model_fn = model_fn
        self.compress = compress
        self.compress_mode = compress_mode
        self.topk_ratio = topk_ratio
        self.alpha_lr = alpha_lr

        # -------- initialize global model --------
        self.global_model = model_fn().to(device)

        # α-params (NAS)
        self.global_alphas = [
            p.clone().detach()
            for p in self.global_model.arch_parameters()
        ]

        # Dice history for plotting
        self.train_dice_history = []
        self.val_dice_history = []

    # --------------------------------------------------------
    # Broadcast global model parameters to client
    # --------------------------------------------------------
    def get_global_weights(self):
        """
        Return weights for clients. Optionally compress.
        """
        state = self.global_model.state_dict()

        # Filter out alpha and gate keys (they're not registered parameters, managed separately)
        filtered_state = {k: v for k, v in state.items()
                          if not k.endswith('.alpha') and not k.endswith('.gate')}

        if not self.compress:
            return filtered_state

        # -------- compression --------
        compressed = {}
        for k, v in filtered_state.items():
            if self.compress_mode == "8bit":
                q, scale, minv = quantize_8bit(v.cpu())
                compressed[k] = (q, scale, minv)
            elif self.compress_mode == "topk":
                sparse, mask = topk_sparsify(v.cpu(), self.topk_ratio)
                compressed[k] = (sparse, mask)
        return compressed

    # --------------------------------------------------------
    # Set global model state after client sends updates
    # --------------------------------------------------------
    def set_global_weights(self, state_dict):
        # Filter out alpha keys as they're not registered parameters
        # Alpha is managed separately via global_alphas
        filtered_state = {k: v for k, v in state_dict.items() if not k.endswith('.alpha')}
        self.global_model.load_state_dict(filtered_state, strict=False)

    # --------------------------------------------------------
    # Aggregation for normal model parameters
    # --------------------------------------------------------
    def aggregate_params(self, client_weights, client_sizes):
        """
        FedAvg weighted by dataset size.

        client_weights: list of state_dict
        client_sizes: list of sizes (total samples each client)
        """
        if not client_weights or len(client_weights) == 0:
            return {}

        total = sum(client_sizes)
        if total == 0:
            return {}

        new_state = {}

        # reference
        keys = client_weights[0].keys()

        for key in keys:
            avg = sum(w[key] * (size / total)
                      for w, size in zip(client_weights, client_sizes))
            new_state[key] = avg

        return new_state

    # --------------------------------------------------------
    # Aggregation for architecture parameters α
    # --------------------------------------------------------
    def aggregate_alpha(self, client_alphas, client_sizes):
        """
        Weighted α update.

        global_alpha = (1-β)*global_alpha + β * Σ_i w_i * alpha_i
        """
        if not client_alphas or len(client_alphas) == 0:
            return  # No alpha to aggregate

        if len(client_alphas[0]) == 0:
            return  # Empty alpha list

        total = sum(client_sizes)
        if total == 0:
            return  # Avoid division by zero

        # Calculate weighted average from clients (on CPU usually)
        weighted_sum = [
            sum(a[j] * (client_sizes[i] / total)
                for i, a in enumerate(client_alphas))
            for j in range(len(client_alphas[0]))
        ]

        # update rule
        new_alphas = []
        for g, w in zip(self.global_alphas, weighted_sum):
            # [Fix] Ensure incoming weight 'w' is on the same device as global 'g' (e.g. MPS/CUDA)
            w = w.to(g.device)
            new = (1 - self.alpha_lr) * g + self.alpha_lr * w
            new_alphas.append(new.detach())

        # update server copy
        self.global_alphas = new_alphas

    # --------------------------------------------------------
    # Apply global α back to the model
    # --------------------------------------------------------
    def apply_alpha_to_model(self):
        if len(self.global_alphas) == 0:
            return  # No alpha to apply

        # Check if model has set_alpha method (new NAS model)
        if hasattr(self.global_model, 'set_alpha'):
            # New NAS model: use set_alpha method
            self.global_model.set_alpha()
        else:
            # Old NAS model: manually assign to MixedOp3D
            from model_swin3d.nas_ops import MixedOp3D
            ptr = 0
            for m in self.global_model.modules():
                if isinstance(m, MixedOp3D):
                    if ptr < len(self.global_alphas):
                        device = next(m.parameters()).device if len(list(m.parameters())) > 0 else self.device
                        m.alpha = self.global_alphas[ptr].clone().detach().to(device).requires_grad_(True)
                        ptr += 1

    # --------------------------------------------------------
    # One training round
    # --------------------------------------------------------
    def federated_round(self, selected_clients, client_results):
        """
        Args:
            selected_clients: list of client indices
            client_results: dict:
                {
                  cid: {
                      "weights": state_dict,
                      "alpha": list_of_alpha_vectors,
                      "size": dataset_size
                  }
                }
        """
        client_weights = []
        client_alphas = []
        client_sizes = []

        for cid in selected_clients:
            info = client_results[cid]
            client_weights.append(info["weights"])
            client_alphas.append(info["alpha"])
            client_sizes.append(info["size"])

        # --- aggregate model parameters ---
        new_state = self.aggregate_params(client_weights, client_sizes)
        self.set_global_weights(new_state)

        # --- aggregate architecture parameters ---
        self.aggregate_alpha(client_alphas, client_sizes)

        # apply α to model
        self.apply_alpha_to_model()

    # --------------------------------------------------------
    # Save / Load
    # --------------------------------------------------------
    def save(self, path):
        # Filter out alpha keys from model state_dict (alpha is managed separately)
        model_state = self.global_model.state_dict()
        filtered_state = {k: v for k, v in model_state.items() if not k.endswith('.alpha')}
        torch.save({
            "model": filtered_state,
            "alpha": self.global_alphas
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        # Filter out alpha keys when loading (alpha is managed separately)
        model_state = ckpt["model"]
        filtered_state = {k: v for k, v in model_state.items() if not k.endswith('.alpha')}
        self.global_model.load_state_dict(filtered_state, strict=False)
        self.global_alphas = ckpt["alpha"]
        self.apply_alpha_to_model()