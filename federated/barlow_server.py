"""
federated/barlow_server.py
--------------------------
Federated server for Barlow Twins pretraining.
Aggregates encoder weights from clients via FedAvg.

Modified:
 - Fixed device mismatch error in aggregate() by ensuring client params move to server device.
"""

import copy
from typing import Any, Dict, List

import torch


class BarlowServer:
    def __init__(self, model_encoder: torch.nn.Module, device: str, args, logger=None):
        self.device = device
        self.global_encoder = model_encoder.to(device)
        self.args = args
        self.logger = logger

    def aggregate(self, client_dicts: List[Dict[str, Any]], sizes: List[int]) -> Dict[str, Any]:
        """
        FedAvg aggregation for encoder weights.

        Args:
            client_dicts: list of dicts each {'encoder': state_dict}
            sizes: list of ints (sample counts per client)
        """
        total = sum(sizes)
        # Initialize new_state on the same device as global_encoder (e.g. CUDA)
        new_state = copy.deepcopy(self.global_encoder.state_dict())

        # Separate floating point and integer parameters
        float_keys = []
        int_keys = []
        for k in new_state.keys():
            if new_state[k].dtype.is_floating_point:
                float_keys.append(k)
                new_state[k] = torch.zeros_like(new_state[k])
            else:
                # For non-floating point (Long, Bool, etc.), keep original value
                int_keys.append(k)

        # Aggregate floating point parameters
        for client_state, sz in zip(client_dicts, sizes):
            st = client_state["encoder"]
            weight = sz / total
            for k in float_keys:
                if k in st:
                    # [Fix] Ensure client_param is on the same device as accumulator new_state[k]
                    target_device = new_state[k].device
                    client_param = st[k].to(target_device)

                    # Ensure it's floating point
                    if not client_param.dtype.is_floating_point:
                        client_param = client_param.float()
                    # Match dtype of new_state
                    if client_param.dtype != new_state[k].dtype:
                        client_param = client_param.to(new_state[k].dtype)

                    new_state[k] += client_param * weight

        # For integer parameters (buffers, etc.), we keep the global model's original values

        # copy back
        self.global_encoder.load_state_dict(new_state)
        return self.global_encoder.state_dict()

    def save(self, path: str) -> None:
        torch.save(self.global_encoder.state_dict(), path)
        if self.logger:
            self.logger.info(f"[BarlowServer] Saved encoder to {path}")