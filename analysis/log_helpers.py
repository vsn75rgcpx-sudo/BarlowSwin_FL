"""
analysis/log_helpers.py

简单日志工具，用于在联邦训练每轮结束时写入 training_log.json。

每轮 log 格式（示例）：
{
  "rounds": [
    {
      "round": 1,
      "stage": "search",                # 或 "retrain"
      "timestamp": 1690000000,
      "server_checkpoint": "fednas_round_1.pth",
      "train_loss": 0.123,              # 可选
      "val_loss": 0.11,                 # 可选
      "val_dice": 0.67,                 # 可选
      "comm_bytes": 1234567,            # 单轮通信量估算（bytes）
      "alpha_summaries": [ ... ]        # 可选：每个 alpha 向量的 summary，比如 top_softmax, entropy
    },
    ...
  ]
}
"""

import json
import os
import time
import math
import torch
import numpy as np

LOG_FILE = "training_log.json"

def safe_load_log(path=LOG_FILE):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    else:
        return {"rounds": []}

def safe_save_log(log, path=LOG_FILE):
    with open(path, "w") as f:
        json.dump(log, f, indent=2)

def tensor_entropy(t):
    """entropy of softmaxed vector (numpy)"""
    p = np.exp(t - np.max(t))
    p = p / (p.sum() + 1e-12)
    ent = -np.sum(p * np.log(p + 1e-12))
    return float(ent)

def alpha_summary(alpha_tensor):
    """
    alpha_tensor: 1D numpy array (raw alphas)
    return: dict with softmax top value, argmax idx, entropy, softmax vector (small)
    """
    a = np.array(alpha_tensor).astype(float)
    ex = np.exp(a - np.max(a))
    soft = ex / (ex.sum() + 1e-12)
    top_idx = int(np.argmax(soft))
    top_val = float(soft[top_idx])
    ent = tensor_entropy(a)
    return {"argmax": top_idx, "top": top_val, "entropy": ent, "soft": soft.tolist()}

def estimate_checkpoint_bytes(checkpoint_path):
    """File size of checkpoint in bytes (approx communication cost per client for upload)."""
    try:
        return os.path.getsize(checkpoint_path)
    except Exception:
        return None

def append_round_log(
    round_idx,
    stage="search",
    ckpt_path=None,
    train_loss=None,
    val_loss=None,
    val_dice=None,
    alpha_list=None,
    comm_bytes=None,
    expected_flops=None,
    temperature=None,
    log_path=LOG_FILE
):
    """
    Append or update an entry for a round in training_log.json.

    alpha_list: list of torch tensors or numpy arrays for each MixedOp alpha (raw values)
    comm_bytes: optional explicit communication bytes for the round (int)
    """
    log = safe_load_log(log_path)
    entry = {
        "round": int(round_idx),
        "stage": stage,
        "timestamp": int(time.time()),
        "server_checkpoint": ckpt_path,
    }
    if train_loss is not None:
        entry["train_loss"] = float(train_loss)
    if val_loss is not None:
        entry["val_loss"] = float(val_loss)
    if val_dice is not None:
        entry["val_dice"] = float(val_dice)
    if comm_bytes is None and ckpt_path is not None:
        entry["comm_bytes"] = estimate_checkpoint_bytes(ckpt_path)
    else:
        entry["comm_bytes"] = comm_bytes

    if alpha_list is not None:
        sums = []
        for a in alpha_list:
            if isinstance(a, torch.Tensor):
                arr = a.detach().cpu().numpy()
            else:
                arr = np.array(a)
            sums.append(alpha_summary(arr))
        entry["alpha_summaries"] = sums
    
    if expected_flops is not None:
        entry["expected_flops"] = float(expected_flops)
    if temperature is not None:
        entry["temperature"] = float(temperature)

    # replace if round exists, else append
    found = False
    for i, e in enumerate(log["rounds"]):
        if e.get("round") == int(round_idx) and e.get("stage") == stage:
            log["rounds"][i] = entry
            found = True
            break
    if not found:
        log["rounds"].append(entry)

    # sort by round
    log["rounds"] = sorted(log["rounds"], key=lambda x: (x["stage"], x["round"]))
    safe_save_log(log, log_path)
    return entry
