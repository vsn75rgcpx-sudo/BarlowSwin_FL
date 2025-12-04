"""
analysis/plot_metrics.py

Plotting utilities for:
 - alpha evolution (softmax per op over rounds)
 - alpha entropy / top-1 confidence
 - loss/dice curves
 - communication bytes per round

Input:
 - training_log.json created by log_helpers.append_round_log
 - or a list of server checkpoint files (path pattern) to extract alpha vectors

Usage examples:
    python -m analysis.plot_metrics --log training_log.json
    python -m analysis.plot_metrics --ckpt_pattern "fednas_round_*.pth"
"""

import os
import json
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from typing import List

# --------------------------
# Helpers
# --------------------------
def load_log(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r") as f:
        return json.load(f)

def extract_alphas_from_checkpoint(ckpt_path):
    """
    Load server checkpoint saved by FederatedServer.save:
    {'model': state_dict, 'alpha': [tensor1, tensor2, ...]}
    Returns list of numpy arrays for each alpha.
    """
    d = torch.load(ckpt_path, map_location="cpu")
    if "alpha" in d:
        alphas = d["alpha"]
        alphas_np = [a.detach().cpu().numpy() if torch.is_tensor(a) else np.array(a) for a in alphas]
        return alphas_np
    else:
        # try to load from model if arch parameters are included as tensors
        # fallback: return []
        return []

def collect_alpha_series_from_ckpt_pattern(pattern):
    """
    pattern: glob pattern to ordered checkpoint files
    returns: list of alpha lists per round: [ [alpha0_round1, alpha1_round1,...], [..round2..], ... ]
    also returns list of round filenames (ordered)
    """
    files = sorted(glob.glob(pattern))
    series = []
    names = []
    for f in files:
        alphas = extract_alphas_from_checkpoint(f)
        if alphas:
            series.append(alphas)
            names.append(f)
    return series, names

def softmax(x):
    ex = np.exp(x - np.max(x))
    return ex / (ex.sum() + 1e-12)

def compute_alpha_softmax_series(alpha_series):
    """
    alpha_series: list (rounds) of list (per-mixed-op) of numpy arrays
    returns:
      - per-op per-round softmax vectors: shape (num_ops, rounds, max_ops) with padded zeros
    """
    rounds = len(alpha_series)
    if rounds == 0:
        return None, None, None

    num_ops = len(alpha_series[0])
    max_k = max([len(a) for a in alpha_series[0]])
    # But ops counts could vary: better compute per op individually
    per_op = []
    for op_idx in range(num_ops):
        op_rounds = []
        for r in range(rounds):
            vec = np.array(alpha_series[r][op_idx])
            s = softmax(vec)
            op_rounds.append(s)
        per_op.append(op_rounds)
    return per_op  # list of num_ops elements; each is list of rounds of softmax arrays

# --------------------------
# Plotting functions
# --------------------------
def plot_alpha_topk(log, out_dir="plots", topk=3, show=False):
    """
    Plot top-k softmax values (confidence) and entropy for each round.
    Uses training_log.json if provided.
    """
    os.makedirs(out_dir, exist_ok=True)

    rounds = []
    top1_conf = []
    entropies = []
    for e in log["rounds"]:
        rounds.append(e["round"])
        if "alpha_summaries" in e and len(e["alpha_summaries"])>0:
            # aggregate across mixed ops: mean top1 and mean entropy
            tops = [a["top"] for a in e["alpha_summaries"]]
            ents = [a["entropy"] for a in e["alpha_summaries"]]
            top1_conf.append(float(np.mean(tops)))
            entropies.append(float(np.mean(ents)))
        else:
            top1_conf.append(None)
            entropies.append(None)

    # top1_conf plot
    plt.figure(figsize=(8,4))
    plt.plot(rounds, top1_conf, marker='o')
    plt.title("Mean alpha top-1 softmax (across MixedOps)")
    plt.xlabel("Round")
    plt.ylabel("Mean top-1 confidence")
    plt.grid(True)
    p = os.path.join(out_dir, "alpha_mean_top1.png")
    plt.savefig(p)
    if show: plt.show()
    plt.close()

    # entropy plot
    plt.figure(figsize=(8,4))
    plt.plot(rounds, entropies, marker='o')
    plt.title("Mean alpha entropy (across MixedOps)")
    plt.xlabel("Round")
    plt.ylabel("Entropy")
    plt.grid(True)
    p = os.path.join(out_dir, "alpha_mean_entropy.png")
    plt.savefig(p)
    if show: plt.show()
    plt.close()

    print("[plot_alpha_topk] saved to", out_dir)

def plot_alpha_heatmap_from_series(alpha_series, out_dir="plots", show=False, max_ops_to_show=6):
    """
    alpha_series: list of rounds, each is list of alpha arrays
    We will plot heatmap for first N mixed-ops: softmax weights (rows=rounds, cols=candidates)
    """
    os.makedirs(out_dir, exist_ok=True)
    if len(alpha_series) == 0:
        print("No alpha series")
        return

    num_mixed = len(alpha_series[0])
    rounds = len(alpha_series)

    for m in range(min(num_mixed, max_ops_to_show)):
        # build matrix rounds x candidates
        mat = []
        max_k = 0
        for r in range(rounds):
            a = np.array(alpha_series[r][m]).astype(float)
            s = softmax(a)
            mat.append(s)
            max_k = max(max_k, len(s))
        # pad
        padded = np.zeros((rounds, max_k), dtype=float)
        for r in range(rounds):
            arr = np.array(alpha_series[r][m]).astype(float)
            s = softmax(arr)
            padded[r, :len(s)] = s

        plt.figure(figsize=(8, max(2, rounds*0.25)))
        plt.imshow(padded, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.ylabel("Round")
        plt.xlabel("Candidate op index")
        plt.title(f"Alpha softmax heatmap - MixedOp {m}")
        p = os.path.join(out_dir, f"alpha_heatmap_m{m}.png")
        plt.savefig(p)
        if show: plt.show()
        plt.close()
    print("[plot_alpha_heatmap_from_series] saved to", out_dir)

def plot_loss_and_dice(log, out_dir="plots", show=False):
    os.makedirs(out_dir, exist_ok=True)
    
    # Separate data by stage
    search_data = {"rounds": [], "train_loss": [], "val_loss": [], "val_dice": []}
    retrain_data = {"rounds": [], "train_loss": [], "val_loss": [], "val_dice": []}
    
    for e in log["rounds"]:
        stage = e.get("stage", "search")
        round_num = int(e["round"])
        
        if stage == "search":
            search_data["rounds"].append(round_num)
            search_data["train_loss"].append(e.get("train_loss", None))
            search_data["val_loss"].append(e.get("val_loss", None))
            search_data["val_dice"].append(e.get("val_dice", None))
        elif stage == "retrain":
            retrain_data["rounds"].append(round_num)
            retrain_data["train_loss"].append(e.get("train_loss", None))
            retrain_data["val_loss"].append(e.get("val_loss", None))
            retrain_data["val_dice"].append(e.get("val_dice", None))

    # train/val loss - plot with stage separation
    plt.figure(figsize=(10, 5))
    has_data = False
    
    # Plot search stage
    if search_data["rounds"]:
        if any([v is not None for v in search_data["train_loss"]]):
            plt.plot(search_data["rounds"], search_data["train_loss"], 
                    label="train_loss (search)", marker='o', linestyle='-', color='blue')
            has_data = True
        if any([v is not None for v in search_data["val_loss"]]):
            plt.plot(search_data["rounds"], search_data["val_loss"], 
                    label="val_loss (search)", marker='s', linestyle='-', color='red')
            has_data = True
    
    # Plot retrain stage (offset x-axis to continue after search)
    if retrain_data["rounds"]:
        max_search_round = max(search_data["rounds"]) if search_data["rounds"] else 0
        retrain_rounds_offset = [r + max_search_round for r in retrain_data["rounds"]]
        
        if any([v is not None for v in retrain_data["train_loss"]]):
            plt.plot(retrain_rounds_offset, retrain_data["train_loss"], 
                    label="train_loss (retrain)", marker='o', linestyle='--', color='cyan')
            has_data = True
        if any([v is not None for v in retrain_data["val_loss"]]):
            plt.plot(retrain_rounds_offset, retrain_data["val_loss"], 
                    label="val_loss (retrain)", marker='s', linestyle='--', color='orange')
            has_data = True
        
        # Add vertical line to separate stages
        if max_search_round > 0:
            plt.axvline(x=max_search_round + 0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)
            plt.text(max_search_round + 0.5, plt.ylim()[1] * 0.95, 'Search→Retrain', 
                    rotation=90, verticalalignment='top', fontsize=8, color='gray')
    
    plt.xlabel("Round")
    plt.ylabel("Loss")
    if has_data:
        plt.legend()
    plt.grid(True)
    plt.title("Loss Curves (Search & Retrain)")
    p = os.path.join(out_dir, "loss_curves.png")
    plt.savefig(p)
    if show: plt.show()
    plt.close()

    # dice - plot with stage separation
    has_dice_data = False
    if any([v is not None for v in search_data["val_dice"]]) or any([v is not None for v in retrain_data["val_dice"]]):
        plt.figure(figsize=(10, 5))
        
        # Plot search stage
        if search_data["rounds"] and any([v is not None for v in search_data["val_dice"]]):
            plt.plot(search_data["rounds"], search_data["val_dice"], 
                    label="val_dice (search)", marker='o', linestyle='-', color='green')
            has_dice_data = True
        
        # Plot retrain stage
        if retrain_data["rounds"] and any([v is not None for v in retrain_data["val_dice"]]):
            max_search_round = max(search_data["rounds"]) if search_data["rounds"] else 0
            retrain_rounds_offset = [r + max_search_round for r in retrain_data["rounds"]]
            
            plt.plot(retrain_rounds_offset, retrain_data["val_dice"], 
                    label="val_dice (retrain)", marker='s', linestyle='--', color='purple')
            has_dice_data = True
            
            # Add vertical line to separate stages
            if max_search_round > 0:
                plt.axvline(x=max_search_round + 0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)
                plt.text(max_search_round + 0.5, plt.ylim()[1] * 0.95, 'Search→Retrain', 
                        rotation=90, verticalalignment='top', fontsize=8, color='gray')
        
        if has_dice_data:
            plt.xlabel("Round")
            plt.ylabel("Dice")
            plt.legend()
            plt.grid(True)
            plt.title("Dice Score Curves (Search & Retrain)")
            p = os.path.join(out_dir, "dice_curve.png")
            plt.savefig(p)
            if show: plt.show()
            plt.close()

    print("[plot_loss_and_dice] saved to", out_dir)

def plot_flops_and_temperature(log, out_dir="plots", show=False):
    """Plot expected FLOPs and temperature curves"""
    os.makedirs(out_dir, exist_ok=True)
    
    # Separate data by stage (FLOPs and temperature are only in search stage)
    search_data = {"rounds": [], "expected_flops": [], "temperature": []}
    
    for e in log["rounds"]:
        stage = e.get("stage", "search")
        if stage == "search":
            search_data["rounds"].append(int(e["round"]))
            search_data["expected_flops"].append(e.get("expected_flops", None))
            search_data["temperature"].append(e.get("temperature", None))
    
    # Expected FLOPs plot (only for search stage)
    if search_data["rounds"] and any([v is not None for v in search_data["expected_flops"]]):
        plt.figure(figsize=(8,4))
        plt.plot(search_data["rounds"], search_data["expected_flops"], 
                label="Expected FLOPs", marker='o', color='green')
        plt.xlabel("Round (Search Stage)")
        plt.ylabel("Expected FLOPs")
        plt.title("Expected FLOPs Evolution (Search Stage)")
        plt.legend()
        plt.grid(True)
        p = os.path.join(out_dir, "flops_curve.png")
        plt.savefig(p)
        if show: plt.show()
        plt.close()
    
    # Temperature plot (only for search stage)
    if search_data["rounds"] and any([v is not None for v in search_data["temperature"]]):
        plt.figure(figsize=(8,4))
        plt.plot(search_data["rounds"], search_data["temperature"], 
                label="Temperature", marker='s', color='orange')
        plt.xlabel("Round (Search Stage)")
        plt.ylabel("Temperature")
        plt.title("Gumbel-Softmax Temperature Schedule (Search Stage)")
        plt.legend()
        plt.grid(True)
        p = os.path.join(out_dir, "temperature_curve.png")
        plt.savefig(p)
        if show: plt.show()
        plt.close()
    
    print("[plot_flops_and_temperature] saved to", out_dir)

def plot_comm_bytes(log, out_dir="plots", show=False):
    os.makedirs(out_dir, exist_ok=True)
    
    # Separate data by stage
    search_data = {"rounds": [], "comm": []}
    retrain_data = {"rounds": [], "comm": []}
    
    for e in log["rounds"]:
        stage = e.get("stage", "search")
        round_num = int(e["round"])
        comm_bytes = e.get("comm_bytes", None)
        
        if stage == "search":
            search_data["rounds"].append(round_num)
            search_data["comm"].append(comm_bytes)
        elif stage == "retrain":
            retrain_data["rounds"].append(round_num)
            retrain_data["comm"].append(comm_bytes)
    
    plt.figure(figsize=(10, 5))
    has_data = False
    
    # Plot search stage
    if search_data["rounds"] and any([v is not None for v in search_data["comm"]]):
        plt.plot(search_data["rounds"], search_data["comm"], 
                label="Comm bytes (search)", marker='o', linestyle='-', color='blue')
        has_data = True
    
    # Plot retrain stage (offset x-axis)
    if retrain_data["rounds"] and any([v is not None for v in retrain_data["comm"]]):
        max_search_round = max(search_data["rounds"]) if search_data["rounds"] else 0
        retrain_rounds_offset = [r + max_search_round for r in retrain_data["rounds"]]
        
        plt.plot(retrain_rounds_offset, retrain_data["comm"], 
                label="Comm bytes (retrain)", marker='s', linestyle='--', color='red')
        has_data = True
        
        # Add vertical line to separate stages
        if max_search_round > 0:
            plt.axvline(x=max_search_round + 0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)
            plt.text(max_search_round + 0.5, plt.ylim()[1] * 0.95, 'Search→Retrain', 
                    rotation=90, verticalalignment='top', fontsize=8, color='gray')
    
    if has_data:
        plt.xlabel("Round")
        plt.ylabel("Comm bytes")
        plt.yscale("log")
        plt.legend()
        plt.grid(True)
        plt.title("Communication Bytes (Search & Retrain)")
        p = os.path.join(out_dir, "comm_bytes.png")
        plt.savefig(p)
        if show: plt.show()
        plt.close()
    
    print("[plot_comm_bytes] saved to", out_dir)


# --------------------------
# CLI
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, default="training_log.json", help="training log JSON")
    parser.add_argument("--ckpt_pattern", type=str, default=None, help="pattern of server ckpt to load alpha series")
    parser.add_argument("--out_dir", type=str, default="plots", help="where to save plots")
    parser.add_argument("--show", action="store_true", help="show plots interactively")
    parser.add_argument("--max_ops_show", type=int, default=6, help="how many MixedOps to show heatmaps for")
    args = parser.parse_args()

    alpha_series = []
    names = []

    if args.ckpt_pattern:
        alpha_series, names = collect_alpha_series_from_ckpt_pattern(args.ckpt_pattern)
    elif os.path.exists(args.log):
        log = load_log(args.log)
        # try to reconstruct alpha_series from log
        # log entries may include "alpha_summaries" but not full softmax vectors; prefer ckpt pattern
        alpha_series = []
        for e in log["rounds"]:
            if "alpha_summaries" in e:
                # alpha_summaries contain 'soft' vectors if logged; otherwise not available
                al = []
                for a in e["alpha_summaries"]:
                    if "soft" in a:
                        al.append(np.array(a["soft"]))
                    else:
                        al.append(np.zeros(1))
                alpha_series.append(al)
        # also keep log for other plots
        log_data = log
    else:
        raise FileNotFoundError("Provide either --log training_log.json or --ckpt_pattern 'fednas_round_*.pth'")

    # If log not loaded from file, try load it for loss/dice/comm plots
    if os.path.exists(args.log):
        log_data = load_log(args.log)
        plot_loss_and_dice(log_data, out_dir=args.out_dir, show=args.show)
        plot_comm_bytes(log_data, out_dir=args.out_dir, show=args.show)
        plot_alpha_topk(log_data, out_dir=args.out_dir, show=args.show)
        plot_flops_and_temperature(log_data, out_dir=args.out_dir, show=args.show)
    else:
        print("[plot_metrics] no training_log.json found; will only plot alpha series & comm estimates if available")

    if alpha_series and len(alpha_series)>0:
        plot_alpha_heatmap_from_series(alpha_series, out_dir=args.out_dir, show=args.show, max_ops_to_show=args.max_ops_show)

    print("All plots saved in", args.out_dir)

if __name__ == "__main__":
    main()
