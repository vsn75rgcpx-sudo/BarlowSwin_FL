"""
Enhanced Architecture Exporter for SwinUNet3D_NAS
--------------------------------------------------
This script exports the best architecture found by FedNAS, including:

Per-block:
    - ffn_choice (argmax)
    - attn_choice (argmax)
    - gate_value (sigmoid of gate_alpha)
    - gate_keep (if gate_value > 0.5)
    - block_index

Per-stage summary:
    - stage_id
    - num_blocks
    - blocks: list of block dicts above

Full model summary:
    - dims
    - depths
    - resolution
    - in_channels
    - num_classes
    - total_blocks
    - stage_summaries

JSON saved at: output_path
"""

import json
import torch
import numpy as np


def export_best_architecture(model, output_path="best_arch.json"):
    """
    Extract NAS architecture (ffn, attn, gate) from SwinUNet3D_NAS model.

    Args:
        model: SwinUNet3D_NAS instance (with nas=True, alpha already set)
        output_path: path to save JSON
    """
    assert hasattr(model, "_nas_blocks"), \
        "Model does not contain NAS blocks. Did you use SwinUNet3D_NAS (extended)?"
    assert hasattr(model, "alpha_mgr"), \
        "Model has no alpha manager. NAS must be enabled."

    print(f"[Export] Exporting best architecture → {output_path}")

    # force reassign alpha to blocks
    model.set_alpha()

    block_records = []
    block_index = 0

    # collect all candidate names (for readability)
    # ffn op names are in each block via blk.ffn.ops
    # attn candidates via blk.attn.atts
    # but we do not store the actual op modules in JSON, only index & description.

    # iterate through each block
    for blk in model._nas_blocks:
        # ----------------------------------
        # FFN choice
        # ----------------------------------
        if blk.ffn.alpha is not None:
            alpha_ffn = blk.ffn.alpha.detach().cpu().numpy()
            ffn_idx = int(np.argmax(alpha_ffn))
        else:
            ffn_idx = 0
            alpha_ffn = None

        # Extract FFN choice names for readability
        ffn_names = [op.__class__.__name__ for op in blk.ffn.ops]
        ffn_choice_name = ffn_names[ffn_idx]

        # ----------------------------------
        # Attention choice
        # ----------------------------------
        if blk.attn.alpha is not None:
            alpha_attn = blk.attn.alpha.detach().cpu().numpy()
            attn_idx = int(np.argmax(alpha_attn))
        else:
            attn_idx = 0
            alpha_attn = None

        attn_cands = []
        for a in blk.attn.atts:
            attn_cands.append({
                "window_size": a.window_size,  # tuple
                "num_heads": a.num_heads
            })
        attn_choice = attn_cands[attn_idx]

        # ----------------------------------
        # Gate value
        # ----------------------------------
        if blk.gate is not None:
            gate_alpha = blk.gate.detach().cpu().numpy().item()
            gate_value = float(torch.sigmoid(blk.gate).item())
        else:
            gate_alpha = 0.0
            gate_value = 1.0

        gate_keep = bool(gate_value > 0.5)

        # ----------------------------------
        # Save block record
        # ----------------------------------
        block_records.append({
            "block_index": block_index,
            "ffn_idx": ffn_idx,
            "ffn_choice": ffn_choice_name,
            "attn_idx": attn_idx,
            "attn_choice": attn_choice,
            "gate_alpha": float(gate_alpha),
            "gate_value": gate_value,
            "gate_keep": gate_keep,
            "resolution": [blk.D, blk.H, blk.W],
            "shift_size": list(blk.shift_size)
        })

        block_index += 1

    # -------------------------
    # Stage-level summary
    # -------------------------
    stage_summaries = []
    idx = 0
    for stage_id, stage in enumerate(model.stages):
        depth = len(stage.blocks)
        stage_blocks = block_records[idx:idx+depth]
        idx += depth

        stage_summaries.append({
            "stage_id": stage_id,
            "num_blocks": depth,
            "blocks": stage_blocks
        })

    # -------------------------
    # Final JSON structure
    # -------------------------
    final_json = {
        "model": {
            "in_channels": model.in_channels,
            "num_classes": model.num_classes,
            "dims": list(model.dims),
            "depths": list(model.depths),
            "resolution": list(model.resolution),
            "total_blocks": len(model._nas_blocks),
        },
        "stages": stage_summaries,
        "description": "Extended NAS architecture export with FFN/Attn/Gate per block."
    }

    # Save JSON
    with open(output_path, "w") as f:
        json.dump(final_json, f, indent=4)

    print(f"[Export] Done. JSON saved → {output_path}")
    
    # Print architecture summary
    print_architecture_summary(final_json)
    
    return final_json


def print_architecture_summary(arch_json):
    """
    Print a human-readable summary of the searched architecture.
    Shows per-block choices: ffn_choice, attn_choice, gate_value, gate_keep
    """
    print("\n" + "="*80)
    print("ARCHITECTURE SEARCH RESULTS")
    print("="*80)
    
    stages = arch_json["stages"]
    model_info = arch_json["model"]
    
    print(f"\nModel Configuration:")
    print(f"  Input channels: {model_info['in_channels']}")
    print(f"  Output classes: {model_info['num_classes']}")
    print(f"  Dims: {model_info['dims']}")
    print(f"  Depths: {model_info['depths']}")
    print(f"  Resolution: {model_info['resolution']}")
    print(f"  Total blocks: {model_info['total_blocks']}")
    
    print(f"\n{'='*80}")
    print("Per-Block Search Results:")
    print(f"{'='*80}")
    
    for stage in stages:
        stage_id = stage["stage_id"]
        num_blocks = stage["num_blocks"]
        blocks = stage["blocks"]
        
        print(f"\n--- Stage {stage_id} ({num_blocks} blocks) ---")
        print(f"{'Block':<8} {'FFN Choice':<20} {'Attn (win,heads)':<25} {'Gate Value':<12} {'Keep':<8}")
        print("-" * 80)
        
        for blk in blocks:
            block_idx = blk["block_index"]
            ffn_choice = blk["ffn_choice"]
            attn = blk["attn_choice"]
            attn_str = f"({attn['window_size']}, {attn['num_heads']})"
            gate_val = blk["gate_value"]
            gate_keep = "✓" if blk["gate_keep"] else "✗"
            
            print(f"{block_idx:<8} {ffn_choice:<20} {attn_str:<25} {gate_val:<12.4f} {gate_keep:<8}")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("Summary Statistics:")
    print(f"{'='*80}")
    
    all_blocks = []
    for stage in stages:
        all_blocks.extend(stage["blocks"])
    
    # FFN choices distribution
    ffn_counts = {}
    for blk in all_blocks:
        ffn = blk["ffn_choice"]
        ffn_counts[ffn] = ffn_counts.get(ffn, 0) + 1
    
    print(f"\nFFN Choices Distribution:")
    for ffn, count in sorted(ffn_counts.items()):
        pct = count / len(all_blocks) * 100
        print(f"  {ffn:<20}: {count:>3} blocks ({pct:>5.1f}%)")
    
    # Attention choices distribution
    attn_counts = {}
    for blk in all_blocks:
        attn = blk["attn_choice"]
        attn_key = f"win{attn['window_size']}_h{attn['num_heads']}"
        attn_counts[attn_key] = attn_counts.get(attn_key, 0) + 1
    
    print(f"\nAttention Choices Distribution:")
    for attn, count in sorted(attn_counts.items()):
        pct = count / len(all_blocks) * 100
        print(f"  {attn:<20}: {count:>3} blocks ({pct:>5.1f}%)")
    
    # Gate statistics
    kept_blocks = sum(1 for blk in all_blocks if blk["gate_keep"])
    avg_gate_val = sum(blk["gate_value"] for blk in all_blocks) / len(all_blocks)
    
    print(f"\nGate Statistics:")
    print(f"  Blocks kept: {kept_blocks}/{len(all_blocks)} ({kept_blocks/len(all_blocks)*100:.1f}%)")
    print(f"  Average gate value: {avg_gate_val:.4f}")
    
    print(f"\n{'='*80}\n")


# ------------------------------------------------------------
# Legacy function for backward compatibility
# ------------------------------------------------------------
def export_best_model(model, json_path="best_architecture.json"):
    """
    Legacy function for backward compatibility.
    Extracts best FFN operations from MixedOp3D and exports to JSON.
    Returns a fixed model (if model has _nas_blocks) or just exports JSON.
    """
    import copy
    from .nas_ops import MixedOp3D
    
    # Check if this is the old-style model (with MixedOp3D in blocks)
    # or new-style model (with _nas_blocks)
    if hasattr(model, "_nas_blocks"):
        # New style: use export_best_architecture
        arch_json = export_best_architecture(model, json_path)
        # For now, return the model itself (caller will handle conversion)
        return model
    else:
        # Old style: extract from MixedOp3D modules
        best_ops = []
        for m in model.modules():
            if isinstance(m, MixedOp3D):
                if m.alpha is None:
                    idx = 0
                else:
                    alpha = m.alpha.detach().cpu()
                    idx = torch.argmax(alpha).item()
                best_ops.append({
                    "index": idx,
                    "op": m.ops[idx].__class__.__name__
                })
        
        # Export JSON
        arch = {
            "num_mixed_ops": len(best_ops),
            "ops": best_ops
        }
        with open(json_path, "w") as f:
            json.dump(arch, f, indent=4)
        print(f"[OK] Architecture exported to {json_path}")
        
        # Return model (caller will handle state_dict conversion)
        return model
