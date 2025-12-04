"""
swin3d_unet_fixed.py
--------------------
Final fixed-architecture SwinUNet3D for retraining/inference.

Reads best_arch.json and constructs:
 - PatchEmbed3D
 - Swin3D blocks (attention same as NAS)
 - FFN replaced by the chosen operator (Conv/Depthwise/1x1/Identity)
 - Downsampling + Upsampling
"""

import torch
import torch.nn as nn
import json
from .layers import PatchMerging3D, PatchExpanding3D, WindowAttention3D, window_partition, window_reverse
from .swin3d_block import LayerNormChannel, DropPath
from .nas_ops import build_op_3d


# ------------------------------------------------------------
# 1. Fixed FFN Block (replaces MixedOp3D)
# ------------------------------------------------------------
class FixedFFN3D(nn.Module):
    """
    Replaces MixedOp3D in FFN.
    Uses exactly ONE chosen operator.
    """
    def __init__(self, chosen_op):
        super().__init__()
        self.op = chosen_op

    def forward(self, x):
        return self.op(x)


# ------------------------------------------------------------
# 2. Swin3DBlock_Fixed
# ------------------------------------------------------------
class Swin3DBlock_Fixed(nn.Module):

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size,
        shift_size,
        ffn_op,
        drop_path
    ):
        super().__init__()

        self.dim = dim
        self.D, self.H, self.W = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = LayerNormChannel(dim)
        self.norm2 = LayerNormChannel(dim)

        # Window Attention remains unchanged
        self.attn = WindowAttention3D(dim, window_size, num_heads)

        self.drop_path = DropPath(drop_path)

        # Fixed FFN (the chosen op)
        self.ffn = FixedFFN3D(ffn_op)

        # cached mask
        self.register_buffer("attn_mask", None, persistent=False)

    def get_mask(self, x):
        if self.attn_mask is not None:
            return self.attn_mask

        from .layers import compute_attn_mask
        mask = compute_attn_mask(
            self.D, self.H, self.W,
            self.window_size, self.shift_size,
            x.device
        )
        self.attn_mask = mask
        return mask

    def forward(self, x):
        B, C, D, H, W = x.shape

        shortcut = x
        x = self.norm1(x)

        sd, sh, sw = self.shift_size
        if sd or sh or sw:
            shifted_x = torch.roll(x, shifts=(-sd,-sh,-sw), dims=(2,3,4))
        else:
            shifted_x = x

        # partition windows
        wd, wh, ww = self.window_size
        N = wd * wh * ww

        windows = window_partition(shifted_x, self.window_size)
        windows = windows.view(-1, N, C)

        mask = self.get_mask(x)
        attn_windows = self.attn(windows, mask)
        attn_windows = attn_windows.view(-1, wd, wh, ww, C)

        shifted_x = window_reverse(attn_windows, self.window_size, B, D, H, W)

        if sd or sh or sw:
            x_attn = torch.roll(shifted_x, shifts=(sd,sh,sw), dims=(2,3,4))
        else:
            x_attn = shifted_x

        x = shortcut + self.drop_path(x_attn)

        # ---- FFN ----
        shortcut2 = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = shortcut2 + self.drop_path(x)
        return x


# ------------------------------------------------------------
# 3. SwinStage_Fixed
# ------------------------------------------------------------
class SwinStage_Fixed(nn.Module):

    def __init__(
        self,
        dim,
        resolution,
        depth,
        num_heads,
        window_size,
        drop_path_list,
        ffn_ops,        # list of chosen FFN operators
        downsample=True
    ):
        super().__init__()

        D, H, W = resolution
        self.blocks = nn.ModuleList()

        for i in range(depth):
            shift = (
                0 if i % 2 == 0 else window_size[0]//2,
                0 if i % 2 == 0 else window_size[1]//2,
                0 if i % 2 == 0 else window_size[2]//2,
            )

            blk = Swin3DBlock_Fixed(
                dim=dim,
                input_resolution=(D,H,W),
                num_heads=num_heads,
                window_size=window_size,
                shift_size=shift,
                ffn_op=ffn_ops[i],
                drop_path=drop_path_list[i]
            )
            self.blocks.append(blk)

        self.downsample = PatchMerging3D(dim) if downsample else None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)

        if self.downsample:
            x = self.downsample(x)
        return x


# ------------------------------------------------------------
# 4. PatchEmbed3D
# ------------------------------------------------------------
class PatchEmbed3D(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size=2):
        super().__init__()
        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        return self.proj(x)


# ------------------------------------------------------------
# 5. Final model: SwinUNet3D_Fixed
# ------------------------------------------------------------
class SwinUNet3D_Fixed(nn.Module):

    def __init__(
        self,
        in_channels,
        num_classes,
        dims,
        depths,
        arch_json,
        window_size=(2,7,7)
    ):
        super().__init__()

        with open(arch_json, "r") as f:
            arch = json.load(f)

        # ---- map class names to build_op_3d names ----
        def map_op_name(class_name):
            """Convert class name to build_op_3d expected name"""
            name_lower = class_name.lower()
            if "conv3x3x3" in name_lower or name_lower == "conv3":
                return "conv3"
            elif "conv1x1x1" in name_lower or name_lower == "conv1":
                return "conv1"
            elif "conv5x5x5" in name_lower or name_lower == "conv5":
                return "conv5"
            elif "depthwise" in name_lower or "dwconv" in name_lower or "depthwiseseparable" in name_lower:
                return "dw"  # build_ffn_op accepts both "dw" and "dwconv"
            elif "identity" in name_lower:
                return "identity"
            else:
                raise ValueError(f"Unknown operator class name: {class_name}")

        # ---- Extract FFN choices from JSON ----
        # Support both old format (arch["ops"]) and new format (arch["stages"])
        if "stages" in arch:
            # New format: extract from stages
            # Note: We use all blocks regardless of gate_keep, as fixed model structure is fixed
            chosen_ops = []
            for stage in sorted(arch["stages"], key=lambda x: x["stage_id"]):
                for blk in stage["blocks"]:
                    chosen_ops.append({
                        "op": blk["ffn_choice"],
                        "index": blk["ffn_idx"]
                    })
        elif "ops" in arch:
            # Old format: direct ops list
            chosen_ops = arch["ops"]
        else:
            raise ValueError(f"Invalid architecture JSON format. Expected 'ops' or 'stages' key.")

        # ---- map to real operators ----
        self.chosen_operators = []
        ptr = 0
        for stage_id, depth in enumerate(depths):
            stage_ops = []
            for i in range(depth):
                if ptr < len(chosen_ops):
                    op_name = chosen_ops[ptr]["op"]
                    mapped_name = map_op_name(op_name)
                    in_c = dims[stage_id]
                    out_c = dims[stage_id]
                    op = build_op_3d(mapped_name, in_c, out_c)
                    stage_ops.append(op)
                    ptr += 1
                else:
                    # Fallback: use conv3 if not enough ops
                    op = build_op_3d("conv3", dims[stage_id], dims[stage_id])
                    stage_ops.append(op)
            self.chosen_operators.append(stage_ops)

        # ------------------------------
        # Patch Embedding
        # ------------------------------
        self.patch_embed = PatchEmbed3D(in_channels, dims[0], patch_size=2)

        # dynamic resolution computed at forward
        self.depths = depths
        self.dims = dims
        self.window_size = window_size

        # ------------------------------
        # Build fixed encoder (4 stages)
        # ------------------------------
        self.stages = nn.ModuleList()
        total_blocks = sum(depths)
        dpr = torch.linspace(0, 0.1, total_blocks).tolist()
        ptr = 0

        for stage_id in range(4):
            dim = dims[stage_id]
            depth = depths[stage_id]
            heads = dim // 32

            drop_list = dpr[ptr:ptr+depth]
            ptr += depth

            # placeholder resolution, will fill in forward
            stage = SwinStage_Fixed(
                dim=dim,
                resolution=(1,1,1),
                depth=depth,
                num_heads=heads,
                window_size=window_size,
                drop_path_list=drop_list,
                ffn_ops=self.chosen_operators[stage_id],
                downsample=(stage_id < 3)
            )
            self.stages.append(stage)

        # ------------------------------
        # Decoder path
        # ------------------------------
        self.up_blocks = nn.ModuleList()
        for i in range(3, 0, -1):
            up = nn.Sequential(
                PatchExpanding3D(dims[i], dims[i-1]),
                nn.Conv3d(dims[i-1], dims[i-1], kernel_size=3, padding=1),
                nn.GELU()
            )
            self.up_blocks.append(up)

        # head
        self.head = nn.Conv3d(dims[0], num_classes, kernel_size=1)


    def forward(self, x):
        B, C, D, H, W = x.shape
        input_resolution = (D, H, W)  # Store original input resolution

        x = self.patch_embed(x)              # /2
        _, C1, D1, H1, W1 = x.shape

        res = [(D1, H1, W1)]
        for _ in range(3):
            d,h,w = res[-1]
            res.append((d//2, h//2, w//2))

        # update stage resolutions
        for stg, r in zip(self.stages, res):
            for blk in stg.blocks:
                blk.D, blk.H, blk.W = r

        # encoder
        feats = []
        out = x
        for stage_idx, stg in enumerate(self.stages):
            # Process through blocks
            for blk in stg.blocks:
                out = blk(out)
            
            # Store feature BEFORE downsampling for skip connections
            if stage_idx < 3:  # First 3 stages have downsampling
                feats.append(out)  # Store before downsampling
            
            # Apply downsampling if present
            if stg.downsample is not None:
                out = stg.downsample(out)
        
        # Store the final stage output (no downsampling)
        feats.append(out)

        # decoder
        x = feats[-1]
        for i, up in enumerate(self.up_blocks):
            x = up(x)
            skip = feats[2-i]
            if x.shape[2:] != skip.shape[2:]:
                x = torch.nn.functional.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = x + skip

        logits = self.head(x)
        
        # Upsample to original input resolution
        if logits.shape[2:] != input_resolution:
            logits = torch.nn.functional.interpolate(
                logits,
                size=input_resolution,
                mode="trilinear",
                align_corners=False
            )
        
        return logits

    def arch_parameters(self):
        """
        Fixed model has no architecture parameters (architecture is already fixed).
        Returns empty list for compatibility with FederatedServer.
        """
        return []
